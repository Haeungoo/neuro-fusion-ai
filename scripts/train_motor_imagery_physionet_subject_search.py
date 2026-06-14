from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np

from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.motor_imagery.metrics import binary_classification_metrics


try:
    from src.common.paths import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "motor_imagery"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "motor_imagery"
    / "physionet"
)


def load_physionet_subject(
    subject: int,
    runs: list[int],
    low_freq: float,
    high_freq: float,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load one PhysioNet EEGBCI subject.

    Label convention:
    0 = left-hand imagery
    1 = right-hand imagery
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_paths = eegbci.load_data(
        subjects=[subject],
        runs=runs,
        path=DATA_DIR,
        verbose=False,
    )

    raw_list = []

    for file_path in file_paths:
        raw = mne.io.read_raw_edf(
            file_path,
            preload=True,
            verbose=False,
        )

        eegbci.standardize(raw)

        raw.set_montage(
            "standard_1005",
            on_missing="ignore",
        )

        raw.filter(
            low_freq,
            high_freq,
            fir_design="firwin",
            verbose=False,
        )

        raw_list.append(raw)

    raw = mne.concatenate_raws(
        raw_list,
    )

    events, event_id = mne.events_from_annotations(
        raw,
        verbose=False,
    )

    if "T1" not in event_id or "T2" not in event_id:
        raise ValueError(
            f"Subject {subject} does not contain T1/T2 events."
        )

    selected_event_id = {
        "left_hand": event_id["T1"],
        "right_hand": event_id["T2"],
    }

    picks = mne.pick_types(
        raw.info,
        eeg=True,
        meg=False,
        stim=False,
        eog=False,
        exclude="bads",
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected_event_id,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()

    raw_labels = epochs.events[:, -1]

    y = np.zeros(
        len(raw_labels),
        dtype=int,
    )

    y[raw_labels == selected_event_id["left_hand"]] = 0
    y[raw_labels == selected_event_id["right_hand"]] = 1

    metadata = {
        "dataset": "physionet_eegbci_motor_imagery",
        "subject": int(subject),
        "runs": runs,
        "low_freq": float(low_freq),
        "high_freq": float(high_freq),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "num_trials": int(len(y)),
        "num_channels": int(X.shape[1]),
        "num_samples_per_trial": int(X.shape[2]),
        "sampling_frequency": float(epochs.info["sfreq"]),
        "label_0": "left_hand_imagery",
        "label_1": "right_hand_imagery",
    }

    return X, y, metadata


def build_pipeline(
    n_components: int,
) -> Pipeline:
    csp = CSP(
        n_components=n_components,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False,
    )

    pipeline = Pipeline(
        steps=[
            (
                "csp",
                csp,
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "lda",
                LinearDiscriminantAnalysis(),
            ),
        ]
    )

    return pipeline


def evaluate_subject(
    subject: int,
    runs: list[int],
    low_freq: float,
    high_freq: float,
    tmin: float,
    tmax: float,
    n_components: int,
    n_splits: int = 5,
) -> dict[str, Any]:
    X, y, metadata = load_physionet_subject(
        subject=subject,
        runs=runs,
        low_freq=low_freq,
        high_freq=high_freq,
        tmin=tmin,
        tmax=tmax,
    )

    pipeline = build_pipeline(
        n_components=n_components,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    y_pred = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict",
    )

    metrics = binary_classification_metrics(
        y,
        y_pred,
    )

    metrics.update(
        {
            "model": "MNE CSP + LDA",
            "dataset": metadata["dataset"],
            "subject": int(subject),
            "runs": runs,
            "low_freq": float(low_freq),
            "high_freq": float(high_freq),
            "tmin": float(tmin),
            "tmax": float(tmax),
            "num_csp_components": int(n_components),
            "num_channels": metadata["num_channels"],
            "num_samples_per_trial": metadata["num_samples_per_trial"],
            "sampling_frequency": metadata["sampling_frequency"],
            "num_cv_splits": int(n_splits),
        }
    )

    return metrics


def save_subject_comparison_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "subject",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
        "balanced_accuracy",
        "num_trials",
        "num_csp_components",
        "low_freq",
        "high_freq",
        "tmin",
        "tmax",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "subject": row.get("subject"),
                    "accuracy": row.get("accuracy"),
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "specificity": row.get("specificity"),
                    "f1_score": row.get("f1_score"),
                    "balanced_accuracy": row.get("balanced_accuracy"),
                    "num_trials": row.get("num_trials"),
                    "num_csp_components": row.get("num_csp_components"),
                    "low_freq": row.get("low_freq"),
                    "high_freq": row.get("high_freq"),
                    "tmin": row.get("tmin"),
                    "tmax": row.get("tmax"),
                }
            )


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def save_comparison_bar_chart(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subjects = [
        str(row["subject"])
        for row in rows
    ]

    accuracies = [
        row["accuracy"]
        for row in rows
    ]

    plt.figure(figsize=(10, 4))
    plt.bar(
        subjects,
        accuracies,
    )

    plt.ylim(0.0, 1.0)
    plt.title("PhysioNet Motor Imagery Subject Search")
    plt.xlabel("Subject")
    plt.ylabel("Accuracy")

    for index, accuracy in enumerate(accuracies):
        plt.text(
            index,
            accuracy + 0.02,
            f"{accuracy:.2f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=160,
    )
    plt.close()


def save_confusion_matrix_image(
    path: Path,
    matrix: list[list[int]],
    subject: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_array = np.asarray(matrix)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix_array)

    plt.title(
        f"Best Subject {subject}: Confusion Matrix"
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    labels = [
        "Left hand",
        "Right hand",
    ]

    plt.xticks(
        [0, 1],
        labels,
        rotation=20,
        ha="right",
    )

    plt.yticks(
        [0, 1],
        labels,
    )

    for row in range(matrix_array.shape[0]):
        for col in range(matrix_array.shape[1]):
            plt.text(
                col,
                row,
                str(matrix_array[row, col]),
                ha="center",
                va="center",
                fontsize=14,
            )

    plt.colorbar(
        fraction=0.046,
        pad=0.04,
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=160,
    )
    plt.close()


def train_final_model_for_best_subject(
    best_subject: int,
    runs: list[int],
    low_freq: float,
    high_freq: float,
    tmin: float,
    tmax: float,
    n_components: int,
) -> None:
    X, y, metadata = load_physionet_subject(
        subject=best_subject,
        runs=runs,
        low_freq=low_freq,
        high_freq=high_freq,
        tmin=tmin,
        tmax=tmax,
    )

    pipeline = build_pipeline(
        n_components=n_components,
    )

    pipeline.fit(
        X,
        y,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = {
        "model": pipeline,
        "metadata": metadata,
    }

    joblib.dump(
        model_bundle,
        MODELS_DIR / "motor_imagery_csp_lda.joblib",
    )

    joblib.dump(
        model_bundle,
        MODELS_DIR / "motor_imagery_physionet_subject_search.joblib",
    )


def write_dashboard_outputs(
    best_metrics: dict[str, Any],
) -> None:
    """
    Write generic output names used by the existing dashboard.
    """

    save_json(
        RESULTS_DIR / "motor_imagery_metrics.json",
        best_metrics,
    )

    save_confusion_matrix_image(
        RESULTS_DIR / "motor_imagery_confusion_matrix.png",
        best_metrics["confusion_matrix"],
        subject=int(best_metrics["subject"]),
    )

    predictions_path = (
        RESULTS_DIR
        / "motor_imagery_predictions.csv"
    )

    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        predictions_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "note",
                "subject",
                "accuracy",
                "f1_score",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "note": (
                    "Subject search used cross_val_predict; "
                    "trial-level probabilities are not saved in this version."
                ),
                "subject": best_metrics["subject"],
                "accuracy": best_metrics["accuracy"],
                "f1_score": best_metrics["f1_score"],
            }
        )


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    subjects = list(
        range(1, 11)
    )

    runs = [
        4,
        8,
        12,
    ]

    low_freq = 8.0
    high_freq = 30.0
    tmin = 0.5
    tmax = 3.5
    n_components = 6

    rows: list[dict[str, Any]] = []

    for subject in subjects:
        print(f"[running] subject {subject}")

        try:
            metrics = evaluate_subject(
                subject=subject,
                runs=runs,
                low_freq=low_freq,
                high_freq=high_freq,
                tmin=tmin,
                tmax=tmax,
                n_components=n_components,
                n_splits=5,
            )

            rows.append(metrics)

            print(
                "[done]",
                "subject",
                subject,
                "accuracy",
                round(metrics["accuracy"], 4),
                "f1",
                round(metrics["f1_score"], 4),
            )

        except Exception as error:
            print(
                "[skip]",
                "subject",
                subject,
                "error:",
                str(error),
            )

    if not rows:
        raise RuntimeError(
            "No subjects were successfully evaluated."
        )

    rows = sorted(
        rows,
        key=lambda item: (
            item["accuracy"],
            item["f1_score"],
        ),
        reverse=True,
    )

    best_metrics = rows[0]

    save_subject_comparison_csv(
        RESULTS_DIR / "physionet_subject_comparison.csv",
        rows,
    )

    save_json(
        RESULTS_DIR / "physionet_subject_comparison_best.json",
        best_metrics,
    )

    save_comparison_bar_chart(
        RESULTS_DIR / "physionet_subject_comparison_accuracy.png",
        rows,
    )

    write_dashboard_outputs(
        best_metrics,
    )

    train_final_model_for_best_subject(
        best_subject=int(best_metrics["subject"]),
        runs=runs,
        low_freq=low_freq,
        high_freq=high_freq,
        tmin=tmin,
        tmax=tmax,
        n_components=n_components,
    )

    print("[complete] PhysioNet subject search finished.")
    print(
        "[best subject]",
        best_metrics["subject"],
        "accuracy:",
        round(best_metrics["accuracy"], 4),
        "f1:",
        round(best_metrics["f1_score"], 4),
    )

    print(
        "[comparison csv]",
        RESULTS_DIR / "physionet_subject_comparison.csv",
    )

    print(
        "[dashboard metrics]",
        RESULTS_DIR / "motor_imagery_metrics.json",
    )


if __name__ == "__main__":
    main()