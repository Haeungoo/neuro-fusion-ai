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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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

MODEL_PATH = (
    MODELS_DIR
    / "motor_imagery_physionet_csp_lda.joblib"
)


def load_physionet_motor_imagery(
    subject: int = 1,
    runs: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load PhysioNet EEGBCI motor imagery data.

    Runs 4, 8, 12 correspond to motor imagery:
    left fist vs right fist.

    Label convention used here:
    0 = left-hand imagery
    1 = right-hand imagery
    """

    if runs is None:
        runs = [4, 8, 12]

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

    raws = []

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
            7.0,
            30.0,
            fir_design="firwin",
            verbose=False,
        )

        raws.append(raw)

    raw = mne.concatenate_raws(
        raws,
    )

    events, event_id = mne.events_from_annotations(
        raw,
        verbose=False,
    )

    # For motor imagery runs 4, 8, 12:
    # T1 = left fist imagery
    # T2 = right fist imagery
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
        tmin=1.0,
        tmax=4.0,
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
        "subject": subject,
        "runs": runs,
        "num_trials": int(len(y)),
        "num_channels": int(X.shape[1]),
        "num_samples_per_trial": int(X.shape[2]),
        "sampling_frequency": float(epochs.info["sfreq"]),
        "label_0": "left_hand_imagery",
        "label_1": "right_hand_imagery",
    }

    return X, y, metadata


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


def save_predictions_csv(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trial_index",
                "y_true",
                "y_pred",
                "probability_right_hand",
            ],
        )

        writer.writeheader()

        for index, (
            true_label,
            predicted_label,
            probability,
        ) in enumerate(
            zip(
                y_true,
                y_pred,
                y_probability,
            )
        ):
            writer.writerow(
                {
                    "trial_index": index,
                    "y_true": int(true_label),
                    "y_pred": int(predicted_label),
                    "probability_right_hand": float(probability),
                }
            )


def save_confusion_matrix_image(
    path: Path,
    matrix: list[list[int]],
    title: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_array = np.asarray(matrix)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix_array)

    plt.title(title)
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


def train_physionet_csp_lda(
    subject: int = 1,
) -> dict[str, Any]:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y, metadata = load_physionet_motor_imagery(
        subject=subject,
        runs=[4, 8, 12],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    csp = CSP(
        n_components=4,
        reg=None,
        log=True,
        norm_trace=False,
    )

    classifier = Pipeline(
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

    classifier.fit(
        X_train,
        y_train,
    )

    y_pred = classifier.predict(
        X_test,
    )

    y_probability = classifier.predict_proba(
        X_test,
    )[:, 1]

    metrics = binary_classification_metrics(
        y_test,
        y_pred,
    )

    metrics.update(
        {
            "model": "MNE CSP + LDA",
            "dataset": metadata["dataset"],
            "subject": int(subject),
            "runs": metadata["runs"],
            "num_csp_components": 4,
            "num_channels": metadata["num_channels"],
            "num_samples_per_trial": metadata["num_samples_per_trial"],
            "sampling_frequency": metadata["sampling_frequency"],
            "num_train_trials": int(len(y_train)),
            "num_test_trials": int(len(y_test)),
        }
    )

    save_json(
        RESULTS_DIR / "motor_imagery_physionet_metrics.json",
        metrics,
    )

    save_predictions_csv(
        RESULTS_DIR / "motor_imagery_physionet_predictions.csv",
        y_test,
        y_pred,
        y_probability,
    )

    save_confusion_matrix_image(
        RESULTS_DIR / "motor_imagery_physionet_confusion_matrix.png",
        metrics["confusion_matrix"],
        title="PhysioNet Motor Imagery Confusion Matrix",
    )

    joblib.dump(
        {
            "model": classifier,
            "metadata": metadata,
        },
        MODEL_PATH,
    )

    return metrics


def also_write_dashboard_aliases() -> None:
    """
    Keep the existing dashboard working without frontend changes.

    The dashboard currently reads:
    motor_imagery_metrics.json
    motor_imagery_predictions.csv
    motor_imagery_confusion_matrix.png
    motor_imagery_csp_lda.joblib

    This function copies PhysioNet outputs to those generic names.
    """

    alias_pairs = [
        (
            RESULTS_DIR / "motor_imagery_physionet_metrics.json",
            RESULTS_DIR / "motor_imagery_metrics.json",
        ),
        (
            RESULTS_DIR / "motor_imagery_physionet_predictions.csv",
            RESULTS_DIR / "motor_imagery_predictions.csv",
        ),
        (
            RESULTS_DIR / "motor_imagery_physionet_confusion_matrix.png",
            RESULTS_DIR / "motor_imagery_confusion_matrix.png",
        ),
        (
            MODEL_PATH,
            MODELS_DIR / "motor_imagery_csp_lda.joblib",
        ),
    ]

    for source, target in alias_pairs:
        if source.exists():
            target.write_bytes(
                source.read_bytes(),
            )


def main() -> None:
    metrics = train_physionet_csp_lda(
        subject=7,
    )

    also_write_dashboard_aliases()

    print("[done] PhysioNet motor imagery CSP + LDA training completed.")
    print(f"[metrics] {RESULTS_DIR / 'motor_imagery_physionet_metrics.json'}")
    print(f"[predictions] {RESULTS_DIR / 'motor_imagery_physionet_predictions.csv'}")
    print(f"[confusion matrix] {RESULTS_DIR / 'motor_imagery_physionet_confusion_matrix.png'}")
    print(f"[model] {MODEL_PATH}")
    print(
        "[accuracy]",
        round(
            metrics["accuracy"],
            4,
        ),
    )


if __name__ == "__main__":
    main()