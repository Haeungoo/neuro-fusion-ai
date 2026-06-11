from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split


def save_window_predictions_csv(
    output_path: Path,
    file_names: list[str],
    window_start_times: list[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> Path:
    """
    Save window-level seizure predictions for later evaluation.
    """

    if not (
        len(file_names)
        == len(window_start_times)
        == len(y_true)
        == len(y_pred)
        == len(probabilities)
    ):
        raise ValueError(
            "Prediction metadata lengths do not match: "
            f"file_names={len(file_names)}, "
            f"window_start_times={len(window_start_times)}, "
            f"y_true={len(y_true)}, "
            f"y_pred={len(y_pred)}, "
            f"probabilities={len(probabilities)}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file_name",
                "window_start_sec",
                "y_true",
                "y_pred",
                "probability",
            ],
        )

        writer.writeheader()

        for (
            file_name,
            window_start_sec,
            true_label,
            predicted_label,
            probability,
        ) in zip(
            file_names,
            window_start_times,
            y_true,
            y_pred,
            probabilities,
        ):
            writer.writerow(
                {
                    "file_name": str(file_name),
                    "window_start_sec": float(
                        window_start_sec
                    ),
                    "y_true": int(true_label),
                    "y_pred": int(predicted_label),
                    "probability": float(probability),
                }
            )

    return output_path


def save_json(
    output_path: Path,
    data: dict[str, Any],
) -> None:
    """
    Save dictionary as JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def save_confusion_matrix_image(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Random Forest Seizure Detection",
) -> None:
    """
    Save binary confusion matrix image.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Non-seizure",
            "Seizure",
        ],
    )

    figure, axis = plt.subplots(
        figsize=(5.5, 5),
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=True,
        values_format="d",
    )

    axis.set_title(title)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def calculate_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    """
    Calculate binary classification metrics.

    Label convention:
        0 = non-seizure
        1 = seizure
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    ).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )

    if y_true.size == 0:
        raise ValueError(
            "Cannot calculate metrics from empty arrays."
        )

    true_positive = int(
        np.logical_and(
            y_true == 1,
            y_pred == 1,
        ).sum()
    )

    true_negative = int(
        np.logical_and(
            y_true == 0,
            y_pred == 0,
        ).sum()
    )

    false_positive = int(
        np.logical_and(
            y_true == 0,
            y_pred == 1,
        ).sum()
    )

    false_negative = int(
        np.logical_and(
            y_true == 1,
            y_pred == 0,
        ).sum()
    )

    epsilon = 1e-12

    sensitivity = true_positive / (
        true_positive + false_negative + epsilon
    )

    recall = sensitivity

    specificity = true_negative / (
        true_negative + false_positive + epsilon
    )

    precision = true_positive / (
        true_positive + false_positive + epsilon
    )

    negative_predictive_value = true_negative / (
        true_negative + false_negative + epsilon
    )

    f1_score = (
        2.0 * precision * recall
        / (precision + recall + epsilon)
    )

    accuracy = (
        true_positive + true_negative
    ) / (
        true_positive
        + true_negative
        + false_positive
        + false_negative
        + epsilon
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    prevalence = (
        true_positive + false_negative
    ) / (
        true_positive
        + true_negative
        + false_positive
        + false_negative
        + epsilon
    )

    false_positive_rate = false_positive / (
        false_positive + true_negative + epsilon
    )

    false_negative_rate = false_negative / (
        false_negative + true_positive + epsilon
    )

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "sensitivity": float(sensitivity),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "negative_predictive_value": float(
            negative_predictive_value
        ),
        "f1_score": float(f1_score),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "prevalence": float(prevalence),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "num_samples": int(y_true.size),
        "num_positive_samples": int(
            (y_true == 1).sum()
        ),
        "num_negative_samples": int(
            (y_true == 0).sum()
        ),
    }


def load_feature_dataset(
    feature_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str] | None,
    list[float] | None,
]:
    """
    Load seizure feature dataset from .npz.

    Required arrays:
        X
        y

    Optional arrays:
        file_names
        window_start_times
    """

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}\n\n"
            "Expected file:\n"
            "results/seizure/chbmit_multi_file_features.npz\n\n"
            "Required arrays inside the .npz:\n"
            "  X\n"
            "  y\n\n"
            "Optional arrays:\n"
            "  file_names\n"
            "  window_start_times"
        )

    data = np.load(
        feature_path,
        allow_pickle=True,
    )

    if "X" not in data or "y" not in data:
        raise KeyError(
            f"{feature_path} must contain arrays named "
            "'X' and 'y'."
        )

    X = np.asarray(
        data["X"],
        dtype=np.float32,
    )

    y = np.asarray(
        data["y"],
        dtype=np.int64,
    ).reshape(-1)

    if X.ndim != 2:
        raise ValueError(
            f"X must be a 2D feature matrix. "
            f"Got shape {X.shape}."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Feature and label length mismatch: "
            f"X has {X.shape[0]} rows, "
            f"y has {y.shape[0]} labels."
        )

    file_names: list[str] | None = None
    window_start_times: list[float] | None = None

    if "file_names" in data:
        file_names = [
            str(value)
            for value in data["file_names"]
        ]

    if "window_start_times" in data:
        window_start_times = [
            float(value)
            for value in data[
                "window_start_times"
            ]
        ]

    return (
        X,
        y,
        file_names,
        window_start_times,
    )


def train_random_forest_seizure(
    X: np.ndarray,
    y: np.ndarray,
    file_names: list[str] | None = None,
    window_start_times: list[float] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
) -> tuple[
    RandomForestClassifier,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[float],
]:
    """
    Train a Random Forest seizure detector.

    Returns:
        model
        X_test
        y_test
        y_pred
        y_prob
        test_file_names
        test_window_start_times
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            f"test_size must be between 0 and 1. "
            f"Got {test_size}."
        )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.int64,
    ).reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y length mismatch: "
            f"X={X.shape[0]}, y={y.shape[0]}"
        )

    unique_labels = set(
        np.unique(y).tolist()
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Labels must be binary 0/1. Got {unique_labels}."
        )

    if len(unique_labels) < 2:
        raise ValueError(
            "Training requires both classes: "
            "0 = non-seizure and 1 = seizure."
        )

    if file_names is None:
        file_names = [
            "chbmit_multi_file"
            for _ in range(len(y))
        ]

    if window_start_times is None:
        window_start_times = [
            index * 2.5
            for index in range(len(y))
        ]

    if not (
        len(X)
        == len(y)
        == len(file_names)
        == len(window_start_times)
    ):
        raise ValueError(
            "Input lengths do not match: "
            f"X={len(X)}, y={len(y)}, "
            f"file_names={len(file_names)}, "
            f"window_start_times={len(window_start_times)}"
        )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_file_names,
        test_file_names,
        train_window_start_times,
        test_window_start_times,
    ) = train_test_split(
        X,
        y,
        file_names,
        window_start_times,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    return (
        model,
        X_test,
        y_test,
        y_pred,
        y_prob,
        list(test_file_names),
        [
            float(value)
            for value in test_window_start_times
        ],
    )


def train_and_save_random_forest_seizure(
    feature_path: Path,
    model_path: Path,
    prediction_csv_path: Path,
    metrics_path: Path,
    confusion_matrix_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
) -> dict[str, Any]:
    """
    Full Random Forest seizure training pipeline.
    """

    (
        X,
        y,
        file_names,
        window_start_times,
    ) = load_feature_dataset(
        feature_path=feature_path,
    )

    (
        model,
        X_test,
        y_test,
        y_pred,
        y_prob,
        test_file_names,
        test_window_start_times,
    ) = train_random_forest_seizure(
        X=X,
        y=y,
        file_names=file_names,
        window_start_times=window_start_times,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    save_window_predictions_csv(
        output_path=prediction_csv_path,
        file_names=test_file_names,
        window_start_times=test_window_start_times,
        y_true=y_test,
        y_pred=y_pred,
        probabilities=y_prob,
    )

    metrics = calculate_basic_metrics(
        y_true=y_test,
        y_pred=y_pred,
    )

    metrics.update(
        {
            "model": "RandomForestClassifier",
            "feature_file": str(feature_path),
            "model_file": str(model_path),
            "prediction_csv": str(
                prediction_csv_path
            ),
            "test_size": float(test_size),
            "random_state": int(random_state),
            "n_estimators": int(n_estimators),
            "num_train_samples": int(
                len(y) - len(y_test)
            ),
            "num_test_samples": int(
                len(y_test)
            ),
            "num_features": int(
                X.shape[1]
            ),
            "probability_min": float(
                np.min(y_prob)
            ),
            "probability_max": float(
                np.max(y_prob)
            ),
            "probability_mean": float(
                np.mean(y_prob)
            ),
        }
    )

    save_json(
        output_path=metrics_path,
        data=metrics,
    )

    save_confusion_matrix_image(
        y_true=y_test,
        y_pred=y_pred,
        output_path=confusion_matrix_path,
    )

    return {
        "model_path": str(model_path),
        "prediction_csv_path": str(
            prediction_csv_path
        ),
        "metrics_path": str(metrics_path),
        "confusion_matrix_path": str(
            confusion_matrix_path
        ),
        "metrics": metrics,
    }