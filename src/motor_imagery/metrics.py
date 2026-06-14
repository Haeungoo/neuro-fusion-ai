from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate binary classification metrics for motor imagery.

    Label convention:
    0 = left-hand motor imagery
    1 = right-hand motor imagery
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    sensitivity = recall

    return {
        "task": "EEG motor imagery classification",
        "label_0": "left_hand_imagery",
        "label_1": "right_hand_imagery",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "confusion_matrix": matrix.tolist(),
        "num_trials": int(len(y_true)),
        "num_left_hand_trials": int(np.sum(y_true == 0)),
        "num_right_hand_trials": int(np.sum(y_true == 1)),
    }


def multiclass_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Optional helper for future multi-class motor imagery tasks.

    Example:
    0 = left hand
    1 = right hand
    2 = feet
    3 = tongue
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    return {
        "task": "EEG motor imagery multiclass classification",
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1_score": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "confusion_matrix": matrix.tolist(),
        "num_trials": int(len(y_true)),
        "classes": sorted(np.unique(y_true).astype(int).tolist()),
    }