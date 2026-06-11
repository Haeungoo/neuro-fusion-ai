from __future__ import annotations

from typing import Sequence

import numpy as np


def binary_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, float | int]:
    """
    Calculate binary classification metrics.

    Expected labels:
        0 = non-seizure
        1 = seizure
    """

    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if truth.shape != prediction.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Got {truth.shape} and {prediction.shape}."
        )

    if truth.size == 0:
        raise ValueError("Cannot calculate metrics from empty arrays.")

    valid_labels = {0, 1}

    if not set(np.unique(truth)).issubset(valid_labels):
        raise ValueError("y_true must contain only 0 and 1.")

    if not set(np.unique(prediction)).issubset(valid_labels):
        raise ValueError("y_pred must contain only 0 and 1.")

    tp = int(np.logical_and(truth == 1, prediction == 1).sum())
    tn = int(np.logical_and(truth == 0, prediction == 0).sum())
    fp = int(np.logical_and(truth == 0, prediction == 1).sum())
    fn = int(np.logical_and(truth == 1, prediction == 0).sum())

    epsilon = 1e-12

    sensitivity = tp / (tp + fn + epsilon)
    recall = sensitivity
    specificity = tn / (tn + fp + epsilon)
    precision = tp / (tp + fp + epsilon)
    negative_predictive_value = tn / (tn + fn + epsilon)

    f1_score = (
        2.0 * precision * recall
        / (precision + recall + epsilon)
    )

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn + epsilon)
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    prevalence = (
        (tp + fn)
        / (tp + tn + fp + fn + epsilon)
    )

    false_positive_rate = fp / (fp + tn + epsilon)
    false_negative_rate = fn / (fn + tp + epsilon)

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "sensitivity": float(sensitivity),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "negative_predictive_value": float(
            negative_predictive_value
        ),
        "f1_score": float(f1_score),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "prevalence": float(prevalence),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "num_samples": int(truth.size),
        "num_positive_samples": int((truth == 1).sum()),
        "num_negative_samples": int((truth == 0).sum()),
    }


def false_alarms_per_hour(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    step_seconds: float,
    merge_consecutive_windows: bool = True,
) -> dict[str, float | int]:
    """
    Estimate false alarms per hour from window-level predictions.

    A false alarm is a predicted seizure window while the true label
    is non-seizure.

    When merge_consecutive_windows=True, consecutive false-positive
    windows are counted as one false-alarm event.
    """

    if step_seconds <= 0:
        raise ValueError(
            f"step_seconds must be positive. Got {step_seconds}."
        )

    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if truth.shape != prediction.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    false_positive_mask = np.logical_and(
        truth == 0,
        prediction == 1,
    )

    false_positive_windows = int(false_positive_mask.sum())

    if merge_consecutive_windows:
        padded = np.concatenate(
            [
                np.array([False]),
                false_positive_mask,
            ]
        )

        false_alarm_events = int(
            np.logical_and(
                padded[1:],
                ~padded[:-1],
            ).sum()
        )
    else:
        false_alarm_events = false_positive_windows

    non_seizure_windows = int((truth == 0).sum())
    non_seizure_duration_seconds = (
        non_seizure_windows * step_seconds
    )
    non_seizure_duration_hours = (
        non_seizure_duration_seconds / 3600.0
    )

    if non_seizure_duration_hours <= 0:
        rate = 0.0
    else:
        rate = (
            false_alarm_events
            / non_seizure_duration_hours
        )

    return {
        "false_positive_windows": false_positive_windows,
        "false_alarm_events": false_alarm_events,
        "non_seizure_windows": non_seizure_windows,
        "non_seizure_duration_seconds": float(
            non_seizure_duration_seconds
        ),
        "non_seizure_duration_hours": float(
            non_seizure_duration_hours
        ),
        "false_alarms_per_hour": float(rate),
        "merge_consecutive_windows": bool(
            merge_consecutive_windows
        ),
        "step_seconds": float(step_seconds),
    }


def evaluate_binary_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    step_seconds: float,
    merge_consecutive_windows: bool = True,
) -> dict:
    """
    Return classification metrics and false-alarm metrics together.
    """

    return {
        "classification": binary_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
        ),
        "false_alarm_analysis": false_alarms_per_hour(
            y_true=y_true,
            y_pred=y_pred,
            step_seconds=step_seconds,
            merge_consecutive_windows=merge_consecutive_windows,
        ),
    }