from __future__ import annotations

import numpy as np
import torch


def binary_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-7,
) -> dict[str, float]:
    """
    Calculate binary segmentation metrics.

    Both prediction and target must already be binary arrays.
    """

    pred = prediction.astype(bool).reshape(-1)
    truth = target.astype(bool).reshape(-1)

    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    tn = float(np.logical_and(~pred, ~truth).sum())

    pred_positive = tp + fp
    truth_positive = tp + fn

    if pred_positive == 0 and truth_positive == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * tp) / (
            2.0 * tp + fp + fn + epsilon
        )
        iou = tp / (
            tp + fp + fn + epsilon
        )

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    accuracy = (tp + tn) / (
        tp + tn + fp + fn + epsilon
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "true_positive_pixels": int(tp),
        "false_positive_pixels": int(fp),
        "false_negative_pixels": int(fn),
        "true_negative_pixels": int(tn),
        "predicted_positive_pixels": int(pred_positive),
        "ground_truth_positive_pixels": int(truth_positive),
    }


def logits_to_binary_mask(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert raw model logits into a binary segmentation mask.
    """

    probabilities = torch.sigmoid(logits)
    return (probabilities >= threshold).float()