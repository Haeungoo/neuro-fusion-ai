from __future__ import annotations
import numpy as np


def normalize_mri_slice(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    mask = image != 0
    if np.any(mask):
        image = (image - image[mask].mean()) / (image[mask].std() + 1e-8)
    else:
        image = (image - image.mean()) / (image.std() + 1e-8)
    return image.astype(np.float32)


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-8) -> float:
    pred = np.asarray(pred_mask).astype(bool)
    true = np.asarray(true_mask).astype(bool)
    inter = np.logical_and(pred, true).sum()
    return float((2 * inter + eps) / (pred.sum() + true.sum() + eps))
