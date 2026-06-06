from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Input:
        logits: raw model output, shape (B, 1, H, W)
        targets: binary mask, shape (B, 1, H, W)

    Dice score:
        2 * intersection / (prediction + target)
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )

        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss + DiceLoss.

    This is commonly used for binary medical image segmentation.

    Total loss:
        BCE loss + Dice loss
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice_loss(logits, targets)

        loss = self.bce_weight * bce + self.dice_weight * dice

        return loss


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Calculate Dice score for binary segmentation.

    This is for evaluation, not training.
    """

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.contiguous().view(preds.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return dice.mean()
