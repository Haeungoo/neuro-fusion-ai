from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.common.paths import PROJECT_ROOT
from src.mri_segmentation.dataset import MRISliceDataset
from src.mri_segmentation.losses import BCEDiceLoss
from src.mri_segmentation.metrics import binary_segmentation_metrics
from src.mri_segmentation.splits import (
    patient_level_split,
    save_split_manifest,
)
from src.mri_segmentation.unet2d import UNet2D


def _resolve_project_path(path: str | Path) -> Path:
    """
    Convert a relative path into a path under PROJECT_ROOT.

    Absolute paths are returned unchanged.
    """

    resolved_path = Path(path)

    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    return resolved_path


def _set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean(values: list[float]) -> float:
    """
    Calculate a safe mean.
    """

    if not values:
        return 0.0

    return float(sum(values) / len(values))


def _save_loss_curve(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """
    Save training and validation loss curves.
    """

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss or not val_loss:
        print(
            "Warning: loss history is empty. "
            "Training curve was not saved."
        )
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    epoch_numbers = range(1, len(train_loss) + 1)

    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    axis.plot(
        epoch_numbers,
        train_loss,
        marker="o",
        label="Training loss",
    )

    axis.plot(
        epoch_numbers,
        val_loss,
        marker="o",
        label="Validation loss",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("MRI 2D U-Net Training Curve")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def _save_validation_curve(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """
    Save validation Dice and IoU curves.
    """

    val_dice = history.get("val_dice", [])
    val_iou = history.get("val_iou", [])

    if not val_dice or not val_iou:
        print(
            "Warning: validation metric history is empty. "
            "Validation curve was not saved."
        )
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    epoch_numbers = range(1, len(val_dice) + 1)

    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    axis.plot(
        epoch_numbers,
        val_dice,
        marker="o",
        label="Validation Dice",
    )

    axis.plot(
        epoch_numbers,
        val_iou,
        marker="o",
        label="Validation IoU",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("MRI Validation Metrics")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def _calculate_batch_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float,
) -> tuple[list[float], list[float]]:
    """
    Calculate slice-level Dice and IoU for one validation batch.
    """

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    prediction_arrays = (
        predictions.detach().cpu().numpy()
    )

    mask_arrays = (
        masks.detach().cpu().numpy()
    )

    dice_scores: list[float] = []
    iou_scores: list[float] = []

    for prediction, target in zip(
        prediction_arrays,
        mask_arrays,
    ):
        # Expected shape for each sample:
        # prediction: (1, H, W)
        # target:     (1, H, W)

        prediction_mask = prediction[0]
        target_mask = target[0]

        metrics = binary_segmentation_metrics(
            prediction=prediction_mask,
            target=target_mask,
        )

        dice_scores.append(
            float(metrics["dice"])
        )

        iou_scores.append(
            float(metrics["iou"])
        )

    return dice_scores, iou_scores


def _save_training_metrics(
    output_path: Path,
    metrics: dict[str, Any],
) -> None:
    """
    Save training metrics as JSON.
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
            metrics,
            file,
            indent=2,
        )


def train_unet2d(
    image_dir: str | Path = "data/mri/processed/images",
    mask_dir: str | Path = "data/mri/processed/masks",
    model_path: str | Path = "models/mri_unet.pt",
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    in_channels: int = 1,
    base_channels: int = 32,
    seed: int = 42,
    threshold: float = 0.5,
    num_workers: int = 0,
) -> dict[str, Any]:
    """
    Train a 2D U-Net using patient-level train/validation splitting.

    Current input:
        One FLAIR channel

    Important:
        All slices belonging to one BraTS case remain entirely in
        either training or validation. A case cannot appear in both.
    """

    if epochs < 1:
        raise ValueError(
            f"epochs must be at least 1. Got {epochs}."
        )

    if batch_size < 1:
        raise ValueError(
            f"batch_size must be at least 1. Got {batch_size}."
        )

    if lr <= 0:
        raise ValueError(
            f"lr must be greater than 0. Got {lr}."
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1. "
            f"Got {threshold}."
        )

    if in_channels != 1:
        print(
            "Warning: this training configuration is currently "
            f"described as FLAIR-only, but in_channels={in_channels}."
        )

    _set_random_seed(seed)

    image_dir_path = _resolve_project_path(
        image_dir
    )

    mask_dir_path = _resolve_project_path(
        mask_dir
    )

    model_output_path = _resolve_project_path(
        model_path
    )

    results_dir = (
        PROJECT_ROOT
        / "results"
        / "mri"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # 1. Load paired MRI slices and masks
    # -------------------------------------------------
    dataset = MRISliceDataset(
        image_dir=image_dir_path,
        mask_dir=mask_dir_path,
    )

    if not hasattr(dataset, "image_paths"):
        raise AttributeError(
            "MRISliceDataset must define dataset.image_paths. "
            "Add self.image_paths inside MRISliceDataset.__init__()."
        )

    # -------------------------------------------------
    # 2. Patient-level train/validation split
    # -------------------------------------------------
    (
        train_indices,
        val_indices,
        train_cases,
        val_cases,
    ) = patient_level_split(
        image_paths=dataset.image_paths,
        val_fraction=val_fraction,
        seed=seed,
    )

    if not train_indices:
        raise RuntimeError(
            "Patient-level split produced no training slices."
        )

    if not val_indices:
        raise RuntimeError(
            "Patient-level split produced no validation slices."
        )

    train_case_set = set(train_cases)
    val_case_set = set(val_cases)

    overlapping_cases = (
        train_case_set.intersection(val_case_set)
    )

    if overlapping_cases:
        raise RuntimeError(
            "Patient-level data leakage detected. "
            f"Cases in both splits: {sorted(overlapping_cases)}"
        )

    train_dataset = Subset(
        dataset,
        train_indices,
    )

    val_dataset = Subset(
        dataset,
        val_indices,
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    # -------------------------------------------------
    # 3. Save reproducible split manifest
    # -------------------------------------------------
    split_manifest_path = (
        results_dir
        / "mri_split_manifest.json"
    )

    save_split_manifest(
        output_path=split_manifest_path,
        train_cases=train_cases,
        val_cases=val_cases,
        train_indices=train_indices,
        val_indices=val_indices,
        seed=seed,
        val_fraction=val_fraction,
    )

    # -------------------------------------------------
    # 4. Model, loss, optimizer
    # -------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = UNet2D(
        in_channels=in_channels,
        out_channels=1,
        base_channels=base_channels,
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }

    best_epoch = 0
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_val_loss = float("inf")

    print("=" * 68)
    print("MRI 2D U-Net Patient-Level Training")
    print("=" * 68)
    print("Modality: FLAIR")
    print("Input channels:", in_channels)
    print("Split type: patient-level")
    print("Random seed:", seed)
    print("Validation fraction:", val_fraction)
    print("Threshold:", threshold)
    print("Device:", device)
    print("-" * 68)
    print("Total cases:", len(train_cases) + len(val_cases))
    print("Training cases:", len(train_cases))
    print("Validation cases:", len(val_cases))
    print("Total slices:", len(dataset))
    print("Training slices:", len(train_indices))
    print("Validation slices:", len(val_indices))
    print("-" * 68)
    print("Training case IDs:")
    print(train_cases)
    print("Validation case IDs:")
    print(val_cases)
    print("=" * 68)

    # -------------------------------------------------
    # 5. Epoch loop
    # -------------------------------------------------
    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1

        # =============================================
        # Training phase
        # =============================================
        model.train()

        train_losses: list[float] = []

        train_progress = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch_number}/{epochs} "
                "training"
            ),
        )

        for images, masks in train_progress:
            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss detected: "
                    f"{loss.item()}"
                )

            loss.backward()
            optimizer.step()

            loss_value = float(
                loss.item()
            )

            train_losses.append(
                loss_value
            )

            train_progress.set_postfix(
                loss=f"{loss_value:.4f}"
            )

        # =============================================
        # Validation phase
        # =============================================
        model.eval()

        val_losses: list[float] = []
        val_dice_scores: list[float] = []
        val_iou_scores: list[float] = []

        val_progress = tqdm(
            val_loader,
            desc=(
                f"Epoch {epoch_number}/{epochs} "
                "validation"
            ),
        )

        with torch.no_grad():
            for images, masks in val_progress:
                images = images.to(
                    device,
                    non_blocking=True,
                )

                masks = masks.to(
                    device,
                    non_blocking=True,
                )

                logits = model(images)

                loss = criterion(
                    logits,
                    masks,
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite validation loss detected: "
                        f"{loss.item()}"
                    )

                loss_value = float(
                    loss.item()
                )

                val_losses.append(
                    loss_value
                )

                batch_dice, batch_iou = (
                    _calculate_batch_metrics(
                        logits=logits,
                        masks=masks,
                        threshold=threshold,
                    )
                )

                val_dice_scores.extend(
                    batch_dice
                )

                val_iou_scores.extend(
                    batch_iou
                )

                val_progress.set_postfix(
                    loss=f"{loss_value:.4f}"
                )

        train_loss = _mean(
            train_losses
        )

        val_loss = _mean(
            val_losses
        )

        val_dice = _mean(
            val_dice_scores
        )

        val_iou = _mean(
            val_iou_scores
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_dice"].append(
            val_dice
        )

        history["val_iou"].append(
            val_iou
        )

        print(
            f"Epoch {epoch_number:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_iou={val_iou:.4f}"
        )

        # -------------------------------------------------
        # Save checkpoint with the best validation Dice
        # -------------------------------------------------
        if val_dice > best_val_dice:
            best_epoch = epoch_number
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "epoch": int(epoch_number),
                "in_channels": int(in_channels),
                "out_channels": 1,
                "base_channels": int(base_channels),
                "threshold": float(threshold),
                "best_val_dice": float(
                    best_val_dice
                ),
                "best_val_iou": float(
                    best_val_iou
                ),
                "best_val_loss": float(
                    best_val_loss
                ),
                "modality": "FLAIR",
                "split_type": "patient_level",
                "seed": int(seed),
                "val_fraction": float(
                    val_fraction
                ),
                "train_cases": list(
                    train_cases
                ),
                "validation_cases": list(
                    val_cases
                ),
            }

            torch.save(
                checkpoint,
                model_output_path,
            )

            print(
                "Saved best checkpoint:",
                model_output_path,
            )

    # -------------------------------------------------
    # 6. Save training figures
    # -------------------------------------------------
    training_curve_path = (
        results_dir
        / "training_curve_mri_unet.png"
    )

    validation_curve_path = (
        results_dir
        / "validation_curve_mri_unet.png"
    )

    _save_loss_curve(
        history=history,
        output_path=training_curve_path,
    )

    _save_validation_curve(
        history=history,
        output_path=validation_curve_path,
    )

    # -------------------------------------------------
    # 7. Save training summary JSON
    # -------------------------------------------------
    metrics_path = (
        results_dir
        / "mri_training_metrics.json"
    )

    training_metrics: dict[str, Any] = {
        "split_type": "patient_level",
        "modality": "FLAIR",
        "input_channels": int(
            in_channels
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_val_dice": float(
            best_val_dice
        ),
        "best_val_iou": float(
            best_val_iou
        ),
        "best_val_loss": float(
            best_val_loss
        ),
        "epochs": int(
            epochs
        ),
        "batch_size": int(
            batch_size
        ),
        "learning_rate": float(
            lr
        ),
        "threshold": float(
            threshold
        ),
        "seed": int(
            seed
        ),
        "val_fraction": float(
            val_fraction
        ),
        "num_samples": int(
            len(dataset)
        ),
        "train_samples": int(
            len(train_indices)
        ),
        "val_samples": int(
            len(val_indices)
        ),
        "num_train_cases": int(
            len(train_cases)
        ),
        "num_validation_cases": int(
            len(val_cases)
        ),
        "train_cases": list(
            train_cases
        ),
        "validation_cases": list(
            val_cases
        ),
        "history": history,
        "files": {
            "model": str(
                model_output_path
            ),
            "split_manifest": str(
                split_manifest_path
            ),
            "training_curve": str(
                training_curve_path
            ),
            "validation_curve": str(
                validation_curve_path
            ),
        },
    }

    _save_training_metrics(
        output_path=metrics_path,
        metrics=training_metrics,
    )

    print()
    print("=" * 68)
    print("Training complete")
    print("=" * 68)
    print("Best epoch:", best_epoch)
    print(
        "Best validation Dice:",
        f"{best_val_dice:.4f}",
    )
    print(
        "Best validation IoU:",
        f"{best_val_iou:.4f}",
    )
    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}",
    )
    print("Model:", model_output_path)
    print("Split manifest:", split_manifest_path)
    print("Training metrics:", metrics_path)
    print("Loss curve:", training_curve_path)
    print("Validation curve:", validation_curve_path)

    return {
        "model_path": str(
            model_output_path
        ),
        "training_metrics_path": str(
            metrics_path
        ),
        "split_manifest_path": str(
            split_manifest_path
        ),
        "training_curve_path": str(
            training_curve_path
        ),
        "validation_curve_path": str(
            validation_curve_path
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_val_dice": float(
            best_val_dice
        ),
        "best_val_iou": float(
            best_val_iou
        ),
        "best_val_loss": float(
            best_val_loss
        ),
    }