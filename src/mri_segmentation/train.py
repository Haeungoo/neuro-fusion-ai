from __future__ import annotations

from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.common.paths import PROJECT_ROOT
from src.common.plotting import save_training_curve
from src.mri_segmentation.dataset import MRISliceDataset
from src.mri_segmentation.unet2d import UNet2D
from src.mri_segmentation.losses import BCEDiceLoss


def train_unet2d(
    image_dir: str | Path = "data/mri/processed/images",
    mask_dir: str | Path = "data/mri/processed/masks",
    model_path: str | Path = "models/mri_unet.pt",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    in_channels: int = 1,
    base_channels: int = 32,
    seed: int = 42,
) -> dict:
    """
    Train a 2D U-Net on processed MRI slice/mask pairs.

    Expected input:
        data/mri/processed/images/*.npy
        data/mri/processed/masks/*.npy

    Expected output:
        models/mri_unet.pt
        results/mri/training_curve_mri_unet.png
        results/mri/mri_training_metrics.json
    """

    image_dir = PROJECT_ROOT / image_dir
    mask_dir = PROJECT_ROOT / mask_dir

    dataset = MRISliceDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
    )

    if len(dataset) < 2:
        raise ValueError(
            f"Need at least 2 image/mask pairs for train/validation split. "
            f"Found {len(dataset)}."
        )

    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size

    if train_size < 1:
        raise ValueError(
            f"Not enough data for training. train_size={train_size}, val_size={val_size}"
        )

    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet2D(
        in_channels=in_channels,
        out_channels=1,
        base_channels=base_channels,
    ).to(device)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")

    save_path = PROJECT_ROOT / model_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("MRI 2D U-Net Training")
    print("========================================")
    print("Image dir:", image_dir)
    print("Mask dir:", mask_dir)
    print("Number of samples:", len(dataset))
    print("Train samples:", train_size)
    print("Val samples:", val_size)
    print("Device:", device)
    print("Model path:", save_path)
    print("========================================")

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for images, masks in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} train",
        ):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for images, masks in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{epochs} val",
            ):
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_losses.append(float(loss.item()))

        train_loss = float(sum(train_losses) / max(1, len(train_losses)))
        val_loss = float(sum(val_losses) / max(1, len(val_losses)))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model: {save_path}")

    curve_path = PROJECT_ROOT / "results/mri/training_curve_mri_unet.png"
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    save_training_curve(
        history=history,
        output_path=curve_path,
    )

    metrics_path = PROJECT_ROOT / "results/mri/mri_training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "best_val_loss": float(best_val_loss),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "num_samples": int(len(dataset)),
        "train_samples": int(train_size),
        "val_samples": int(val_size),
        "in_channels": int(in_channels),
        "base_channels": int(base_channels),
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": str(save_path),
        "training_curve_path": str(curve_path),
        "metrics_path": str(metrics_path),
        "best_val_loss": float(best_val_loss),
    }
