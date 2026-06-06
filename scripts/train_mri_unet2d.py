from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.mri_segmentation.train import train_unet2d


def main() -> None:
    """
    Train 2D U-Net on processed BraTS MRI slices.

    Expected input folders:
        data/mri/processed/images/*.npy
        data/mri/processed/masks/*.npy

    Expected outputs:
        models/mri_unet.pt
        results/mri/training_curve_mri_unet.png
        results/mri/mri_training_metrics.json
    """

    result = train_unet2d(
        image_dir="data/mri/processed/images",
        mask_dir="data/mri/processed/masks",
        model_path="models/mri_unet.pt",
        epochs=10,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.2,
        in_channels=1,
        base_channels=32,
    )

    print()
    print("MRI U-Net training complete.")
    print("--------------------------------")
    print("Model path:", result.get("model_path"))
    print("Training curve:", result.get("training_curve_path"))
    print("Metrics path:", result.get("metrics_path"))
    print("Best val loss:", result.get("best_val_loss"))


if __name__ == "__main__":
    main()
