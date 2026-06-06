from __future__ import annotations

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.mri_segmentation.unet2d import UNet2D
from src.mri_segmentation.preprocess import normalize_mri_slice


def load_sample_image_and_mask() -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load one processed BraTS MRI image/mask pair.

    Expected:
        data/mri/processed/images/*.npy
        data/mri/processed/masks/*.npy

    If no processed real data exists, create synthetic fallback data.
    """

    image_dir = PROJECT_ROOT / "data/mri/processed/images"
    mask_dir = PROJECT_ROOT / "data/mri/processed/masks"

    image_files = sorted(image_dir.glob("*.npy"))

    if image_files:
        image_path = image_files[0]
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        return image, mask, image_path.name

    # Synthetic fallback
    size = 160
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    image = np.exp(-((xx / 0.75) ** 2 + (yy / 0.9) ** 2)).astype(np.float32)
    mask = (((xx - 0.2) ** 2 + (yy + 0.1) ** 2) < 0.08).astype(np.float32)

    return image, mask, "synthetic_fallback"


def save_grayscale_image(
    image: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_mask_image(
    mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_overlay_image(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.imshow(mask, alpha=0.45)
    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def calculate_dice_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Calculate Dice score between predicted mask and ground-truth mask.

    Dice = 2 * intersection / (prediction + ground truth)
    """

    pred_mask = (pred_mask > 0).astype(np.float32)
    gt_mask = (gt_mask > 0).astype(np.float32)

    intersection = float((pred_mask * gt_mask).sum())
    denominator = float(pred_mask.sum() + gt_mask.sum())

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return float(dice)


def main() -> None:
    """
    Run MRI U-Net inference on one processed BraTS slice and save four figures.

    Outputs:
        results/mri/mri_input_slice.png
        results/mri/mri_ground_truth_mask.png
        results/mri/mri_predicted_mask.png
        results/mri/mri_prediction_overlay.png
    """

    output_dir = PROJECT_ROOT / "results/mri"
    output_dir.mkdir(parents=True, exist_ok=True)

    image, gt_mask, sample_name = load_sample_image_and_mask()

    print("Sample:", sample_name)
    print("Image shape:", image.shape)
    print("Mask shape:", gt_mask.shape)
    print("Mask unique:", np.unique(gt_mask)[:10])

    image_norm = normalize_mri_slice(image)

    model_path = PROJECT_ROOT / "models/mri_unet.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet2D(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print("Loaded trained model:", model_path)
    else:
        print("Warning: mri_unet.pt not found. Using random U-Net.")

    model.eval()

    x = torch.from_numpy(image_norm[None, None, :, :]).float().to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

    pred_mask = (prob >= 0.5).astype(np.float32)
    
    dice = calculate_dice_score(
        pred_mask=pred_mask,
        gt_mask=gt_mask,
    )

    input_path = output_dir / "mri_input_slice.png"
    gt_path = output_dir / "mri_ground_truth_mask.png"
    pred_path = output_dir / "mri_predicted_mask.png"
    overlay_path = output_dir / "mri_prediction_overlay.png"

    save_grayscale_image(
        image=image,
        output_path=input_path,
        title="Input MRI slice",
    )

    save_mask_image(
        mask=gt_mask,
        output_path=gt_path,
        title="Ground-truth tumor mask",
    )

    save_mask_image(
        mask=pred_mask,
        output_path=pred_path,
        title="Predicted tumor mask",
    )

    save_overlay_image(
        image=image,
        mask=pred_mask,
        output_path=overlay_path,
        title="MRI + predicted tumor overlay",
    )

    metrics_path = output_dir / "mri_inference_metrics.json"

    metrics = {
        "sample_name": sample_name,
        "dice_score": float(dice),
        "threshold": 0.5,
        "prediction_pixels": int(pred_mask.sum()),
        "ground_truth_pixels": int((gt_mask > 0).sum()),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print("Saved:")
    print(input_path)
    print(gt_path)
    print(pred_path)
    print(overlay_path)
    print("Dice score:", dice)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()