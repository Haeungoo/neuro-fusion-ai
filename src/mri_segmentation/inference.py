from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from src.common.paths import PROJECT_ROOT
from src.common.plotting import save_mri_overlay
from src.mri_segmentation.unet2d import UNet2D
from src.mri_segmentation.preprocess import normalize_mri_slice


def load_unet_model(
    model_path: str | Path = "models/mri_unet.pt",
    device: str | None = None,
    in_channels: int = 1,
    base_channels: int = 32,
) -> UNet2D:
    """
    Load a 2D U-Net model.

    If models/mri_unet.pt does not exist, this returns a randomly initialized model.
    That is okay for checking whether the pipeline works.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet2D(
        in_channels=in_channels,
        out_channels=1,
        base_channels=base_channels,
    )

    model_path = PROJECT_ROOT / model_path

    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    else:
        print(f"Warning: {model_path} not found. Using randomly initialized U-Net.")

    model.to(device)
    model.eval()

    return model


def run_mri_inference(
    image_2d: np.ndarray,
    model_path: str | Path = "models/mri_unet.pt",
    threshold: float = 0.5,
    output_path: str | Path = "results/mri/mri_prediction_overlay.png",
) -> dict:
    """
    Real MRI inference function.

    Input:
        image_2d: one 2D MRI slice, shape (H, W)

    Output:
        predicted binary mask and overlay image path
    """
    image_2d = np.asarray(image_2d, dtype=np.float32)

    if image_2d.ndim != 2:
        raise ValueError(
            f"image_2d must be 2D with shape (H, W). Got shape {image_2d.shape}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_unet_model(
        model_path=model_path,
        device=device,
        in_channels=1,
    )

    image_norm = normalize_mri_slice(image_2d)

    x = torch.from_numpy(image_norm[None, None, :, :]).float().to(device)

    with torch.no_grad():
        logits = model(x)
        probability = torch.sigmoid(logits).cpu().numpy()[0, 0]

    binary_mask = probability >= threshold

    overlay_path = PROJECT_ROOT / output_path
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    save_mri_overlay(
        image_2d=image_2d,
        mask_2d=binary_mask,
        output_path=overlay_path,
    )

    return {
        "prediction": "Tumor mask generated",
        "overlay_path": str(overlay_path),
        "threshold": threshold,
        "mask_probability": probability,
        "binary_mask": binary_mask,
        "note": "If models/mri_unet.pt is not trained, this is only a structural demo.",
    }


def create_synthetic_mri_slice(size: int = 160, seed: int = 1) -> np.ndarray:
    """
    Create synthetic MRI-like 2D image for testing.
    This is not real MRI.
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    brain = np.exp(-((xx / 0.75) ** 2 + (yy / 0.9) ** 2))
    brain += 0.05 * rng.normal(size=brain.shape)
    brain = np.clip(brain, 0, 1).astype(np.float32)

    return brain


def run_mri_dashboard_demo() -> dict:
    """
    Dashboard-safe MRI demo.

    This can be called without arguments.
    """
    image = create_synthetic_mri_slice()
    return run_mri_inference(image_2d=image)