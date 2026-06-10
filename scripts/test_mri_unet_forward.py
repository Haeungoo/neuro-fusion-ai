from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.mri_segmentation.dataset import normalize_mri_slice
from src.mri_segmentation.metrics import binary_segmentation_metrics
from src.mri_segmentation.splits import (
    extract_case_id,
    load_split_manifest,
)
from src.mri_segmentation.unet2d import UNet2D


def load_json_safe(path: Path) -> dict[str, Any]:
    """
    Read a JSON file safely.

    Returns an empty dictionary if the file does not exist
    or cannot be decoded.
    """

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {}

    except (OSError, json.JSONDecodeError):
        return {}


def load_model(
    model_path: Path,
    device: torch.device,
) -> tuple[UNet2D, dict[str, Any]]:
    """
    Load a trained MRI U-Net.

    Supported checkpoint formats:

    1. New checkpoint dictionary:
       {
           "model_state_dict": ...,
           "in_channels": 1,
           "base_channels": 32,
           "threshold": 0.5,
           ...
       }

    2. Legacy plain state dictionary.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"MRI model not found: {model_path}\n"
            "Run training first:\n"
            "python -m scripts.train_mri_unet2d"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        metadata: dict[str, Any] = checkpoint

        in_channels = int(
            checkpoint.get("in_channels", 1)
        )

        out_channels = int(
            checkpoint.get("out_channels", 1)
        )

        base_channels = int(
            checkpoint.get("base_channels", 32)
        )

        state_dict = checkpoint["model_state_dict"]

    else:
        metadata = {
            "checkpoint_format": "legacy_state_dict",
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 32,
            "threshold": 0.5,
        }

        in_channels = 1
        out_channels = 1
        base_channels = 32
        state_dict = checkpoint

    model = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model, metadata


def find_candidate_image_paths(
    image_dir: Path,
    manifest_path: Path,
) -> list[Path]:
    """
    Prefer images belonging to validation patients.

    If the split manifest is unavailable, all processed images
    are returned as fallback candidates.
    """

    image_paths = sorted(
        image_dir.glob("*.npy")
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No processed MRI slices found in {image_dir}"
        )

    manifest = load_json_safe(manifest_path)

    validation_cases = set(
        manifest.get("validation_cases", [])
    )

    if not validation_cases:
        print(
            "Warning: validation case list not found. "
            "Selecting from all processed slices."
        )

        return image_paths

    validation_paths = [
        path
        for path in image_paths
        if extract_case_id(path.name)
        in validation_cases
    ]

    if not validation_paths:
        print(
            "Warning: no processed files matched the "
            "validation cases. Selecting from all slices."
        )

        return image_paths

    return validation_paths


def choose_evaluation_slice(
    candidate_paths: list[Path],
    mask_dir: Path,
) -> tuple[Path, Path]:
    """
    Select a representative tumor-containing slice.

    The slice with the largest ground-truth tumor area is chosen.
    This makes the visual demo more informative than selecting
    an arbitrary slice.
    """

    best_image_path: Path | None = None
    best_mask_path: Path | None = None
    largest_mask_pixels = -1

    for image_path in candidate_paths:
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            continue

        mask = np.load(mask_path)

        if mask.ndim != 2:
            continue

        tumor_pixels = int(
            (mask > 0).sum()
        )

        if tumor_pixels > largest_mask_pixels:
            best_image_path = image_path
            best_mask_path = mask_path
            largest_mask_pixels = tumor_pixels

    if best_image_path is None or best_mask_path is None:
        raise RuntimeError(
            "No valid image/mask pair was found for inference."
        )

    return best_image_path, best_mask_path


def prepare_input_tensor(
    image: np.ndarray,
    expected_channels: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Normalize an MRI slice and convert it to a batched tensor.

    Current baseline:
        FLAIR input stored as (H, W)

    Also supports future multi-channel input stored as:
        (C, H, W)

    Returns:
        input_tensor: (1, C, H, W)
        display_image: 2D image used for visualization
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim == 2:
        normalized_image = normalize_mri_slice(
            image
        )

        channel_array = normalized_image[
            None, :, :
        ]

        display_image = normalized_image

    elif image.ndim == 3:
        normalized_channels: list[np.ndarray] = []

        for channel in image:
            normalized_channels.append(
                normalize_mri_slice(channel)
            )

        channel_array = np.stack(
            normalized_channels,
            axis=0,
        ).astype(np.float32)

        # Channel 0 is assumed to be FLAIR for display.
        display_image = channel_array[0]

    else:
        raise ValueError(
            "Expected MRI array shape (H, W) or "
            f"(C, H, W). Got {image.shape}."
        )

    actual_channels = int(
        channel_array.shape[0]
    )

    if actual_channels != expected_channels:
        raise ValueError(
            "Input channel mismatch. "
            f"Processed image has {actual_channels} channels, "
            f"but the model expects {expected_channels}.\n"
            "For the current FLAIR baseline, both should be 1."
        )

    input_tensor = torch.from_numpy(
        channel_array[None, :, :, :]
    ).float().to(device)

    return input_tensor, display_image


def normalize_for_display(
    image: np.ndarray,
) -> np.ndarray:
    """
    Scale a 2D image to the range 0–1 for visualization.
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    finite_mask = np.isfinite(image)

    if not np.any(finite_mask):
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    finite_values = image[finite_mask]

    lower = float(
        np.percentile(finite_values, 1)
    )

    upper = float(
        np.percentile(finite_values, 99)
    )

    if upper <= lower:
        lower = float(
            finite_values.min()
        )

        upper = float(
            finite_values.max()
        )

    if upper <= lower:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    scaled = (
        image - lower
    ) / (
        upper - lower
    )

    return np.clip(
        scaled,
        0.0,
        1.0,
    ).astype(np.float32)


def save_grayscale_image(
    image: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """
    Save one grayscale result image.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(5, 5),
    )

    axis.imshow(
        image,
        cmap="gray",
        interpolation="nearest",
    )

    axis.set_title(title)
    axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_binary_mask(
    mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """
    Save one binary segmentation mask.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(5, 5),
    )

    axis.imshow(
        mask,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    axis.set_title(title)
    axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save an MRI overlay containing:

    Green:
        Ground-truth tumor mask

    Red:
        Predicted tumor mask

    Yellow:
        Areas where prediction and ground truth overlap
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    display_image = normalize_for_display(
        image
    )

    ground_truth_binary = (
        ground_truth > 0
    )

    prediction_binary = (
        prediction > 0
    )

    overlap = np.logical_and(
        ground_truth_binary,
        prediction_binary,
    )

    ground_truth_only = np.logical_and(
        ground_truth_binary,
        ~prediction_binary,
    )

    prediction_only = np.logical_and(
        prediction_binary,
        ~ground_truth_binary,
    )

    overlay = np.stack(
        [
            display_image,
            display_image,
            display_image,
        ],
        axis=-1,
    )

    # Ground truth only: green
    overlay[ground_truth_only] = [
        0.1,
        1.0,
        0.1,
    ]

    # Prediction only: red
    overlay[prediction_only] = [
        1.0,
        0.1,
        0.1,
    ]

    # Correct overlap: yellow
    overlay[overlap] = [
        1.0,
        0.9,
        0.1,
    ]

    figure, axis = plt.subplots(
        figsize=(6, 6),
    )

    axis.imshow(
        overlay,
        interpolation="nearest",
    )

    axis.set_title(
        "MRI Tumor Segmentation Overlay\n"
        "Green: GT only | Red: Prediction only | Yellow: Overlap"
    )

    axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_comparison_figure(
    image: np.ndarray,
    ground_truth: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save a four-panel inference comparison figure.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(
        image,
        cmap="gray",
    )
    axes[0].set_title("Input FLAIR")

    axes[1].imshow(
        ground_truth,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Ground Truth")

    probability_plot = axes[2].imshow(
        probability,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Tumor Probability")

    axes[3].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[3].set_title("Predicted Mask")

    for axis in axes:
        axis.axis("off")

    figure.colorbar(
        probability_plot,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """
    Run single-slice MRI inference using the best trained checkpoint.
    """

    image_dir = (
        PROJECT_ROOT
        / "data"
        / "mri"
        / "processed"
        / "images"
    )

    mask_dir = (
        PROJECT_ROOT
        / "data"
        / "mri"
        / "processed"
        / "masks"
    )

    model_path = (
        PROJECT_ROOT
        / "models"
        / "mri_unet.pt"
    )

    manifest_path = (
        PROJECT_ROOT
        / "results"
        / "mri"
        / "mri_split_manifest.json"
    )

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "mri"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, checkpoint_metadata = load_model(
        model_path=model_path,
        device=device,
    )

    expected_channels = int(
        checkpoint_metadata.get(
            "in_channels",
            1,
        )
    )

    threshold = float(
        checkpoint_metadata.get(
            "threshold",
            0.5,
        )
    )

    candidate_paths = find_candidate_image_paths(
        image_dir=image_dir,
        manifest_path=manifest_path,
    )

    image_path, mask_path = choose_evaluation_slice(
        candidate_paths=candidate_paths,
        mask_dir=mask_dir,
    )

    image = np.load(
        image_path
    ).astype(np.float32)

    ground_truth = (
        np.load(mask_path) > 0
    ).astype(np.float32)

    if ground_truth.ndim != 2:
        raise ValueError(
            "Expected ground-truth mask shape (H, W). "
            f"Got {ground_truth.shape} from {mask_path}."
        )

    input_tensor, display_image = prepare_input_tensor(
        image=image,
        expected_channels=expected_channels,
        device=device,
    )

    with torch.no_grad():
        logits = model(input_tensor)

        probability_tensor = torch.sigmoid(
            logits
        )

    probability = (
        probability_tensor
        .detach()
        .cpu()
        .numpy()[0, 0]
        .astype(np.float32)
    )

    prediction = (
        probability >= threshold
    ).astype(np.float32)

    metrics = binary_segmentation_metrics(
        prediction=prediction,
        target=ground_truth,
    )

    input_output_path = (
        output_dir
        / "mri_input_slice.png"
    )

    ground_truth_output_path = (
        output_dir
        / "mri_ground_truth_mask.png"
    )

    prediction_output_path = (
        output_dir
        / "mri_predicted_mask.png"
    )

    overlay_output_path = (
        output_dir
        / "mri_prediction_overlay.png"
    )

    probability_output_path = (
        output_dir
        / "mri_probability_map.png"
    )

    comparison_output_path = (
        output_dir
        / "mri_inference_comparison.png"
    )

    metrics_output_path = (
        output_dir
        / "mri_inference_metrics.json"
    )

    save_grayscale_image(
        image=display_image,
        output_path=input_output_path,
        title="Input FLAIR MRI Slice",
    )

    save_binary_mask(
        mask=ground_truth,
        output_path=ground_truth_output_path,
        title="Ground-Truth Tumor Mask",
    )

    save_binary_mask(
        mask=prediction,
        output_path=prediction_output_path,
        title="Predicted Tumor Mask",
    )

    save_grayscale_image(
        image=probability,
        output_path=probability_output_path,
        title="Predicted Tumor Probability",
    )

    save_overlay(
        image=display_image,
        ground_truth=ground_truth,
        prediction=prediction,
        output_path=overlay_output_path,
    )

    save_comparison_figure(
        image=display_image,
        ground_truth=ground_truth,
        probability=probability,
        prediction=prediction,
        output_path=comparison_output_path,
    )

    case_id = extract_case_id(
        image_path.name
    )

    inference_metrics: dict[str, Any] = {
        "evaluation_type": "single_slice_demo",
        "sample_name": image_path.name,
        "case_id": case_id,
        "modality": checkpoint_metadata.get(
            "modality",
            "FLAIR",
        ),
        "model_path": str(model_path),
        "checkpoint_epoch": checkpoint_metadata.get(
            "epoch"
        ),
        "checkpoint_best_val_dice": (
            checkpoint_metadata.get(
                "best_val_dice"
            )
        ),
        "checkpoint_best_val_iou": (
            checkpoint_metadata.get(
                "best_val_iou"
            )
        ),
        "split_type": checkpoint_metadata.get(
            "split_type",
            "unknown",
        ),
        "input_channels": expected_channels,
        "threshold": threshold,
        "device": str(device),
        "probability_min": float(
            probability.min()
        ),
        "probability_max": float(
            probability.max()
        ),
        "probability_mean": float(
            probability.mean()
        ),
        "prediction_pixels": int(
            prediction.sum()
        ),
        "ground_truth_pixels": int(
            ground_truth.sum()
        ),
        "dice_score": float(
            metrics["dice"]
        ),
        "iou_score": float(
            metrics["iou"]
        ),
        "precision": float(
            metrics["precision"]
        ),
        "recall": float(
            metrics["recall"]
        ),
        "specificity": float(
            metrics["specificity"]
        ),
        "accuracy": float(
            metrics["accuracy"]
        ),
        "true_positive_pixels": int(
            metrics["true_positive_pixels"]
        ),
        "false_positive_pixels": int(
            metrics["false_positive_pixels"]
        ),
        "false_negative_pixels": int(
            metrics["false_negative_pixels"]
        ),
        "true_negative_pixels": int(
            metrics["true_negative_pixels"]
        ),
        "outputs": {
            "input_slice": str(
                input_output_path
            ),
            "ground_truth_mask": str(
                ground_truth_output_path
            ),
            "predicted_mask": str(
                prediction_output_path
            ),
            "probability_map": str(
                probability_output_path
            ),
            "overlay": str(
                overlay_output_path
            ),
            "comparison": str(
                comparison_output_path
            ),
        },
    }

    with open(
        metrics_output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            inference_metrics,
            file,
            indent=2,
        )

    print("=" * 68)
    print("MRI single-slice inference complete")
    print("=" * 68)
    print("Sample:", image_path.name)
    print("Case:", case_id)
    print("Device:", device)
    print("Threshold:", threshold)
    print(
        "Probability range:",
        f"{probability.min():.4f}",
        "to",
        f"{probability.max():.4f}",
    )
    print(
        "Predicted pixels:",
        int(prediction.sum()),
    )
    print(
        "Ground-truth pixels:",
        int(ground_truth.sum()),
    )
    print(
        "Dice:",
        f"{metrics['dice']:.4f}",
    )
    print(
        "IoU:",
        f"{metrics['iou']:.4f}",
    )
    print(
        "Precision:",
        f"{metrics['precision']:.4f}",
    )
    print(
        "Recall:",
        f"{metrics['recall']:.4f}",
    )
    print("-" * 68)
    print("Saved:", input_output_path)
    print("Saved:", ground_truth_output_path)
    print("Saved:", prediction_output_path)
    print("Saved:", probability_output_path)
    print("Saved:", overlay_output_path)
    print("Saved:", comparison_output_path)
    print("Saved:", metrics_output_path)


if __name__ == "__main__":
    main()