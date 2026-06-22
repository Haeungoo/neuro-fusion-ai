from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.mri_segmentation.dataset import MRISliceDataset
from src.mri_segmentation.metrics import binary_segmentation_metrics
from src.mri_segmentation.splits import (
    extract_case_id,
    load_split_manifest,
)
from src.mri_segmentation.unet2d import UNet2D


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown_case"


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))

    if max_value - min_value < 1e-8:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_value) / (max_value - min_value)


def make_overlay(
    image: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
) -> np.ndarray:
    base = normalize_image(image)

    rgb = np.stack(
        [base, base, base],
        axis=-1,
    )

    truth_mask = truth > 0.5
    pred_mask = pred > 0.5
    overlap_mask = truth_mask & pred_mask

    # Ground truth: green
    rgb[truth_mask, 0] = 0.1
    rgb[truth_mask, 1] = 0.9
    rgb[truth_mask, 2] = 0.2

    # Prediction: red
    rgb[pred_mask, 0] = 0.95
    rgb[pred_mask, 1] = 0.15
    rgb[pred_mask, 2] = 0.1

    # Overlap: yellow
    rgb[overlap_mask, 0] = 1.0
    rgb[overlap_mask, 1] = 0.9
    rgb[overlap_mask, 2] = 0.1

    return np.clip(rgb, 0.0, 1.0)


def save_grayscale_png(
    output_path: Path,
    image: np.ndarray,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.imsave(
        output_path,
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def save_rgb_png(
    output_path: Path,
    image: np.ndarray,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.imsave(
        output_path,
        image,
    )


def save_case_panel_images(
    output_dir: Path,
    case_id: str,
    image: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
) -> dict[str, str]:
    safe_case_id = safe_filename(case_id)

    image_norm = normalize_image(image)
    truth_mask = (truth > 0.5).astype(np.float32)
    pred_mask = (pred > 0.5).astype(np.float32)
    overlay = make_overlay(
        image=image,
        truth=truth_mask,
        pred=pred_mask,
    )

    input_path = output_dir / f"{safe_case_id}_input.png"
    ground_truth_path = output_dir / f"{safe_case_id}_ground_truth.png"
    predicted_mask_path = output_dir / f"{safe_case_id}_predicted_mask.png"
    overlay_only_path = output_dir / f"{safe_case_id}_overlay_only.png"

    save_grayscale_png(
        input_path,
        image_norm,
    )

    save_grayscale_png(
        ground_truth_path,
        truth_mask,
    )

    save_grayscale_png(
        predicted_mask_path,
        pred_mask,
    )

    save_rgb_png(
        overlay_only_path,
        overlay,
    )

    return {
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "ground_truth_path": str(ground_truth_path.relative_to(PROJECT_ROOT)),
        "predicted_mask_path": str(predicted_mask_path.relative_to(PROJECT_ROOT)),
        "overlay_only_path": str(overlay_only_path.relative_to(PROJECT_ROOT)),
    }


def save_case_composite_overlay(
    output_path: Path,
    image: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_norm = normalize_image(image)
    truth_mask = (truth > 0.5).astype(np.float32)
    pred_mask = (pred > 0.5).astype(np.float32)

    overlay = make_overlay(
        image=image,
        truth=truth_mask,
        pred=pred_mask,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14, 4),
    )

    panels = [
        ("Input MRI", image_norm, "gray"),
        ("Ground Truth", truth_mask, "gray"),
        ("Predicted Mask", pred_mask, "gray"),
        ("Overlay", overlay, None),
    ]

    for axis, (panel_title, panel_image, cmap) in zip(
        axes,
        panels,
    ):
        if cmap is None:
            axis.imshow(panel_image)
        else:
            axis.imshow(
                panel_image,
                cmap=cmap,
                vmin=0,
                vmax=1,
            )

        axis.axis("off")
        axis.set_title(
            panel_title,
            fontsize=10,
        )

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.03,
        subtitle,
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[UNet2D, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and (
        "model_state_dict" in checkpoint
    ):
        metadata = checkpoint

        model = UNet2D(
            in_channels=int(
                checkpoint.get("in_channels", 1)
            ),
            out_channels=int(
                checkpoint.get("out_channels", 1)
            ),
            base_channels=int(
                checkpoint.get("base_channels", 32)
            ),
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:
        metadata = {
            "threshold": 0.5,
            "checkpoint_format": "legacy_state_dict",
        }

        model = UNet2D(
            in_channels=1,
            out_channels=1,
            base_channels=32,
        )

        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, metadata


def copy_case_outputs_to_final_names(
    case_row: dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> dict[str, str]:
    mapping = {
        "input_path": output_dir / f"mri_{prefix}_case_input.png",
        "ground_truth_path": output_dir / f"mri_{prefix}_case_ground_truth.png",
        "predicted_mask_path": output_dir / f"mri_{prefix}_case_predicted_mask.png",
        "overlay_only_path": output_dir / f"mri_{prefix}_case_overlay_only.png",
        "overlay_path": output_dir / f"mri_{prefix}_case_overlay.png",
    }

    copied_paths: dict[str, str] = {}

    for source_key, destination_path in mapping.items():
        source_value = case_row.get(source_key)

        if not isinstance(source_value, str):
            continue

        source_path = PROJECT_ROOT / source_value

        if not source_path.exists():
            continue

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copyfile(
            source_path,
            destination_path,
        )

        copied_paths[source_key] = str(
            destination_path.relative_to(PROJECT_ROOT)
        )

    return copied_paths


def main() -> None:
    image_dir = PROJECT_ROOT / "data/mri/processed/images"
    mask_dir = PROJECT_ROOT / "data/mri/processed/masks"

    model_path = PROJECT_ROOT / "models/mri_unet.pt"

    manifest_path = (
        PROJECT_ROOT
        / "results/mri/mri_split_manifest.json"
    )

    output_dir = PROJECT_ROOT / "results/mri"
    composite_overlay_dir = output_dir / "case_overlays"
    panel_dir = output_dir / "case_panels"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    composite_overlay_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"MRI model not found: {model_path}"
        )

    dataset = MRISliceDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        return_metadata=True,
    )

    manifest = load_split_manifest(manifest_path)

    validation_cases = set(
        manifest["validation_cases"]
    )

    validation_indices: list[int] = []

    for index, image_path in enumerate(
        dataset.image_paths
    ):
        case_id = extract_case_id(
            image_path.name
        )

        if case_id in validation_cases:
            validation_indices.append(index)

    if not validation_indices:
        raise RuntimeError(
            "No validation slices matched the split manifest."
        )

    validation_dataset = Subset(
        dataset,
        validation_indices,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, checkpoint_metadata = load_checkpoint(
        checkpoint_path=model_path,
        device=device,
    )

    threshold = float(
        checkpoint_metadata.get(
            "threshold",
            0.5,
        )
    )

    per_slice_rows: list[dict[str, Any]] = []

    per_case_values: dict[
        str,
        dict[str, list[float]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    case_representatives: dict[str, dict[str, Any]] = {}

    print("=" * 55)
    print("MRI Validation Evaluation")
    print("=" * 55)
    print("Validation cases:", sorted(validation_cases))
    print("Validation slices:", len(validation_indices))
    print("Threshold:", threshold)
    print("Device:", device)
    print("=" * 55)

    with torch.no_grad():
        for batch in validation_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            filename = batch["filename"][0]

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= threshold
            ).float()

            image_np = images.cpu().numpy()[0, 0]
            pred_np = predictions.cpu().numpy()[0, 0]
            truth_np = masks.cpu().numpy()[0, 0]

            metrics = binary_segmentation_metrics(
                prediction=pred_np,
                target=truth_np,
            )

            case_id = extract_case_id(filename)

            row = {
                "case_id": case_id,
                "filename": filename,
                "threshold": threshold,
                **metrics,
            }

            per_slice_rows.append(row)

            for metric_name in [
                "dice",
                "iou",
                "precision",
                "recall",
                "specificity",
                "accuracy",
            ]:
                per_case_values[case_id][
                    metric_name
                ].append(
                    float(metrics[metric_name])
                )

            truth_area = float(
                np.sum(truth_np > 0.5)
            )

            pred_area = float(
                np.sum(pred_np > 0.5)
            )

            representative_score = truth_area + pred_area

            current_representative = case_representatives.get(
                case_id
            )

            if (
                current_representative is None
                or representative_score
                > current_representative["representative_score"]
            ):
                case_representatives[case_id] = {
                    "case_id": case_id,
                    "filename": filename,
                    "image": image_np,
                    "truth": truth_np,
                    "pred": pred_np,
                    "dice": float(metrics["dice"]),
                    "iou": float(metrics["iou"]),
                    "representative_score": representative_score,
                }

    per_case_rows: list[dict[str, Any]] = []

    for case_id, values in sorted(
        per_case_values.items()
    ):
        representative = case_representatives.get(case_id)

        composite_overlay_path = (
            composite_overlay_dir
            / f"{safe_filename(case_id)}_overlay.png"
        )

        mean_dice = mean(values["dice"])
        mean_iou = mean(values["iou"])

        panel_paths: dict[str, str] = {}

        if representative is not None:
            save_case_composite_overlay(
                output_path=composite_overlay_path,
                image=representative["image"],
                truth=representative["truth"],
                pred=representative["pred"],
                title=f"Validation Case: {case_id}",
                subtitle=(
                    f"Case mean Dice: {mean_dice:.4f} | "
                    f"Representative slice Dice: "
                    f"{representative['dice']:.4f} | "
                    f"File: {representative['filename']}"
                ),
            )

            panel_paths = save_case_panel_images(
                output_dir=panel_dir,
                case_id=case_id,
                image=representative["image"],
                truth=representative["truth"],
                pred=representative["pred"],
            )

        case_row = {
            "case_id": case_id,
            "num_slices": len(values["dice"]),
            "mean_dice": mean_dice,
            "mean_iou": mean_iou,
            "mean_precision": mean(values["precision"]),
            "mean_recall": mean(values["recall"]),
            "mean_specificity": mean(values["specificity"]),
            "mean_accuracy": mean(values["accuracy"]),
            "representative_filename": (
                representative["filename"]
                if representative is not None
                else None
            ),
            "representative_slice_dice": (
                representative["dice"]
                if representative is not None
                else None
            ),
            "representative_slice_iou": (
                representative["iou"]
                if representative is not None
                else None
            ),
            "overlay_path": str(
                composite_overlay_path.relative_to(PROJECT_ROOT)
            ),
            **panel_paths,
        }

        per_case_rows.append(case_row)

    summary = {
        "evaluation_type": "patient_level_validation",
        "threshold": threshold,
        "num_validation_cases": len(validation_cases),
        "num_validation_slices": len(per_slice_rows),
        "validation_cases": sorted(validation_cases),
        "slice_level": {
            "mean_dice": mean([row["dice"] for row in per_slice_rows]),
            "mean_iou": mean([row["iou"] for row in per_slice_rows]),
            "mean_precision": mean([row["precision"] for row in per_slice_rows]),
            "mean_recall": mean([row["recall"] for row in per_slice_rows]),
            "mean_specificity": mean(
                [row["specificity"] for row in per_slice_rows]
            ),
            "mean_accuracy": mean([row["accuracy"] for row in per_slice_rows]),
        },
        "case_level": {
            "mean_dice": mean([row["mean_dice"] for row in per_case_rows]),
            "mean_iou": mean([row["mean_iou"] for row in per_case_rows]),
            "mean_precision": mean(
                [row["mean_precision"] for row in per_case_rows]
            ),
            "mean_recall": mean([row["mean_recall"] for row in per_case_rows]),
            "mean_specificity": mean(
                [row["mean_specificity"] for row in per_case_rows]
            ),
            "mean_accuracy": mean(
                [row["mean_accuracy"] for row in per_case_rows]
            ),
        },
        "case_results": per_case_rows,
    }

    sorted_cases = sorted(
        per_case_rows,
        key=lambda row: float(row["mean_dice"]),
    )

    worst_case = sorted_cases[0]
    best_case = sorted_cases[-1]

    best_copied_paths = copy_case_outputs_to_final_names(
        case_row=best_case,
        output_dir=output_dir,
        prefix="best",
    )

    worst_copied_paths = copy_case_outputs_to_final_names(
        case_row=worst_case,
        output_dir=output_dir,
        prefix="worst",
    )

    best_worst_metadata = {
        "mode": "case_metrics",
        "best_case": {
            "case_id": best_case["case_id"],
            "dice": best_case["mean_dice"],
            "source": best_case,
            **best_copied_paths,
        },
        "worst_case": {
            "case_id": worst_case["case_id"],
            "dice": worst_case["mean_dice"],
            "source": worst_case,
            **worst_copied_paths,
        },
        "note": (
            "Best and worst cases were selected using case-level mean Dice. "
            "Separate input, ground-truth, predicted-mask, and overlay images "
            "were saved for dashboard display."
        ),
    }

    summary_path = output_dir / "mri_validation_metrics.json"
    per_slice_path = output_dir / "mri_validation_per_slice.csv"
    per_case_path = output_dir / "mri_validation_per_case.csv"
    case_metrics_path = output_dir / "mri_case_validation_metrics.json"
    best_worst_path = output_dir / "mri_best_worst_cases.json"

    case_metrics_json = {
        "evaluation_type": "case_level_validation",
        "threshold": threshold,
        "num_cases": len(per_case_rows),
        "validation_cases": per_case_rows,
    }

    write_json(summary_path, summary)
    write_json(case_metrics_path, case_metrics_json)
    write_json(best_worst_path, best_worst_metadata)

    write_csv(per_slice_path, per_slice_rows)
    write_csv(per_case_path, per_case_rows)

    print()
    print("Validation complete")
    print("------------------------------")
    print(
        "Slice mean Dice:",
        f"{summary['slice_level']['mean_dice']:.4f}",
    )
    print(
        "Slice mean IoU:",
        f"{summary['slice_level']['mean_iou']:.4f}",
    )
    print(
        "Case mean Dice:",
        f"{summary['case_level']['mean_dice']:.4f}",
    )
    print(
        "Case mean IoU:",
        f"{summary['case_level']['mean_iou']:.4f}",
    )
    print()
    print("Best case:", best_case["case_id"], best_case["mean_dice"])
    print("Worst case:", worst_case["case_id"], worst_case["mean_dice"])
    print()
    print("Saved:", summary_path)
    print("Saved:", per_slice_path)
    print("Saved:", per_case_path)
    print("Saved:", case_metrics_path)
    print("Saved:", best_worst_path)
    print("Saved panels:", panel_dir)
    print("Saved case overlays:", composite_overlay_dir)


if __name__ == "__main__":
    main()