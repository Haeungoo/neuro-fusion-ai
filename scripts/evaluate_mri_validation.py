from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.mri_segmentation.dataset import MRISliceDataset
from src.mri_segmentation.metrics import (
    binary_segmentation_metrics,
)
from src.mri_segmentation.splits import (
    extract_case_id,
    load_split_manifest,
)
from src.mri_segmentation.unet2d import UNet2D


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[UNet2D, dict]:
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


def main() -> None:
    image_dir = (
        PROJECT_ROOT
        / "data/mri/processed/images"
    )

    mask_dir = (
        PROJECT_ROOT
        / "data/mri/processed/masks"
    )

    model_path = (
        PROJECT_ROOT
        / "models/mri_unet.pt"
    )

    manifest_path = (
        PROJECT_ROOT
        / "results/mri/mri_split_manifest.json"
    )

    output_dir = (
        PROJECT_ROOT
        / "results/mri"
    )

    output_dir.mkdir(
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

    per_slice_rows: list[dict] = []
    per_case_values: dict[
        str,
        dict[str, list[float]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

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

            pred = predictions.cpu().numpy()[0, 0]
            truth = masks.cpu().numpy()[0, 0]

            metrics = binary_segmentation_metrics(
                prediction=pred,
                target=truth,
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

    per_case_rows: list[dict] = []

    for case_id, values in sorted(
        per_case_values.items()
    ):
        per_case_rows.append(
            {
                "case_id": case_id,
                "num_slices": len(
                    values["dice"]
                ),
                "mean_dice": mean(
                    values["dice"]
                ),
                "mean_iou": mean(
                    values["iou"]
                ),
                "mean_precision": mean(
                    values["precision"]
                ),
                "mean_recall": mean(
                    values["recall"]
                ),
                "mean_specificity": mean(
                    values["specificity"]
                ),
                "mean_accuracy": mean(
                    values["accuracy"]
                ),
            }
        )

    summary = {
        "evaluation_type": "patient_level_validation",
        "threshold": threshold,
        "num_validation_cases": len(
            validation_cases
        ),
        "num_validation_slices": len(
            per_slice_rows
        ),
        "validation_cases": sorted(
            validation_cases
        ),
        "slice_level": {
            "mean_dice": mean(
                [
                    row["dice"]
                    for row in per_slice_rows
                ]
            ),
            "mean_iou": mean(
                [
                    row["iou"]
                    for row in per_slice_rows
                ]
            ),
            "mean_precision": mean(
                [
                    row["precision"]
                    for row in per_slice_rows
                ]
            ),
            "mean_recall": mean(
                [
                    row["recall"]
                    for row in per_slice_rows
                ]
            ),
            "mean_specificity": mean(
                [
                    row["specificity"]
                    for row in per_slice_rows
                ]
            ),
            "mean_accuracy": mean(
                [
                    row["accuracy"]
                    for row in per_slice_rows
                ]
            ),
        },
        "case_level": {
            "mean_dice": mean(
                [
                    row["mean_dice"]
                    for row in per_case_rows
                ]
            ),
            "mean_iou": mean(
                [
                    row["mean_iou"]
                    for row in per_case_rows
                ]
            ),
            "mean_precision": mean(
                [
                    row["mean_precision"]
                    for row in per_case_rows
                ]
            ),
            "mean_recall": mean(
                [
                    row["mean_recall"]
                    for row in per_case_rows
                ]
            ),
            "mean_specificity": mean(
                [
                    row["mean_specificity"]
                    for row in per_case_rows
                ]
            ),
            "mean_accuracy": mean(
                [
                    row["mean_accuracy"]
                    for row in per_case_rows
                ]
            ),
        },
    }

    summary_path = (
        output_dir
        / "mri_validation_metrics.json"
    )

    per_slice_path = (
        output_dir
        / "mri_validation_per_slice.csv"
    )

    per_case_path = (
        output_dir
        / "mri_validation_per_case.csv"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    write_csv(
        per_slice_path,
        per_slice_rows,
    )

    write_csv(
        per_case_path,
        per_case_rows,
    )

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
    print("Saved:", summary_path)
    print("Saved:", per_slice_path)
    print("Saved:", per_case_path)


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()