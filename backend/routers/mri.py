from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter


try:
    from src.common.paths import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


router = APIRouter(
    tags=["MRI Tumor Segmentation"],
)


RESULTS_DIR = PROJECT_ROOT / "results" / "mri"
MODELS_DIR = PROJECT_ROOT / "models"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else None

    except Exception:
        return None


def file_info(path: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    results_root = (PROJECT_ROOT / "results").resolve()

    if not resolved_path.exists():
        return {
            "exists": False,
            "path": str(resolved_path),
            "url": None,
        }

    try:
        relative_path = resolved_path.relative_to(results_root)

        return {
            "exists": True,
            "path": str(resolved_path),
            "url": f"/media/results/{relative_path.as_posix()}",
        }

    except ValueError:
        return {
            "exists": True,
            "path": str(resolved_path),
            "url": None,
        }


def first_existing_file(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path

    return None


def find_metrics_file() -> Path | None:
    candidates = [
        RESULTS_DIR / "mri_validation_metrics.json",
        RESULTS_DIR / "mri_metrics.json",
        RESULTS_DIR / "mri_unet_metrics.json",
        RESULTS_DIR / "mri_case_metrics.json",
        RESULTS_DIR / "metrics.json",
    ]

    return first_existing_file(candidates)


def find_model_file() -> Path | None:
    candidates = [
        MODELS_DIR / "mri_unet.pt",
        MODELS_DIR / "mri_unet.pth",
        MODELS_DIR / "mri_segmentation_unet.pt",
        MODELS_DIR / "mri_segmentation_unet.pth",
    ]

    return first_existing_file(candidates)


def normalize_mri_metrics(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None

    metrics = dict(data)

    case_level = data.get("case_level")
    slice_level = data.get("slice_level")
    validation = data.get("validation")
    summary = data.get("summary")
    dataset = data.get("dataset")

    nested_sources = [
        validation,
        summary,
    ]

    for source in nested_sources:
        if isinstance(source, dict):
            for key, value in source.items():
                if metrics.get(key) is None:
                    metrics[key] = value

    if isinstance(case_level, dict):
        if metrics.get("case_mean_dice") is None:
            metrics["case_mean_dice"] = case_level.get("mean_dice")

        if metrics.get("case_mean_iou") is None:
            metrics["case_mean_iou"] = case_level.get("mean_iou")

        if metrics.get("case_mean_precision") is None:
            metrics["case_mean_precision"] = case_level.get("mean_precision")

        if metrics.get("case_mean_recall") is None:
            metrics["case_mean_recall"] = case_level.get("mean_recall")

        if metrics.get("case_mean_specificity") is None:
            metrics["case_mean_specificity"] = case_level.get("mean_specificity")

        if metrics.get("case_mean_accuracy") is None:
            metrics["case_mean_accuracy"] = case_level.get("mean_accuracy")

    if isinstance(slice_level, dict):
        if metrics.get("slice_mean_dice") is None:
            metrics["slice_mean_dice"] = slice_level.get("mean_dice")

        if metrics.get("slice_mean_iou") is None:
            metrics["slice_mean_iou"] = slice_level.get("mean_iou")

        if metrics.get("slice_mean_precision") is None:
            metrics["slice_mean_precision"] = slice_level.get("mean_precision")

        if metrics.get("slice_mean_recall") is None:
            metrics["slice_mean_recall"] = slice_level.get("mean_recall")

        if metrics.get("slice_mean_specificity") is None:
            metrics["slice_mean_specificity"] = slice_level.get("mean_specificity")

        if metrics.get("slice_mean_accuracy") is None:
            metrics["slice_mean_accuracy"] = slice_level.get("mean_accuracy")

    if isinstance(dataset, dict):
        for key, value in dataset.items():
            if metrics.get(key) is None:
                metrics[key] = value
    elif isinstance(dataset, str):
        metrics["dataset"] = dataset

    if metrics.get("num_cases") is None:
        validation_cases = data.get("validation_cases")

        if isinstance(validation_cases, list):
            metrics["num_cases"] = len(validation_cases)
        elif data.get("num_validation_cases") is not None:
            metrics["num_cases"] = data.get("num_validation_cases")

    if metrics.get("num_slices") is None:
        if data.get("num_validation_slices") is not None:
            metrics["num_slices"] = data.get("num_validation_slices")
        elif data.get("num_samples") is not None:
            metrics["num_slices"] = data.get("num_samples")

    alias_map = {
        "case_mean_dice": [
            "case_mean_dice",
            "mean_case_dice",
            "case_dice",
            "mean_dice_case",
            "validation_case_dice",
        ],
        "case_mean_iou": [
            "case_mean_iou",
            "mean_case_iou",
            "case_iou",
            "mean_iou_case",
            "validation_case_iou",
        ],
        "slice_mean_dice": [
            "slice_mean_dice",
            "mean_slice_dice",
            "slice_dice",
            "mean_dice_slice",
            "validation_slice_dice",
            "dice",
            "dice_score",
            "mean_dice",
        ],
        "slice_mean_iou": [
            "slice_mean_iou",
            "mean_slice_iou",
            "slice_iou",
            "mean_iou_slice",
            "validation_slice_iou",
            "iou",
            "iou_score",
            "mean_iou",
        ],
        "num_cases": [
            "num_cases",
            "n_cases",
            "case_count",
            "num_validation_cases",
            "n_validation_cases",
        ],
        "num_slices": [
            "num_slices",
            "n_slices",
            "slice_count",
            "validation_slices",
            "num_validation_slices",
            "n_validation_slices",
            "num_samples",
        ],
    }

    for standard_key, aliases in alias_map.items():
        if metrics.get(standard_key) is not None:
            continue

        for alias in aliases:
            value = metrics.get(alias)

            if value is None:
                continue

            if isinstance(value, list):
                continue

            metrics[standard_key] = value
            break

    if metrics.get("model") is None:
        metrics["model"] = "2D U-Net"

    if metrics.get("dataset") is None or isinstance(metrics.get("dataset"), dict):
        metrics["dataset"] = "BraTS-style MRI slices"

    return metrics


def load_best_worst_metadata() -> dict[str, Any] | None:
    metadata_path = RESULTS_DIR / "mri_best_worst_cases.json"

    return read_json(metadata_path)


@router.get("/status")
def get_mri_status() -> dict[str, Any]:
    metrics_path = find_metrics_file()
    model_path = find_model_file()

    metrics = normalize_mri_metrics(
        read_json(metrics_path) if metrics_path is not None else None
    )

    input_slice_path = RESULTS_DIR / "mri_input_slice.png"
    ground_truth_mask_path = RESULTS_DIR / "mri_ground_truth_mask.png"
    predicted_mask_path = RESULTS_DIR / "mri_predicted_mask.png"
    prediction_overlay_path = RESULTS_DIR / "mri_prediction_overlay.png"

    best_case_overlay_path = RESULTS_DIR / "mri_best_case_overlay.png"
    worst_case_overlay_path = RESULTS_DIR / "mri_worst_case_overlay.png"
    best_worst_metadata_path = RESULTS_DIR / "mri_best_worst_cases.json"

    best_worst_metadata = load_best_worst_metadata()

    has_basic_visuals = all(
        [
            input_slice_path.exists(),
            ground_truth_mask_path.exists(),
            predicted_mask_path.exists(),
            prediction_overlay_path.exists(),
        ]
    )

    has_best_worst_visuals = all(
        [
            best_case_overlay_path.exists(),
            worst_case_overlay_path.exists(),
            best_worst_metadata_path.exists(),
        ]
    )

    return {
        "module": "MRI Tumor Segmentation",
        "status": "ready" if has_basic_visuals else "missing_outputs",
        "description": (
            "Brain tumor segmentation dashboard using a 2D U-Net style MRI "
            "segmentation workflow."
        ),
        "dataset_note": (
            "This module is designed for BraTS-style MRI slices. Large raw MRI "
            "datasets are not included in the repository."
        ),
        "model": {
            "name": "2D U-Net",
            "task": "Brain tumor segmentation",
            "input": "MRI slice",
            "output": "Tumor segmentation mask",
            "model_file": file_info(model_path) if model_path is not None else {
                "exists": False,
                "path": None,
                "url": None,
            },
        },
        "metrics": metrics,
        "outputs": {
            "metrics_json": file_info(metrics_path) if metrics_path is not None else {
                "exists": False,
                "path": None,
                "url": None,
            },
            "input_slice": file_info(input_slice_path),
            "ground_truth_mask": file_info(ground_truth_mask_path),
            "predicted_mask": file_info(predicted_mask_path),
            "prediction_overlay": file_info(prediction_overlay_path),
            "best_case_overlay": file_info(best_case_overlay_path),
            "worst_case_overlay": file_info(worst_case_overlay_path),
            "best_worst_metadata": file_info(best_worst_metadata_path),
            "model_file": file_info(model_path) if model_path is not None else {
                "exists": False,
                "path": None,
                "url": None,
            },
        },
        "visualization": {
            "available": has_basic_visuals,
            "input_slice": file_info(input_slice_path),
            "ground_truth_mask": file_info(ground_truth_mask_path),
            "predicted_mask": file_info(predicted_mask_path),
            "prediction_overlay": file_info(prediction_overlay_path),
        },
        "best_worst_cases": {
            "available": has_best_worst_visuals,
            "metadata": best_worst_metadata,
            "metadata_file": file_info(best_worst_metadata_path),
            "best_case_overlay": file_info(best_case_overlay_path),
            "worst_case_overlay": file_info(worst_case_overlay_path),
        },
        "pipeline": [
            "MRI slice loading",
            "Preprocessing",
            "U-Net segmentation",
            "Prediction mask generation",
            "Dice and IoU evaluation",
            "Overlay visualization",
            "Best and worst case review",
        ],
        "disclaimer": (
            "This MRI segmentation module is for research and educational use "
            "only and is not intended for clinical diagnosis."
        ),
    }