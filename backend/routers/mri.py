from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from src.common.paths import PROJECT_ROOT
from backend.routers.files import file_info


router = APIRouter()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@router.get("/status")
def mri_status() -> dict:
    model_path = PROJECT_ROOT / "models/mri_unet.pt"

    input_path = PROJECT_ROOT / "results/mri/mri_input_slice.png"
    gt_path = PROJECT_ROOT / "results/mri/mri_ground_truth_mask.png"
    pred_path = PROJECT_ROOT / "results/mri/mri_predicted_mask.png"
    overlay_path = PROJECT_ROOT / "results/mri/mri_prediction_overlay.png"

    training_metrics_path = PROJECT_ROOT / "results/mri/mri_training_metrics.json"
    inference_metrics_path = PROJECT_ROOT / "results/mri/mri_inference_metrics.json"

    return {
        "module": "MRI Tumor Segmentation",
        "dataset": "BraTS 2020",
        "model": "2D U-Net",
        "model_file": file_info(model_path),
        "outputs": {
            "input_slice": file_info(input_path),
            "ground_truth_mask": file_info(gt_path),
            "predicted_mask": file_info(pred_path),
            "overlay": file_info(overlay_path),
        },
        "training_metrics": read_json(training_metrics_path),
        "inference_metrics": read_json(inference_metrics_path),
        "disclaimer": "Educational prototype only. Not for clinical use.",
    }