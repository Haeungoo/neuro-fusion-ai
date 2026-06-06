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
def motor_status() -> dict:
    metrics_path = PROJECT_ROOT / "results/motor/motor_metrics.json"

    return {
        "module": "EEG Motor Imagery BCI",
        "dataset": "PhysioNet EEGBCI",
        "model": "CSP + LDA",
        "pipeline": [
            "8-30 Hz band-pass filtering",
            "T1/T2 epoch extraction",
            "CSP spatial filtering",
            "LDA classification",
        ],
        "model_file": file_info(PROJECT_ROOT / "models/motor_csp_lda.pkl"),
        "outputs": {
            "confusion_matrix": file_info(PROJECT_ROOT / "results/motor/confusion_matrix_motor.png"),
            "csp_patterns": file_info(PROJECT_ROOT / "results/motor/csp_patterns.png"),
            "metrics": file_info(metrics_path),
        },
        "metrics": read_json(metrics_path),
        "note": "CSP spatial patterns are discriminative EEG spatial patterns, not direct brain activation maps.",
    }