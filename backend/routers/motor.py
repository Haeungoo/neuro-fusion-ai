from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.common.paths import PROJECT_ROOT


router = APIRouter(
    tags=["motor-imagery"],
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "motor_imagery"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)


def read_json(
    path: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError:
        return None


def file_info(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "url": None,
        }

    try:
        relative_path = path.relative_to(
            PROJECT_ROOT / "results"
        )

        return {
            "exists": True,
            "path": str(path),
            "url": f"/media/results/{relative_path.as_posix()}",
        }

    except ValueError:
        return {
            "exists": True,
            "path": str(path),
            "url": None,
        }


@router.get("/status")
def motor_status() -> dict[str, Any]:
    metrics_path = (
        RESULTS_DIR
        / "motor_imagery_metrics.json"
    )

    predictions_path = (
        RESULTS_DIR
        / "motor_imagery_predictions.csv"
    )

    confusion_matrix_path = (
        RESULTS_DIR
        / "motor_imagery_confusion_matrix.png"
    )

    model_path = (
        MODELS_DIR
        / "motor_imagery_csp_lda.joblib"
    )

    metrics = read_json(
        metrics_path
    )

    is_ready = (
        metrics_path.exists()
        and predictions_path.exists()
        and confusion_matrix_path.exists()
    )

    return {
        "module": "EEG Motor Imagery BCI",
        "status": "ready" if is_ready else "missing_outputs",
        "description": (
            "CSP + LDA baseline for left-hand versus "
            "right-hand motor imagery classification."
        ),
        "dataset_note": (
            "Current prototype uses synthetic motor imagery "
            "EEG-like data. Later this can be replaced with "
            "real motor imagery EEG data."
        ),
        "labels": {
            "0": "left_hand_imagery",
            "1": "right_hand_imagery",
        },
        "pipeline": [
            "EEG trials",
            "CSP spatial filtering",
            "Log-variance features",
            "Linear Discriminant Analysis",
            "Left/right imagery prediction",
        ],
        "metrics": metrics,
        "outputs": {
            "metrics_json": file_info(
                metrics_path
            ),
            "predictions_csv": file_info(
                predictions_path
            ),
            "confusion_matrix": file_info(
                confusion_matrix_path
            ),
            "model": file_info(
                model_path
            ),
            "model_file": file_info(
                model_path
            ),
        },
    }