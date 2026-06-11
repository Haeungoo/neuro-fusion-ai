from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.common.paths import PROJECT_ROOT


router = APIRouter(
    # prefix="/api/seizure",
    tags=["seizure"],
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "seizure"
)


def read_json(
    path: Path,
) -> dict[str, Any] | None:
    """
    Safely read a JSON file.
    """

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
    """
    Return file existence and browser-accessible media path.
    """

    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "url": None,
        }

    relative_path = path.relative_to(
        PROJECT_ROOT / "results"
    )

    return {
        "exists": True,
        "path": str(path),
        "url": f"/media/results/{relative_path.as_posix()}",
    }


@router.get("/status")
def seizure_status() -> dict[str, Any]:
    """
    Return seizure detection module status and result files.
    """

    synthetic_waveform_path = (
        RESULTS_DIR
        / "synthetic_waveform.png"
    )

    synthetic_timeline_path = (
        RESULTS_DIR
        / "synthetic_probability_timeline.png"
    )

    synthetic_confusion_matrix_path = (
        RESULTS_DIR
        / "synthetic_confusion_matrix.png"
    )

    chbmit_one_file_waveform_path = (
        RESULTS_DIR
        / "chbmit_chb01_03_waveform.png"
    )

    chbmit_one_file_timeline_path = (
        RESULTS_DIR
        / "chbmit_chb01_03_probability_timeline.png"
    )

    chbmit_one_file_confusion_matrix_path = (
        RESULTS_DIR
        / "chbmit_chb01_03_confusion_matrix.png"
    )

    chbmit_multi_file_waveform_path = (
        RESULTS_DIR
        / "chbmit_multi_file_waveform.png"
    )

    chbmit_multi_file_timeline_path = (
        RESULTS_DIR
        / "chbmit_multi_file_probability_timeline.png"
    )

    chbmit_multi_file_confusion_matrix_path = (
        RESULTS_DIR
        / "chbmit_multi_file_confusion_matrix.png"
    )

    random_forest_metrics_path = (
        RESULTS_DIR
        / "seizure_random_forest_metrics.json"
    )

    random_forest_confusion_matrix_path = (
        RESULTS_DIR
        / "seizure_random_forest_confusion_matrix.png"
    )

    evaluation_metrics_path = (
        RESULTS_DIR
        / "seizure_metrics.json"
    )

    evaluation_per_file_path = (
        RESULTS_DIR
        / "seizure_metrics_per_file.csv"
    )

    evaluation_confusion_matrix_path = (
        RESULTS_DIR
        / "seizure_evaluation_confusion_matrix.png"
    )

    prediction_csv_path = (
        RESULTS_DIR
        / "chbmit_multi_file_predictions.csv"
    )

    feature_npz_path = (
        RESULTS_DIR
        / "chbmit_multi_file_features.npz"
    )

    return {
        "module": "EEG Seizure Detection",
        "status": "ready",
        "description": (
            "Random Forest-based EEG seizure detection "
            "with CHB-MIT window-level evaluation metrics."
        ),
        "available_modes": [
            "synthetic",
            "chbmit_one_file",
            "chbmit_multi_file",
        ],
        "results": {
            "synthetic": {
                "waveform": file_info(
                    synthetic_waveform_path
                ),
                "probability_timeline": file_info(
                    synthetic_timeline_path
                ),
                "confusion_matrix": file_info(
                    synthetic_confusion_matrix_path
                ),
            },
            "chbmit_one_file": {
                "waveform": file_info(
                    chbmit_one_file_waveform_path
                ),
                "probability_timeline": file_info(
                    chbmit_one_file_timeline_path
                ),
                "confusion_matrix": file_info(
                    chbmit_one_file_confusion_matrix_path
                ),
            },
            "chbmit_multi_file": {
                "waveform": file_info(
                    chbmit_multi_file_waveform_path
                ),
                "probability_timeline": file_info(
                    chbmit_multi_file_timeline_path
                ),
                "confusion_matrix": file_info(
                    chbmit_multi_file_confusion_matrix_path
                ),
            },
        },
        "training_outputs": {
            "feature_dataset": file_info(
                feature_npz_path
            ),
            "prediction_csv": file_info(
                prediction_csv_path
            ),
            "random_forest_metrics": read_json(
                random_forest_metrics_path
            ),
            "random_forest_confusion_matrix": file_info(
                random_forest_confusion_matrix_path
            ),
        },
        "evaluation_metrics": read_json(
            evaluation_metrics_path
        ),
        "evaluation_outputs": {
            "per_file_metrics_csv": file_info(
                evaluation_per_file_path
            ),
            "confusion_matrix": file_info(
                evaluation_confusion_matrix_path
            ),
        },
    }