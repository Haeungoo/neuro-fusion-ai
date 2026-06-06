from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from src.common.paths import PROJECT_ROOT


router = APIRouter()


def display_path(path: Path) -> str:
    """
    Convert absolute path to project-relative display path.

    Example:
        /Users/name/Downloads/neuro-fusion-ai/results/mri/file.png
    becomes:
        neuro-fusion-ai/results/mri/file.png
    """

    try:
        relative = path.relative_to(PROJECT_ROOT)
        return str(Path(PROJECT_ROOT.name) / relative)
    except ValueError:
        return str(path)


def file_info(path: Path) -> dict:
    return {
        "path": display_path(path),
        "exists": path.exists(),
    }


@router.get("/status")
def files_status() -> dict:
    """
    Return status of key output files.
    """

    files = {
        "mri_model": PROJECT_ROOT / "models/mri_unet.pt",
        "mri_overlay": PROJECT_ROOT / "results/mri/mri_prediction_overlay.png",
        "seizure_model": PROJECT_ROOT / "models/seizure_rf.pkl",
        "seizure_chbmit_model": PROJECT_ROOT / "models/seizure_rf_chbmit_one_file.pkl",
        "motor_model": PROJECT_ROOT / "models/motor_csp_lda.pkl",
        "motor_confusion_matrix": PROJECT_ROOT / "results/motor/confusion_matrix_motor.png",
    }

    return {
        name: file_info(path)
        for name, path in files.items()
    }