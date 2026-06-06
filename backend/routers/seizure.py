from __future__ import annotations

from fastapi import APIRouter

from src.common.paths import PROJECT_ROOT
from backend.routers.files import file_info


router = APIRouter()


@router.get("/status")
def seizure_status() -> dict:
    return {
        "module": "EEG Seizure Detection",
        "model": "Random Forest",
        "features": [
            "bandpower",
            "entropy",
            "line length",
            "Hjorth parameters",
        ],
        "modes": {
            "synthetic": {
                "model_file": file_info(PROJECT_ROOT / "models/seizure_rf.pkl"),
                "waveform": file_info(PROJECT_ROOT / "results/seizure/synthetic_waveform.png"),
                "timeline": file_info(PROJECT_ROOT / "results/seizure/synthetic_probability_timeline.png"),
                "confusion_matrix": file_info(PROJECT_ROOT / "results/seizure/synthetic_confusion_matrix.png"),
            },
            "chbmit_one_file": {
                "dataset": "CHB-MIT",
                "file": "chb01_03.edf",
                "known_seizure_interval_sec": [2996, 3036],
                "model_file": file_info(PROJECT_ROOT / "models/seizure_rf_chbmit_one_file.pkl"),
                "waveform": file_info(PROJECT_ROOT / "results/seizure/chbmit_chb01_03_waveform.png"),
                "timeline": file_info(PROJECT_ROOT / "results/seizure/chbmit_chb01_03_probability_timeline.png"),
                "confusion_matrix": file_info(PROJECT_ROOT / "results/seizure/chbmit_chb01_03_confusion_matrix.png"),
            },
            "chbmit_multi_file": {
                "dataset": "CHB-MIT",
                "subject": "chb01",
                "model_file": file_info(PROJECT_ROOT / "models/seizure_rf_chbmit_multi_file.pkl"),
                "waveform": file_info(PROJECT_ROOT / "results/seizure/chbmit_multi_file_waveform.png"),
                "timeline": file_info(PROJECT_ROOT / "results/seizure/chbmit_multi_file_probability_timeline.png"),
                "confusion_matrix": file_info(PROJECT_ROOT / "results/seizure/chbmit_multi_file_confusion_matrix.png"),
            },
        },
        "disclaimer": "Educational prototype only. Not for clinical seizure detection.",
    }