from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mne
import numpy as np

from src.seizure_detection.windowing import make_sliding_windows, label_windows_from_intervals
from src.seizure_detection.features import extract_features_from_windows
from src.seizure_detection.train import train_random_forest_seizure


def load_chbmit_edf(edf_path: str | Path, max_channels: int = 8):
    """
    Load one CHB-MIT EDF file.

    Parameters
    ----------
    edf_path:
        Path to CHB-MIT .edf file.

    max_channels:
        Use only first few EEG channels for a simple prototype.
    """

    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(
            f"EDF file not found: {edf_path}\n"
            "Please place your CHB-MIT EDF file under data/seizure_eeg/"
        )

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    raw.pick_types(eeg=True)

    if len(raw.ch_names) > max_channels:
        raw.pick_channels(raw.ch_names[:max_channels])

    raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)

    eeg = raw.get_data()
    fs = float(raw.info["sfreq"])

    return eeg, fs, raw.ch_names


def main() -> None:
    """
    Train seizure detector from one CHB-MIT EDF file.

    Example target file:
        data/seizure_eeg/chb01_03.edf

    Known seizure interval for chb01_03.edf:
        2996 sec to 3036 sec
    """

    edf_path = PROJECT_ROOT / "data/seizure_eeg/chb01_03.edf"

    seizure_intervals = [
        (2996, 3036),
    ]

    print("Loading EDF file...")
    eeg, fs, channels = load_chbmit_edf(edf_path)

    print("Loaded EEG")
    print("Shape:", eeg.shape)
    print("Sampling rate:", fs)
    print("Channels:", channels)

    print("Creating sliding windows...")
    window_sec = 5.0
    step_sec = 2.5

    windows, starts_sec = make_sliding_windows(
        eeg=eeg,
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
    )

    y = label_windows_from_intervals(
        starts_sec=starts_sec,
        window_sec=window_sec,
        seizure_intervals=seizure_intervals,
        overlap_threshold=0.25,
    )

    print("Windows:", windows.shape)
    print("Label counts:", np.bincount(y))

    print("Extracting features...")
    X_features = extract_features_from_windows(windows, fs)

    print("Feature matrix:", X_features.shape)

    print("Training Random Forest seizure classifier...")
    result = train_random_forest_seizure(
        X_features=X_features,
        y=y,
        model_path="models/seizure_rf_chbmit_one_file.pkl",
        test_size=0.25,
        random_state=42,
    )

    print()
    print("Training complete.")
    print("Model path:", result.get("model_path"))
    print("Confusion matrix:", result.get("confusion_matrix_path"))
    print("Metrics path:", result.get("metrics_path"))
    print("Accuracy:", result.get("accuracy"))
    print("F1-score:", result.get("f1_score"))
    print("Sensitivity:", result.get("sensitivity"))
    print("Specificity:", result.get("specificity"))
    print("Precision:", result.get("precision"))


if __name__ == "__main__":
    main()