from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mne

from src.seizure_detection.inference import run_seizure_inference


def load_chbmit_edf(edf_path: str | Path, max_channels: int = 8):
    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True)

    if len(raw.ch_names) > max_channels:
        raw.pick_channels(raw.ch_names[:max_channels])

    raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)

    eeg = raw.get_data()
    fs = float(raw.info["sfreq"])

    return eeg, fs, raw.ch_names


def main() -> None:
    edf_path = PROJECT_ROOT / "data/seizure_eeg/chb01_03.edf"

    print("Loading CHB-MIT EDF...")
    eeg, fs, channels = load_chbmit_edf(edf_path)

    print("EEG shape:", eeg.shape)
    print("Sampling rate:", fs)
    print("Channels:", channels)

    print("Running CHB-MIT seizure inference...")

    result = run_seizure_inference(
        eeg=eeg,
        fs=fs,
        model_path="models/seizure_rf_chbmit_one_file.pkl",
        window_sec=5.0,
        step_sec=2.5,
        threshold=0.5,
    )

    print("Prediction:", result.get("prediction"))
    print("Max probability:", result.get("max_probability"))
    print("Detected windows:", result.get("num_detected_windows"))
    print("First detection sec:", result.get("first_detection_sec"))
    print("Waveform path:", result.get("waveform_path"))
    print("Timeline path:", result.get("timeline_path"))


if __name__ == "__main__":
    main()