from __future__ import annotations

import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.seizure_detection.chbmit_loader import load_chbmit_edf
from src.seizure_detection.inference import run_seizure_inference


def main() -> None:
    """
    Test CHB-MIT multi-file model on one EDF file.

    Default test file:
        chb01_03.edf
    """

    edf_path = PROJECT_ROOT / "data/seizure_eeg/chb01/chb01_03.edf"

    print("Loading test EDF:", edf_path)

    eeg, fs, channels = load_chbmit_edf(
        edf_path=edf_path,
        max_channels=8,
    )

    print("EEG shape:", eeg.shape)
    print("Sampling rate:", fs)
    print("Channels:", channels)

    print("Running inference with multi-file CHB-MIT model...")

    result = run_seizure_inference(
        eeg=eeg,
        fs=fs,
        model_path="models/seizure_rf_chbmit_multi_file.pkl",
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

    default_waveform = PROJECT_ROOT / "results/seizure/eeg_waveform_input.png"
    default_timeline = PROJECT_ROOT / "results/seizure/seizure_probability_timeline.png"

    target_waveform = PROJECT_ROOT / "results/seizure/chbmit_multi_file_waveform.png"
    target_timeline = PROJECT_ROOT / "results/seizure/chbmit_multi_file_probability_timeline.png"

    if default_waveform.exists():
        shutil.copyfile(default_waveform, target_waveform)

    if default_timeline.exists():
        shutil.copyfile(default_timeline, target_timeline)

    print("Saved multi-file waveform:", target_waveform)
    print("Saved multi-file timeline:", target_timeline)

    print("Done.")


if __name__ == "__main__":
    main()
