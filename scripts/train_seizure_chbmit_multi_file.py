from __future__ import annotations

import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from src.seizure_detection.chbmit_loader import (
    parse_chbmit_summary,
    load_chbmit_edf,
    list_available_seizure_edfs,
)
from src.seizure_detection.windowing import (
    make_sliding_windows,
    label_windows_from_intervals,
)
from src.seizure_detection.features import extract_features_from_windows
from src.seizure_detection.train import train_random_forest_seizure


def main() -> None:
    """
    Train seizure detector using multiple CHB-MIT EDF files from chb01.

    Expected folder:
        data/seizure_eeg/chb01/

    Required:
        chb01-summary.txt
        several chb01_XX.edf files
    """

    data_dir = PROJECT_ROOT / "data/seizure_eeg/chb01"
    summary_path = data_dir / "chb01-summary.txt"

    window_sec = 5.0
    step_sec = 2.5
    max_channels = 8

    print("========================================")
    print("CHB-MIT Multi-file Seizure Training")
    print("========================================")

    print("Data dir:", data_dir)
    print("Summary:", summary_path)

    seizure_map = parse_chbmit_summary(summary_path)

    print()
    print("Parsed seizure files:")
    for file_name, intervals in seizure_map.items():
        print(file_name, intervals)

    edf_paths = list_available_seizure_edfs(data_dir, seizure_map)

    if not edf_paths:
        raise RuntimeError(
            "No seizure EDF files found. "
            "Download CHB-MIT EDF files into data/seizure_eeg/chb01/"
        )

    all_features = []
    all_labels = []

    for edf_path in edf_paths:
        file_name = edf_path.name
        intervals = seizure_map[file_name]

        print()
        print("----------------------------------------")
        print("Processing:", file_name)
        print("Seizure intervals:", intervals)

        eeg, fs, channels = load_chbmit_edf(
            edf_path=edf_path,
            max_channels=max_channels,
        )

        print("EEG shape:", eeg.shape)
        print("Sampling rate:", fs)
        print("Channels:", channels)

        windows, starts_sec = make_sliding_windows(
            eeg=eeg,
            fs=fs,
            window_sec=window_sec,
            step_sec=step_sec,
        )

        y = label_windows_from_intervals(
            starts_sec=starts_sec,
            window_sec=window_sec,
            seizure_intervals=intervals,
            overlap_threshold=0.25,
        )

        print("Windows:", windows.shape)
        print("Label counts:", np.bincount(y))

        X_features = extract_features_from_windows(windows, fs)

        all_features.append(X_features)
        all_labels.append(y)

    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_labels)

    print()
    print("========================================")
    print("Combined dataset")
    print("========================================")
    print("X_all:", X_all.shape)
    print("y_all:", y_all.shape)
    print("Label counts:", np.bincount(y_all))

    print()
    print("Training Random Forest on CHB-MIT multi-file data...")

    result = train_random_forest_seizure(
        X_features=X_all,
        y=y_all,
        model_path="models/seizure_rf_chbmit_multi_file.pkl",
        test_size=0.25,
        random_state=42,
    )

    # Preserve confusion matrix with CHB-MIT multi-file name
    default_cm = PROJECT_ROOT / "results/seizure/confusion_matrix_seizure.png"
    multi_cm = PROJECT_ROOT / "results/seizure/chbmit_multi_file_confusion_matrix.png"

    if default_cm.exists():
        shutil.copyfile(default_cm, multi_cm)

    print()
    print("Training complete.")
    print("Model path:", result.get("model_path"))
    print("Confusion matrix:", multi_cm)
    print("Metrics path:", result.get("metrics_path"))
    print("Accuracy:", result.get("accuracy"))
    print("F1-score:", result.get("f1_score"))
    print("Sensitivity:", result.get("sensitivity"))
    print("Specificity:", result.get("specificity"))
    print("Precision:", result.get("precision"))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
