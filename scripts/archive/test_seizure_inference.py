from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.seizure_detection.inference import run_seizure_dashboard_demo


def main() -> None:
    """
    Test seizure inference using the dashboard-safe demo function.

    Before running this file, train the seizure model:

        python -m scripts.train_seizure_demo

    This script expects:

        models/seizure_rf.pkl

    This script creates:

        results/seizure/eeg_waveform_input.png
        results/seizure/seizure_probability_timeline.png
    """

    print("Running seizure inference test...")
    print("--------------------------------")

    result = run_seizure_dashboard_demo()

    print("Seizure inference completed.")
    print("--------------------------------")
    print("Prediction:", result.get("prediction"))
    print("Number of windows:", result.get("num_windows"))
    print("Detected windows:", result.get("num_detected_windows"))
    print("Max probability:", result.get("max_probability"))
    print("First detection time, sec:", result.get("first_detection_sec"))
    print("Waveform path:", result.get("waveform_path"))
    print("Timeline path:", result.get("timeline_path"))

    metrics = result.get("model_metrics", {})

    if metrics:
        print()
        print("Model metrics")
        print("--------------------------------")
        print("Accuracy:", metrics.get("accuracy"))
        print("F1-score:", metrics.get("f1_score"))
        print("Sensitivity:", metrics.get("sensitivity"))
        print("Specificity:", metrics.get("specificity"))
        print("Precision:", metrics.get("precision"))
    else:
        print()
        print("No saved model metrics found in model bundle.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()