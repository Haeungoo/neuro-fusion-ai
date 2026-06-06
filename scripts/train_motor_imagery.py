from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.motor_imagery.config import (
    SUBJECTS,
    RUNS,
    FMIN,
    FMAX,
    TMIN,
    TMAX,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
)
from src.motor_imagery.loader import load_physionet_eegbci
from src.motor_imagery.preprocess import preprocess_raw, create_left_right_epochs
from src.motor_imagery.train import train_csp_lda


def main() -> None:
    """
    Train CSP + LDA motor imagery classifier using PhysioNet EEGBCI data.

    Pipeline:
        PhysioNet EEGBCI
        ↓
        Load EDF files
        ↓
        8–30 Hz band-pass filtering
        ↓
        T1/T2 epoch extraction
        ↓
        CSP feature extraction
        ↓
        LDA classifier
        ↓
        Save model and results

    Expected outputs:
        models/motor_csp_lda.pkl
        results/motor/confusion_matrix_motor.png
        results/motor/motor_metrics.json
    """

    print("========================================")
    print("Motor Imagery Training: CSP + LDA")
    print("========================================")

    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Configuration")
    print("----------------------------------------")
    print("Subjects:", SUBJECTS)
    print("Runs:", RUNS)
    print("Filter band:", FMIN, "-", FMAX, "Hz")
    print("Epoch window:", TMIN, "-", TMAX, "sec")
    print("Test size:", TEST_SIZE)
    print("Random state:", RANDOM_STATE)
    print("Model path:", MODEL_PATH)

    print()
    print("Step 1. Loading PhysioNet EEGBCI data...")
    raw = load_physionet_eegbci(
        subjects=SUBJECTS,
        runs=RUNS,
    )

    print("Loaded raw EEG:")
    print(raw)

    print()
    print("Step 2. Filtering EEG...")
    raw_filtered = preprocess_raw(
        raw=raw,
        fmin=FMIN,
        fmax=FMAX,
    )

    print("Filtering complete.")

    print()
    print("Step 3. Creating left/right motor imagery epochs...")
    X, y, epochs = create_left_right_epochs(
        raw=raw_filtered,
        tmin=TMIN,
        tmax=TMAX,
    )

    print("Epoch creation complete.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Number of epochs:", len(y))
    print("Unique labels:", sorted(set(y.tolist())))

    print()
    print("Step 4. Training CSP + LDA model...")
    result = train_csp_lda(
        X=X,
        y=y,
        model_path=MODEL_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        info=epochs.info,
    )

    print()
    print("Training complete.")
    print("========================================")
    print("Results")
    print("========================================")
    print("Model path:", result.get("model_path"))
    print("Confusion matrix path:", result.get("confusion_matrix_path"))
    print("CSP patterns path:", result.get("csp_patterns_path"))
    print("Metrics path:", result.get("metrics_path"))
    print("Accuracy:", result.get("accuracy"))
    print("F1-score:", result.get("f1_score"))

    print()
    print("Classification report")
    print("----------------------------------------")
    print(result.get("classification_report"))

    print()
    print("Done.")


if __name__ == "__main__":
    main()