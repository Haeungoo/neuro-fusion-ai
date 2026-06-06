from __future__ import annotations

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.motor_imagery.loader import load_physionet_eegbci
from src.motor_imagery.preprocess import preprocess_raw, create_left_right_epochs
from src.motor_imagery.train import train_csp_lda


def main() -> None:
    subjects = [1, 2, 3, 4, 5]
    runs = [4, 8, 12]

    results = {}

    for subject in subjects:
        print()
        print("========================================")
        print(f"Training subject {subject}")
        print("========================================")

        try:
            raw = load_physionet_eegbci(
                subjects=[subject],
                runs=runs,
            )

            raw = preprocess_raw(
                raw=raw,
                fmin=8.0,
                fmax=30.0,
            )

            X, y, epochs = create_left_right_epochs(
                raw=raw,
                tmin=1.0,
                tmax=4.0,
            )

            result = train_csp_lda(
                X=X,
                y=y,
                model_path=f"models/motor_subject_{subject}_csp_lda.pkl",
                test_size=0.25,
                random_state=42,
            )

            results[f"subject_{subject}"] = {
                "accuracy": result.get("accuracy"),
                "f1_score": result.get("f1_score"),
                "model_path": result.get("model_path"),
                "confusion_matrix_path": result.get("confusion_matrix_path"),
            }

            print("Accuracy:", result.get("accuracy"))
            print("F1-score:", result.get("f1_score"))

        except Exception as e:
            print(f"Subject {subject} failed.")
            print(e)

            results[f"subject_{subject}"] = {
                "error": str(e),
            }

    output_path = PROJECT_ROOT / "results/motor/motor_subject_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("Saved benchmark:")
    print(output_path)


if __name__ == "__main__":
    main()