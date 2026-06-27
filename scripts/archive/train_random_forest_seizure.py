from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.seizure_detection.train import (
    train_and_save_random_forest_seizure,
)


def main() -> None:
    results_dir = (
        PROJECT_ROOT
        / "results"
        / "seizure"
    )

    models_dir = (
        PROJECT_ROOT
        / "models"
    )

    feature_path = (
        results_dir
        / "chbmit_multi_file_features.npz"
    )

    model_path = (
        models_dir
        / "seizure_random_forest.joblib"
    )

    prediction_csv_path = (
        results_dir
        / "chbmit_multi_file_predictions.csv"
    )

    metrics_path = (
        results_dir
        / "seizure_random_forest_metrics.json"
    )

    confusion_matrix_path = (
        results_dir
        / "seizure_random_forest_confusion_matrix.png"
    )

    result = train_and_save_random_forest_seizure(
        feature_path=feature_path,
        model_path=model_path,
        prediction_csv_path=prediction_csv_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        test_size=0.2,
        random_state=42,
        n_estimators=300,
    )

    metrics = result["metrics"]

    print("=" * 68)
    print("Random Forest seizure training complete")
    print("=" * 68)
    print("Model:", result["model_path"])
    print("Prediction CSV:", result["prediction_csv_path"])
    print("Metrics:", result["metrics_path"])
    print("Confusion matrix:", result["confusion_matrix_path"])
    print("-" * 68)
    print(
        "Sensitivity:",
        f"{metrics['sensitivity']:.4f}",
    )
    print(
        "Specificity:",
        f"{metrics['specificity']:.4f}",
    )
    print(
        "Precision:",
        f"{metrics['precision']:.4f}",
    )
    print(
        "F1 score:",
        f"{metrics['f1_score']:.4f}",
    )
    print(
        "Accuracy:",
        f"{metrics['accuracy']:.4f}",
    )


if __name__ == "__main__":
    main()