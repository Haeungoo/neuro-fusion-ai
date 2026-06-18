from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


try:
    from src.common.paths import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


RESULTS_DIR = PROJECT_ROOT / "results" / "seizure"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data if isinstance(data, dict) else None


def find_metrics_file() -> Path | None:
    candidates = [
        RESULTS_DIR / "seizure_multi_file_metrics.json",
        RESULTS_DIR / "seizure_metrics.json",
        RESULTS_DIR / "seizure_evaluation_metrics.json",
        RESULTS_DIR / "multi_file_metrics.json",
        RESULTS_DIR / "metrics.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def get_nested_value(
    data: dict[str, Any],
    section_name: str,
    key: str,
) -> Any:
    section = data.get(section_name)

    if isinstance(section, dict):
        return section.get(key)

    return None


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = find_metrics_file()

    if metrics_path is None:
        raise FileNotFoundError(
            "No seizure metrics JSON file found in results/seizure."
        )

    metrics = read_json(metrics_path)

    if metrics is None:
        raise RuntimeError(
            f"Could not read metrics file: {metrics_path}"
        )

    accuracy = (
        get_nested_value(metrics, "classification", "accuracy")
        or metrics.get("accuracy")
    )

    precision = (
        get_nested_value(metrics, "classification", "precision")
        or metrics.get("precision")
    )

    recall = (
        get_nested_value(metrics, "classification", "recall")
        or get_nested_value(metrics, "classification", "sensitivity")
        or metrics.get("recall")
        or metrics.get("sensitivity")
    )

    sensitivity = (
        get_nested_value(metrics, "classification", "sensitivity")
        or get_nested_value(metrics, "classification", "recall")
        or metrics.get("sensitivity")
        or metrics.get("recall")
    )

    specificity = (
        get_nested_value(metrics, "classification", "specificity")
        or metrics.get("specificity")
    )

    f1_score = (
        get_nested_value(metrics, "classification", "f1_score")
        or get_nested_value(metrics, "classification", "f1")
        or metrics.get("f1_score")
        or metrics.get("f1")
    )

    balanced_accuracy = (
        get_nested_value(metrics, "classification", "balanced_accuracy")
        or metrics.get("balanced_accuracy")
    )

    num_files = (
        get_nested_value(metrics, "file_level_summary", "num_files")
        or get_nested_value(metrics, "data_summary", "num_files")
        or metrics.get("num_files")
        or 1
    )

    num_samples_or_windows = (
        get_nested_value(metrics, "window_level_summary", "num_samples")
        or get_nested_value(metrics, "window_level_summary", "num_windows")
        or metrics.get("num_samples")
        or metrics.get("num_windows")
        or "N/A"
    )

    true_positive = (
        get_nested_value(metrics, "classification", "true_positive")
        or get_nested_value(metrics, "classification", "tp")
        or metrics.get("true_positive")
        or metrics.get("tp")
    )

    true_negative = (
        get_nested_value(metrics, "classification", "true_negative")
        or get_nested_value(metrics, "classification", "tn")
        or metrics.get("true_negative")
        or metrics.get("tn")
    )

    false_positive = (
        get_nested_value(metrics, "classification", "false_positive")
        or get_nested_value(metrics, "classification", "fp")
        or metrics.get("false_positive")
        or metrics.get("fp")
    )

    false_negative = (
        get_nested_value(metrics, "classification", "false_negative")
        or get_nested_value(metrics, "classification", "fn")
        or metrics.get("false_negative")
        or metrics.get("fn")
    )

    output_path = RESULTS_DIR / "seizure_predictions.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "row_type",
                "source_metrics_file",
                "num_files",
                "num_samples_or_windows",
                "accuracy",
                "precision",
                "recall",
                "sensitivity",
                "specificity",
                "f1_score",
                "balanced_accuracy",
                "true_positive",
                "true_negative",
                "false_positive",
                "false_negative",
                "note",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "row_type": "summary",
                "source_metrics_file": str(metrics_path),
                "num_files": num_files,
                "num_samples_or_windows": num_samples_or_windows,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1_score": f1_score,
                "balanced_accuracy": balanced_accuracy,
                "true_positive": true_positive,
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "note": (
                    "Summary-level prediction CSV generated from existing "
                    "seizure metrics. True window-level predictions require "
                    "saved y_true, y_pred, and predict_proba outputs from the "
                    "evaluation script."
                ),
            }
        )

    print("[saved]", output_path)


if __name__ == "__main__":
    main()