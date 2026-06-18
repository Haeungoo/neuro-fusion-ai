from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.common.paths import PROJECT_ROOT


router = APIRouter(
    tags=["seizure-detection"],
)


RESULTS_DIR = PROJECT_ROOT / "results" / "seizure"
MODELS_DIR = PROJECT_ROOT / "models"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

        return None

    except json.JSONDecodeError:
        return None


def first_existing_path(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path

    return paths[0]


def first_existing_json(paths: list[Path]) -> dict[str, Any] | None:
    for path in paths:
        data = read_json(path)

        if data is not None:
            return data

    return None


def file_info(path: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    results_root = (PROJECT_ROOT / "results").resolve()

    if not resolved_path.exists():
        return {
            "exists": False,
            "path": str(resolved_path),
            "url": None,
        }

    try:
        relative_path = resolved_path.relative_to(results_root)

        return {
            "exists": True,
            "path": str(resolved_path),
            "url": f"/media/results/{relative_path.as_posix()}",
        }

    except ValueError:
        return {
            "exists": True,
            "path": str(resolved_path),
            "url": None,
        }

def normalize_metrics(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None

    metrics = dict(data)

    classification = data.get("classification")
    file_level_summary = data.get("file_level_summary")
    window_level_summary = data.get("window_level_summary")
    data_summary = data.get("data_summary")
    evaluation = data.get("evaluation")
    dataset = data.get("dataset")

    files = data.get("files")
    file_names = data.get("file_names")
    test_file_names = data.get("test_file_names")

    nested_sources = [
        classification,
        file_level_summary,
        window_level_summary,
        data_summary,
        evaluation,
    ]

    for source in nested_sources:
        if isinstance(source, dict):
            for key, value in source.items():
                if metrics.get(key) is None:
                    metrics[key] = value

    if isinstance(dataset, dict):
        for key, value in dataset.items():
            if metrics.get(key) is None:
                metrics[key] = value
    elif isinstance(dataset, str):
        metrics["dataset"] = dataset

    if metrics.get("num_files") is None:
        if isinstance(file_names, list):
            metrics["num_files"] = len(file_names)
        elif isinstance(test_file_names, list):
            metrics["num_files"] = len(test_file_names)
        elif isinstance(files, list):
            metrics["num_files"] = len(files)

    alias_map = {
        "accuracy": [
            "accuracy",
            "acc",
            "test_accuracy",
            "mean_accuracy",
        ],
        "precision": [
            "precision",
            "mean_precision",
        ],
        "recall": [
            "recall",
            "sensitivity",
            "mean_recall",
            "mean_sensitivity",
        ],
        "sensitivity": [
            "sensitivity",
            "recall",
            "mean_sensitivity",
            "mean_recall",
        ],
        "specificity": [
            "specificity",
            "mean_specificity",
        ],
        "f1_score": [
            "f1_score",
            "f1",
            "f1_macro",
            "mean_f1",
        ],
        "balanced_accuracy": [
            "balanced_accuracy",
            "mean_balanced_accuracy",
        ],
        "true_positive": [
            "true_positive",
            "tp",
        ],
        "true_negative": [
            "true_negative",
            "tn",
        ],
        "false_positive": [
            "false_positive",
            "fp",
        ],
        "false_negative": [
            "false_negative",
            "fn",
        ],
        "num_files": [
            "num_files",
            "n_files",
            "file_count",
            "num_test_files",
            "n_test_files",
            "evaluated_files",
            "num_evaluated_files",
        ],
        "num_windows": [
            "num_windows",
            "n_windows",
            "window_count",
            "num_segments",
            "n_segments",
            "total_windows",
            "total_segments",
            "num_test_windows",
            "n_test_windows",
            "evaluated_windows",
            "num_evaluated_windows",
            "num_samples",
            "n_samples",
            "sample_count",
        ],
        "num_samples": [
            "num_samples",
            "n_samples",
            "sample_count",
            "num_windows",
            "n_windows",
            "window_count",
            "num_segments",
            "n_segments",
        ],
    }

    for standard_key, aliases in alias_map.items():
        if metrics.get(standard_key) is not None:
            continue

        for alias in aliases:
            if metrics.get(alias) is not None:
                metrics[standard_key] = metrics[alias]
                break

    if metrics.get("num_windows") is None and metrics.get("num_samples") is not None:
        metrics["num_windows"] = metrics["num_samples"]

    if metrics.get("num_samples") is None and metrics.get("num_windows") is not None:
        metrics["num_samples"] = metrics["num_windows"]

    if metrics.get("num_files") is None:
        metrics["num_files"] = 1

    if metrics.get("model") is None:
        metrics["model"] = "Random Forest"

    if metrics.get("dataset") is None or isinstance(metrics.get("dataset"), dict):
        metrics["dataset"] = "CHB-MIT"

    return metrics

@router.get("/status")
def seizure_status() -> dict[str, Any]:
    metrics_candidates = [
        RESULTS_DIR / "seizure_multi_file_metrics.json",
        RESULTS_DIR / "seizure_metrics.json",
        RESULTS_DIR / "seizure_evaluation_metrics.json",
        RESULTS_DIR / "multi_file_metrics.json",
        RESULTS_DIR / "metrics.json",
    ]

    confusion_matrix_candidates = [
        RESULTS_DIR / "seizure_multi_file_confusion_matrix.png",
        RESULTS_DIR / "seizure_confusion_matrix.png",
        RESULTS_DIR / "seizure_evaluation_confusion_matrix.png",
        RESULTS_DIR / "confusion_matrix.png",
    ]

    predictions_candidates = [
        RESULTS_DIR / "seizure_predictions.csv",
        RESULTS_DIR / "seizure_multi_file_predictions.csv",
        RESULTS_DIR / "predictions.csv",
    ]

    waveform_path = RESULTS_DIR / "seizure_eeg_waveform.png"
    probability_timeline_path = RESULTS_DIR / "seizure_probability_timeline.png"
    visualization_metadata_path = RESULTS_DIR / "seizure_visualization_metadata.json"

    model_candidates = [
        MODELS_DIR / "seizure_random_forest.joblib",
        MODELS_DIR / "seizure_rf.joblib",
        MODELS_DIR / "seizure_model.joblib",
    ]

    metrics_path = first_existing_path(metrics_candidates)
    confusion_matrix_path = first_existing_path(confusion_matrix_candidates)
    predictions_path = first_existing_path(predictions_candidates)
    model_path = first_existing_path(model_candidates)

    raw_metrics = first_existing_json(metrics_candidates)
    metrics = normalize_metrics(raw_metrics)

    visualization_metadata = read_json(visualization_metadata_path)

    is_ready = metrics is not None

    has_visualization = (
        waveform_path.exists()
        and probability_timeline_path.exists()
    )

    return {
        "module": "EEG Seizure Detection",
        "status": "ready" if is_ready else "missing_outputs",
        "description": (
            "EEG seizure detection module using machine learning features "
            "extracted from CHB-MIT EEG recordings."
        ),
        "dataset_note": (
            "This module uses EEG recordings and derived signal features "
            "to classify seizure and non-seizure segments. The waveform and "
            "probability timeline provide visual context for model output."
        ),
        "labels": {
            "0": "non_seizure",
            "1": "seizure",
        },
        "pipeline": [
            "EEG recording",
            "Signal preprocessing",
            "Window segmentation",
            "Feature extraction",
            "Random Forest classification",
            "Seizure probability visualization",
        ],
        "metrics": metrics,
        "multi_file_metrics": metrics,
        "visualization_metadata": visualization_metadata,
        "outputs": {
            "metrics_json": file_info(metrics_path),
            "predictions_csv": file_info(predictions_path),
            "confusion_matrix": file_info(confusion_matrix_path),
            "multi_file_metrics_json": file_info(metrics_path),
            "multi_file_confusion_matrix": file_info(confusion_matrix_path),
            "waveform": file_info(waveform_path),
            "probability_timeline": file_info(probability_timeline_path),
            "visualization_metadata": file_info(visualization_metadata_path),
            "model": file_info(model_path),
            "model_file": file_info(model_path),
        },
        "visualization": {
            "available": has_visualization,
            "waveform": file_info(waveform_path),
            "probability_timeline": file_info(probability_timeline_path),
            "metadata": visualization_metadata,
        },
        "disclaimer": (
            "This model is for research and educational use only and is not "
            "intended for clinical diagnosis."
        ),
    }