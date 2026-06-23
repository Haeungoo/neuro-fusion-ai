from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter


router = APIRouter(tags=["motor"])


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "motor_imagery"
MODELS_DIR = PROJECT_ROOT / "models"


def file_info(path: Path, media_path: str) -> dict[str, Any]:
    """
    Return file availability and media URL information.

    media_path is relative to /media/results/.
    Example:
    motor_imagery/motor_imagery_csp_topomap.png
    """
    exists = path.exists()

    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "exists": exists,
        "url": media_path if exists else None,
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def count_subjects_from_csv(path: Path) -> int | None:
    """
    Count unique subjects from the PhysioNet subject comparison CSV.
    """
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            subjects: set[str] = set()

            for row in reader:
                subject = (
                    row.get("subject")
                    or row.get("subject_id")
                    or row.get("Subject")
                    or row.get("id")
                )

                if subject is not None and str(subject).strip():
                    subjects.add(str(subject).strip())

            return len(subjects) if subjects else None

    except Exception:
        return None


def count_motor_labels_from_predictions_csv(path: Path) -> dict[str, int]:
    """
    Count left-hand and right-hand samples from motor prediction CSV.

    This function is intentionally flexible because different scripts may use
    different column names such as true_label, label, y_true, target, or class.
    """
    counts = {
        "left_hand_count": 0,
        "right_hand_count": 0,
    }

    if not path.exists():
        return counts

    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                label = (
                    row.get("true_label")
                    or row.get("label")
                    or row.get("y_true")
                    or row.get("target")
                    or row.get("class")
                    or row.get("event")
                    or row.get("condition")
                )

                if label is None:
                    continue

                label_text = str(label).strip().lower()

                if label_text in {"left", "left_hand", "left hand", "1", "t1"}:
                    counts["left_hand_count"] += 1

                elif label_text in {"right", "right_hand", "right hand", "2", "t2"}:
                    counts["right_hand_count"] += 1

    except Exception:
        return counts

    return counts

def first_value(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def normalize_motor_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize different possible motor metrics JSON structures into
    frontend-friendly keys.
    """
    if not raw:
        return {}

    metrics: dict[str, Any] = {}

    classification = raw.get("classification")
    if isinstance(classification, dict):
        metrics.update(classification)

    performance = raw.get("performance")
    if isinstance(performance, dict):
        metrics.update(performance)

    model = raw.get("model")
    if isinstance(model, dict):
        metrics.update(
            {
                "num_csp_components": model.get("num_csp_components")
                or model.get("csp_components"),
                "num_channels": model.get("num_channels"),
            }
        )

    dataset = raw.get("dataset")
    if isinstance(dataset, dict):
        metrics.update(
            {
                "num_subjects": dataset.get("num_subjects"),
                "num_channels": dataset.get("num_channels")
                or metrics.get("num_channels"),
                "left_hand_count": dataset.get("left_hand_count"),
                "right_hand_count": dataset.get("right_hand_count"),
            }
        )

    # Top-level aliases
    alias_map = {
        "accuracy": ["accuracy", "test_accuracy", "mean_accuracy"],
        "precision": ["precision"],
        "recall": ["recall", "sensitivity"],
        "f1_score": ["f1_score", "f1", "macro_f1"],
        "best_subject": ["best_subject", "subject", "subject_id"],
        "num_subjects": ["num_subjects", "subjects_evaluated"],
        "num_channels": ["num_channels", "channels"],
        "num_csp_components": ["num_csp_components", "csp_components"],
        "left_hand_count": ["left_hand_count", "left_count"],
        "right_hand_count": ["right_hand_count", "right_count"],
    }

    for normalized_key, possible_keys in alias_map.items():
        if metrics.get(normalized_key) is None:
            value = first_value(raw, possible_keys)
            if value is not None:
                metrics[normalized_key] = value

    # If a subject-search JSON exists, it may store the best result separately.
    best_result = raw.get("best_result")
    if isinstance(best_result, dict):
        if metrics.get("best_subject") is None:
            metrics["best_subject"] = first_value(
                best_result,
                ["subject", "subject_id", "best_subject"],
            )

        if metrics.get("accuracy") is None:
            metrics["accuracy"] = first_value(
                best_result,
                ["accuracy", "test_accuracy", "mean_accuracy"],
            )

        if metrics.get("f1_score") is None:
            metrics["f1_score"] = first_value(
                best_result,
                ["f1_score", "f1", "macro_f1"],
            )

    return metrics


def load_motor_metrics() -> dict[str, Any]:
    """
    Load motor imagery metrics from the most useful available JSON file.
    """
    candidate_paths = [
        RESULTS_DIR / "motor_imagery_metrics.json",
        RESULTS_DIR / "physionet_subject_comparison_best.json",
    ]

    for path in candidate_paths:
        data = read_json(path)
        if data:
            metrics = normalize_motor_metrics(data)
            if metrics:
                return metrics

    return {}


def output_files() -> dict[str, Any]:
    return {
        "confusion_matrix": file_info(
            RESULTS_DIR / "motor_imagery_confusion_matrix.png",
            "motor_imagery/motor_imagery_confusion_matrix.png",
        ),
        "subject_comparison_plot": file_info(
            RESULTS_DIR / "physionet_subject_comparison_accuracy.png",
            "motor_imagery/physionet_subject_comparison_accuracy.png",
        ),
        "csp_topomap": file_info(
            RESULTS_DIR / "motor_imagery_csp_topomap.png",
            "motor_imagery/motor_imagery_csp_topomap.png",
        ),
        "csp_3d_topomap": file_info(
            RESULTS_DIR / "motor_imagery_csp_3d_topomap.html",
            "motor_imagery/motor_imagery_csp_3d_topomap.html",
        ),
        "csp_3d_topomap_metadata": file_info(
            RESULTS_DIR / "motor_imagery_csp_3d_topomap_metadata.json",
            "motor_imagery/motor_imagery_csp_3d_topomap_metadata.json",
        ),
        "predictions_csv": file_info(
            RESULTS_DIR / "motor_imagery_predictions.csv",
            "motor_imagery/motor_imagery_predictions.csv",
        ),
        "metrics_json": file_info(
            RESULTS_DIR / "motor_imagery_metrics.json",
            "motor_imagery/motor_imagery_metrics.json",
        ),
        "subject_comparison_csv": file_info(
            RESULTS_DIR / "physionet_subject_comparison.csv",
            "motor_imagery/physionet_subject_comparison.csv",
        ),
        "subject_comparison_best_json": file_info(
            RESULTS_DIR / "physionet_subject_comparison_best.json",
            "motor_imagery/physionet_subject_comparison_best.json",
        ),
        "model_file": file_info(
            MODELS_DIR / "motor_imagery_csp_lda.joblib",
            "../models/motor_imagery_csp_lda.joblib",
        ),
        "subject_search_model_file": file_info(
            MODELS_DIR / "motor_imagery_physionet_subject_search.joblib",
            "../models/motor_imagery_physionet_subject_search.joblib",
        ),
    }


@router.get("/status")
def get_motor_status() -> dict[str, Any]:
    metrics = load_motor_metrics()
    outputs = output_files()

    subject_count = count_subjects_from_csv(
        RESULTS_DIR / "physionet_subject_comparison.csv"
    )

    if metrics.get("num_subjects") is None and subject_count is not None:
        metrics["num_subjects"] = subject_count

    label_counts = count_motor_labels_from_predictions_csv(
        RESULTS_DIR / "motor_imagery_predictions.csv"
    )

    if metrics.get("left_hand_count") is None and label_counts["left_hand_count"] > 0:
        metrics["left_hand_count"] = label_counts["left_hand_count"]

    if metrics.get("right_hand_count") is None and label_counts["right_hand_count"] > 0:
        metrics["right_hand_count"] = label_counts["right_hand_count"]

    ready = any(
        [
            outputs["confusion_matrix"]["exists"],
            outputs["subject_comparison_plot"]["exists"],
            outputs["csp_topomap"]["exists"],
            outputs["csp_3d_topomap"]["exists"],
            bool(metrics),
        ]
    )

    return {
        "module": "EEG Motor Imagery BCI",
        "status": "ready" if ready else "missing_results",
        "metrics": metrics,
        "outputs": outputs,
    }


@router.get("/health")
def motor_health() -> dict[str, str]:
    return {
        "module": "motor",
        "status": "ok",
    }