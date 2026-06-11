from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.seizure_detection.metrics import (
    evaluate_binary_predictions,
)


DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "seizure"
    / "chbmit_multi_file_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "seizure"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EEG seizure predictions and calculate "
            "classification and false-alarm metrics."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Prediction CSV path. Required columns: "
            "y_true and y_pred."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation outputs.",
    )

    parser.add_argument(
        "--step-seconds",
        type=float,
        default=2.5,
        help=(
            "Time difference between consecutive windows. "
            "Default: 2.5 seconds."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold used when y_pred is absent. "
            "Default: 0.5."
        ),
    )

    parser.add_argument(
        "--do-not-merge-false-alarms",
        action="store_true",
        help=(
            "Count each false-positive window as a separate "
            "false alarm."
        ),
    )

    return parser.parse_args()


def parse_binary_value(
    value: str,
    column_name: str,
    row_number: int,
) -> int:
    """
    Parse a CSV value as binary integer 0 or 1.
    """

    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {column_name} value at CSV row "
            f"{row_number}: {value!r}"
        ) from error

    if parsed not in (0, 1):
        raise ValueError(
            f"{column_name} must be 0 or 1 at CSV row "
            f"{row_number}. Got {parsed}."
        )

    return parsed


def parse_probability(
    value: str,
    row_number: int,
) -> float:
    """
    Parse a probability value between 0 and 1.
    """

    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid probability at CSV row "
            f"{row_number}: {value!r}"
        ) from error

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Probability must be between 0 and 1 at "
            f"CSV row {row_number}. Got {probability}."
        )

    return probability


def load_prediction_csv(
    path: Path,
    probability_threshold: float,
) -> list[dict[str, Any]]:
    """
    Load seizure prediction rows from CSV.

    Required:
        y_true

    One of:
        y_pred
        probability

    Optional:
        file_name
        window_start_sec
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: {path}\n"
            "Generate seizure predictions first."
        )

    rows: list[dict[str, Any]] = []

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {path}"
            )

        fieldnames = set(reader.fieldnames)

        if "y_true" not in fieldnames:
            raise ValueError(
                "Prediction CSV must contain a y_true column."
            )

        if (
            "y_pred" not in fieldnames
            and "probability" not in fieldnames
        ):
            raise ValueError(
                "Prediction CSV must contain either y_pred "
                "or probability."
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            y_true = parse_binary_value(
                value=row["y_true"],
                column_name="y_true",
                row_number=row_number,
            )

            probability: float | None = None

            if row.get("probability", "").strip():
                probability = parse_probability(
                    value=row["probability"],
                    row_number=row_number,
                )

            if row.get("y_pred", "").strip():
                y_pred = parse_binary_value(
                    value=row["y_pred"],
                    column_name="y_pred",
                    row_number=row_number,
                )
            elif probability is not None:
                y_pred = int(
                    probability >= probability_threshold
                )
            else:
                raise ValueError(
                    f"CSV row {row_number} has neither "
                    "y_pred nor probability."
                )

            file_name = (
                row.get("file_name", "").strip()
                or "unknown_file"
            )

            window_start_text = (
                row.get("window_start_sec", "").strip()
            )

            if window_start_text:
                try:
                    window_start_sec = float(
                        window_start_text
                    )
                except ValueError as error:
                    raise ValueError(
                        "Invalid window_start_sec at CSV row "
                        f"{row_number}: {window_start_text!r}"
                    ) from error
            else:
                window_start_sec = None

            rows.append(
                {
                    "file_name": file_name,
                    "window_start_sec": window_start_sec,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "probability": probability,
                }
            )

    if not rows:
        raise ValueError(
            f"Prediction CSV contains no data rows: {path}"
        )

    return rows


def calculate_file_level_metrics(
    rows: list[dict[str, Any]],
    step_seconds: float,
    merge_consecutive_windows: bool,
) -> list[dict[str, Any]]:
    """
    Calculate metrics separately for each EDF file.
    """

    grouped_rows: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped_rows[row["file_name"]].append(row)

    file_metrics: list[dict[str, Any]] = []

    for file_name, file_rows in sorted(
        grouped_rows.items()
    ):
        y_true = np.asarray(
            [row["y_true"] for row in file_rows],
            dtype=np.int64,
        )

        y_pred = np.asarray(
            [row["y_pred"] for row in file_rows],
            dtype=np.int64,
        )

        evaluation = evaluate_binary_predictions(
            y_true=y_true,
            y_pred=y_pred,
            step_seconds=step_seconds,
            merge_consecutive_windows=(
                merge_consecutive_windows
            ),
        )

        classification = evaluation["classification"]
        false_alarm = evaluation[
            "false_alarm_analysis"
        ]

        file_metrics.append(
            {
                "file_name": file_name,
                "num_windows": int(len(file_rows)),
                "sensitivity": float(
                    classification["sensitivity"]
                ),
                "specificity": float(
                    classification["specificity"]
                ),
                "precision": float(
                    classification["precision"]
                ),
                "f1_score": float(
                    classification["f1_score"]
                ),
                "accuracy": float(
                    classification["accuracy"]
                ),
                "balanced_accuracy": float(
                    classification[
                        "balanced_accuracy"
                    ]
                ),
                "true_positive": int(
                    classification["true_positive"]
                ),
                "true_negative": int(
                    classification["true_negative"]
                ),
                "false_positive": int(
                    classification["false_positive"]
                ),
                "false_negative": int(
                    classification["false_negative"]
                ),
                "false_alarm_events": int(
                    false_alarm[
                        "false_alarm_events"
                    ]
                ),
                "false_alarms_per_hour": float(
                    false_alarm[
                        "false_alarms_per_hour"
                    ]
                ),
            }
        )

    return file_metrics


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(
    true_negative: int,
    false_positive: int,
    false_negative: int,
    true_positive: int,
    output_path: Path,
) -> None:
    """
    Save a two-class confusion matrix.
    """

    matrix = np.asarray(
        [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        dtype=np.int64,
    )

    figure, axis = plt.subplots(
        figsize=(5, 4.5),
    )

    image = axis.imshow(
        matrix,
        cmap="Blues",
    )

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])

    axis.set_xticklabels(
        ["Non-seizure", "Seizure"]
    )

    axis.set_yticklabels(
        ["Non-seizure", "Seizure"]
    )

    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(
        "EEG Seizure Detection Confusion Matrix"
    )

    maximum = int(matrix.max())

    threshold = (
        maximum / 2.0
        if maximum > 0
        else 0.0
    )

    for row_index in range(2):
        for column_index in range(2):
            value = int(
                matrix[row_index, column_index]
            )

            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
                fontsize=13,
                fontweight="bold",
            )

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def safe_mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        sum(values) / len(values)
    )


def main() -> None:
    args = parse_arguments()

    input_path = args.input

    if not input_path.is_absolute():
        input_path = (
            PROJECT_ROOT / input_path
        )

    output_dir = args.output_dir

    if not output_dir.is_absolute():
        output_dir = (
            PROJECT_ROOT / output_dir
        )

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    if args.step_seconds <= 0:
        raise ValueError(
            "--step-seconds must be greater than 0."
        )

    merge_consecutive_windows = (
        not args.do_not_merge_false_alarms
    )

    rows = load_prediction_csv(
        path=input_path,
        probability_threshold=args.threshold,
    )

    y_true = np.asarray(
        [row["y_true"] for row in rows],
        dtype=np.int64,
    )

    y_pred = np.asarray(
        [row["y_pred"] for row in rows],
        dtype=np.int64,
    )

    evaluation = evaluate_binary_predictions(
        y_true=y_true,
        y_pred=y_pred,
        step_seconds=args.step_seconds,
        merge_consecutive_windows=(
            merge_consecutive_windows
        ),
    )

    classification = evaluation[
        "classification"
    ]

    false_alarm_analysis = evaluation[
        "false_alarm_analysis"
    ]

    file_level_metrics = (
        calculate_file_level_metrics(
            rows=rows,
            step_seconds=args.step_seconds,
            merge_consecutive_windows=(
                merge_consecutive_windows
            ),
        )
    )

    file_level_summary = {
        "num_files": len(
            file_level_metrics
        ),
        "mean_sensitivity": safe_mean(
            [
                row["sensitivity"]
                for row in file_level_metrics
            ]
        ),
        "mean_specificity": safe_mean(
            [
                row["specificity"]
                for row in file_level_metrics
            ]
        ),
        "mean_precision": safe_mean(
            [
                row["precision"]
                for row in file_level_metrics
            ]
        ),
        "mean_f1_score": safe_mean(
            [
                row["f1_score"]
                for row in file_level_metrics
            ]
        ),
        "mean_accuracy": safe_mean(
            [
                row["accuracy"]
                for row in file_level_metrics
            ]
        ),
        "mean_balanced_accuracy": safe_mean(
            [
                row["balanced_accuracy"]
                for row in file_level_metrics
            ]
        ),
        "mean_false_alarms_per_hour": safe_mean(
            [
                row["false_alarms_per_hour"]
                for row in file_level_metrics
            ]
        ),
    }

    result = {
        "evaluation_type": (
            "window_level_multi_file"
        ),
        "input_file": str(input_path),
        "probability_threshold": float(
            args.threshold
        ),
        "step_seconds": float(
            args.step_seconds
        ),
        "merge_consecutive_false_alarms": bool(
            merge_consecutive_windows
        ),
        "classification": classification,
        "false_alarm_analysis": (
            false_alarm_analysis
        ),
        "file_level_summary": (
            file_level_summary
        ),
    }

    metrics_path = (
        output_dir
        / "seizure_metrics.json"
    )

    per_file_path = (
        output_dir
        / "seizure_metrics_per_file.csv"
    )

    confusion_matrix_path = (
        output_dir
        / "seizure_evaluation_confusion_matrix.png"
    )

    save_json(
        path=metrics_path,
        data=result,
    )

    save_csv(
        path=per_file_path,
        rows=file_level_metrics,
    )

    save_confusion_matrix(
        true_negative=int(
            classification["true_negative"]
        ),
        false_positive=int(
            classification["false_positive"]
        ),
        false_negative=int(
            classification["false_negative"]
        ),
        true_positive=int(
            classification["true_positive"]
        ),
        output_path=confusion_matrix_path,
    )

    print("=" * 68)
    print("Seizure evaluation complete")
    print("=" * 68)
    print("Input:", input_path)
    print("Windows:", classification["num_samples"])
    print(
        "Sensitivity:",
        f"{classification['sensitivity']:.4f}",
    )
    print(
        "Specificity:",
        f"{classification['specificity']:.4f}",
    )
    print(
        "Precision:",
        f"{classification['precision']:.4f}",
    )
    print(
        "F1 score:",
        f"{classification['f1_score']:.4f}",
    )
    print(
        "Accuracy:",
        f"{classification['accuracy']:.4f}",
    )
    print(
        "Balanced accuracy:",
        f"{classification['balanced_accuracy']:.4f}",
    )
    print(
        "False alarms/hour:",
        (
            f"{false_alarm_analysis['false_alarms_per_hour']:.4f}"
        ),
    )
    print("-" * 68)
    print("Saved:", metrics_path)
    print("Saved:", per_file_path)
    print("Saved:", confusion_matrix_path)


if __name__ == "__main__":
    main()