from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


try:
    from src.common.paths import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


RESULTS_DIR = PROJECT_ROOT / "results" / "mri"


CASE_METRICS_CANDIDATES = [
    RESULTS_DIR / "mri_case_validation_metrics.json",
    RESULTS_DIR / "mri_validation_cases.json",
    RESULTS_DIR / "mri_best_worst_source_cases.json",
    RESULTS_DIR / "case_validation_metrics.json",
]


OVERLAY_CANDIDATES = [
    RESULTS_DIR / "mri_prediction_overlay.png",
    RESULTS_DIR / "prediction_overlay.png",
    RESULTS_DIR / "overlay.png",
]


INPUT_CANDIDATES = [
    RESULTS_DIR / "mri_input_slice.png",
    RESULTS_DIR / "input_slice.png",
]


GROUND_TRUTH_CANDIDATES = [
    RESULTS_DIR / "mri_ground_truth_mask.png",
    RESULTS_DIR / "ground_truth_mask.png",
]


PREDICTED_MASK_CANDIDATES = [
    RESULTS_DIR / "mri_predicted_mask.png",
    RESULTS_DIR / "predicted_mask.png",
]


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path

    return None


def load_case_metrics() -> list[dict[str, Any]]:
    """
    Expected flexible formats:

    1. List:
    [
      {"case_id": "BraTS_001", "dice": 0.91, "overlay_path": "..."},
      {"case_id": "BraTS_002", "dice": 0.44, "overlay_path": "..."}
    ]

    2. Dict with cases:
    {
      "cases": [
        {"case_id": "BraTS_001", "dice": 0.91, "overlay_path": "..."}
      ]
    }

    3. Dict with validation_cases:
    {
      "validation_cases": [
        {"case_id": "BraTS_001", "mean_dice": 0.91}
      ]
    }
    """

    for path in CASE_METRICS_CANDIDATES:
        data = read_json(path)

        if data is None:
            continue

        if isinstance(data, list):
            return [
                item
                for item in data
                if isinstance(item, dict)
            ]

        if isinstance(data, dict):
            if isinstance(data.get("cases"), list):
                return [
                    item
                    for item in data["cases"]
                    if isinstance(item, dict)
                ]

            if isinstance(data.get("validation_cases"), list):
                return [
                    item
                    for item in data["validation_cases"]
                    if isinstance(item, dict)
                ]

    return []


def get_case_score(case: dict[str, Any]) -> float | None:
    score_keys = [
        "dice",
        "dice_score",
        "mean_dice",
        "case_dice",
        "case_mean_dice",
        "validation_dice",
    ]

    for key in score_keys:
        value = case.get(key)

        if isinstance(value, int | float):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue

    return None


def resolve_case_overlay_path(case: dict[str, Any]) -> Path | None:
    path_keys = [
        "overlay_path",
        "prediction_overlay_path",
        "overlay",
        "image_path",
        "output_path",
    ]

    for key in path_keys:
        value = case.get(key)

        if not isinstance(value, str):
            continue

        path = Path(value)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if path.exists():
            return path

    return None


def copy_or_create_labeled_overlay(
    source_path: Path,
    output_path: Path,
    title: str,
    subtitle: str,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        image = Image.open(source_path).convert("RGB")
        image_array = np.asarray(image)

        plt.figure(figsize=(7, 6))
        plt.imshow(image_array)
        plt.axis("off")
        plt.title(
            title,
            fontsize=14,
            fontweight="bold",
        )

        plt.figtext(
            0.5,
            0.03,
            subtitle,
            ha="center",
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=170,
            bbox_inches="tight",
        )
        plt.close()

    except Exception:
        shutil.copyfile(
            source_path,
            output_path,
        )


def create_composite_from_existing_mri_outputs(
    output_path: Path,
    title: str,
    subtitle: str,
) -> bool:
    input_path = first_existing_path(INPUT_CANDIDATES)
    ground_truth_path = first_existing_path(GROUND_TRUTH_CANDIDATES)
    predicted_path = first_existing_path(PREDICTED_MASK_CANDIDATES)
    overlay_path = first_existing_path(OVERLAY_CANDIDATES)

    panel_paths = [
        input_path,
        ground_truth_path,
        predicted_path,
        overlay_path,
    ]

    if any(path is None for path in panel_paths):
        return False

    titles = [
        "Input MRI",
        "Ground Truth",
        "Predicted Mask",
        "Overlay",
    ]

    images = [
        Image.open(path).convert("RGB")
        for path in panel_paths
        if path is not None
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14, 4),
    )

    for axis, image, panel_title in zip(axes, images, titles):
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            panel_title,
            fontsize=10,
        )

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.03,
        subtitle,
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)

    return True


def create_best_worst_from_case_metrics(
    cases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    scored_cases = []

    for case in cases:
        score = get_case_score(case)

        if score is None:
            continue

        scored_cases.append(
            {
                "case": case,
                "score": score,
            }
        )

    if not scored_cases:
        return None

    scored_cases = sorted(
        scored_cases,
        key=lambda item: item["score"],
    )

    worst = scored_cases[0]
    best = scored_cases[-1]

    best_case = best["case"]
    worst_case = worst["case"]

    best_overlay_source = resolve_case_overlay_path(best_case)
    worst_overlay_source = resolve_case_overlay_path(worst_case)

    fallback_overlay = first_existing_path(OVERLAY_CANDIDATES)

    best_output_path = RESULTS_DIR / "mri_best_case_overlay.png"
    worst_output_path = RESULTS_DIR / "mri_worst_case_overlay.png"

    if best_overlay_source is not None:
        copy_or_create_labeled_overlay(
            source_path=best_overlay_source,
            output_path=best_output_path,
            title="Best Validation Case",
            subtitle=f"Dice score: {best['score']:.4f}",
        )
    elif fallback_overlay is not None:
        copy_or_create_labeled_overlay(
            source_path=fallback_overlay,
            output_path=best_output_path,
            title="Best Validation Case",
            subtitle=(
                f"Dice score: {best['score']:.4f} "
                "(fallback overlay image)"
            ),
        )
    else:
        create_composite_from_existing_mri_outputs(
            output_path=best_output_path,
            title="Best Validation Case",
            subtitle=f"Dice score: {best['score']:.4f}",
        )

    if worst_overlay_source is not None:
        copy_or_create_labeled_overlay(
            source_path=worst_overlay_source,
            output_path=worst_output_path,
            title="Worst Validation Case",
            subtitle=f"Dice score: {worst['score']:.4f}",
        )
    elif fallback_overlay is not None:
        copy_or_create_labeled_overlay(
            source_path=fallback_overlay,
            output_path=worst_output_path,
            title="Worst Validation Case",
            subtitle=(
                f"Dice score: {worst['score']:.4f} "
                "(fallback overlay image)"
            ),
        )
    else:
        create_composite_from_existing_mri_outputs(
            output_path=worst_output_path,
            title="Worst Validation Case",
            subtitle=f"Dice score: {worst['score']:.4f}",
        )

    return {
        "mode": "case_metrics",
        "best_case": {
            "case_id": (
                best_case.get("case_id")
                or best_case.get("case")
                or best_case.get("name")
                or "unknown"
            ),
            "dice": best["score"],
            "source": best_case,
            "output_image": str(best_output_path),
        },
        "worst_case": {
            "case_id": (
                worst_case.get("case_id")
                or worst_case.get("case")
                or worst_case.get("name")
                or "unknown"
            ),
            "dice": worst["score"],
            "source": worst_case,
            "output_image": str(worst_output_path),
        },
        "note": (
            "Best and worst cases were selected from case-level validation "
            "metrics found in results/mri."
        ),
    }


def create_fallback_best_worst() -> dict[str, Any]:
    best_output_path = RESULTS_DIR / "mri_best_case_overlay.png"
    worst_output_path = RESULTS_DIR / "mri_worst_case_overlay.png"

    overlay_path = first_existing_path(OVERLAY_CANDIDATES)

    if overlay_path is not None:
        copy_or_create_labeled_overlay(
            source_path=overlay_path,
            output_path=best_output_path,
            title="Best Validation Case",
            subtitle="Fallback example overlay. Case-level ranking not available.",
        )

        copy_or_create_labeled_overlay(
            source_path=overlay_path,
            output_path=worst_output_path,
            title="Worst Validation Case",
            subtitle="Fallback example overlay. Case-level ranking not available.",
        )

    else:
        best_created = create_composite_from_existing_mri_outputs(
            output_path=best_output_path,
            title="Best Validation Case",
            subtitle="Fallback example. Case-level ranking not available.",
        )

        worst_created = create_composite_from_existing_mri_outputs(
            output_path=worst_output_path,
            title="Worst Validation Case",
            subtitle="Fallback example. Case-level ranking not available.",
        )

        if not best_created or not worst_created:
            raise FileNotFoundError(
                "No MRI overlay or panel images were found. "
                "Run the MRI inference script first."
            )

    return {
        "mode": "fallback_example",
        "best_case": {
            "case_id": "example_case",
            "dice": None,
            "output_image": str(best_output_path),
        },
        "worst_case": {
            "case_id": "example_case",
            "dice": None,
            "output_image": str(worst_output_path),
        },
        "note": (
            "Case-level validation metrics were not found. "
            "The same available MRI example was used as a placeholder for "
            "dashboard layout testing. To enable true best/worst selection, "
            "generate per-case Dice metrics and case-specific overlay images."
        ),
    }


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = load_case_metrics()

    metadata = create_best_worst_from_case_metrics(
        cases,
    )

    if metadata is None:
        metadata = create_fallback_best_worst()

    metadata_path = RESULTS_DIR / "mri_best_worst_cases.json"

    save_json(
        metadata_path,
        metadata,
    )

    print("[saved]", RESULTS_DIR / "mri_best_case_overlay.png")
    print("[saved]", RESULTS_DIR / "mri_worst_case_overlay.png")
    print("[saved]", metadata_path)


if __name__ == "__main__":
    main()