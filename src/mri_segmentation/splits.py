from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence


def extract_case_id(filename: str) -> str:
    """
    Extract BraTS case ID from a processed slice filename.

    Example:
        BraTS20_Training_001_slice_080.npy

    Returns:
        BraTS20_Training_001
    """

    if "_slice_" not in filename:
        raise ValueError(
            f"Invalid processed filename: {filename}"
        )

    return filename.split("_slice_", maxsplit=1)[0]


def build_case_groups(
    image_paths: Sequence[Path],
) -> dict[str, list[int]]:
    """
    Group dataset indices by BraTS case ID.
    """

    case_groups: dict[str, list[int]] = {}

    for index, image_path in enumerate(image_paths):
        case_id = extract_case_id(image_path.name)
        case_groups.setdefault(case_id, []).append(index)

    return case_groups


def patient_level_split(
    image_paths: Sequence[Path],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int], list[str], list[str]]:
    """
    Split dataset by patient/case rather than by individual MRI slices.

    This prevents slices from the same patient appearing in both
    training and validation datasets.
    """

    case_groups = {}

    for index, image_path in enumerate(image_paths):
        case_id = extract_case_id(image_path.name)
        case_groups.setdefault(case_id, []).append(index)

    case_ids = sorted(case_groups)

    rng = random.Random(seed)
    rng.shuffle(case_ids)

    val_case_count = max(
        1,
        round(len(case_ids) * val_fraction),
    )

    if val_case_count >= len(case_ids):
        val_case_count = len(case_ids) - 1

    val_cases = sorted(case_ids[:val_case_count])
    train_cases = sorted(case_ids[val_case_count:])

    train_indices = []
    val_indices = []

    for case_id in train_cases:
        train_indices.extend(case_groups[case_id])

    for case_id in val_cases:
        val_indices.extend(case_groups[case_id])

    return (
        train_indices,
        val_indices,
        train_cases,
        val_cases,
    )


def save_split_manifest(
    output_path: str | Path,
    train_cases: Sequence[str],
    val_cases: Sequence[str],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    seed: int,
    val_fraction: float,
) -> Path:
    """
    Save the exact patient-level split for reproducibility.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "split_type": "patient_level",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train_cases": list(train_cases),
        "validation_cases": list(val_cases),
        "num_train_cases": len(train_cases),
        "num_validation_cases": len(val_cases),
        "num_train_slices": len(train_indices),
        "num_validation_slices": len(val_indices),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return output_path


def load_split_manifest(
    manifest_path: str | Path,
) -> dict:
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)