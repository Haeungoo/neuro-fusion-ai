from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import nibabel as nib


def find_first_existing(case_dir: Path, patterns: list[str]) -> Path | None:
    """
    Find the first file matching one of the given patterns.
    This helps support different BraTS naming conventions.
    """
    for pattern in patterns:
        matches = sorted(case_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def normalize_slice(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalize one MRI slice using nonzero brain pixels.
    """
    image = image.astype(np.float32)

    nonzero = image != 0

    if np.any(nonzero):
        mean = image[nonzero].mean()
        std = image[nonzero].std() + 1e-8
        image = (image - mean) / std
    else:
        image = (image - image.mean()) / (image.std() + 1e-8)

    return image.astype(np.float32)


def process_case(
    case_dir: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    min_mask_pixels: int = 20,
) -> int:
    """
    Convert one BraTS case folder into 2D .npy image/mask pairs.

    This version uses one modality:
        FLAIR / T2-FLAIR / t2f

    Mask:
        seg > 0 becomes tumor mask.
    """

    image_path = find_first_existing(
        case_dir,
        patterns=[
            "*t2f*.nii.gz",
            "*flair*.nii.gz",
            "*FLAIR*.nii.gz",
            "*t2f*.nii",
            "*flair*.nii",
        ],
    )

    seg_path = find_first_existing(
        case_dir,
        patterns=[
            "*seg*.nii.gz",
            "*Seg*.nii.gz",
            "*mask*.nii.gz",
            "*seg*.nii",
            "*mask*.nii",
        ],
    )

    if image_path is None:
        print(f"Skipping {case_dir.name}: no FLAIR/T2F image found")
        return 0

    if seg_path is None:
        print(f"Skipping {case_dir.name}: no segmentation mask found")
        return 0

    print()
    print("----------------------------------------")
    print(f"Processing case: {case_dir.name}")
    print("Image:", image_path.name)
    print("Seg:", seg_path.name)

    image_vol = nib.load(str(image_path)).get_fdata()
    seg_vol = nib.load(str(seg_path)).get_fdata()

    if image_vol.shape != seg_vol.shape:
        print(
            f"Skipping {case_dir.name}: shape mismatch "
            f"{image_vol.shape} vs {seg_vol.shape}"
        )
        return 0

    count = 0

    for z in range(image_vol.shape[2]):
        image_slice = image_vol[:, :, z]
        mask_slice = seg_vol[:, :, z] > 0

        # Skip slices without enough tumor pixels
        if int(mask_slice.sum()) < min_mask_pixels:
            continue

        image_slice = normalize_slice(image_slice)
        mask_slice = mask_slice.astype(np.float32)

        stem = f"{case_dir.name}_slice_{z:03d}.npy"

        np.save(output_image_dir / stem, image_slice)
        np.save(output_mask_dir / stem, mask_slice)

        count += 1

    print(f"Saved {count} tumor-containing slices from {case_dir.name}")

    return count


def main() -> None:
    raw_dir = PROJECT_ROOT / "data/mri/brats_raw"
    output_image_dir = PROJECT_ROOT / "data/mri/processed/images"
    output_mask_dir = PROJECT_ROOT / "data/mri/processed/masks"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    if not case_dirs:
        raise RuntimeError(
            f"No BraTS case folders found in {raw_dir}.\n"
            "Put one BraTS case folder inside data/mri/brats_raw/ first."
        )

    total = 0

    for case_dir in case_dirs:
        total += process_case(
            case_dir=case_dir,
            output_image_dir=output_image_dir,
            output_mask_dir=output_mask_dir,
            min_mask_pixels=20,
        )

    print()
    print("========================================")
    print("BraTS preprocessing complete")
    print("========================================")
    print("Total saved slices:", total)
    print("Images:", output_image_dir)
    print("Masks:", output_mask_dir)


if __name__ == "__main__":
    main()
