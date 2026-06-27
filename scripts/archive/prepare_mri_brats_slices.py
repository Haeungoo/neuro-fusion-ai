from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import nibabel as nib

def find_case_dirs(raw_dir: Path) -> list[Path]:
    """
    Find BraTS case directories.

    It searches recursively for folders that contain FLAIR and segmentation files.
    """

    case_dirs: list[Path] = []

    for path in raw_dir.rglob("*"):
        if not path.is_dir():
            continue

        has_flair = any(path.glob("*_flair.nii")) or any(
            path.glob("*_flair.nii.gz")
        )

        has_seg = any(path.glob("*_seg.nii")) or any(
            path.glob("*_seg.nii.gz")
        )

        if has_flair and has_seg:
            case_dirs.append(path)

    return sorted(case_dirs)

def find_modality_file(
    case_dir: Path,
    modality: str,
) -> Path | None:
    """
    Find a BraTS modality file inside one case folder.

    Expected examples:
        BraTS20_Training_001_flair.nii
        BraTS20_Training_001_t1.nii
        BraTS20_Training_001_t1ce.nii
        BraTS20_Training_001_t2.nii
        BraTS20_Training_001_seg.nii

    This function also supports .nii.gz files.
    """

    modality = modality.lower()

    patterns = [
        f"*_{modality}.nii",
        f"*_{modality}.nii.gz",
    ]

    for pattern in patterns:
        matches = sorted(case_dir.glob(pattern))

        if matches:
            return matches[0]

    return None

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
    modality: str = "flair",
    min_mask_pixels: int = 20,
) -> int:
    """
    Process one BraTS case into paired 2D MRI image and mask slices.

    The saved filename preserves the BraTS case ID so that
    patient-level train/validation splitting can be performed later.
    """

    image_path = find_modality_file(
        case_dir=case_dir,
        modality=modality,
    )

    seg_path = find_modality_file(
        case_dir=case_dir,
        modality="seg",
    )

    if image_path is None:
        print(
            f"Skipping {case_dir.name}: "
            f"{modality} image not found"
        )
        return 0

    if seg_path is None:
        print(
            f"Skipping {case_dir.name}: "
            "segmentation file not found"
        )
        return 0

    image_volume = nib.load(
        str(image_path)
    ).get_fdata().astype(np.float32)

    seg_volume = nib.load(
        str(seg_path)
    ).get_fdata().astype(np.float32)

    if image_volume.shape != seg_volume.shape:
        print(
            f"Skipping {case_dir.name}: "
            f"image shape {image_volume.shape} "
            f"does not match mask shape {seg_volume.shape}"
        )
        return 0

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_count = 0

    for slice_index in range(image_volume.shape[2]):
        image_slice = image_volume[
            :, :, slice_index
        ]

        mask_slice = (
            seg_volume[:, :, slice_index] > 0
        ).astype(np.float32)

        if int(mask_slice.sum()) < min_mask_pixels:
            continue

        image_slice = normalize_slice(
            image_slice
        ).astype(np.float32)

        filename = (
            f"{case_dir.name}"
            f"_slice_{slice_index:03d}.npy"
        )

        np.save(
            output_image_dir / filename,
            image_slice,
        )

        np.save(
            output_mask_dir / filename,
            mask_slice,
        )

        saved_count += 1

    print(
        f"{case_dir.name}: "
        f"saved {saved_count} {modality} slices"
    )

    return saved_count


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
        
    print("Found total case folders:", len(case_dirs))
    
    max_cases = 20
    case_dirs = case_dirs[:max_cases]
    
    print("Cases selected for preprocessing:", len(case_dirs))

    total = 0

    for case_dir in case_dirs:
        total += process_case(
            case_dir=case_dir,
            output_image_dir=output_image_dir,
            output_mask_dir=output_mask_dir,
            modality="flair",
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
