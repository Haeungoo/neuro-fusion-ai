from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_mri_slice(image: np.ndarray) -> np.ndarray:
    """
    Normalize one MRI slice using nonzero pixels.

    Input:
        image: 2D MRI slice

    Output:
        normalized 2D MRI slice
    """
    image = np.asarray(image, dtype=np.float32)

    nonzero = image != 0

    if np.any(nonzero):
        mean = float(image[nonzero].mean())
        std = float(image[nonzero].std()) + 1e-8
        image = (image - mean) / std
    else:
        image = (image - image.mean()) / (image.std() + 1e-8)

    return image.astype(np.float32)


class MRISliceDataset(Dataset):
    """
    Dataset for paired MRI slice and tumor mask .npy files.

    Expected structure:

        data/mri/processed/images/*.npy
        data/mri/processed/masks/*.npy

    The image and mask file names must match.

    Example:

        images/BraTS20_Training_001_slice_080.npy
        masks/BraTS20_Training_001_slice_080.npy
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        image_paths = sorted(self.image_dir.glob("*.npy"))

        if not image_paths:
            raise FileNotFoundError(f"No .npy image files found in {self.image_dir}")

        self.pairs: list[tuple[Path, Path]] = []

        for image_path in image_paths:
            mask_path = self.mask_dir / image_path.name

            if mask_path.exists():
                self.pairs.append((image_path, mask_path))
            else:
                print(f"Warning: missing mask for {image_path.name}")

        if not self.pairs:
            raise FileNotFoundError(
                "No matching image/mask pairs found. "
                f"Image dir={self.image_dir}, mask dir={self.mask_dir}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if image.ndim != 2:
            raise ValueError(
                f"Expected image shape (H, W). Got {image.shape} from {image_path}"
            )

        if mask.ndim != 2:
            raise ValueError(
                f"Expected mask shape (H, W). Got {mask.shape} from {mask_path}"
            )

        image = normalize_mri_slice(image)
        mask = (mask > 0).astype(np.float32)

        # PyTorch expects channel-first format: (C, H, W)
        image = image[None, :, :]
        mask = mask[None, :, :]

        return torch.from_numpy(image), torch.from_numpy(mask)
