from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_mri_slice(image: np.ndarray) -> np.ndarray:
    """
    Normalize one MRI slice using nonzero brain pixels.
    """

    image = np.asarray(image, dtype=np.float32)
    nonzero = image != 0

    if np.any(nonzero):
        mean = float(image[nonzero].mean())
        std = float(image[nonzero].std()) + 1e-8
        image = (image - mean) / std
    else:
        image = (image - image.mean()) / (
            image.std() + 1e-8
        )

    return image.astype(np.float32)


class MRISliceDataset(Dataset):
    """
    Dataset for paired MRI image and tumor mask .npy files.

    Expected folders:

        data/mri/processed/images/*.npy
        data/mri/processed/masks/*.npy
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        return_metadata: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.return_metadata = return_metadata

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask directory not found: {self.mask_dir}"
            )

        image_paths = sorted(self.image_dir.glob("*.npy"))

        if not image_paths:
            raise FileNotFoundError(
                f"No .npy image files found in {self.image_dir}"
            )

        self.pairs: list[tuple[Path, Path]] = []

        for image_path in image_paths:
            mask_path = self.mask_dir / image_path.name

            if mask_path.exists():
                self.pairs.append((image_path, mask_path))
            else:
                print(
                    f"Warning: missing mask for {image_path.name}"
                )

        if not self.pairs:
            raise FileNotFoundError(
                "No matching MRI image/mask pairs were found."
            )

        self.image_paths = [
            image_path for image_path, _ in self.pairs
        ]
        self.mask_paths = [
            mask_path for _, mask_path in self.pairs
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if image.ndim != 2:
            raise ValueError(
                f"Expected 2D MRI slice. Got {image.shape} "
                f"from {image_path}"
            )

        if mask.ndim != 2:
            raise ValueError(
                f"Expected 2D mask. Got {mask.shape} "
                f"from {mask_path}"
            )

        image = normalize_mri_slice(image)

        image_tensor = torch.from_numpy(
            image[None, :, :]
        ).float()

        mask_tensor = torch.from_numpy(
            mask[None, :, :]
        ).float()

        if self.return_metadata:
            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "filename": image_path.name,
            }

        return image_tensor, mask_tensor