"""Leaf-branch image transforms: CLAHE, augmentation, optional HSV masking."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CLAHETransform:
    """Apply CLAHE contrast equalization on the L channel in LAB color space."""

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        l_equalized = clahe.apply(l_channel)
        merged = cv2.merge((l_equalized, a_channel, b_channel))
        equalized_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(equalized_rgb)


def hsv_canopy_mask(
    image: Image.Image,
    lower_green: tuple[int, int, int] = (25, 40, 40),
    upper_green: tuple[int, int, int] = (95, 255, 255),
    background_value: int = 0,
) -> Image.Image:
    """Optional leaf canopy mask using HSV green thresholds.

    Not part of the default training pipeline — validate usefulness before enabling.
    Pixels outside the green canopy range are replaced with ``background_value``.
    """
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower = np.array(lower_green, dtype=np.uint8)
    upper = np.array(upper_green, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    masked = rgb.copy()
    masked[mask == 0] = background_value
    return Image.fromarray(masked)


def build_leaf_transforms(
    train: bool,
    image_size: int = 224,
    clahe: CLAHETransform | None = None,
) -> Callable:
    """Build a torchvision transform pipeline for the leaf classifier."""
    clahe_transform = clahe or CLAHETransform()
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train:
        return transforms.Compose(
            [
                clahe_transform,
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            clahe_transform,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean
