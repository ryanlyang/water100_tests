"""Lightweight source-data primitives for the original DecoyMNIST PNGs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


NUM_CLASSES = 10
PATCH_SIZE = 5


def expected_patch_intensity(label: int, split: str) -> int:
    if not 0 <= int(label) < NUM_CLASSES:
        raise ValueError(f"Invalid DecoyMNIST label: {label}")
    if split == "train":
        return 255 - 25 * int(label)
    if split == "test":
        return 25 * int(label)
    raise ValueError(f"split must be 'train' or 'test', found {split!r}")


def corner_slices(height: int, width: int) -> List[Tuple[slice, slice]]:
    if height < PATCH_SIZE or width < PATCH_SIZE:
        raise ValueError(f"Image is too small for a {PATCH_SIZE}x{PATCH_SIZE} corner patch")
    return [
        (slice(0, PATCH_SIZE), slice(0, PATCH_SIZE)),
        (slice(0, PATCH_SIZE), slice(width - PATCH_SIZE, width)),
        (slice(height - PATCH_SIZE, height), slice(0, PATCH_SIZE)),
        (
            slice(height - PATCH_SIZE, height),
            slice(width - PATCH_SIZE, width),
        ),
    ]


def locate_decoy_patch(
    grayscale: np.ndarray, label: int, split: str, *, tolerance: float = 1.0
) -> Tuple[slice, slice]:
    """Locate and validate the published class-coded corner patch."""

    if grayscale.ndim != 2:
        raise ValueError(f"Expected a grayscale HxW image, got {grayscale.shape}")
    expected = float(expected_patch_intensity(label, split))
    corners = corner_slices(*grayscale.shape)
    errors = [
        float(np.abs(grayscale[rows, cols].astype(np.float32) - expected).mean())
        for rows, cols in corners
    ]
    selected = int(np.argmin(np.asarray(errors)))
    rows, cols = corners[selected]
    max_error = float(
        np.abs(grayscale[rows, cols].astype(np.float32) - expected).max()
    )
    if max_error > tolerance:
        raise ValueError(
            "Image does not match the unmodified DecoyMNIST encoding: "
            f"label={label} split={split} expected={expected:.1f} "
            f"best_mean_error={errors[selected]:.3f} max_error={max_error:.3f}"
        )
    return rows, cols


def discover_samples(root: str | Path, split: str) -> Dict[int, List[Path]]:
    split_root = Path(root).expanduser().resolve() / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing DecoyMNIST split: {split_root}")
    by_label: Dict[int, List[Path]] = {}
    for label in range(NUM_CLASSES):
        class_root = split_root / str(label)
        paths = sorted(class_root.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No PNGs found under {class_root}")
        by_label[label] = paths
    return by_label

