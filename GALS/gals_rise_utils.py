#!/usr/bin/env python3
"""Shared utilities for deterministic, GALS-style RISE explanations."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
from skimage.transform import resize


def generate_gals_rise_masks(
    num_masks: int,
    grid_size: int,
    height: int,
    width: int,
    p1: float,
    seed: int,
) -> np.ndarray:
    """Reproduce the random-mask construction in ``utils/rise.py``."""
    if num_masks <= 0:
        raise ValueError("num_masks must be positive")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if not 0.0 < p1 <= 1.0:
        raise ValueError("p1 must be in (0, 1]")

    rng = np.random.RandomState(int(seed))
    cell_size = np.ceil(np.asarray([height, width]) / int(grid_size)).astype(int)
    up_size = (int(grid_size) + 1) * cell_size
    grids = (rng.rand(int(num_masks), int(grid_size), int(grid_size)) < float(p1)).astype(
        np.float32
    )
    masks = np.empty((int(num_masks), 1, int(height), int(width)), dtype=np.float32)

    for index in range(int(num_masks)):
        shift_row = int(rng.randint(0, cell_size[0]))
        shift_col = int(rng.randint(0, cell_size[1]))
        upsampled = resize(
            grids[index],
            tuple(int(value) for value in up_size),
            order=1,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        )
        masks[index, 0] = upsampled[
            shift_row : shift_row + int(height),
            shift_col : shift_col + int(width),
        ]
    return masks


def validate_mask_bank(
    masks: np.ndarray,
    num_masks: int,
    height: int,
    width: int,
) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.float32)
    expected = (int(num_masks), 1, int(height), int(width))
    if masks.shape != expected:
        raise ValueError(f"RISE mask bank has shape {masks.shape}; expected {expected}")
    if not np.isfinite(masks).all():
        raise ValueError("RISE mask bank contains non-finite values")
    if float(masks.min()) < -1e-6 or float(masks.max()) > 1.0 + 1e-6:
        raise ValueError("RISE mask bank values must lie in [0, 1]")
    return np.ascontiguousarray(masks)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_create_mask_bank(
    path: Path,
    num_masks: int,
    grid_size: int,
    height: int,
    width: int,
    p1: float,
    seed: int,
) -> Tuple[np.ndarray, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        masks = validate_mask_bank(np.load(path), num_masks, height, width)
    else:
        masks = generate_gals_rise_masks(
            num_masks=num_masks,
            grid_size=grid_size,
            height=height,
            width=width,
            p1=p1,
            seed=seed,
        )
        masks = validate_mask_bank(masks, num_masks, height, width)
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
        np.save(temporary, masks)
        os.replace(temporary, path)
    return masks, sha256_file(path)


@torch.no_grad()
def rise_from_probabilities_batch(
    probability_fn: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    p1: float,
    max_masked_batch: int,
) -> torch.Tensor:
    """Compute class-targeted RISE maps for a batch of source images."""
    batch_size, channels, height, width = images.shape
    num_masks = int(masks.shape[0])
    masks_per_step = max(1, int(max_masked_batch) // max(batch_size, 1))
    total = torch.zeros((batch_size, height, width), device=images.device)

    for start in range(0, num_masks, masks_per_step):
        mask_chunk = masks[start : start + masks_per_step]
        chunk_size = int(mask_chunk.shape[0])
        masked = (images[:, None] * mask_chunk[None]).reshape(
            batch_size * chunk_size, channels, height, width
        )
        probabilities = probability_fn(masked)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise RuntimeError(
                f"Probability adapter returned invalid shape {tuple(probabilities.shape)}"
            )
        probabilities = probabilities.view(batch_size, chunk_size, -1)
        gather_index = targets.view(batch_size, 1, 1).expand(-1, chunk_size, 1)
        scores = probabilities.gather(2, gather_index).squeeze(2)
        total += torch.einsum("bm,mhw->bhw", scores, mask_chunk[:, 0])

    return total / max(float(num_masks) * float(p1), 1e-12)
