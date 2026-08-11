#!/usr/bin/env python3
"""Evaluate a DecoyMNIST checkpoint with GALS-style RISE Pointing Game.

The explanation target is the ground-truth class by default. One deterministic
RISE mask bank can be shared across every model and seed. Pointing Game masks
are reconstructed from the corresponding clean torchvision MNIST digit, so the
synthetic corner shortcut is excluded from the task-relevant region.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset, Subset

from decoymnist_pointing_game_eval import (
    ABNLeNet,
    CleanDigitMaskDataset,
    LeNet,
    SUPPORTED_METHODS,
    load_checkpoint_strict,
    pointing_result,
)


MASK_PROTOCOL_VERSION = 1
PRIMARY_PG_PROTOCOL = "rise_pixel_argmax"
EXPLAINER = "rise"


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def generate_gals_rise_masks(
    num_masks: int,
    grid_size: int,
    height: int,
    width: int,
    p1: float,
    seed: int,
) -> np.ndarray:
    """Reproduce the random-mask construction in the original GALS RISE code."""
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
    digest = hashlib.sha256(masks.tobytes(order="C")).hexdigest()
    return masks, digest


@torch.no_grad()
def rise_batch(
    model: nn.Module,
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
        masked = images[:, None] * mask_chunk[None]
        masked = masked.reshape(batch_size * chunk_size, channels, height, width)
        logits = model.forward_logits(masked)  # type: ignore[attr-defined]
        probabilities = torch.softmax(logits, dim=1).view(batch_size, chunk_size, -1)
        gather_index = targets.view(batch_size, 1, 1).expand(-1, chunk_size, 1)
        scores = probabilities.gather(2, gather_index).squeeze(2)
        total += torch.einsum("bm,mhw->bhw", scores, mask_chunk[:, 0])

    return total / max(float(num_masks) * float(p1), 1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-root", required=True)
    parser.add_argument("--mnist-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", required=True, choices=SUPPORTED_METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--target-mode", choices=("label", "prediction"), default="label")
    parser.add_argument("--mask-threshold", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--max-masked-batch", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--rise-num-masks", type=int, default=2000)
    parser.add_argument("--rise-grid-size", type=int, default=8)
    parser.add_argument("--rise-p1", type=float, default=0.1)
    parser.add_argument("--rise-seed", type=int, default=0)
    parser.add_argument("--rise-masks-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.sample_seed))
    np.random.seed(int(args.sample_seed))
    torch.manual_seed(int(args.seed))

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    png_root = Path(args.png_root).expanduser().resolve()
    mnist_root = Path(args.mnist_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mask_bank_path = Path(args.rise_masks_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not (png_root / ("test" if args.split == "test" else "train")).is_dir():
        raise FileNotFoundError(f"Missing DecoyMNIST split under: {png_root}")

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    dataset: Dataset = CleanDigitMaskDataset(
        png_root=png_root,
        mnist_root=mnist_root,
        split=args.split,
        mask_threshold=args.mask_threshold,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
    )
    if 0 < int(args.max_samples) < len(dataset):
        indices = list(range(len(dataset)))
        random.Random(int(args.sample_seed)).shuffle(indices)
        dataset = Subset(dataset, sorted(indices[: int(args.max_samples)]))

    loader = DataLoader(
        dataset,
        batch_size=int(args.image_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    model: nn.Module = ABNLeNet() if args.method == "abn" else LeNet()
    load_meta = load_checkpoint_strict(model, checkpoint)
    model.to(device).eval()

    masks_np, mask_hash = load_or_create_mask_bank(
        path=mask_bank_path,
        num_masks=args.rise_num_masks,
        grid_size=args.rise_grid_size,
        height=28,
        width=28,
        p1=args.rise_p1,
        seed=args.rise_seed,
    )
    rise_masks = torch.from_numpy(masks_np).to(device=device, dtype=torch.float32)

    class_hits = np.zeros(10, dtype=np.int64)
    class_totals = np.zeros(10, dtype=np.int64)
    class_correct = np.zeros(10, dtype=np.int64)
    class_mass_sum = np.zeros(10, dtype=np.float64)
    mask_pixels_total = 0
    zero_maps = 0
    rows: List[Dict[str, object]] = []

    print(
        f"[INFO] method={args.method} seed={args.seed} split={args.split} "
        f"samples={len(dataset)} device={device} explainer={EXPLAINER}"
    )
    print(f"[INFO] checkpoint={checkpoint} loaded_keys={load_meta['loaded_keys']}")
    print(
        f"[INFO] rise_masks={mask_bank_path} sha256={mask_hash} "
        f"N={args.rise_num_masks} grid={args.rise_grid_size} "
        f"p1={args.rise_p1} seed={args.rise_seed}"
    )
    print(f"[INFO] mask_source=clean_torchvision_mnist threshold>{args.mask_threshold}")

    for images, labels, digit_masks, paths, source_indices in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with torch.no_grad():
            prediction_logits = model.forward_logits(images)  # type: ignore[attr-defined]
            predictions = prediction_logits.argmax(dim=1)
        targets = labels_device if args.target_mode == "label" else predictions
        saliency = rise_batch(
            model=model,
            images=images,
            targets=targets,
            masks=rise_masks,
            p1=float(args.rise_p1),
            max_masked_batch=int(args.max_masked_batch),
        ).cpu().numpy()

        digit_masks_np = digit_masks.numpy()
        labels_np = labels.numpy()
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        source_indices_np = source_indices.numpy()

        for sal, mask, label, prediction, target, path, source_index in zip(
            saliency,
            digit_masks_np,
            labels_np,
            predictions_np,
            targets_np,
            paths,
            source_indices_np,
        ):
            hit, peak_row, peak_col, is_zero = pointing_result(sal, mask)
            label_int = int(label)
            positive_saliency = np.maximum(sal.astype(np.float64), 0.0)
            saliency_sum = float(positive_saliency.sum())
            mass_inside = (
                float(positive_saliency[mask > 0].sum() / saliency_sum)
                if saliency_sum > 0.0
                else 0.0
            )
            class_totals[label_int] += 1
            class_hits[label_int] += int(hit)
            class_correct[label_int] += int(int(prediction) == label_int)
            class_mass_sum[label_int] += mass_inside
            mask_pixels_total += int(np.count_nonzero(mask))
            zero_maps += int(is_zero)
            rows.append(
                {
                    "dataset": "decoymnist",
                    "method": args.method,
                    "seed": int(args.seed),
                    "split": args.split,
                    "target_mode": args.target_mode,
                    "explainer": EXPLAINER,
                    "image_path": str(path),
                    "source_index": int(source_index),
                    "label": label_int,
                    "prediction": int(prediction),
                    "classification_correct": int(int(prediction) == label_int),
                    "saliency_target": int(target),
                    "pointing_hit": int(hit),
                    "peak_row": peak_row,
                    "peak_col": peak_col,
                    "saliency_max": float(np.max(sal)),
                    "saliency_mass_in_digit": mass_inside,
                    "zero_saliency": int(is_zero),
                    "digit_mask_pixels": int(np.count_nonzero(mask)),
                }
            )

    if not rows:
        raise RuntimeError("No DecoyMNIST samples were evaluated.")
    class_pg = np.divide(
        class_hits,
        class_totals,
        out=np.full(10, np.nan, dtype=np.float64),
        where=class_totals > 0,
    )
    class_acc = np.divide(
        class_correct,
        class_totals,
        out=np.full(10, np.nan, dtype=np.float64),
        where=class_totals > 0,
    )
    class_mass = np.divide(
        class_mass_sum,
        class_totals,
        out=np.full(10, np.nan, dtype=np.float64),
        where=class_totals > 0,
    )
    finite_pg = np.flatnonzero(np.isfinite(class_pg))
    worst_digit = int(finite_pg[np.argmin(class_pg[finite_pg])])
    total = int(class_totals.sum())

    summary: Dict[str, object] = {
        "dataset": "decoymnist",
        "method": args.method,
        "seed": int(args.seed),
        "split": args.split,
        "target_mode": args.target_mode,
        "explainer": EXPLAINER,
        "mask_protocol_version": MASK_PROTOCOL_VERSION,
        "primary_pg_protocol": PRIMARY_PG_PROTOCOL,
        "map_height": 28,
        "map_width": 28,
        "pg_hits": int(class_hits.sum()),
        "pg_total": total,
        "pg_acc": float(class_hits.sum() / max(total, 1)),
        "pg_macro_class_acc": float(np.nanmean(class_pg)),
        "pg_worst_class_acc": float(class_pg[worst_digit]),
        "pg_worst_class": worst_digit,
        "pg_random_acc": float(mask_pixels_total / max(total * 28 * 28, 1)),
        "classification_acc": float(class_correct.sum() / max(total, 1)),
        "saliency_mass_in_digit": float(np.nanmean(class_mass)),
        "zero_saliency_maps": int(zero_maps),
        "mask_source": "clean_torchvision_mnist_foreground",
        "mask_threshold": int(args.mask_threshold),
        "max_samples": int(args.max_samples),
        "sample_seed": int(args.sample_seed),
        "val_frac": float(args.val_frac),
        "split_seed": int(args.split_seed),
        "checkpoint": str(checkpoint),
        "rise_num_masks": int(args.rise_num_masks),
        "rise_grid_size": int(args.rise_grid_size),
        "rise_p1": float(args.rise_p1),
        "rise_seed": int(args.rise_seed),
        "rise_masks_path": str(mask_bank_path),
        "rise_masks_sha256": mask_hash,
        "errors": 0,
    }
    for digit in range(10):
        summary[f"digit_{digit}_pg_hits"] = int(class_hits[digit])
        summary[f"digit_{digit}_total"] = int(class_totals[digit])
        summary[f"digit_{digit}_pg_acc"] = float(class_pg[digit])
        summary[f"digit_{digit}_classification_acc"] = float(class_acc[digit])
        summary[f"digit_{digit}_saliency_mass_in_digit"] = float(class_mass[digit])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pointing_game_per_image.csv", rows)
    write_csv(output_dir / "pointing_game_summary.csv", [summary])
    temporary_json = output_dir / f".run_summary.{os.getpid()}.tmp"
    with open(temporary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_json, output_dir / "run_summary.json")

    print(
        f"[RESULT] method={args.method} seed={args.seed} "
        f"rise_pg={100.0 * float(summary['pg_acc']):.2f}% "
        f"macro={100.0 * float(summary['pg_macro_class_acc']):.2f}% "
        f"worst_digit={worst_digit} "
        f"worst={100.0 * float(summary['pg_worst_class_acc']):.2f}% "
        f"random={100.0 * float(summary['pg_random_acc']):.2f}% "
        f"mass_inside={100.0 * float(summary['saliency_mass_in_digit']):.2f}% "
        f"zero_maps={zero_maps}/{total}"
    )
    print(f"[DONE] {output_dir / 'pointing_game_summary.csv'}")


if __name__ == "__main__":
    main()
