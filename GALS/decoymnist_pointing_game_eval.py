#!/usr/bin/env python3
"""Evaluate DecoyMNIST checkpoints with clean-digit Pointing Game masks.

The exported DecoyMNIST PNG filenames preserve the original torchvision
MNIST index (for example, ``005462_y0.png``).  This evaluator uses that index
to recover the corresponding clean digit and defines the task-relevant mask
as its nonzero foreground.  The synthetic corner patch is therefore never
part of the evaluation mask.

All supported LeNet-style methods are explained with standard Grad-CAM at
``conv2`` for a common, architecture-matched comparison.  The primary metric
is resolution-matched: the clean foreground mask is max-pooled to the native
8x8 Grad-CAM grid before testing the peak location.  The original pixel-level
score is retained as a diagnostic.  An all-zero map is counted as a miss.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder, MNIST
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor


SUPPORTED_METHODS = ("vanilla", "elrep", "upweight", "abn", "gals", "afr", "r4rr")
FILENAME_RE = re.compile(r"^(?P<index>\d+)_y(?P<label>\d+)$")
MASK_PROTOCOL_VERSION = 2
PRIMARY_PG_PROTOCOL = "native_resolution_overlap"


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward_logits(x), dim=1)


class ABNLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.att_conv = nn.Conv2d(50, 1, kernel_size=1, stride=1)
        self.abn_fc = nn.Linear(50, 10)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        features = F.relu(self.conv2(x))
        attention = torch.sigmoid(self.att_conv(features))
        x = F.max_pool2d(features * (1.0 + attention), 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_logits(x)
        return F.log_softmax(logits, dim=1), logits.new_zeros((x.size(0), 10))


def torch_load_compat(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def extract_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(isinstance(key, str) for key in payload):
            return payload  # type: ignore[return-value]
    raise RuntimeError("Could not extract a model state_dict from the checkpoint.")


def state_dict_candidates(state: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    prefixes = ("module.", "model.", "net.", "base.")
    queue = [state]
    candidates: List[Dict[str, torch.Tensor]] = []
    seen = set()
    while queue:
        candidate = queue.pop(0)
        signature = tuple(sorted(candidate))
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(candidate)
        for prefix in prefixes:
            if candidate and all(key.startswith(prefix) for key in candidate):
                queue.append({key[len(prefix) :]: value for key, value in candidate.items()})
    return candidates


def load_checkpoint_strict(model: nn.Module, checkpoint: Path) -> Dict[str, object]:
    state = extract_state_dict(torch_load_compat(checkpoint))
    errors = []
    for candidate in state_dict_candidates(state):
        try:
            model.load_state_dict(candidate, strict=True)
            return {"loaded_keys": len(candidate), "checkpoint": str(checkpoint)}
        except RuntimeError as exc:
            errors.append(str(exc).splitlines()[0])
    raise RuntimeError(
        "Checkpoint does not exactly match the expected model architecture: "
        f"{checkpoint}; attempts={errors[:4]}"
    )


class CleanDigitMaskDataset(Dataset):
    def __init__(
        self,
        png_root: Path,
        mnist_root: Path,
        split: str,
        mask_threshold: int,
        val_frac: float,
        split_seed: int,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.split = split
        source_split = "test" if split == "test" else "train"
        self.transform = Compose(
            [Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)]
        )
        self.images = ImageFolder(str(png_root / source_split), transform=self.transform)
        self.clean = MNIST(
            root=str(mnist_root), train=(source_split == "train"), download=False, transform=None
        )
        self.mask_threshold = int(mask_threshold)

        indices = list(range(len(self.images)))
        if split in ("train", "val"):
            n_val = int(float(val_frac) * len(indices))
            generator = torch.Generator().manual_seed(int(split_seed))
            permutation = torch.randperm(len(indices), generator=generator).tolist()
            n_train = len(indices) - n_val
            indices = permutation[:n_train] if split == "train" else permutation[n_train:]
        self.indices = indices

        if len(self.clean) != len(self.images):
            raise RuntimeError(
                f"Clean MNIST and PNG split lengths differ: {len(self.clean)} != {len(self.images)}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        imagefolder_index = int(self.indices[item])
        image, label = self.images[imagefolder_index]
        image_path = Path(self.images.samples[imagefolder_index][0])
        match = FILENAME_RE.match(image_path.stem)
        if match is None:
            raise RuntimeError(
                f"Expected '<source_index>_y<label>.png' filename, found: {image_path.name}"
            )
        source_index = int(match.group("index"))
        encoded_label = int(match.group("label"))
        clean_image = self.clean.data[source_index]
        clean_label = int(self.clean.targets[source_index])
        if int(label) != encoded_label or clean_label != int(label):
            raise RuntimeError(
                "Decoy/Clean MNIST label mismatch for "
                f"{image_path}: folder={label}, filename={encoded_label}, clean={clean_label}"
            )
        mask = clean_image.gt(self.mask_threshold).to(torch.uint8)
        if int(mask.sum()) == 0:
            raise RuntimeError(f"Clean digit mask is empty: {image_path}")
        return image, int(label), mask, str(image_path), source_index


@dataclass
class GradCAMBatch:
    saliency: torch.Tensor
    native_saliency: torch.Tensor
    logits: torch.Tensor


def normalize_saliency_batch(saliency: torch.Tensor) -> torch.Tensor:
    flat = saliency.flatten(1)
    minima = flat.min(dim=1, keepdim=True).values
    maxima = flat.max(dim=1, keepdim=True).values
    ranges = maxima - minima
    return torch.where(
        ranges > 1e-12,
        (flat - minima) / ranges.clamp_min(1e-12),
        torch.zeros_like(flat),
    ).view_as(saliency)


def resolution_match_masks(masks: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    if masks.ndim != 3:
        raise ValueError(f"Expected BxHxW masks, found shape {tuple(masks.shape)}")
    pooled = F.adaptive_max_pool2d(masks.unsqueeze(1).float(), output_size=output_size)
    return pooled.squeeze(1).gt(0).to(torch.uint8)


def gradcam_batch(
    model: nn.Module, images: torch.Tensor, targets: torch.Tensor
) -> GradCAMBatch:
    captured: List[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output)

    handle = model.conv2.register_forward_hook(capture)  # type: ignore[attr-defined]
    try:
        model.zero_grad(set_to_none=True)
        logits = model.forward_logits(images)  # type: ignore[attr-defined]
        if len(captured) != 1:
            raise RuntimeError(f"Expected one conv2 activation, captured {len(captured)}")
        features = captured[0]
        scores = logits.gather(1, targets.view(-1, 1)).sum()
        gradients = torch.autograd.grad(scores, features, retain_graph=False, create_graph=False)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        native_saliency = torch.relu((weights * features).sum(dim=1))
        pixel_saliency = F.interpolate(
            native_saliency.unsqueeze(1),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return GradCAMBatch(
            saliency=normalize_saliency_batch(pixel_saliency).detach(),
            native_saliency=normalize_saliency_batch(native_saliency).detach(),
            logits=logits.detach(),
        )
    finally:
        handle.remove()


def pointing_result(saliency: np.ndarray, mask: np.ndarray) -> Tuple[bool, int, int, bool]:
    maximum = float(np.max(saliency))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return False, -1, -1, True
    maxima = np.argwhere(np.isclose(saliency, maximum, rtol=1e-6, atol=1e-8))
    if maxima.size == 0:
        return False, -1, -1, True
    hit = any(mask[int(row), int(col)] > 0 for row, col in maxima)
    row, col = maxima[0]
    return bool(hit), int(row), int(col), False


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
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
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    model: nn.Module = ABNLeNet() if args.method == "abn" else LeNet()
    load_meta = load_checkpoint_strict(model, checkpoint)
    model.to(device).eval()

    class_hits = np.zeros(10, dtype=np.int64)
    native_class_hits = np.zeros(10, dtype=np.int64)
    class_totals = np.zeros(10, dtype=np.int64)
    class_correct = np.zeros(10, dtype=np.int64)
    digit_mask_pixels_total = 0
    native_mask_cells_total = 0
    zero_maps = 0
    native_map_size: Optional[Tuple[int, int]] = None
    pixel_map_size: Optional[Tuple[int, int]] = None
    rows: List[Dict[str, object]] = []

    print(
        f"[INFO] method={args.method} seed={args.seed} split={args.split} "
        f"samples={len(dataset)} device={device}"
    )
    print(f"[INFO] mask_source=clean_torchvision_mnist threshold>{args.mask_threshold}")
    print(f"[INFO] checkpoint={checkpoint} loaded_keys={load_meta['loaded_keys']}")

    for images, labels, masks, paths, source_indices in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with torch.no_grad():
            prediction_logits = model.forward_logits(images)  # type: ignore[attr-defined]
            predictions = prediction_logits.argmax(dim=1)
        targets = labels_device if args.target_mode == "label" else predictions
        explained = gradcam_batch(model, images, targets)

        batch_native_size = tuple(explained.native_saliency.shape[-2:])
        batch_pixel_size = tuple(explained.saliency.shape[-2:])
        if native_map_size is None:
            native_map_size = batch_native_size
            pixel_map_size = batch_pixel_size
        elif native_map_size != batch_native_size or pixel_map_size != batch_pixel_size:
            raise RuntimeError(
                "Grad-CAM spatial resolution changed between batches: "
                f"native={native_map_size}->{batch_native_size}, "
                f"pixel={pixel_map_size}->{batch_pixel_size}"
            )

        saliency_batch = explained.saliency.cpu().numpy()
        native_saliency_batch = explained.native_saliency.cpu().numpy()
        masks_batch = masks.numpy()
        native_masks_batch = resolution_match_masks(
            masks,
            output_size=batch_native_size,
        ).numpy()
        labels_batch = labels.numpy()
        predictions_batch = predictions.cpu().numpy()
        targets_batch = targets.cpu().numpy()
        source_batch = source_indices.numpy()

        batch_items = zip(
            saliency_batch,
            native_saliency_batch,
            masks_batch,
            native_masks_batch,
            labels_batch,
            predictions_batch,
            targets_batch,
            paths,
            source_batch,
        )
        for (
            saliency,
            native_saliency,
            mask,
            native_mask,
            label,
            prediction,
            target,
            path,
            source_index,
        ) in batch_items:
            hit, peak_row, peak_col, is_zero = pointing_result(saliency, mask)
            native_hit, native_peak_row, native_peak_col, native_is_zero = pointing_result(
                native_saliency,
                native_mask,
            )
            if is_zero != native_is_zero:
                raise RuntimeError("Pixel and native Grad-CAM zero-map status disagree.")
            label_int = int(label)
            class_totals[label_int] += 1
            class_hits[label_int] += int(hit)
            native_class_hits[label_int] += int(native_hit)
            class_correct[label_int] += int(int(prediction) == label_int)
            digit_mask_pixels_total += int(np.count_nonzero(mask))
            native_mask_cells_total += int(np.count_nonzero(native_mask))
            zero_maps += int(is_zero)
            rows.append(
                {
                    "dataset": "decoymnist",
                    "method": args.method,
                    "seed": int(args.seed),
                    "split": args.split,
                    "target_mode": args.target_mode,
                    "image_path": str(path),
                    "source_index": int(source_index),
                    "label": label_int,
                    "prediction": int(prediction),
                    "classification_correct": int(int(prediction) == label_int),
                    "saliency_target": int(target),
                    "pointing_hit": int(hit),
                    "peak_row": peak_row,
                    "peak_col": peak_col,
                    "saliency_max": float(np.max(saliency)),
                    "pg_native_hit": int(native_hit),
                    "native_peak_row": native_peak_row,
                    "native_peak_col": native_peak_col,
                    "native_saliency_max": float(np.max(native_saliency)),
                    "zero_saliency": int(is_zero),
                    "digit_mask_pixels": int(np.count_nonzero(mask)),
                    "native_digit_mask_cells": int(np.count_nonzero(native_mask)),
                }
            )

    if not rows:
        raise RuntimeError("No DecoyMNIST samples were evaluated.")
    if native_map_size is None or pixel_map_size is None:
        raise RuntimeError("Grad-CAM map dimensions were not recorded.")
    class_pg = np.divide(
        class_hits,
        class_totals,
        out=np.full(10, np.nan, dtype=np.float64),
        where=class_totals > 0,
    )
    native_class_pg = np.divide(
        native_class_hits,
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
    finite_pg = np.flatnonzero(np.isfinite(class_pg))
    worst_digit = int(finite_pg[np.argmin(class_pg[finite_pg])])
    finite_native_pg = np.flatnonzero(np.isfinite(native_class_pg))
    native_worst_digit = int(
        finite_native_pg[np.argmin(native_class_pg[finite_native_pg])]
    )
    total = int(class_totals.sum())
    native_height, native_width = native_map_size
    pixel_height, pixel_width = pixel_map_size
    summary: Dict[str, object] = {
        "dataset": "decoymnist",
        "method": args.method,
        "seed": int(args.seed),
        "split": args.split,
        "target_mode": args.target_mode,
        "mask_protocol_version": MASK_PROTOCOL_VERSION,
        "primary_pg_protocol": PRIMARY_PG_PROTOCOL,
        "native_map_height": int(native_height),
        "native_map_width": int(native_width),
        "pg_native_hits": int(native_class_hits.sum()),
        "pg_native_total": total,
        "pg_native_acc": float(native_class_hits.sum() / max(total, 1)),
        "pg_native_macro_class_acc": float(np.nanmean(native_class_pg)),
        "pg_native_worst_class_acc": float(native_class_pg[native_worst_digit]),
        "pg_native_worst_class": native_worst_digit,
        "pg_native_random_acc": float(
            native_mask_cells_total / max(total * native_height * native_width, 1)
        ),
        "pg_hits": int(class_hits.sum()),
        "pg_total": total,
        "pg_acc": float(class_hits.sum() / max(total, 1)),
        "pg_macro_class_acc": float(np.nanmean(class_pg)),
        "pg_worst_class_acc": float(class_pg[worst_digit]),
        "pg_worst_class": worst_digit,
        "pg_pixel_random_acc": float(
            digit_mask_pixels_total / max(total * pixel_height * pixel_width, 1)
        ),
        "classification_acc": float(class_correct.sum() / max(total, 1)),
        "zero_saliency_maps": int(zero_maps),
        "mask_source": "clean_torchvision_mnist_foreground",
        "mask_threshold": int(args.mask_threshold),
        "max_samples": int(args.max_samples),
        "sample_seed": int(args.sample_seed),
        "val_frac": float(args.val_frac),
        "split_seed": int(args.split_seed),
        "checkpoint": str(checkpoint),
        "errors": 0,
    }
    for digit in range(10):
        summary[f"digit_{digit}_pg_native_hits"] = int(native_class_hits[digit])
        summary[f"digit_{digit}_pg_native_acc"] = float(native_class_pg[digit])
        summary[f"digit_{digit}_hits"] = int(class_hits[digit])
        summary[f"digit_{digit}_total"] = int(class_totals[digit])
        summary[f"digit_{digit}_pg_acc"] = float(class_pg[digit])
        summary[f"digit_{digit}_classification_acc"] = float(class_acc[digit])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pointing_game_per_image.csv", rows)
    write_csv(output_dir / "pointing_game_summary.csv", [summary])
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"[RESULT] method={args.method} seed={args.seed} "
        f"native_pg={100.0 * float(summary['pg_native_acc']):.2f}% "
        f"native_macro={100.0 * float(summary['pg_native_macro_class_acc']):.2f}% "
        f"native_worst_digit={native_worst_digit} "
        f"native_worst={100.0 * float(summary['pg_native_worst_class_acc']):.2f}% "
        f"native_random={100.0 * float(summary['pg_native_random_acc']):.2f}% "
        f"pixel_pg={100.0 * float(summary['pg_acc']):.2f}% "
        f"zero_maps={zero_maps}/{total}"
    )
    print(f"[DONE] {output_dir / 'pointing_game_summary.csv'}")


if __name__ == "__main__":
    main()
