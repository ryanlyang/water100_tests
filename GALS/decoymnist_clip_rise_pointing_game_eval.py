#!/usr/bin/env python3
"""Evaluate deterministic RN50 CLIP baselines on DecoyMNIST with RISE."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from decoymnist_pointing_game_eval import CleanDigitMaskDataset, pointing_result
from decoymnist_rise_pointing_game_eval import load_or_create_mask_bank
from run_decoymnist_clip_vit_lr_fixed import _try_import_clip


DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
TEMPLATES = (
    "a handwritten digit {}.",
    "a photo of the handwritten digit {}.",
    "the number {}.",
)
METHODS = ("clip_zs", "clip_lr")
IMAGE_SIZE = 28
MASK_PROTOCOL_VERSION = 1
PRIMARY_PG_PROTOCOL = "rise_pixel_argmax"


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(denominator, 1e-12)


@torch.no_grad()
def extract_clip_features(
    dataset: Dataset,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract one C-contiguous float64 feature matrix without list/concat copies."""
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
    )
    features: Optional[np.ndarray] = None
    labels = np.empty(len(dataset), dtype=np.int64)
    offset = 0
    model.eval()
    for images, batch_labels in loader:
        images = images.to(device)
        encoded = model.encode_image(images).float()
        encoded /= encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        batch_features = encoded.cpu().numpy().astype(np.float32, copy=False)
        if features is None:
            features = np.empty(
                (len(dataset), int(batch_features.shape[1])),
                dtype=np.float64,
                order="C",
            )
        end = offset + int(batch_features.shape[0])
        features[offset:end] = batch_features
        labels[offset:end] = batch_labels.numpy()
        offset = end
    if features is None or offset != len(dataset):
        raise RuntimeError(
            f"Feature extraction produced {offset} rows for a {len(dataset)}-sample dataset"
        )
    if not np.isfinite(features).all():
        np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return features, labels


def build_text_features(clip_module, model: nn.Module, device: torch.device) -> np.ndarray:
    class_features: List[torch.Tensor] = []
    with torch.no_grad():
        for digit, word in enumerate(DIGIT_WORDS):
            variants = (str(digit), word)
            prompts = [template.format(value) for value in variants for template in TEMPLATES]
            tokens = clip_module.tokenize(prompts).to(device)
            features = model.encode_text(tokens).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            class_feature = features.mean(dim=0)
            class_feature = class_feature / class_feature.norm().clamp_min(1e-12)
            class_features.append(class_feature)
    return torch.stack(class_features).cpu().numpy().astype(np.float32)


class CLIPZeroShot(nn.Module):
    def __init__(self, model: nn.Module, text_features: np.ndarray) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("text_features", torch.from_numpy(text_features), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(images).float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = self.model.logit_scale.exp().float().clamp(max=100.0)
        return torch.softmax(scale * features @ self.text_features.t(), dim=1)


class CLIPLinear(nn.Module):
    def __init__(self, model: nn.Module, coefficients: np.ndarray, intercept: np.ndarray) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "coefficients",
            torch.from_numpy(np.asarray(coefficients, dtype=np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "intercept",
            torch.from_numpy(np.asarray(intercept, dtype=np.float32)),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(images).float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.softmax(features @ self.coefficients.t() + self.intercept, dim=1)


def build_probability_model(
    method: str,
    png_root: Path,
    clip_model_name: str,
    clip_c: float,
    feature_batch_size: int,
    num_workers: int,
    split_seed: int,
    device: torch.device,
):
    print(f"[STAGE] Loading OpenAI CLIP model={clip_model_name}", flush=True)
    clip_module = _try_import_clip()
    try:
        model, preprocess = clip_module.load(clip_model_name, device=str(device), jit=False)
    except TypeError:
        model, preprocess = clip_module.load(clip_model_name, device=str(device))
    model.eval()

    details: Dict[str, object] = {
        "clip_model": clip_model_name,
        "clip_c": float(clip_c) if method == "clip_lr" else "",
        "clip_penalty": "l2" if method == "clip_lr" else "",
        "clip_solver": "lbfgs" if method == "clip_lr" else "",
        "clip_fit_intercept": True if method == "clip_lr" else "",
        "clip_num_templates": len(TEMPLATES) if method == "clip_zs" else "",
    }
    if method == "clip_zs":
        text_features = build_text_features(clip_module, model, device)
        return CLIPZeroShot(model, text_features), preprocess, details

    from sklearn.linear_model import LogisticRegression

    full_train = ImageFolder(str(png_root / "train"), transform=preprocess)
    permutation = np.random.default_rng(int(split_seed)).permutation(len(full_train))
    validation_count = int(0.10 * len(full_train))
    train_indices = permutation[validation_count:]
    train_subset = Subset(full_train, train_indices.tolist())
    print(
        f"[STAGE] Extracting CLIP features samples={len(train_subset)} "
        f"batch_size={feature_batch_size} workers={num_workers} dtype=float64",
        flush=True,
    )
    train_features, train_labels = extract_clip_features(
        train_subset,
        model,
        device,
        int(feature_batch_size),
        int(num_workers),
    )
    print(
        f"[STAGE] Fitting CLIP-LR shape={train_features.shape} "
        f"matrix_gib={train_features.nbytes / (1024 ** 3):.3f}",
        flush=True,
    )
    classifier = LogisticRegression(
        random_state=0,
        C=float(clip_c),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        max_iter=5000,
        n_jobs=1,
        verbose=0,
    )
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            classifier.fit(train_features, train_labels)
    except Exception:
        classifier.fit(train_features, train_labels)
    print("[STAGE] CLIP-LR fit complete", flush=True)
    details["clip_train_samples"] = int(train_features.shape[0])
    return CLIPLinear(model, classifier.coef_, classifier.intercept_), preprocess, details


@torch.no_grad()
def rise_batch(
    probability_model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    masks_28: torch.Tensor,
    p1: float,
    max_masked_batch: int,
) -> torch.Tensor:
    batch_size, channels, height, width = images.shape
    masks_per_step = max(1, int(max_masked_batch) // max(batch_size, 1))
    total = torch.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE), device=images.device)

    for start in range(0, int(masks_28.shape[0]), masks_per_step):
        mask_chunk_28 = masks_28[start : start + masks_per_step]
        chunk_size = int(mask_chunk_28.shape[0])
        mask_chunk_native = F.interpolate(
            mask_chunk_28,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        masked = (images[:, None] * mask_chunk_native[None]).reshape(
            batch_size * chunk_size,
            channels,
            height,
            width,
        )
        probabilities = probability_model(masked).view(batch_size, chunk_size, -1)
        scores = probabilities.gather(
            2,
            targets.view(batch_size, 1, 1).expand(-1, chunk_size, 1),
        ).squeeze(2)
        total += torch.einsum("bm,mhw->bhw", scores, mask_chunk_28[:, 0])
    return total / max(float(masks_28.shape[0]) * float(p1), 1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-root", type=Path, required=True)
    parser.add_argument("--mnist-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-model", default="RN50")
    parser.add_argument("--clip-c", type=float, default=0.2515000498909345)
    parser.add_argument("--clip-feature-batch-size", type=int, default=128)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--target-mode", choices=("label", "prediction"), default="label")
    parser.add_argument("--mask-threshold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--max-masked-batch", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--rise-num-masks", type=int, default=2000)
    parser.add_argument("--rise-grid-size", type=int, default=8)
    parser.add_argument("--rise-p1", type=float, default=0.1)
    parser.add_argument("--rise-seed", type=int, default=0)
    parser.add_argument("--rise-masks-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.sample_seed))
    np.random.seed(int(args.sample_seed))
    torch.manual_seed(int(args.seed))

    png_root = args.png_root.expanduser().resolve()
    mnist_root = args.mnist_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mask_bank_path = args.rise_masks_path.expanduser().resolve()
    if not (png_root / "train").is_dir() or not (png_root / "test").is_dir():
        raise FileNotFoundError(f"Expected train/test under {png_root}")

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    probability_model, preprocess, clip_details = build_probability_model(
        method=args.method,
        png_root=png_root,
        clip_model_name=str(args.clip_model),
        clip_c=float(args.clip_c),
        feature_batch_size=int(args.clip_feature_batch_size),
        num_workers=int(args.num_workers),
        split_seed=int(args.split_seed),
        device=device,
    )
    probability_model = probability_model.to(device).eval()

    base_dataset = CleanDigitMaskDataset(
        png_root=png_root,
        mnist_root=mnist_root,
        split="test",
        mask_threshold=int(args.mask_threshold),
        val_frac=0.10,
        split_seed=int(args.split_seed),
    )
    base_dataset.images.transform = preprocess
    dataset: Dataset = base_dataset
    if 0 < int(args.max_samples) < len(dataset):
        indices = list(range(len(dataset)))
        random.Random(int(args.sample_seed)).shuffle(indices)
        dataset = Subset(dataset, sorted(indices[: int(args.max_samples)]))
    loader = DataLoader(
        dataset,
        batch_size=int(args.image_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )

    masks_np, mask_hash = load_or_create_mask_bank(
        mask_bank_path,
        int(args.rise_num_masks),
        int(args.rise_grid_size),
        IMAGE_SIZE,
        IMAGE_SIZE,
        float(args.rise_p1),
        int(args.rise_seed),
    )
    rise_masks = torch.from_numpy(masks_np).to(device=device, dtype=torch.float32)

    class_hits = np.zeros(10, dtype=np.int64)
    class_totals = np.zeros(10, dtype=np.int64)
    class_correct = np.zeros(10, dtype=np.int64)
    class_mass_sum = np.zeros(10, dtype=np.float64)
    mask_pixels_total = 0
    zero_maps = 0
    rows: List[Dict[str, object]] = []

    print(f"[STAGE] Starting RISE evaluation", flush=True)
    print(f"[INFO] method={args.method} samples={len(dataset)} device={device} clip={clip_details}")
    print(f"[INFO] RISE 28x28 bank={mask_bank_path} sha256={mask_hash}")
    for images, labels, digit_masks, paths, source_indices in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        probabilities = probability_model(images)
        predictions = probabilities.argmax(dim=1)
        targets = labels_device if args.target_mode == "label" else predictions
        saliency = rise_batch(
            probability_model,
            images,
            targets,
            rise_masks,
            float(args.rise_p1),
            int(args.max_masked_batch),
        ).cpu().numpy()

        for sal, mask, label, prediction, target, path, source_index in zip(
            saliency,
            digit_masks.numpy(),
            labels.numpy(),
            predictions.cpu().numpy(),
            targets.cpu().numpy(),
            paths,
            source_indices.numpy(),
        ):
            hit, peak_row, peak_col, is_zero = pointing_result(sal, mask)
            label_int = int(label)
            positive = np.maximum(sal.astype(np.float64), 0.0)
            total_saliency = float(positive.sum())
            mass_inside = (
                float(positive[mask > 0].sum() / total_saliency) if total_saliency > 0 else 0.0
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
                    "seed": 0,
                    "split": "test",
                    "target_mode": args.target_mode,
                    "explainer": "rise",
                    "image_path": str(path),
                    "source_index": int(source_index),
                    "label": label_int,
                    "prediction": int(prediction),
                    "classification_correct": int(int(prediction) == label_int),
                    "saliency_target": int(target),
                    "pointing_hit": int(hit),
                    "peak_row": int(peak_row),
                    "peak_col": int(peak_col),
                    "saliency_max": float(np.max(sal)),
                    "saliency_mass_in_digit": mass_inside,
                    "zero_saliency": int(is_zero),
                    "digit_mask_pixels": int(np.count_nonzero(mask)),
                }
            )

    class_pg = np.divide(class_hits, class_totals, out=np.full(10, np.nan), where=class_totals > 0)
    class_acc = np.divide(
        class_correct, class_totals, out=np.full(10, np.nan), where=class_totals > 0
    )
    class_mass = np.divide(
        class_mass_sum, class_totals, out=np.full(10, np.nan), where=class_totals > 0
    )
    worst_digit = int(np.nanargmin(class_pg))
    total = int(class_totals.sum())
    summary: Dict[str, object] = {
        "dataset": "decoymnist",
        "method": args.method,
        "seed": 0,
        "split": "test",
        "target_mode": args.target_mode,
        "explainer": "rise",
        "mask_protocol_version": MASK_PROTOCOL_VERSION,
        "primary_pg_protocol": PRIMARY_PG_PROTOCOL,
        "map_height": IMAGE_SIZE,
        "map_width": IMAGE_SIZE,
        "pg_hits": int(class_hits.sum()),
        "pg_total": total,
        "pg_acc": float(class_hits.sum() / max(total, 1)),
        "pg_macro_class_acc": float(np.nanmean(class_pg)),
        "pg_worst_class_acc": float(class_pg[worst_digit]),
        "pg_worst_class": worst_digit,
        "pg_random_acc": float(mask_pixels_total / max(total * IMAGE_SIZE * IMAGE_SIZE, 1)),
        "classification_acc": float(class_correct.sum() / max(total, 1)),
        "saliency_mass_in_digit": float(np.nanmean(class_mass)),
        "zero_saliency_maps": int(zero_maps),
        "mask_source": "clean_torchvision_mnist_foreground",
        "mask_threshold": int(args.mask_threshold),
        "max_samples": int(args.max_samples),
        "sample_seed": int(args.sample_seed),
        "val_frac": 0.10,
        "split_seed": int(args.split_seed),
        "checkpoint": "",
        **clip_details,
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
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_json, output_dir / "run_summary.json")
    print(
        f"[RESULT] {args.method} PG={100*float(summary['pg_acc']):.2f}% "
        f"macro={100*float(summary['pg_macro_class_acc']):.2f}% "
        f"worst={100*float(summary['pg_worst_class_acc']):.2f}% "
        f"classification={100*float(summary['classification_acc']):.2f}%"
    )


if __name__ == "__main__":
    main()
