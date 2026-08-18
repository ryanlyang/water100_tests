#!/usr/bin/env python3
"""RISE Pointing Game for one WB95-to-ImageNet-9 transfer checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gals_rise_utils import load_or_create_mask_bank, rise_from_probabilities_batch
from imagenet9_data import CLASS_NAMES, build_eval_transform
from imagenet9_final_utils import atomic_json
from imagenet9_pointing_game_utils import (
    METHODS,
    PRIMARY_VARIANTS,
    index_foreground_masks,
    parse_progress_jsonl,
    read_manifest,
    resolve_foreground_mask,
    write_csv,
)
from train_imagenet9_baseline import _forward


IMAGE_SIZE = 224
MASK_PROTOCOL_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_transform(resize_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
    )


class OfficialForegroundDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        official_root: Path,
        mask_root: Path,
        variant: str,
        image_transform,
        resize_size: int,
        completed_keys: Sequence[str],
        max_samples: int,
        sample_seed: int,
    ) -> None:
        rows = read_manifest(manifest, variant)
        if max_samples > 0 and max_samples < len(rows):
            rng = random.Random(sample_seed)
            selected = sorted(rng.sample(range(len(rows)), max_samples))
            rows = [rows[index] for index in selected]
        mask_index = index_foreground_masks(mask_root)
        completed = set(completed_keys)
        self.image_transform = image_transform
        self.mask_transform = mask_transform(resize_size)
        self.records: List[Dict[str, object]] = []
        all_keys = set()

        for row in rows:
            image_path = official_root / row["relative_path"]
            mask_path, foreground_id = resolve_foreground_mask(row, mask_index)
            sample_key = row["relative_path"]
            if sample_key in all_keys:
                raise RuntimeError(f"Duplicate official sample key: {sample_key}")
            all_keys.add(sample_key)
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing image/mask pair: {image_path}, {mask_path}")
            if sample_key in completed:
                continue
            self.records.append(
                {
                    "sample_key": sample_key,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "foreground_id": foreground_id,
                    "label": int(row["label"]),
                    "class_name": row["class_name"],
                }
            )
        unknown = completed - all_keys
        if unknown:
            raise RuntimeError(f"Progress contains unknown sample keys: {sorted(unknown)[:5]}")
        self.requested_total = len(rows)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["image_path"]) as handle:
            image = handle.convert("RGB")
            native_size = image.size
            image_tensor = self.image_transform(image)
        mask = np.load(record["mask_path"], allow_pickle=False)
        mask = np.asarray(mask).squeeze()
        if mask.ndim != 2:
            raise RuntimeError(f"Foreground mask is not 2D: {record['mask_path']} {mask.shape}")
        expected_shape = (native_size[1], native_size[0])
        if tuple(mask.shape) != expected_shape:
            raise RuntimeError(
                f"Image/mask native shape mismatch for {record['sample_key']}: "
                f"image={expected_shape} mask={mask.shape}"
            )
        mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        transformed_mask = np.asarray(self.mask_transform(mask_image), dtype=np.uint8) > 0
        if transformed_mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"Unexpected transformed mask shape: {transformed_mask.shape}")
        if not transformed_mask.any():
            raise RuntimeError(f"Foreground vanished after preprocessing: {record['sample_key']}")
        return (
            image_tensor,
            torch.from_numpy(transformed_mask.copy()),
            int(record["label"]),
            str(record["sample_key"]),
            str(record["image_path"]),
            str(record["mask_path"]),
            str(record["foreground_id"]),
            str(record["class_name"]),
        )


class ImageNet9ProbabilityModel(nn.Module):
    def __init__(self, method: str, model: nn.Module) -> None:
        super().__init__()
        self.method = method
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        native_method = "erm" if self.method in {"r4rr", "afr"} else self.method
        logits = _forward(native_method, self.model, images)[0]
        return torch.softmax(logits, dim=1)


class ClipLinearProbabilityModel(nn.Module):
    def __init__(self, model: nn.Module, classifier: object) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "coefficients",
            torch.as_tensor(np.asarray(classifier.coef_), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "intercept",
            torch.as_tensor(np.asarray(classifier.intercept_), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(images).float()
        features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        logits = features @ self.coefficients.t() + self.intercept
        return torch.softmax(logits, dim=1)


def load_probability_model(
    method: str,
    evaluation: Mapping[str, object],
    device: torch.device,
) -> Tuple[nn.Module, object, int, Dict[str, str]]:
    checkpoint = Path(str(evaluation["checkpoint"])).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    details = {"checkpoint": str(checkpoint), "afr_classifier_checkpoint": ""}
    if method == "clip_lr":
        from sweep_imagenet9_clip_lr import _load_clip

        with checkpoint.open("rb") as handle:
            classifier = pickle.load(handle)
        clip = _load_clip()
        model, preprocess = clip.load("RN50", device=str(device), jit=False)
        model.eval()
        return ClipLinearProbabilityModel(model, classifier).to(device), preprocess, 224, details

    from evaluate_imagenet9_final_checkpoint import load_model

    load_args = argparse.Namespace(
        method=method,
        checkpoint=checkpoint,
        afr_classifier_checkpoint=None,
    )
    if method == "afr":
        classifier_path = Path(str(evaluation.get("afr_classifier_checkpoint", ""))).resolve()
        if not classifier_path.is_file():
            raise FileNotFoundError(classifier_path)
        load_args.afr_classifier_checkpoint = classifier_path
        details["afr_classifier_checkpoint"] = str(classifier_path)
    model, _forward_fn = load_model(load_args, device)
    model.eval()
    return ImageNet9ProbabilityModel(method, model).to(device), build_eval_transform(), 256, details


def pointing_result(saliency: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int, int]:
    if saliency.shape != mask.shape or not np.isfinite(saliency).all():
        raise RuntimeError("Invalid RISE saliency/mask pair")
    if float(np.max(saliency)) <= 1e-12:
        return 0, -1, -1, 1
    row, col = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
    return int(mask[row, col] > 0), int(row), int(col), 0


def build_summary(
    rows: Sequence[Mapping[str, object]], args: argparse.Namespace, details: Mapping[str, str],
    mask_hash: str, seconds: int,
) -> Dict[str, object]:
    by_class: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["label"])].append(row)
    if sorted(by_class) != list(range(9)):
        raise RuntimeError(f"Pointing Game results are missing classes: {sorted(by_class)}")
    per_class_pg = []
    per_class_acc = []
    summary: Dict[str, object] = {
        "dataset": "imagenet9",
        "transfer_source": "waterbirds95",
        "method": args.method,
        "seed": args.seed,
        "variant": args.variant,
        "target_mode": args.target_mode,
        "explainer": "rise",
        "primary_pg_protocol": "rise_pixel_argmax_official_foreground_mask",
        "mask_protocol_version": MASK_PROTOCOL_VERSION,
        "mask_source": "backgrounds_challenge_fg_mask",
        "mask_root": str(args.mask_root.resolve()),
        "pg_hits": sum(int(row["pointing_hit"]) for row in rows),
        "pg_total": len(rows),
        "classification_correct": sum(int(row["classification_correct"]) for row in rows),
        "zero_saliency_maps": sum(int(row["zero_saliency"]) for row in rows),
        "saliency_mass_in_foreground": float(np.mean([float(row["saliency_mass_in_foreground"]) for row in rows])),
        "foreground_mask_fraction": float(np.mean([float(row["foreground_mask_fraction"]) for row in rows])),
        "rise_num_masks": args.rise_num_masks,
        "rise_grid_size": args.rise_grid_size,
        "rise_p1": args.rise_p1,
        "rise_seed": args.rise_seed,
        "rise_masks_path": str(args.rise_masks_path.resolve()),
        "rise_masks_sha256": mask_hash,
        "max_samples": args.max_samples,
        "sample_seed": args.sample_seed,
        "seconds": seconds,
        "missing_images": 0,
        "missing_masks": 0,
        "errors": 0,
        **details,
    }
    summary["pg_acc"] = float(summary["pg_hits"]) / len(rows)
    summary["classification_acc"] = float(summary["classification_correct"]) / len(rows)
    summary["pg_random_acc"] = float(summary["foreground_mask_fraction"])
    for label in range(9):
        class_rows = by_class[label]
        pg = float(np.mean([int(row["pointing_hit"]) for row in class_rows]))
        acc = float(np.mean([int(row["classification_correct"]) for row in class_rows]))
        per_class_pg.append(pg)
        per_class_acc.append(acc)
        token = CLASS_NAMES[label].replace(" ", "_")
        summary[f"class_{label}_{token}_total"] = len(class_rows)
        summary[f"class_{label}_{token}_pg_acc"] = pg
        summary[f"class_{label}_{token}_classification_acc"] = acc
    summary["pg_macro_class_acc"] = float(np.mean(per_class_pg))
    summary["pg_worst_class_acc"] = float(np.min(per_class_pg))
    summary["pg_worst_class"] = CLASS_NAMES[int(np.argmin(per_class_pg))]
    summary["classification_macro_class_acc"] = float(np.mean(per_class_acc))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=PRIMARY_VARIANTS, required=True)
    parser.add_argument("--evaluation-json", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-test-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-mode", choices=("label", "prediction"), default="label")
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--max-masked-batch", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--rise-num-masks", type=int, default=2000)
    parser.add_argument("--rise-grid-size", type=int, default=8)
    parser.add_argument("--rise-p1", type=float, default=0.1)
    parser.add_argument("--rise-seed", type=int, default=0)
    parser.add_argument("--rise-masks-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.time()
    for path in (args.evaluation_json, args.official_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.official_test_root, args.mask_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    evaluation = json.loads(args.evaluation_json.read_text())
    if int(evaluation.get("seed", -1)) != args.seed:
        raise RuntimeError("Evaluation JSON seed does not match requested seed")
    if evaluation.get("official_variants_used_for_selection") is not False:
        raise RuntimeError("Source evaluation does not certify held-out official variants")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    probability_model, image_transform, resize_size, details = load_probability_model(
        args.method, evaluation, device
    )
    probability_model.eval()
    masks_np, mask_hash = load_or_create_mask_bank(
        args.rise_masks_path,
        args.rise_num_masks,
        args.rise_grid_size,
        IMAGE_SIZE,
        IMAGE_SIZE,
        args.rise_p1,
        args.rise_seed,
    )
    rise_masks = torch.from_numpy(masks_np).to(device=device, dtype=torch.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "pointing_game_progress.jsonl"
    existing_rows = parse_progress_jsonl(progress_path)
    contract = {
        "method": args.method,
        "seed": args.seed,
        "variant": args.variant,
        "target_mode": args.target_mode,
        "evaluation_json": str(args.evaluation_json.resolve()),
        "evaluation_json_sha256": sha256(args.evaluation_json),
        "official_manifest": str(args.official_manifest.resolve()),
        "official_manifest_sha256": sha256(args.official_manifest),
        "official_test_root": str(args.official_test_root.resolve()),
        "mask_root": str(args.mask_root.resolve()),
        "mask_protocol_version": MASK_PROTOCOL_VERSION,
        "rise_num_masks": args.rise_num_masks,
        "rise_grid_size": args.rise_grid_size,
        "rise_p1": args.rise_p1,
        "rise_seed": args.rise_seed,
        "rise_masks_sha256": mask_hash,
        "max_samples": args.max_samples,
        "sample_seed": args.sample_seed,
    }
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError(f"Refusing to resume changed RISE contract: {contract_path}")
    if not contract_path.is_file():
        atomic_json(contract_path, contract)

    dataset = OfficialForegroundDataset(
        args.official_manifest,
        args.official_test_root,
        args.mask_root,
        args.variant,
        image_transform,
        resize_size,
        [str(row["sample_key"]) for row in existing_rows],
        args.max_samples,
        args.sample_seed,
    )
    print(
        f"[INFO] method={args.method} seed={args.seed} variant={args.variant} "
        f"complete={len(existing_rows)}/{dataset.requested_total} remaining={len(dataset)}",
        flush=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    with progress_path.open("a", encoding="utf-8") as progress:
        for batch in loader:
            images, foreground_masks, labels, keys, image_paths, mask_paths, foreground_ids, class_names = batch
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            with torch.no_grad():
                probabilities = probability_model(images)
                predictions = probabilities.argmax(dim=1)
            targets = labels_device if args.target_mode == "label" else predictions
            saliency_batch = rise_from_probabilities_batch(
                probability_model,
                images,
                targets,
                rise_masks,
                args.rise_p1,
                args.max_masked_batch,
            ).cpu().numpy()
            labels_np = labels.numpy()
            predictions_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()
            for index in range(len(keys)):
                saliency = saliency_batch[index]
                foreground = foreground_masks[index].numpy().astype(bool)
                hit, peak_row, peak_col, zero = pointing_result(saliency, foreground)
                positive = np.maximum(saliency.astype(np.float64), 0.0)
                total_mass = float(positive.sum())
                mass_inside = float(positive[foreground].sum() / total_mass) if total_mass > 0 else 0.0
                row = {
                    "dataset": "imagenet9",
                    "transfer_source": "waterbirds95",
                    "method": args.method,
                    "seed": args.seed,
                    "variant": args.variant,
                    "target_mode": args.target_mode,
                    "explainer": "rise",
                    "sample_key": keys[index],
                    "foreground_id": foreground_ids[index],
                    "label": int(labels_np[index]),
                    "class_name": class_names[index],
                    "prediction": int(predictions_np[index]),
                    "classification_correct": int(predictions_np[index] == labels_np[index]),
                    "saliency_target": int(targets_np[index]),
                    "pointing_hit": hit,
                    "peak_row": peak_row,
                    "peak_col": peak_col,
                    "saliency_max": float(np.max(saliency)),
                    "saliency_mass_in_foreground": mass_inside,
                    "foreground_mask_pixels": int(foreground.sum()),
                    "foreground_mask_fraction": float(foreground.mean()),
                    "zero_saliency": zero,
                    "image_path": image_paths[index],
                    "mask_path": mask_paths[index],
                }
                progress.write(json.dumps(row, sort_keys=True) + "\n")
                existing_rows.append(row)
            progress.flush()
            os.fsync(progress.fileno())
            print(f"[PROGRESS] {len(existing_rows)}/{dataset.requested_total}", flush=True)

    if len(existing_rows) != dataset.requested_total:
        raise RuntimeError(
            f"Incomplete Pointing Game result: {len(existing_rows)}/{dataset.requested_total}"
        )
    existing_rows.sort(key=lambda row: str(row["sample_key"]))
    summary = build_summary(
        existing_rows, args, details, mask_hash, int(time.time() - started)
    )
    write_csv(args.output_dir / "pointing_game_per_image.csv", existing_rows)
    write_csv(args.output_dir / "pointing_game_summary.csv", [summary])
    atomic_json(args.output_dir / "pointing_game_summary.json", summary)
    print(
        f"[RESULT] method={args.method} seed={args.seed} variant={args.variant} "
        f"pg={100 * float(summary['pg_acc']):.2f}% "
        f"macro={100 * float(summary['pg_macro_class_acc']):.2f}% "
        f"worst={100 * float(summary['pg_worst_class_acc']):.2f}% "
        f"random={100 * float(summary['pg_random_acc']):.2f}%",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
