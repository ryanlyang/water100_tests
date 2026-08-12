#!/usr/bin/env python3
"""Evaluate one Waterbirds checkpoint with GALS-style RISE Pointing Game."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gals_rise_utils import load_or_create_mask_bank, rise_from_probabilities_batch
from waterbirds_pointing_game_eval import (
    AFRRunner,
    ABNRunner,
    GROUP_NAMES,
    GALSRunner,
    GuidedRunner,
    MethodRunnerBase,
    SPLIT_MAP,
    VanillaRunner,
    _pick_rows,
    build_preprocess,
    open_pil_with_retry,
    resolve_mask_path,
)


SUPPORTED_METHODS = (
    "vanilla",
    "elrep",
    "upweight",
    "abn",
    "gals",
    "afr",
    "r4rr",
    "clip_zs",
    "clip_lr",
)
MASK_PROTOCOL_VERSION = 1
PRIMARY_PG_PROTOCOL = "rise_pixel_argmax"
EXPLAINER = "rise"
IMAGE_SIZE = 224


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


class WaterbirdsPointingDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        mask_root: Path,
        split: str,
        max_samples: int,
        sample_seed: int,
        mask_threshold: int,
        preprocess: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        mask_preprocess: Optional[Callable[[Image.Image], Image.Image]] = None,
    ) -> None:
        metadata_path = data_path / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing Waterbirds metadata: {metadata_path}")
        frame = pd.read_csv(metadata_path)
        frame = frame[frame["split"] == SPLIT_MAP[split]].copy()
        frame = _pick_rows(frame, max_samples=max_samples, seed=sample_seed)
        if frame.empty:
            raise RuntimeError(f"No Waterbirds rows for split={split}")

        self.data_path = data_path
        self.mask_threshold = int(mask_threshold)
        self.preprocess = preprocess if preprocess is not None else build_preprocess()
        self.mask_preprocess = mask_preprocess
        self.records: List[Dict[str, object]] = []
        missing_images: List[str] = []
        missing_masks: List[str] = []

        for _, row in frame.iterrows():
            relative_path = str(row["img_filename"])
            image_path = data_path / relative_path
            mask_path, _ = resolve_mask_path(mask_root, image_path, data_path)
            if not image_path.is_file():
                missing_images.append(str(image_path))
                continue
            if mask_path is None:
                missing_masks.append(relative_path)
                continue
            label = int(row["y"])
            place = int(row["place"])
            self.records.append(
                {
                    "relative_path": relative_path,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "label": label,
                    "place": place,
                    "group": label * 2 + place,
                }
            )

        if missing_images or missing_masks:
            raise RuntimeError(
                "Waterbirds RISE evaluation requires complete image and CUB mask coverage. "
                f"missing_images={len(missing_images)} preview={missing_images[:5]} "
                f"missing_masks={len(missing_masks)} preview={missing_masks[:5]}"
            )
        if not self.records:
            raise RuntimeError("No valid Waterbirds image/mask pairs were resolved")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = open_pil_with_retry(Path(str(record["image_path"])), mode="RGB")
        image_tensor = self.preprocess(image)
        mask_image = open_pil_with_retry(Path(str(record["mask_path"])), mode="L")
        if self.mask_preprocess is None:
            mask_image = mask_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        else:
            mask_image = self.mask_preprocess(mask_image)
        mask = (np.asarray(mask_image, dtype=np.uint8) > self.mask_threshold).astype(np.uint8)
        return (
            image_tensor,
            torch.from_numpy(mask),
            int(record["label"]),
            int(record["place"]),
            int(record["group"]),
            str(record["relative_path"]),
            str(record["image_path"]),
            str(record["mask_path"]),
        )


class WaterbirdsProbabilityModel(nn.Module):
    """Normalize each repository model's classifier output to two probabilities."""

    def __init__(self, method: str, runner: MethodRunnerBase) -> None:
        super().__init__()
        self.method = method
        self.model = runner.model  # type: ignore[attr-defined]
        self.num_outputs = int(getattr(runner, "num_outputs", 2))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.method == "r4rr":
            logits, _features = self.model(images)
        elif self.method in ("gals", "upweight"):
            logits, _features = self.model(images)
        elif self.method == "abn":
            _attention_logits, logits, _attention_data = self.model(images)
        else:
            logits = self.model(images)

        if self.method == "abn" or (
            self.method in ("gals", "upweight") and self.num_outputs == 1
        ):
            probability_1 = torch.sigmoid(logits.reshape(-1))
            return torch.stack((1.0 - probability_1, probability_1), dim=1)
        return torch.softmax(logits, dim=1)


class CLIPZeroShotProbabilityModel(nn.Module):
    """OpenAI CLIP RN50 with the same prompt ensemble as the ZS baseline."""

    def __init__(self, model: nn.Module, text_features: np.ndarray) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "text_features",
            torch.from_numpy(text_features).float(),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(images).float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = self.model.logit_scale.exp().float().clamp(max=100.0)
        logits = scale * features @ self.text_features.t()
        return torch.softmax(logits, dim=1)


class CLIPLinearProbabilityModel(nn.Module):
    """Frozen CLIP image encoder plus an sklearn-fitted logistic head."""

    def __init__(
        self,
        model: nn.Module,
        coefficients: np.ndarray,
        intercept: np.ndarray,
    ) -> None:
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
        logits = features @ self.coefficients.t() + self.intercept
        if logits.shape[1] == 1:
            probability_1 = torch.sigmoid(logits[:, 0])
            return torch.stack((1.0 - probability_1, probability_1), dim=1)
        return torch.softmax(logits, dim=1)


def build_clip_mask_preprocess(image_size: int) -> Callable[[Image.Image], Image.Image]:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
        ]
    )


def build_clip_probability_model(
    method: str,
    data_path: Path,
    clip_model_name: str,
    clip_c: float,
    clip_feature_batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, Callable[[Image.Image], torch.Tensor], Dict[str, object]]:
    import run_clip_lr_sweep as clip_lr
    import run_clip_zeroshot_waterbirds as clip_zs

    clip_module = clip_lr._try_import_clip()
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
    }
    if method == "clip_zs":
        class_names = ("landbird", "waterbird")
        templates = clip_zs._default_templates()
        text_features = clip_zs._build_text_features(
            clip_module=clip_module,
            model=model,
            device=str(device),
            class_names=class_names,
            templates=templates,
        )
        details["clip_class_names"] = "|".join(class_names)
        details["clip_num_templates"] = len(templates)
        return CLIPZeroShotProbabilityModel(model, text_features), preprocess, details

    from sklearn.linear_model import LogisticRegression

    root, cfg, Waterbirds = clip_lr._load_waterbirds(str(data_path))
    train_dataset = Waterbirds(root=root, cfg=cfg, split="train", transform=preprocess)
    train_features, train_labels, _groups = clip_lr._extract_features(
        train_dataset,
        model,
        str(device),
        int(clip_feature_batch_size),
        int(num_workers),
    )
    train_features = np.ascontiguousarray(
        clip_lr._l2_normalize(train_features),
        dtype=np.float64,
    )
    classifier = LogisticRegression(
        random_state=int(seed),
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
    details["clip_train_samples"] = int(train_labels.shape[0])
    return (
        CLIPLinearProbabilityModel(model, classifier.coef_, classifier.intercept_),
        preprocess,
        details,
    )


def build_runner(
    method: str,
    checkpoint: Path,
    stage1_checkpoint: Optional[Path],
    afr_root: Path,
    device: torch.device,
) -> MethodRunnerBase:
    if method == "r4rr":
        return GuidedRunner(checkpoint, num_classes=2, device=device)
    if method in ("vanilla", "elrep"):
        return VanillaRunner(checkpoint, num_classes=2, device=device)
    if method in ("gals", "upweight"):
        return GALSRunner(checkpoint, device=device)
    if method == "abn":
        return ABNRunner(checkpoint, device=device)
    if method == "afr":
        if stage1_checkpoint is None:
            raise ValueError("AFR requires --afr-stage1-checkpoint")
        return AFRRunner(
            afr_root=afr_root,
            stage1_checkpoint=stage1_checkpoint,
            stage2_last_layer_checkpoint=checkpoint,
            num_classes=2,
            device=device,
        )
    raise ValueError(f"Unsupported method: {method}")


def pointing_result(saliency: np.ndarray, mask: np.ndarray) -> Tuple[bool, int, int, bool]:
    if saliency.shape != mask.shape:
        raise ValueError(f"Saliency/mask shape mismatch: {saliency.shape} != {mask.shape}")
    if not np.isfinite(saliency).all():
        raise ValueError("RISE saliency contains non-finite values")
    is_zero = float(np.max(saliency)) <= 1e-12
    if is_zero:
        return False, -1, -1, True
    peak_row, peak_col = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
    return bool(mask[peak_row, peak_col] > 0), int(peak_row), int(peak_col), False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tag", choices=("95", "100"), required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--afr-stage1-checkpoint", type=Path)
    parser.add_argument("--afr-root", type=Path, required=True)
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--clip-model", default="RN50")
    parser.add_argument("--clip-c", type=float, default=1.0)
    parser.add_argument("--clip-feature-batch-size", type=int, default=256)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--target-mode", choices=("label", "prediction"), default="label")
    parser.add_argument("--mask-threshold", type=int, default=0)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--max-masked-batch", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
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
    started = time.time()
    random.seed(int(args.sample_seed))
    np.random.seed(int(args.sample_seed))
    torch.manual_seed(int(args.seed))

    data_path = args.data_path.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else None
    stage1_checkpoint = (
        args.afr_stage1_checkpoint.expanduser().resolve()
        if args.afr_stage1_checkpoint is not None
        else None
    )
    afr_root = args.afr_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mask_bank_path = args.rise_masks_path.expanduser().resolve()
    for required in (data_path, mask_root):
        if not required.exists():
            raise FileNotFoundError(f"Missing required path: {required}")
    is_clip = args.method in ("clip_zs", "clip_lr")
    if not is_clip and (checkpoint is None or not checkpoint.is_file()):
        raise FileNotFoundError(f"Missing required checkpoint: {checkpoint}")
    if args.method == "afr" and (stage1_checkpoint is None or not stage1_checkpoint.is_file()):
        raise FileNotFoundError(f"Missing AFR stage-1 checkpoint: {stage1_checkpoint}")

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    runner: Optional[MethodRunnerBase] = None
    clip_details: Dict[str, object] = {
        "clip_model": "",
        "clip_c": "",
        "clip_penalty": "",
        "clip_solver": "",
        "clip_fit_intercept": "",
    }
    if is_clip:
        probability_model, preprocess, clip_details = build_clip_probability_model(
            method=args.method,
            data_path=data_path,
            clip_model_name=str(args.clip_model),
            clip_c=float(args.clip_c),
            clip_feature_batch_size=int(args.clip_feature_batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            device=device,
        )
        mask_preprocess = build_clip_mask_preprocess(IMAGE_SIZE)
    else:
        assert checkpoint is not None
        runner = build_runner(
            method=args.method,
            checkpoint=checkpoint,
            stage1_checkpoint=stage1_checkpoint,
            afr_root=afr_root,
            device=device,
        )
        probability_model = WaterbirdsProbabilityModel(args.method, runner)
        preprocess = None
        mask_preprocess = None

    dataset = WaterbirdsPointingDataset(
        data_path=data_path,
        mask_root=mask_root,
        split=args.split,
        max_samples=int(args.max_samples),
        sample_seed=int(args.sample_seed),
        mask_threshold=int(args.mask_threshold),
        preprocess=preprocess,
        mask_preprocess=mask_preprocess,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.image_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    probability_model = probability_model.to(device).eval()

    masks_np, mask_hash = load_or_create_mask_bank(
        path=mask_bank_path,
        num_masks=int(args.rise_num_masks),
        grid_size=int(args.rise_grid_size),
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        p1=float(args.rise_p1),
        seed=int(args.rise_seed),
    )
    rise_masks = torch.from_numpy(masks_np).to(device=device, dtype=torch.float32)

    group_hits = np.zeros(4, dtype=np.int64)
    group_totals = np.zeros(4, dtype=np.int64)
    group_correct = np.zeros(4, dtype=np.int64)
    group_mass_sum = np.zeros(4, dtype=np.float64)
    mask_pixels_total = 0
    zero_maps = 0
    processed = 0
    rows: List[Dict[str, object]] = []

    print(
        f"[INFO] dataset=waterbirds_{args.dataset_tag} method={args.method} seed={args.seed} "
        f"split={args.split} samples={len(dataset)} device={device} explainer={EXPLAINER}",
        flush=True,
    )
    print(f"[INFO] checkpoint={checkpoint or ''}", flush=True)
    if is_clip:
        print(f"[INFO] clip={clip_details}", flush=True)
    if stage1_checkpoint is not None:
        print(f"[INFO] afr_stage1_checkpoint={stage1_checkpoint}", flush=True)
    print(f"[INFO] mask_source={mask_root} protocol=CUB segmentation", flush=True)
    print(
        f"[INFO] rise_masks={mask_bank_path} sha256={mask_hash} "
        f"N={args.rise_num_masks} grid={args.rise_grid_size} "
        f"p1={args.rise_p1} seed={args.rise_seed}",
        flush=True,
    )

    try:
        for batch in loader:
            images, bird_masks, labels, places, groups, rel_paths, image_paths, mask_paths = batch
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            with torch.no_grad():
                probabilities = probability_model(images)
                predictions = probabilities.argmax(dim=1)
            targets = labels_device if args.target_mode == "label" else predictions
            saliency_batch = rise_from_probabilities_batch(
                probability_fn=probability_model,
                images=images,
                targets=targets,
                masks=rise_masks,
                p1=float(args.rise_p1),
                max_masked_batch=int(args.max_masked_batch),
            ).cpu().numpy()

            bird_masks_np = bird_masks.numpy()
            labels_np = labels.numpy()
            places_np = places.numpy()
            groups_np = groups.numpy()
            predictions_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for saliency, bird_mask, label, place, group, prediction, target, rel_path, image_path, mask_path in zip(
                saliency_batch,
                bird_masks_np,
                labels_np,
                places_np,
                groups_np,
                predictions_np,
                targets_np,
                rel_paths,
                image_paths,
                mask_paths,
            ):
                hit, peak_row, peak_col, is_zero = pointing_result(saliency, bird_mask)
                group_int = int(group)
                positive_saliency = np.maximum(saliency.astype(np.float64), 0.0)
                saliency_sum = float(positive_saliency.sum())
                mass_inside = (
                    float(positive_saliency[bird_mask > 0].sum() / saliency_sum)
                    if saliency_sum > 0.0
                    else 0.0
                )
                correct = int(int(prediction) == int(label))
                group_totals[group_int] += 1
                group_hits[group_int] += int(hit)
                group_correct[group_int] += correct
                group_mass_sum[group_int] += mass_inside
                mask_pixels_total += int(np.count_nonzero(bird_mask))
                zero_maps += int(is_zero)
                rows.append(
                    {
                        "dataset": f"waterbirds_{args.dataset_tag}",
                        "method": args.method,
                        "seed": int(args.seed),
                        "split": args.split,
                        "target_mode": args.target_mode,
                        "explainer": EXPLAINER,
                        "img_filename": rel_path,
                        "label": int(label),
                        "place": int(place),
                        "group": group_int,
                        "group_name": GROUP_NAMES[group_int],
                        "prediction": int(prediction),
                        "classification_correct": correct,
                        "saliency_target": int(target),
                        "pointing_hit": int(hit),
                        "peak_row": peak_row,
                        "peak_col": peak_col,
                        "saliency_max": float(np.max(saliency)),
                        "saliency_mass_in_bird": mass_inside,
                        "zero_saliency": int(is_zero),
                        "bird_mask_pixels": int(np.count_nonzero(bird_mask)),
                        "image_path": image_path,
                        "mask_path": mask_path,
                    }
                )
            processed += len(labels_np)
            if processed == len(dataset) or processed % 100 <= len(labels_np):
                print(f"[PROGRESS] {processed}/{len(dataset)}", flush=True)
    finally:
        if runner is not None:
            runner.close()

    if not rows:
        raise RuntimeError("No Waterbirds samples were evaluated")
    group_pg = np.divide(
        group_hits,
        group_totals,
        out=np.full(4, np.nan, dtype=np.float64),
        where=group_totals > 0,
    )
    group_acc = np.divide(
        group_correct,
        group_totals,
        out=np.full(4, np.nan, dtype=np.float64),
        where=group_totals > 0,
    )
    group_mass = np.divide(
        group_mass_sum,
        group_totals,
        out=np.full(4, np.nan, dtype=np.float64),
        where=group_totals > 0,
    )
    finite_groups = np.flatnonzero(np.isfinite(group_pg))
    worst_group = int(finite_groups[np.argmin(group_pg[finite_groups])])
    total = int(group_totals.sum())
    summary: Dict[str, object] = {
        "dataset": f"waterbirds_{args.dataset_tag}",
        "method": args.method,
        "seed": int(args.seed),
        "split": args.split,
        "target_mode": args.target_mode,
        "explainer": EXPLAINER,
        "mask_protocol_version": MASK_PROTOCOL_VERSION,
        "primary_pg_protocol": PRIMARY_PG_PROTOCOL,
        "map_height": IMAGE_SIZE,
        "map_width": IMAGE_SIZE,
        "pg_hits": int(group_hits.sum()),
        "pg_total": total,
        "pg_acc": float(group_hits.sum() / max(total, 1)),
        "pg_macro_group_acc": float(np.nanmean(group_pg)),
        "pg_worst_group_acc": float(group_pg[worst_group]),
        "pg_worst_group": GROUP_NAMES[worst_group],
        "pg_random_acc": float(mask_pixels_total / max(total * IMAGE_SIZE * IMAGE_SIZE, 1)),
        "classification_acc": float(group_correct.sum() / max(total, 1)),
        "saliency_mass_in_bird": float(np.nansum(group_mass_sum) / max(total, 1)),
        "zero_saliency_maps": int(zero_maps),
        "mask_source": "CUB_200_2011_segmentations",
        "mask_root": str(mask_root),
        "mask_threshold": int(args.mask_threshold),
        "max_samples": int(args.max_samples),
        "sample_seed": int(args.sample_seed),
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "afr_stage1_checkpoint": str(stage1_checkpoint) if stage1_checkpoint else "",
        **clip_details,
        "rise_num_masks": int(args.rise_num_masks),
        "rise_grid_size": int(args.rise_grid_size),
        "rise_p1": float(args.rise_p1),
        "rise_seed": int(args.rise_seed),
        "rise_masks_path": str(mask_bank_path),
        "rise_masks_sha256": mask_hash,
        "image_batch_size": int(args.image_batch_size),
        "max_masked_batch": int(args.max_masked_batch),
        "missing_images": 0,
        "missing_masks": 0,
        "errors": 0,
        "seconds": int(time.time() - started),
    }
    for group in range(4):
        name = GROUP_NAMES[group]
        summary[f"group_{name}_pg_hits"] = int(group_hits[group])
        summary[f"group_{name}_total"] = int(group_totals[group])
        summary[f"group_{name}_pg_acc"] = float(group_pg[group])
        summary[f"group_{name}_classification_acc"] = float(group_acc[group])
        summary[f"group_{name}_saliency_mass_in_bird"] = float(group_mass[group])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pointing_game_per_image.csv", rows)
    write_csv(output_dir / "pointing_game_summary.csv", [summary])
    temporary_json = output_dir / f".run_summary.{os.getpid()}.tmp"
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_json, output_dir / "run_summary.json")

    print(
        f"[RESULT] dataset=waterbirds_{args.dataset_tag} method={args.method} seed={args.seed} "
        f"rise_pg={100.0 * float(summary['pg_acc']):.2f}% "
        f"macro_group={100.0 * float(summary['pg_macro_group_acc']):.2f}% "
        f"worst_group={summary['pg_worst_group']} "
        f"worst={100.0 * float(summary['pg_worst_group_acc']):.2f}% "
        f"random={100.0 * float(summary['pg_random_acc']):.2f}% "
        f"mass_inside={100.0 * float(summary['saliency_mass_in_bird']):.2f}% "
        f"zero_maps={zero_maps}/{total}",
        flush=True,
    )
    print(f"[DONE] {output_dir / 'pointing_game_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
