#!/usr/bin/env python3
"""Resumable two-stage AFR training and validation selection for ImageNet-9."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import models

from imagenet9_data import (
    CLASS_NAMES,
    NUM_CLASSES,
    ImageNet9Dataset,
    build_eval_transform,
    build_train_transform,
    load_original_samples,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=21)
    parser.add_argument("--stage1-prop", type=float, default=0.8)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=0.003)
    parser.add_argument("--stage1-weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-momentum", type=float, default=0.9)
    parser.add_argument("--stage2-epochs", type=int, default=500)
    parser.add_argument("--stage2-lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=94.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resnet50() -> nn.Module:
    if hasattr(models, "ResNet50_Weights"):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def _features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model.conv1(images)
    output = model.bn1(output)
    output = model.relu(output)
    output = model.maxpool(output)
    output = model.layer1(output)
    output = model.layer2(output)
    output = model.layer3(output)
    output = model.layer4(output)
    return torch.flatten(model.avgpool(output), 1)


def _split_samples(samples, stage1_prop: float, split_seed: int):
    if not 0.0 < stage1_prop < 1.0:
        raise ValueError("--stage1-prop must be between zero and one")
    indices = list(range(len(samples)))
    random.Random(split_seed).shuffle(indices)
    first_count = int(len(indices) * stage1_prop)
    first = sorted(indices[:first_count])
    second = sorted(indices[first_count:])
    if set(first).intersection(second) or len(first) + len(second) != len(samples):
        raise RuntimeError("AFR stage-1/stage-2 partition is invalid")
    return [samples[index] for index in first], [samples[index] for index in second]


def _loader(samples, transform, args, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ImageNet9Dataset(samples, transform),
        batch_size=args.batch_size if shuffle else args.embedding_batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, List[float]]:
    predictions = logits.argmax(dim=1)
    accuracy = float((predictions == labels).float().mean().item())
    per_class = []
    for label in range(NUM_CLASSES):
        mask = labels == label
        if not torch.any(mask):
            raise RuntimeError(f"Metric input is missing class {label}")
        per_class.append(float((predictions[mask] == labels[mask]).float().mean().item()))
    return accuracy, float(np.mean(per_class)), per_class


@torch.no_grad()
def _evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits.append(model(images).cpu())
        labels.append(batch["label"].cpu())
    return _metrics(torch.cat(logits), torch.cat(labels))


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _train_stage1(args, stage1_samples, val_samples, checkpoint: Path, device: torch.device) -> None:
    model = _resnet50().to(device)
    train_loader = _loader(stage1_samples, build_train_transform(), args, True, args.seed)
    val_loader = _loader(val_samples, build_eval_transform(), args, False, args.seed)
    optimizer = SGD(
        model.parameters(),
        lr=args.stage1_lr,
        momentum=args.stage1_momentum,
        weight_decay=args.stage1_weight_decay,
    )
    for epoch in range(args.stage1_epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * images.shape[0]
            count += images.shape[0]
        val_accuracy, val_macro, _ = _evaluate_model(model, val_loader, device)
        print(
            f"[STAGE1] epoch={epoch + 1}/{args.stage1_epochs} "
            f"train_loss={loss_sum / max(count, 1):.6f} "
            f"val_acc={val_accuracy:.6f} val_macro={val_macro:.6f}",
            flush=True,
        )
    _atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "stage1_prop": args.stage1_prop,
            "stage1_epochs": args.stage1_epochs,
            "stage1_lr": args.stage1_lr,
            "stage1_weight_decay": args.stage1_weight_decay,
        },
        checkpoint,
    )


@torch.no_grad()
def _encode(model, samples, args, device):
    loader = _loader(samples, build_eval_transform(), args, False, args.seed)
    features, logits, labels = [], [], []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        batch_features = _features(model, images)
        features.append(batch_features.cpu())
        logits.append(model.fc(batch_features).cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(features), torch.cat(logits), torch.cat(labels)


def _build_embedding_cache(args, stage2_samples, val_samples, checkpoint, cache, device):
    payload = torch.load(checkpoint, map_location="cpu")
    model = _resnet50()
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    stage2_features, stage2_logits, stage2_labels = _encode(
        model, stage2_samples, args, device
    )
    val_features, _val_logits, val_labels = _encode(model, val_samples, args, device)
    _atomic_torch_save(
        {
            "stage1_checkpoint_sha256": _sha256(checkpoint),
            "stage2_features": stage2_features,
            "stage2_logits": stage2_logits,
            "stage2_labels": stage2_labels,
            "val_features": val_features,
            "val_labels": val_labels,
            "initial_weight": model.fc.weight.detach().cpu(),
            "initial_bias": model.fc.bias.detach().cpu(),
        },
        cache,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _afr_weights(logits: torch.Tensor, labels: torch.Tensor, gamma: float) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    weights = torch.exp(-gamma * true_probability)
    for label in range(NUM_CLASSES):
        mask = labels == label
        weights[mask] *= labels.numel() / int(mask.sum().item())
    return weights / weights.sum()


def _run_stage2_configuration(
    args,
    cache,
    gamma: float,
    reg_coeff: float,
    output_path: Path,
    device: torch.device,
) -> Dict[str, object]:
    features = cache["stage2_features"].to(device)
    base_logits = cache["stage2_logits"].to(device)
    labels = cache["stage2_labels"].to(device)
    val_features = cache["val_features"].to(device)
    val_labels = cache["val_labels"].to(device)
    initial_weight = cache["initial_weight"].to(device)
    initial_bias = cache["initial_bias"].to(device)
    classifier = nn.Linear(initial_weight.shape[1], initial_weight.shape[0]).to(device)
    classifier.weight.data.copy_(initial_weight)
    classifier.bias.data.copy_(initial_bias)
    optimizer = SGD(classifier.parameters(), lr=args.stage2_lr, momentum=0.0, weight_decay=0.0)
    weights = _afr_weights(base_logits, labels, gamma).detach()
    best_macro = -1.0
    best_epoch = -1
    best_accuracy = -1.0
    best_per_class: List[float] = []
    best_state = None
    started = time.time()
    for epoch in range(args.stage2_epochs):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(features)
        weighted_ce = (weights * F.cross_entropy(logits, labels, reduction="none")).sum()
        regularizer = (
            (classifier.weight - initial_weight).square().sum()
            + (classifier.bias - initial_bias).square().sum()
        )
        loss = weighted_ce + reg_coeff * regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            val_logits = classifier(val_features)
            val_accuracy, val_macro, per_class = _metrics(val_logits, val_labels)
        if val_macro > best_macro:
            best_macro = val_macro
            best_epoch = epoch + 1
            best_accuracy = val_accuracy
            best_per_class = per_class
            best_state = copy.deepcopy(classifier.state_dict())
    result: Dict[str, object] = {
        "gamma": gamma,
        "reg_coeff": reg_coeff,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "best_val_macro_class_accuracy": best_macro,
        "best_val_per_class_accuracy": best_per_class,
        "seconds": time.time() - started,
        "classifier_checkpoint": str(output_path.with_suffix(".pt").resolve()),
    }
    _atomic_torch_save(
        {
            "model_state_dict": best_state,
            "gamma": gamma,
            "reg_coeff": reg_coeff,
            "result": result,
        },
        output_path.with_suffix(".pt"),
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return result


def _write_summary(results: List[Dict[str, object]], args) -> None:
    fieldnames = [
        "gamma", "reg_coeff", "best_epoch", "best_val_accuracy",
        "best_val_macro_class_accuracy", "best_val_per_class_accuracy",
        "seconds", "classifier_checkpoint",
    ]
    csv_path = args.run_root / "stage2_results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["best_val_per_class_accuracy"] = json.dumps(row["best_val_per_class_accuracy"])
            writer.writerow(row)
    temporary.replace(csv_path)
    best = max(results, key=lambda row: float(row["best_val_macro_class_accuracy"]))
    summary = {
        "method": "afr",
        "selection_objective": "val_macro_class_accuracy",
        "official_variants_used_for_selection": False,
        "stage1_train_fraction": args.stage1_prop,
        "stage2_train_fraction": 1.0 - args.stage1_prop,
        "completed_stage2_configurations": len(results),
        "target_stage2_configurations": 1 if args.smoke else 165,
        "best": best,
    }
    path = args.run_root / "summary.json"
    temporary_json = path.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_json.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.run_root.mkdir(parents=True, exist_ok=True)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    contract = {
        "method": "afr",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "stage1_prop": args.stage1_prop,
        "stage1_epochs": args.stage1_epochs,
        "stage1_lr": args.stage1_lr,
        "stage1_weight_decay": args.stage1_weight_decay,
        "stage1_momentum": args.stage1_momentum,
        "stage2_epochs": args.stage2_epochs,
        "stage2_lr": args.stage2_lr,
        "batch_size": args.batch_size,
        "smoke": args.smoke,
        "objective": "val_macro_class_accuracy",
        "official_variants_used": False,
    }
    contract_path = args.run_root / "contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract:
            raise RuntimeError("Refusing to resume AFR with a changed experiment contract")
    else:
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_everything(args.seed)
    all_train = load_original_samples(args.manifest, "train", verify_files=True)
    val_samples = load_original_samples(args.manifest, "val", verify_files=True)
    stage1_samples, stage2_samples = _split_samples(all_train, args.stage1_prop, args.split_seed)
    print(
        f"[AFR] stage1_train={len(stage1_samples)} stage2_train={len(stage2_samples)} "
        f"val={len(val_samples)} official_variants_used=NO",
        flush=True,
    )
    checkpoint = args.run_root / "stage1_final.pt"
    if not checkpoint.is_file():
        _train_stage1(args, stage1_samples, val_samples, checkpoint, device)
    else:
        print(f"[RESUME] stage-1 checkpoint: {checkpoint}", flush=True)
    cache_path = args.run_root / "embeddings.pt"
    if not cache_path.is_file():
        _build_embedding_cache(
            args, stage2_samples, val_samples, checkpoint, cache_path, device
        )
    cache = torch.load(cache_path, map_location="cpu")
    if cache.get("stage1_checkpoint_sha256") != _sha256(checkpoint):
        raise RuntimeError("AFR embedding cache does not match the stage-1 checkpoint")
    cache = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in cache.items()
    }

    configurations = [(4.0 + 0.5 * index, reg) for index in range(33) for reg in (0.0, 0.1, 0.2, 0.3, 0.4)]
    if args.smoke:
        configurations = [(4.0, 0.0)]
    result_dir = args.run_root / "stage2"
    result_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: List[Dict[str, object]] = []
    for index, (gamma, reg_coeff) in enumerate(configurations, start=1):
        name = f"gamma_{gamma:g}_reg_{reg_coeff:g}".replace(".", "p")
        output_path = result_dir / f"{name}.json"
        if output_path.is_file():
            result = json.loads(output_path.read_text())
            print(f"[RESUME] {index}/{len(configurations)} {name}", flush=True)
        else:
            if time.time() - started >= args.max_hours * 3600:
                print("[INCOMPLETE] Wall-clock budget reached; resubmit to continue.", flush=True)
                break
            result = _run_stage2_configuration(
                args, cache, gamma, reg_coeff, output_path, device
            )
            print(
                f"[STAGE2] {index}/{len(configurations)} gamma={gamma:g} "
                f"reg={reg_coeff:g} val_macro={result['best_val_macro_class_accuracy']:.6f}",
                flush=True,
            )
        results.append(result)
        _write_summary(results, args)
    print(f"[DONE] AFR stage2={len(results)}/{len(configurations)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
