#!/usr/bin/env python3
"""Evaluate one frozen ImageNet-9 checkpoint on every official test variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from imagenet9_data import (
    FORBIDDEN_SELECTION_VARIANTS,
    ImageNet9Dataset,
    build_eval_transform,
    classification_metrics,
    load_official_variant_samples,
)
from imagenet9_final_utils import atomic_json
from train_imagenet9_baseline import _forward, build_model


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--afr-classifier-checkpoint", type=Path)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-test-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--selection-value", type=float)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-file-checks", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    if args.method == "afr":
        if args.afr_classifier_checkpoint is None:
            raise ValueError("AFR evaluation requires --afr-classifier-checkpoint")
        model, _ = build_model("erm", pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        classifier = torch.load(str(args.afr_classifier_checkpoint), map_location="cpu")
        model.fc.load_state_dict(classifier["model_state_dict"], strict=True)
        return model.to(device), lambda images: model(images)

    stored_method = checkpoint.get("method")
    if stored_method != args.method:
        raise RuntimeError(
            f"Checkpoint method mismatch: requested={args.method} stored={stored_method}"
        )
    model, _ = build_model(args.method, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    return model, lambda images: _forward(args.method, model, images)[0]


@torch.no_grad()
def evaluate_variant(args, model, forward, variant: str, device: torch.device) -> Dict[str, object]:
    samples = load_official_variant_samples(
        args.official_manifest,
        args.official_test_root,
        variant,
        verify_files=not args.skip_file_checks,
    )
    loader = DataLoader(
        ImageNet9Dataset(samples, build_eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    predictions = []
    targets = []
    started = time.time()
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        predictions.append(forward(images).argmax(dim=1).cpu())
        targets.append(batch["label"].cpu())
    metrics = classification_metrics(torch.cat(predictions), torch.cat(targets))
    metrics["samples"] = len(samples)
    metrics["seconds"] = time.time() - started
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    for path in (args.checkpoint, args.official_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.method == "afr" and (
        args.afr_classifier_checkpoint is None
        or not args.afr_classifier_checkpoint.is_file()
    ):
        raise FileNotFoundError(args.afr_classifier_checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, forward = load_model(args, device)
    variant_results: Dict[str, Mapping[str, object]] = {}
    for variant in FORBIDDEN_SELECTION_VARIANTS:
        metrics = evaluate_variant(args, model, forward, variant, device)
        variant_results[variant] = metrics
        print(
            f"[EVAL] method={args.method} seed={args.seed} variant={variant} "
            f"acc={100.0 * float(metrics['accuracy']):.2f} "
            f"macro={100.0 * float(metrics['macro_class_accuracy']):.2f}",
            flush=True,
        )
    payload: Dict[str, object] = {
        "method": args.method,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "selection_objective": "val_macro_class_accuracy",
        "selection_value": args.selection_value,
        "official_variants_used_for_selection": False,
        "variant_results": variant_results,
    }
    if args.afr_classifier_checkpoint is not None:
        payload["afr_classifier_checkpoint"] = str(args.afr_classifier_checkpoint.resolve())
        payload["afr_classifier_checkpoint_sha256"] = sha256(
            args.afr_classifier_checkpoint
        )
    atomic_json(args.output_json, payload)
    print(f"[DONE] {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
