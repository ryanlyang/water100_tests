#!/usr/bin/env python3
"""Audit prepared ImageNet-9 manifests through the training/evaluation loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imagenet9_data import (
    CLASS_NAMES,
    FORBIDDEN_SELECTION_VARIANTS,
    TUNING_OBJECTIVE,
    ImageNet9Dataset,
    build_eval_transform,
    class_counts,
    load_official_variant_samples,
    load_original_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ryreu/guided_cnn/data/imagenet9"),
    )
    parser.add_argument("--protocol-name", default="reconstructed_original_bbox1_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = args.data_root / "metadata" / args.protocol_name
    manifest = metadata / "manifest.csv"
    official_manifest = metadata / "official_test_manifest.csv"
    official_root = args.data_root / "official_test" / "bg_challenge"

    train = load_original_samples(manifest, "train")
    val = load_original_samples(manifest, "val")
    expected_train = {name: 5045 for name in CLASS_NAMES}
    expected_val = {name: 450 for name in CLASS_NAMES}
    if class_counts(train) != expected_train:
        raise RuntimeError(f"Unexpected train counts: {class_counts(train)}")
    if class_counts(val) != expected_val:
        raise RuntimeError(f"Unexpected validation counts: {class_counts(val)}")
    overlap = {sample.sample_id for sample in train} & {sample.sample_id for sample in val}
    if overlap:
        raise RuntimeError(f"Train/validation overlap: {sorted(overlap)[:10]}")

    variant_counts = {}
    transform = build_eval_transform()
    for variant in FORBIDDEN_SELECTION_VARIANTS:
        samples = load_official_variant_samples(official_manifest, official_root, variant)
        counts = class_counts(samples)
        expected = {name: 450 for name in CLASS_NAMES}
        if counts != expected:
            raise RuntimeError(f"Unexpected {variant} counts: {counts}")
        variant_counts[variant] = len(samples)

    # Decode and preprocess representative samples through the real loader.
    for samples in (train[:1], val[:1]):
        item = ImageNet9Dataset(samples, transform)[0]
        if tuple(item["image"].shape) != (3, 224, 224):
            raise RuntimeError(f"Unexpected transformed image shape: {item['image'].shape}")

    report = {
        "status": "ok",
        "train_total": len(train),
        "validation_total": len(val),
        "train_counts": class_counts(train),
        "validation_counts": class_counts(val),
        "official_variant_totals": variant_counts,
        "tuning_objective": TUNING_OBJECTIVE,
        "official_variants_allowed_for_selection": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

