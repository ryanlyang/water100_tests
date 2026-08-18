#!/usr/bin/env python3
"""Audit official ImageNet-9 image-to-foreground-mask joins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from imagenet9_pointing_game_utils import (
    PRIMARY_VARIANTS,
    index_foreground_masks,
    read_manifest,
    resolve_foreground_mask,
)
from imagenet9_final_utils import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-test-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=list(PRIMARY_VARIANTS))
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    mask_index = index_foreground_masks(args.mask_root)
    report = {
        "status": "ok",
        "official_manifest": str(args.official_manifest.resolve()),
        "official_test_root": str(args.official_test_root.resolve()),
        "mask_root": str(args.mask_root.resolve()),
        "mask_count": len(mask_index),
        "first_source_id_is_foreground": True,
        "variants": {},
    }
    for variant in args.variants:
        if variant not in PRIMARY_VARIANTS:
            raise ValueError(f"Unsupported Pointing Game variant: {variant}")
        rows = read_manifest(args.official_manifest, variant)
        unique_masks = set()
        class_counts = {}
        for row in rows:
            image_path = args.official_test_root / row["relative_path"]
            mask_path, foreground_id = resolve_foreground_mask(row, mask_index)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                image_shape = (image.height, image.width)
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            if tuple(np.asarray(mask).squeeze().shape) != image_shape:
                raise RuntimeError(
                    f"Native image/mask shape mismatch: variant={variant} "
                    f"image={image_path} image_shape={image_shape} "
                    f"mask={mask_path} mask_shape={mask.shape}"
                )
            unique_masks.add(str(mask_path.resolve()))
            label = int(row["label"])
            class_counts[str(label)] = class_counts.get(str(label), 0) + 1
            if not foreground_id:
                raise RuntimeError(f"Empty foreground ID for {image_path}")
        report["variants"][variant] = {
            "images": len(rows),
            "resolved_masks": len(rows),
            "unique_foreground_masks": len(unique_masks),
            "class_counts": class_counts,
            "native_shapes_match": True,
        }
        print(
            f"[AUDIT] variant={variant} images={len(rows)} "
            f"unique_masks={len(unique_masks)} native_shapes_match=true",
            flush=True,
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"[DONE] {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
