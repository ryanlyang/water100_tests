#!/usr/bin/env python3
"""Audit ImageNet-9 WeCLIP+ maps against the complete training manifest."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


EXPECTED_IMAGES = 45405
CLASS_NAMES: Tuple[str, ...] = (
    "dog",
    "bird",
    "vehicle",
    "reptile",
    "carnivore",
    "insect",
    "instrument",
    "primate",
    "fish",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-manifest", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--inference-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def voc_colormap(count: int = 256) -> np.ndarray:
    colors = np.zeros((count, 3), dtype=np.uint8)
    for index in range(count):
        red = green = blue = 0
        value = index
        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        colors[index] = (red, green, blue)
    return colors


def decode_voc_colors(image: np.ndarray, num_labels: int = 10) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB map, got shape {image.shape}")
    palette = voc_colormap(num_labels)
    labels = np.full(image.shape[:2], -1, dtype=np.int16)
    for label, color in enumerate(palette):
        matches = np.all(image == color, axis=2)
        labels[matches] = label
    if np.any(labels < 0):
        unknown = np.unique(image[labels < 0].reshape(-1, 3), axis=0)
        raise ValueError(f"Map contains colors outside labels 0..{num_labels - 1}: {unknown[:10]}")
    return labels


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = (
        "sample_id",
        "class_name",
        "expected_label",
        "width",
        "height",
        "background_fraction",
        "expected_foreground_fraction",
        "unexpected_class_fraction",
        "unique_labels",
        "empty_expected_foreground",
        "map_path",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    contract = json.loads(args.inference_contract.read_text())
    if contract.get("num_source_images") != EXPECTED_IMAGES:
        raise RuntimeError("Inference contract does not cover all 45,405 training images")
    if contract.get("official_validation_used") or contract.get("official_test_variants_used"):
        raise RuntimeError("Inference contract indicates held-out data leakage")

    with args.workspace_manifest.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} manifest rows, found {len(manifest_rows)}"
        )
    expected_ids = [row["sample_id"] for row in manifest_rows]
    if len(set(expected_ids)) != EXPECTED_IMAGES:
        raise RuntimeError("Workspace manifest contains duplicate IDs")

    all_png_paths = list(args.map_root.glob("*.png"))
    temporary_paths = [path for path in all_png_paths if path.name.startswith(".")]
    actual_paths = [path for path in all_png_paths if not path.name.startswith(".")]
    actual_by_id = {path.stem: path for path in actual_paths}
    if len(actual_by_id) != len(actual_paths):
        raise RuntimeError("Duplicate map stems detected")
    missing = sorted(set(expected_ids) - set(actual_by_id))
    extras = sorted(set(actual_by_id) - set(expected_ids))
    if missing or extras:
        raise RuntimeError(
            f"Map set mismatch: missing={len(missing)} {missing[:10]}, "
            f"extras={len(extras)} {extras[:10]}"
        )

    audit_rows: List[Dict[str, object]] = []
    class_stats: Dict[str, Dict[str, float]] = {
        name: {
            "images": 0,
            "empty_maps": 0,
            "foreground_fraction_sum": 0.0,
            "background_fraction_sum": 0.0,
            "unexpected_fraction_sum": 0.0,
        }
        for name in CLASS_NAMES
    }
    for index, row in enumerate(manifest_rows, start=1):
        sample_id = row["sample_id"]
        class_name = row["class_name"]
        expected_label = int(row["label"]) + 1
        source_path = Path(row["source_path"])
        map_path = actual_by_id[sample_id]
        with Image.open(source_path) as source:
            expected_size = source.size
        with Image.open(map_path) as map_file:
            if map_file.mode != "RGB":
                raise RuntimeError(f"Map is not RGB: {map_path} mode={map_file.mode}")
            if map_file.size != expected_size:
                raise RuntimeError(
                    f"Map/source size mismatch for {sample_id}: {map_file.size} != {expected_size}"
                )
            encoded = np.asarray(map_file, dtype=np.uint8)
        labels = decode_voc_colors(encoded, num_labels=10)
        total = float(labels.size)
        background_fraction = float(np.count_nonzero(labels == 0) / total)
        foreground_fraction = float(np.count_nonzero(labels == expected_label) / total)
        unexpected_fraction = float(
            np.count_nonzero((labels != 0) & (labels != expected_label)) / total
        )
        unique_labels = [int(value) for value in np.unique(labels)]
        empty = foreground_fraction == 0.0
        audit_rows.append(
            {
                "sample_id": sample_id,
                "class_name": class_name,
                "expected_label": expected_label,
                "width": expected_size[0],
                "height": expected_size[1],
                "background_fraction": background_fraction,
                "expected_foreground_fraction": foreground_fraction,
                "unexpected_class_fraction": unexpected_fraction,
                "unique_labels": json.dumps(unique_labels),
                "empty_expected_foreground": int(empty),
                "map_path": str(map_path),
            }
        )
        stats = class_stats[class_name]
        stats["images"] += 1
        stats["empty_maps"] += int(empty)
        stats["foreground_fraction_sum"] += foreground_fraction
        stats["background_fraction_sum"] += background_fraction
        stats["unexpected_fraction_sum"] += unexpected_fraction
        if index % 2500 == 0 or index == EXPECTED_IMAGES:
            print(f"[AUDIT] {index}/{EXPECTED_IMAGES}", flush=True)

    per_class = {}
    for class_name, stats in class_stats.items():
        count = int(stats["images"])
        per_class[class_name] = {
            "images": count,
            "empty_expected_foreground_maps": int(stats["empty_maps"]),
            "empty_expected_foreground_rate": stats["empty_maps"] / count,
            "mean_expected_foreground_fraction": stats["foreground_fraction_sum"] / count,
            "mean_background_fraction": stats["background_fraction_sum"] / count,
            "mean_unexpected_class_fraction": stats["unexpected_fraction_sum"] / count,
        }
    empty_count = sum(int(row["empty_expected_foreground"]) for row in audit_rows)
    summary = {
        "status": "ok",
        "expected_maps": EXPECTED_IMAGES,
        "valid_maps": len(audit_rows),
        "missing_maps": 0,
        "extra_maps": 0,
        "ignored_atomic_temporary_files": len(temporary_paths),
        "empty_expected_foreground_maps": empty_count,
        "empty_expected_foreground_rate": empty_count / EXPECTED_IMAGES,
        "mean_expected_foreground_fraction": float(
            np.mean([float(row["expected_foreground_fraction"]) for row in audit_rows])
        ),
        "mean_background_fraction": float(
            np.mean([float(row["background_fraction"]) for row in audit_rows])
        ),
        "mean_unexpected_class_fraction": float(
            np.mean([float(row["unexpected_class_fraction"]) for row in audit_rows])
        ),
        "per_class": per_class,
        "inference_contract": str(args.inference_contract.resolve()),
        "workspace_manifest": str(args.workspace_manifest.resolve()),
        "map_root": str(args.map_root.resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(args.output_dir / "map_audit.csv", audit_rows)
    _atomic_json(args.output_dir / "map_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[DONE] {args.output_dir / 'map_audit_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
