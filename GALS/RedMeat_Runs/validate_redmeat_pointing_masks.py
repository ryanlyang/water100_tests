#!/usr/bin/env python3
"""Validate a canonical RedMeat Pointing Game mask package.

The validator can operate in two modes:

1. Local/source mode, using ``--source-root`` with ``<class>/<image>`` files.
2. Research-compute mode, using ``--data-root`` and ``all_images.csv``.

In either mode, the package itself is checked for exact coverage, binary and
nonempty masks, dimensions, class counts, uniqueness, and recorded checksums.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


DEFAULT_CLASSES = (
    "baby_back_ribs",
    "filet_mignon",
    "pork_chop",
    "prime_rib",
    "steak",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_classes(value: str) -> Tuple[str, ...]:
    classes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not classes:
        raise argparse.ArgumentTypeError("At least one class is required")
    if len(classes) != len(set(classes)):
        raise argparse.ArgumentTypeError(f"Duplicate class names: {classes}")
    return classes


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_index(source_root: Path) -> Dict[Tuple[str, str], Path]:
    result: Dict[Tuple[str, str], Path] = {}
    for path in sorted(source_root.glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = (path.parent.name, path.stem)
        if key in result:
            raise RuntimeError(f"Duplicate source image key {key}: {result[key]} and {path}")
        result[key] = path
    return result


def _metadata_image_candidates(data_root: Path, raw_path: str, label: str) -> Iterable[Path]:
    raw = Path(str(raw_path).replace("\\", "/"))
    basename = raw.name
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(data_root / raw)
    candidates.extend(
        [
            data_root / str(raw).lstrip("/"),
            data_root / "images" / label / basename,
            data_root / label / basename,
        ]
    )
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            yield candidate


def metadata_image_index(
    data_root: Path,
    metadata_csv: Path,
    split: str,
    split_col: str,
    label_col: str,
    path_col: str,
) -> Dict[Tuple[str, str], Path]:
    rows = load_csv(metadata_csv)
    if not rows:
        raise RuntimeError(f"Empty metadata CSV: {metadata_csv}")
    required = {split_col, label_col, path_col}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise RuntimeError(f"Metadata is missing columns: {sorted(missing_columns)}")

    result: Dict[Tuple[str, str], Path] = {}
    unresolved: List[str] = []
    for row in rows:
        if str(row[split_col]) != split:
            continue
        label = str(row[label_col])
        image_id = Path(str(row[path_col]).replace("\\", "/")).stem
        key = (label, image_id)
        if key in result:
            raise RuntimeError(f"Duplicate metadata key for split={split}: {key}")
        resolved = next(
            (path for path in _metadata_image_candidates(data_root, row[path_col], label) if path.is_file()),
            None,
        )
        if resolved is None:
            unresolved.append(f"{label}/{image_id}: {row[path_col]}")
        else:
            result[key] = resolved
    if unresolved:
        raise RuntimeError(
            f"Could not resolve {len(unresolved)} metadata images under {data_root}; "
            f"preview={unresolved[:10]}"
        )
    return result


def validate_package(
    package_root: Path,
    expected_images: int,
    expected_per_class: int,
    classes: Sequence[str],
    external_images: Optional[Mapping[Tuple[str, str], Path]] = None,
    verify_source_checksum: bool = True,
) -> Dict[str, object]:
    manifest_path = package_root / "manifest.csv"
    metadata_path = package_root / "package_metadata.json"
    rows = load_csv(manifest_path)
    errors: List[str] = []

    if not metadata_path.is_file():
        errors.append(f"Missing package metadata: {metadata_path}")
        metadata: Dict[str, object] = {}
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            errors.append(
                f"Unsupported schema_version={metadata.get('schema_version')}; expected {SCHEMA_VERSION}"
            )

    required_columns = {
        "image_id",
        "class_name",
        "dataset_split",
        "image_filename",
        "mask_relative_path",
        "width",
        "height",
        "foreground_pixels",
        "foreground_fraction",
        "image_sha256",
        "mask_sha256",
    }
    if rows:
        missing_columns = required_columns - set(rows[0])
        if missing_columns:
            errors.append(f"Manifest is missing columns: {sorted(missing_columns)}")
    else:
        errors.append("Manifest has no rows")

    keys: List[Tuple[str, str]] = []
    counts: Counter[str] = Counter()
    actual_mask_paths = set()
    foreground_fractions: List[float] = []

    for row_number, row in enumerate(rows, start=2):
        if not required_columns.issubset(row):
            continue
        image_id = row["image_id"]
        class_name = row["class_name"]
        key = (class_name, image_id)
        keys.append(key)
        counts[class_name] += 1

        if row["dataset_split"] != "test":
            errors.append(f"row {row_number}: dataset_split={row['dataset_split']!r}, expected 'test'")
        if class_name not in classes:
            errors.append(f"row {row_number}: unknown class {class_name!r}")
        if Path(row["image_filename"]).stem != image_id:
            errors.append(f"row {row_number}: image filename does not match image_id")

        mask_path = package_root / row["mask_relative_path"]
        actual_mask_paths.add(mask_path.resolve())
        if not mask_path.is_file():
            errors.append(f"row {row_number}: missing mask {mask_path}")
            continue

        try:
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
        except Exception as exc:
            errors.append(f"row {row_number}: could not read {mask_path}: {exc}")
            continue

        expected_width = int(row["width"])
        expected_height = int(row["height"])
        if mask.shape != (expected_height, expected_width):
            errors.append(
                f"row {row_number}: mask shape={mask.shape}, expected "
                f"{(expected_height, expected_width)}"
            )
        unique_values = set(int(value) for value in np.unique(mask))
        if not unique_values.issubset({0, 255}):
            errors.append(f"row {row_number}: nonbinary values {sorted(unique_values)[:20]}")
        foreground_pixels = int(np.count_nonzero(mask))
        if foreground_pixels <= 0:
            errors.append(f"row {row_number}: empty mask")
        if foreground_pixels != int(row["foreground_pixels"]):
            errors.append(
                f"row {row_number}: foreground_pixels={foreground_pixels}, "
                f"manifest={row['foreground_pixels']}"
            )
        foreground_fraction = foreground_pixels / float(max(mask.size, 1))
        foreground_fractions.append(foreground_fraction)
        if not np.isclose(foreground_fraction, float(row["foreground_fraction"]), atol=1e-12):
            errors.append(f"row {row_number}: foreground fraction mismatch")
        if sha256_file(mask_path) != row["mask_sha256"]:
            errors.append(f"row {row_number}: mask checksum mismatch")

        if external_images is not None:
            image_path = external_images.get(key)
            if image_path is None:
                errors.append(f"row {row_number}: no external image for {class_name}/{image_id}")
                continue
            try:
                with Image.open(image_path) as image:
                    image_size = image.size
            except Exception as exc:
                errors.append(f"row {row_number}: could not read image {image_path}: {exc}")
                continue
            if image_size != (expected_width, expected_height):
                errors.append(
                    f"row {row_number}: image size={image_size}, expected "
                    f"{(expected_width, expected_height)}"
                )
            if verify_source_checksum and sha256_file(image_path) != row["image_sha256"]:
                errors.append(f"row {row_number}: source image checksum mismatch: {image_path}")

    duplicate_keys = [key for key, count in Counter(keys).items() if count != 1]
    if duplicate_keys:
        errors.append(f"Duplicate manifest keys: {duplicate_keys[:10]}")
    if len(rows) != expected_images:
        errors.append(f"Manifest rows={len(rows)}, expected {expected_images}")
    if external_images is not None:
        manifest_keys = set(keys)
        external_keys = set(external_images)
        missing_masks = sorted(external_keys - manifest_keys)
        extra_masks = sorted(manifest_keys - external_keys)
        if missing_masks:
            errors.append(f"External images without masks={len(missing_masks)} preview={missing_masks[:10]}")
        if extra_masks:
            errors.append(f"Masks without external images={len(extra_masks)} preview={extra_masks[:10]}")

    for class_name in classes:
        if counts[class_name] != expected_per_class:
            errors.append(
                f"Class {class_name}: masks={counts[class_name]}, expected {expected_per_class}"
            )
    unexpected_classes = sorted(set(counts) - set(classes))
    if unexpected_classes:
        errors.append(f"Unexpected classes: {unexpected_classes}")

    disk_masks = {
        path.resolve() for path in (package_root / "test").glob("*/*.png") if path.is_file()
    }
    orphan_masks = sorted(str(path) for path in disk_masks - actual_mask_paths)
    if orphan_masks:
        errors.append(f"Unreferenced masks on disk={len(orphan_masks)} preview={orphan_masks[:10]}")
    missing_disk_entries = sorted(str(path) for path in actual_mask_paths - disk_masks)
    if missing_disk_entries:
        errors.append(
            f"Manifest mask paths outside canonical test tree={len(missing_disk_entries)} "
            f"preview={missing_disk_entries[:10]}"
        )

    report: Dict[str, object] = {
        "valid": not errors,
        "schema_version": SCHEMA_VERSION,
        "package_root": str(package_root.resolve()),
        "images": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "foreground_fraction_min": min(foreground_fractions) if foreground_fractions else None,
        "foreground_fraction_median": float(np.median(foreground_fractions)) if foreground_fractions else None,
        "foreground_fraction_max": max(foreground_fractions) if foreground_fractions else None,
        "external_images_checked": len(external_images) if external_images is not None else 0,
        "source_checksums_verified": bool(external_images is not None and verify_source_checksum),
        "errors": errors,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--source-root", type=Path)
    source_group.add_argument("--data-root", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--path-col", default="abs_file_path")
    parser.add_argument("--expected-images", type=int, default=1250)
    parser.add_argument("--expected-per-class", type=int, default=250)
    parser.add_argument("--classes", type=parse_classes, default=DEFAULT_CLASSES)
    parser.add_argument(
        "--skip-source-checksum",
        action="store_true",
        help="Allow matching dimensions/IDs when RC image bytes intentionally differ.",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_images: Optional[Mapping[Tuple[str, str], Path]] = None
    if args.source_root is not None:
        external_images = image_index(args.source_root)
    elif args.data_root is not None:
        metadata_csv = args.metadata_csv or (args.data_root / "all_images.csv")
        external_images = metadata_image_index(
            data_root=args.data_root,
            metadata_csv=metadata_csv,
            split=args.split,
            split_col=args.split_col,
            label_col=args.label_col,
            path_col=args.path_col,
        )
    elif args.metadata_csv is not None:
        raise SystemExit("--metadata-csv requires --data-root")

    report = validate_package(
        package_root=args.package_root,
        expected_images=args.expected_images,
        expected_per_class=args.expected_per_class,
        classes=args.classes,
        external_images=external_images,
        verify_source_checksum=not args.skip_source_checksum,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
