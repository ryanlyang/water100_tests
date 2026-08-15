#!/usr/bin/env python3
"""Build the canonical test-only RedMeat Pointing Game mask package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from validate_redmeat_pointing_masks import (
    DEFAULT_CLASSES,
    image_index,
    parse_classes,
    sha256_file,
    validate_package,
)


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive_package(package_root: Path, archive_path: Path) -> str:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        archive.add(package_root, arcname=package_root.name)
    checksum = sha256_file(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
    return checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-images", type=int, default=1250)
    parser.add_argument("--expected-per-class", type=int, default=250)
    parser.add_argument("--classes", type=parse_classes, default=DEFAULT_CLASSES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_manifest_path = args.review_root / "review_manifest.csv"
    binary_root = args.review_root / "binary_masks"
    if not review_manifest_path.is_file():
        raise FileNotFoundError(review_manifest_path)
    if not binary_root.is_dir():
        raise FileNotFoundError(binary_root)
    if args.output_root.exists():
        raise RuntimeError(
            f"Output already exists: {args.output_root}. Use a new path or remove it explicitly."
        )

    review_rows = load_manifest(review_manifest_path)
    source_images = image_index(args.source_root)
    if len(review_rows) != args.expected_images:
        raise RuntimeError(
            f"Review manifest rows={len(review_rows)}, expected {args.expected_images}"
        )
    if len(source_images) != args.expected_images:
        raise RuntimeError(
            f"Source images={len(source_images)}, expected {args.expected_images}"
        )

    output_parent = args.output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.staging.", dir=output_parent))
    try:
        package_rows: List[Dict[str, object]] = []
        seen: Set[Tuple[str, str]] = set()
        roboflow_counts: Counter[str] = Counter()
        for row in sorted(review_rows, key=lambda item: (item["class_name"], int(item["image_id"]))):
            image_id = str(row["image_id"])
            class_name = str(row["class_name"])
            key = (class_name, image_id)
            if key in seen:
                raise RuntimeError(f"Duplicate review-manifest key: {key}")
            seen.add(key)
            if class_name not in args.classes:
                raise RuntimeError(f"Unexpected class in review manifest: {class_name}")

            source_path = source_images.get(key)
            if source_path is None:
                raise RuntimeError(f"Missing source image for {class_name}/{image_id}")
            source_mask = binary_root / class_name / f"{image_id}.png"
            if not source_mask.is_file():
                raise FileNotFoundError(source_mask)

            with Image.open(source_path) as image:
                width, height = image.size
            with Image.open(source_mask) as mask_image:
                mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
            if mask.shape != (height, width):
                raise RuntimeError(
                    f"Dimension mismatch for {key}: image={(width, height)} mask={mask.shape[::-1]}"
                )
            values = set(int(value) for value in np.unique(mask))
            if not values.issubset({0, 255}):
                raise RuntimeError(f"Nonbinary mask for {key}: {sorted(values)}")
            foreground_pixels = int(np.count_nonzero(mask))
            if foreground_pixels <= 0:
                raise RuntimeError(f"Empty mask for {key}")

            relative_mask = Path("test") / class_name / f"{image_id}.png"
            output_mask = staging / relative_mask
            output_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_mask, output_mask)

            roboflow_split = str(row.get("roboflow_split", ""))
            roboflow_counts[roboflow_split] += 1
            package_rows.append(
                {
                    "image_id": image_id,
                    "class_name": class_name,
                    "dataset_split": "test",
                    "image_filename": source_path.name,
                    "source_relative_path": str(source_path.relative_to(args.source_root)),
                    "mask_relative_path": str(relative_mask),
                    "width": width,
                    "height": height,
                    "foreground_pixels": foreground_pixels,
                    "foreground_fraction": foreground_pixels / float(width * height),
                    "image_sha256": sha256_file(source_path),
                    "mask_sha256": sha256_file(output_mask),
                    "annotation_count": int(row["annotation_count"]),
                    "polygon_count": int(row["polygon_count"]),
                    "roboflow_export_split": roboflow_split,
                }
            )

        write_csv(staging / "manifest.csv", package_rows)
        class_counts = Counter(row["class_name"] for row in package_rows)
        metadata = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "food-101-redmeat",
            "dataset_split": "test",
            "mask_semantics": "union of all reviewed meat foreground polygons",
            "mask_values": [0, 255],
            "image_preprocessing": "Resize((224, 224))",
            "mask_preprocessing": "Resize((224, 224), nearest-neighbor)",
            "images": len(package_rows),
            "classes": list(args.classes),
            "class_counts": dict(sorted(class_counts.items())),
            "roboflow_export_split_counts": dict(sorted(roboflow_counts.items())),
            "roboflow_export_split_note": (
                "Ignored for evaluation. Every packaged image belongs to the original RedMeat test split."
            ),
            "source_review_manifest": str(review_manifest_path.resolve()),
        }
        (staging / "package_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "README.txt").write_text(
            "RedMeat Pointing Game masks\n"
            "===========================\n\n"
            "This package contains reviewed, test-only binary foreground masks.\n"
            "All polygons for an image are unioned. Values are 0 (background) and\n"
            "255 (meat foreground). Roboflow export split names are provenance only;\n"
            "all 1,250 images are from the original RedMeat test split. These masks\n"
            "must be used only for evaluation and never for model training or model\n"
            "selection.\n\n"
            "After extracting this package on research compute, validate it against\n"
            "the original dataset with:\n\n"
            "  python RedMeat_Runs/validate_redmeat_pointing_masks.py \\\n"
            "    --package-root /path/to/redmeat_pointing_masks \\\n"
            "    --data-root /home/ryreu/guided_cnn/Food101/data/food-101-redmeat\n\n"
            "The command must report valid=true and errors=[]. Source checksums are\n"
            "verified by default; use --skip-source-checksum only if the dataset\n"
            "images were intentionally re-encoded after download.\n",
            encoding="utf-8",
        )

        report = validate_package(
            package_root=staging,
            expected_images=args.expected_images,
            expected_per_class=args.expected_per_class,
            classes=args.classes,
            external_images=source_images,
            verify_source_checksum=True,
        )
        if not report["valid"]:
            raise RuntimeError(f"Staged package failed validation: {report['errors'][:10]}")

        staging.rename(args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_report = validate_package(
        package_root=args.output_root,
        expected_images=args.expected_images,
        expected_per_class=args.expected_per_class,
        classes=args.classes,
        external_images=source_images,
        verify_source_checksum=True,
    )
    if not final_report["valid"]:
        raise RuntimeError(f"Final package failed validation: {final_report['errors'][:10]}")
    (args.output_root / "validation_report.json").write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    archive_path = args.archive or args.output_root.with_suffix(".tar.gz")
    checksum = archive_package(args.output_root, archive_path)
    print(f"[DONE] package={args.output_root}")
    print(f"[DONE] archive={archive_path}")
    print(f"[DONE] archive_sha256={checksum}")
    print(f"[DONE] images={len(package_rows)} class_counts={dict(sorted(class_counts.items()))}")


if __name__ == "__main__":
    main()
