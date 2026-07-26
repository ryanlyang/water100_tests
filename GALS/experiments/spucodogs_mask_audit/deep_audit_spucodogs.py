#!/usr/bin/env python3
"""Read-only, evidence-producing audit of the complete SpuCoDogs dataset.

This is the second-stage audit.  The first-stage mask audit established the
pickle schema, filename-keyed lookup, polarity, and complete dog-image
coverage.  This script verifies the live split hierarchy and official group
counts, checks every image/mask geometry pair, screens for split leakage,
quantifies mask-quality warning signs, inventories reusable loaders/manifests,
rechecks integrity receipts, and projects compact-artifact storage.

No source image, mask, checksum, or repository file is modified.  Pickle
loading can execute code; only run this against the trusted author-provided
artifact after verifying its recorded checksum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pickle
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
TARGETS = {
    "small_dogs": 0,
    "big_dogs": 1,
}
ENVIRONMENTS = {
    "indoor": 0,
    "outdoor": 1,
}
EXPECTED_COUNTS = {
    ("train", "small_dogs", "indoor"): 10_000,
    ("train", "small_dogs", "outdoor"): 500,
    ("train", "big_dogs", "indoor"): 500,
    ("train", "big_dogs", "outdoor"): 10_000,
    ("val", "small_dogs", "indoor"): 500,
    ("val", "small_dogs", "outdoor"): 25,
    ("val", "big_dogs", "indoor"): 25,
    ("val", "big_dogs", "outdoor"): 500,
    ("test", "small_dogs", "indoor"): 500,
    ("test", "small_dogs", "outdoor"): 500,
    ("test", "big_dogs", "indoor"): 500,
    ("test", "big_dogs", "outdoor"): 500,
}
OFFICIAL_DOGS_SOURCE = (
    "https://github.com/BigML-CS-UCLA/SpuCo/blob/master/"
    "src/spuco/datasets/spuco_dogs.py"
)
OFFICIAL_MASK_SOURCE = (
    "https://github.com/BigML-CS-UCLA/SpuCo/blob/master/"
    "src/spuco/datasets/spuco_animals.py"
)
EXIF_ORIENTATION_TAG = 274
EXIF_IDENTITY_TAG_NAMES = {
    "Make",
    "Model",
    "Software",
    "DateTime",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "ImageUniqueID",
    "ImageDescription",
    "Artist",
    "Copyright",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative_path: str
    split: str
    target_name: str
    target_label: int
    environment_name: str
    environment_label: int
    mask_id: int


class BKTree:
    """Small BK-tree for 64-bit perceptual hashes."""

    def __init__(self) -> None:
        self.root: tuple[int, list[int], dict[int, Any]] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return int((left ^ right).bit_count())

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = (value, [index], {})
            return
        node = self.root
        while True:
            node_value, indices, children = node
            distance = self.distance(value, node_value)
            if distance == 0:
                indices.append(index)
                return
            child = children.get(distance)
            if child is None:
                children[distance] = (value, [index], {})
                return
            node = child

    def query(self, value: int, radius: int) -> Iterator[tuple[int, int]]:
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node_value, indices, children = stack.pop()
            distance = self.distance(value, node_value)
            if distance <= radius:
                for index in indices:
                    yield distance, index
            lower = distance - radius
            upper = distance + radius
            stack.extend(
                child
                for edge, child in children.items()
                if lower <= edge <= upper
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--mask-checksum", type=Path, required=True)
    parser.add_argument("--archive-checksum", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--perceptual-hamming-threshold",
        type=int,
        default=4,
        help="Maximum 64-bit dHash distance for a review candidate.",
    )
    parser.add_argument(
        "--max-perceptual-pairs",
        type=int,
        default=100_000,
        help="Safety cap; truncation is reported explicitly.",
    )
    parser.add_argument("--quality-preview-count", type=int, default=40)
    parser.add_argument(
        "--minimum-core-area-fraction",
        type=float,
        default=0.01,
        help="Broad audit warning threshold, not a training filter.",
    )
    parser.add_argument(
        "--maximum-core-area-fraction",
        type=float,
        default=0.95,
        help="Broad audit warning threshold, not a training filter.",
    )
    parser.add_argument(
        "--meaningful-component-fraction",
        type=float,
        default=0.002,
        help="Minimum image fraction for a connected core component to count.",
    )
    parser.add_argument("--storage-reserve-gib", type=float, default=10.0)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_receipt(path: Path | None) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "receipt": str(path) if path is not None else None,
        "expected_sha256": None,
        "recorded_filename": None,
    }
    if path is None or not path.is_file():
        return result
    tokens = path.read_text(errors="replace").strip().split()
    for token in tokens:
        lowered = token.lower()
        if len(lowered) == 64 and all(char in "0123456789abcdef" for char in lowered):
            result["expected_sha256"] = lowered
            break
    if len(tokens) >= 2:
        result["recorded_filename"] = tokens[-1].lstrip("*")
    return result


def max_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def iter_images(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def classify_image(image_root: Path, path: Path) -> tuple[ImageRecord | None, str | None]:
    relative = path.relative_to(image_root)
    parts = relative.parts
    if len(parts) != 4:
        return None, f"expected four path components, found {len(parts)}"
    split, target_name, environment_name, _filename = parts
    if split not in SPLITS:
        return None, f"unknown split {split!r}"
    if target_name not in TARGETS:
        return None, f"unknown target directory {target_name!r}"
    if environment_name not in ENVIRONMENTS:
        return None, f"unknown environment directory {environment_name!r}"
    try:
        mask_id = int(path.stem)
    except ValueError:
        return None, f"filename stem is not an integer: {path.stem!r}"
    return (
        ImageRecord(
            path=path,
            relative_path=relative.as_posix(),
            split=split,
            target_name=target_name,
            target_label=TARGETS[target_name],
            environment_name=environment_name,
            environment_label=ENVIRONMENTS[environment_name],
            mask_id=mask_id,
        ),
        None,
    )


def load_trusted_masks(path: Path) -> dict[int, np.ndarray]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected pickle root dict, found {type(value)!r}")
    return value


def bool_mask(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Expected NumPy array, found {type(value)!r}")
    if value.dtype != np.bool_:
        raise TypeError(f"Expected bool dtype, found {value.dtype}")
    if value.ndim != 2:
        raise ValueError(f"Expected 2-D mask, found shape {value.shape}")
    return value


def dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel(order="C"):
        value = (value << 1) | int(bit)
    return value


def canonical_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB\0".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def connected_component_stats(core_mask: np.ndarray, meaningful_fraction: float) -> dict[str, Any]:
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for all-mask fragmentation auditing in fcv_gh200"
        ) from exc

    labels, component_count = ndimage.label(
        core_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if component_count == 0:
        return {
            "component_count": 0,
            "meaningful_component_count": 0,
            "largest_component_core_fraction": 0.0,
        }
    sizes = np.bincount(labels.ravel())[1:]
    core_pixels = int(core_mask.sum())
    meaningful_minimum = max(8, int(math.ceil(core_mask.size * meaningful_fraction)))
    meaningful_count = int(np.count_nonzero(sizes >= meaningful_minimum))
    largest_fraction = float(sizes.max() / core_pixels) if core_pixels else 0.0
    return {
        "component_count": int(component_count),
        "meaningful_component_count": meaningful_count,
        "largest_component_core_fraction": largest_fraction,
    }


def json_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_numeric(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def cross_split_duplicate_rows(
    groups: dict[str, list[int]],
    rows: Sequence[dict[str, Any]],
    hash_kind: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for digest, indices in groups.items():
        splits = sorted({str(rows[index]["split"]) for index in indices})
        if len(splits) < 2:
            continue
        paths = [str(rows[index]["relative_path"]) for index in indices]
        output.append(
            {
                "hash_kind": hash_kind,
                "digest": digest,
                "image_count": len(indices),
                "splits": "|".join(splits),
                "relative_paths": "|".join(paths),
            }
        )
    return output


def candidate_quality_indices(
    rows: Sequence[dict[str, Any]],
    count: int,
) -> list[int]:
    if count <= 0 or not rows:
        return []
    eligible = [
        index
        for index, row in enumerate(rows)
        if row.get("image_decode_ok") and row.get("mask_decode_ok")
    ]
    if not eligible:
        return []
    candidates: list[int] = []
    by_low_core = sorted(
        eligible,
        key=lambda index: float(rows[index].get("core_area_fraction") or 0.0),
    )
    by_high_core = list(reversed(by_low_core))
    by_fragment = sorted(
        range(len(rows)),
        key=lambda index: (
            int(rows[index].get("meaningful_component_count") or 0),
            1.0 - float(rows[index].get("largest_component_core_fraction") or 0.0),
        ),
        reverse=True,
    )
    per_ranking = max(1, count // 4)
    candidates.extend(by_low_core[:per_ranking])
    candidates.extend(by_high_core[:per_ranking])
    candidates.extend(by_fragment[:per_ranking])

    # Deterministic coverage of every split x target x environment cell.
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index in eligible:
        row = rows[index]
        grouped[(str(row["split"]), str(row["target_name"]), str(row["environment_name"]))].append(index)
    for indices in grouped.values():
        candidates.append(indices[len(indices) // 2])

    unique: list[int] = []
    seen: set[int] = set()
    for index in candidates:
        if index not in seen:
            seen.add(index)
            unique.append(index)
        if len(unique) >= count:
            return unique
    for index in eligible:
        if index not in seen:
            unique.append(index)
        if len(unique) >= count:
            break
    return unique


def mask_overlay(image: Image.Image, core_mask: np.ndarray) -> Image.Image:
    image = image.convert("RGB")
    mask_image = Image.fromarray(core_mask.astype(np.uint8) * 255, mode="L")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_image, dtype=np.float32)[..., None] / 255.0
    base = np.asarray(image, dtype=np.float32)
    blue = np.array([0, 120, 255], dtype=np.float32)
    blended = base * (1.0 - 0.45 * mask) + blue * (0.45 * mask)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def make_quality_gallery(
    path: Path,
    image_root: Path,
    masks: dict[int, np.ndarray],
    rows: Sequence[dict[str, Any]],
    indices: Sequence[int],
) -> None:
    if not indices:
        return
    tile_width = 250
    image_height = 170
    label_height = 70
    columns = 2
    sheet = Image.new(
        "RGB",
        (columns * tile_width, len(indices) * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for output_row, index in enumerate(indices):
        row = rows[index]
        image_path = image_root / str(row["relative_path"])
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            core = ~bool_mask(masks[int(row["mask_id"])])
            overlay = mask_overlay(image, core)
        except Exception as exc:
            y = output_row * (image_height + label_height)
            draw.multiline_text(
                (4, y + 3),
                f"{row['relative_path']}\nGallery error: {type(exc).__name__}: {exc}",
                fill="red",
                spacing=2,
            )
            continue
        image = ImageOps.contain(image, (tile_width, image_height))
        overlay = ImageOps.contain(overlay, (tile_width, image_height))
        y = output_row * (image_height + label_height)
        sheet.paste(image, ((tile_width - image.width) // 2, y + label_height))
        sheet.paste(
            overlay,
            (tile_width + (tile_width - overlay.width) // 2, y + label_height),
        )
        relative = str(row["relative_path"])
        label = (
            f"{relative}\n"
            f"core={float(row['core_area_fraction']):.3f} "
            f"components={row['meaningful_component_count']} "
            f"border={row['core_touches_border']}"
        )
        draw.multiline_text((4, y + 3), label, fill="black", spacing=2)
        draw.text((4, y + 52), "Original", fill="black")
        draw.text((tile_width + 4, y + 52), "BLUE = core/dog", fill="black")
    sheet.save(path, quality=90, optimize=True)


def inspect_repository(repo_root: Path) -> dict[str, Any]:
    patterns = r"SpuCoDogs|spuco_dogs|spuco_animals_masks|spucodogs"
    command = [
        "rg",
        "-l",
        "-i",
        "--glob",
        "!.git/**",
        "--glob",
        "!download_logs/**",
        "--glob",
        "!logs*/**",
        patterns,
        str(repo_root),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        matches = [
            str(Path(line).resolve())
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
        rg_status: int | None = completed.returncode
        rg_error = completed.stderr.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        matches = []
        rg_status = None
        rg_error = f"{type(exc).__name__}: {exc}"

    named_candidates: list[str] = []
    name_tokens = ("spuco", "manifest", "metadata")
    pruned_names = {".git", "__pycache__", "download_logs"}
    stop = False
    for directory, directory_names, filenames in os.walk(repo_root):
        directory_names[:] = [
            name
            for name in directory_names
            if name not in pruned_names and not name.startswith("logs")
        ]
        for filename in filenames:
            if any(token in filename.lower() for token in name_tokens):
                named_candidates.append(str((Path(directory) / filename).resolve()))
                if len(named_candidates) >= 1_000:
                    stop = True
                    break
        if stop:
            break

    spec = importlib.util.find_spec("spuco")
    try:
        version = importlib.metadata.version("spuco")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "repository_root": str(repo_root),
        "content_match_count": len(matches),
        "content_matches": sorted(matches)[:1_000],
        "named_candidate_count": len(named_candidates),
        "named_candidates": sorted(named_candidates),
        "rg_returncode": rg_status,
        "rg_stderr": rg_error,
        "installed_spuco_module": spec is not None,
        "installed_spuco_origin": spec.origin if spec is not None else None,
        "installed_spuco_version": version,
        "reuse_rule": (
            "Prefer the official SpuCoDogs loader's hierarchy and label semantics. "
            "Reuse a local manifest only after verifying it reproduces the official "
            "12 split/target/environment counts and all 24,050 numeric image IDs."
        ),
    }


def main() -> None:
    args = parse_args()
    image_root = args.image_root.expanduser().resolve()
    mask_pickle = args.mask_pickle.expanduser().resolve()
    mask_checksum = args.mask_checksum.expanduser().resolve()
    archive_checksum = (
        args.archive_checksum.expanduser().resolve()
        if args.archive_checksum is not None
        else None
    )
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    for required_file in (mask_pickle, mask_checksum):
        if not required_file.is_file():
            raise FileNotFoundError(required_file)
    for required_directory in (image_root, repo_root):
        if not required_directory.is_dir():
            raise FileNotFoundError(required_directory)
    if not (0 <= args.perceptual_hamming_threshold <= 16):
        raise ValueError("--perceptual-hamming-threshold must be in [0, 16]")
    if args.max_perceptual_pairs <= 0:
        raise ValueError("--max-perceptual-pairs must be positive")
    if not (
        0.0 <= args.minimum_core_area_fraction
        < args.maximum_core_area_fraction
        <= 1.0
    ):
        raise ValueError("core-area warning thresholds must satisfy 0 <= min < max <= 1")
    if not (0.0 < args.meaningful_component_fraction < 1.0):
        raise ValueError("--meaningful-component-fraction must be in (0, 1)")

    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report: dict[str, Any] = {
        "audit_version": 2,
        "read_only": True,
        "inputs_modified": False,
        "image_root": str(image_root),
        "mask_pickle": str(mask_pickle),
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "official_sources": {
            "spucodogs_loader": OFFICIAL_DOGS_SOURCE,
            "mask_loader": OFFICIAL_MASK_SOURCE,
        },
        "official_semantics": {
            "target_labels": TARGETS,
            "environment_labels": ENVIRONMENTS,
            "hierarchy": (
                "spuco_dogs/{train,val,test}/{small_dogs,big_dogs}/"
                "{indoor,outdoor}/{integer_mask_id}.<image extension>"
            ),
            "mask_lookup": "masks[int(image filename stem)]",
            "mask_true": "spurious/context/background",
            "mask_false": "core/animal/dog",
            "official_pre_mask_geometry": (
                "The official loader multiplies the native decoded image by its "
                "native mask before Resize(256,256) and CenterCrop(224,224)."
            ),
        },
        "quality_thresholds_are_audit_warnings_not_filters": {
            "minimum_core_area_fraction": args.minimum_core_area_fraction,
            "maximum_core_area_fraction": args.maximum_core_area_fraction,
            "meaningful_component_fraction": args.meaningful_component_fraction,
        },
    }

    mask_receipt = parse_checksum_receipt(mask_checksum)
    observed_mask_sha = sha256_file(mask_pickle)
    mask_receipt["observed_sha256"] = observed_mask_sha
    mask_receipt["matches"] = (
        mask_receipt["expected_sha256"] == observed_mask_sha
        if mask_receipt["expected_sha256"] is not None
        else None
    )
    report["mask_integrity"] = mask_receipt
    if mask_receipt["matches"] is False:
        raise RuntimeError("Mask pickle SHA-256 does not match its receipt")

    archive_receipt = parse_checksum_receipt(archive_checksum)
    archive_target: Path | None = None
    recorded_archive = archive_receipt.get("recorded_filename")
    if archive_checksum is not None and recorded_archive:
        candidate = archive_checksum.parent / str(recorded_archive)
        if candidate.is_file():
            archive_target = candidate
    archive_receipt["live_artifact"] = str(archive_target) if archive_target else None
    if archive_target is not None:
        archive_observed = sha256_file(archive_target)
        archive_receipt["observed_sha256"] = archive_observed
        archive_receipt["matches"] = (
            archive_observed == archive_receipt["expected_sha256"]
            if archive_receipt["expected_sha256"] is not None
            else None
        )
    else:
        archive_receipt["observed_sha256"] = None
        archive_receipt["matches"] = None
        archive_receipt["status"] = (
            "receipt present but recorded archive is not live; verify extracted "
            "tree through the canonical image manifest digest"
        )
    report["archive_integrity"] = archive_receipt

    all_paths = list(iter_images(image_root))
    records: list[ImageRecord] = []
    hierarchy_errors: list[dict[str, str]] = []
    for path in all_paths:
        record, error = classify_image(image_root, path)
        if record is None:
            hierarchy_errors.append(
                {
                    "relative_path": path.relative_to(image_root).as_posix(),
                    "error": str(error),
                }
            )
        else:
            records.append(record)

    live_counts = Counter(
        (record.split, record.target_name, record.environment_name)
        for record in records
    )
    count_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for target_name in TARGETS:
            for environment_name in ENVIRONMENTS:
                key = (split, target_name, environment_name)
                observed = int(live_counts[key])
                expected = int(EXPECTED_COUNTS[key])
                count_rows.append(
                    {
                        "split": split,
                        "target_name": target_name,
                        "target_label": TARGETS[target_name],
                        "environment_name": environment_name,
                        "environment_label": ENVIRONMENTS[environment_name],
                        "expected_count": expected,
                        "observed_count": observed,
                        "matches_expected": observed == expected,
                    }
                )
    group_counts_csv = output_dir / "official_group_counts.csv"
    write_csv(group_counts_csv, list(count_rows[0].keys()), count_rows)
    report["hierarchy_and_counts"] = {
        "all_image_files": len(all_paths),
        "schema_valid_images": len(records),
        "hierarchy_error_count": len(hierarchy_errors),
        "hierarchy_errors": hierarchy_errors[:1_000],
        "expected_total": sum(EXPECTED_COUNTS.values()),
        "all_12_counts_match": all(row["matches_expected"] for row in count_rows),
        "counts_csv": str(group_counts_csv),
        "totals_by_split": json_counter(Counter(record.split for record in records)),
    }

    mask_ids = [record.mask_id for record in records]
    duplicate_image_mask_ids = {
        str(mask_id): count
        for mask_id, count in Counter(mask_ids).items()
        if count > 1
    }

    print(f"[INFO] Loading trusted mask pickle: {mask_pickle}", flush=True)
    masks = load_trusted_masks(mask_pickle)
    report["mask_schema"] = {
        "root_type": f"{type(masks).__module__}.{type(masks).__qualname__}",
        "root_length": len(masks),
        "key_type_counts": json_counter(Counter(type(key).__name__ for key in masks)),
        "duplicate_keys_observable_after_unpickle": False,
        "duplicate_key_note": (
            "A Python dict has unique final keys; overwritten duplicate entries "
            "cannot be recovered from the materialized pickle. Dog-image numeric "
            "ID reuse is checked separately."
        ),
        "duplicate_dog_image_numeric_ids": duplicate_image_mask_ids,
        "dog_mask_ids_unique": not duplicate_image_mask_ids,
        "dog_only_filter": "retain masks[int(image filename stem)] for the 24,050 image paths",
    }

    inventory_rows: list[dict[str, Any]] = []
    file_hash_groups: dict[str, list[int]] = defaultdict(list)
    pixel_hash_groups: dict[str, list[int]] = defaultdict(list)
    dhashes: list[int] = []
    exif_identity_counts: Counter[str] = Counter()
    exif_orientation_counts: Counter[str] = Counter()
    image_format_counts: Counter[str] = Counter()
    image_mode_counts: Counter[str] = Counter()
    mask_shape_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    missing_mask_count = 0
    mask_decode_failure_count = 0
    image_decode_failure_count = 0
    empty_context_count = 0
    empty_core_count = 0
    full_context_count = 0
    full_core_count = 0
    core_border_count = 0
    fragmented_count = 0
    implausible_area_count = 0
    packed_mask_bytes = 0
    uncompressed_bool_bytes = 0
    image_tree_bytes = 0
    manifest_digest = hashlib.sha256()

    for position, record in enumerate(records):
        row: dict[str, Any] = {
            "relative_path": record.relative_path,
            "split": record.split,
            "target_name": record.target_name,
            "target_label": record.target_label,
            "environment_name": record.environment_name,
            "environment_label": record.environment_label,
            "group": f"{record.target_label}_{record.environment_label}",
            "mask_id": record.mask_id,
        }
        file_size = record.path.stat().st_size
        image_tree_bytes += file_size
        file_sha = sha256_file(record.path, chunk_size=1024 * 1024)
        manifest_digest.update(record.relative_path.encode("utf-8"))
        manifest_digest.update(b"\0")
        manifest_digest.update(str(file_size).encode("ascii"))
        manifest_digest.update(b"\0")
        manifest_digest.update(file_sha.encode("ascii"))
        manifest_digest.update(b"\n")
        row["file_size_bytes"] = file_size
        row["file_sha256"] = file_sha

        try:
            with Image.open(record.path) as opened:
                opened.load()
                raw = opened.convert("RGB")
                raw_size = raw.size
                image_format = opened.format or "UNKNOWN"
                image_mode = opened.mode
                exif = opened.getexif()
                orientation = int(exif.get(EXIF_ORIENTATION_TAG, 1) or 1)
                oriented = ImageOps.exif_transpose(opened).convert("RGB")
                oriented_size = oriented.size
                for tag_id, value in exif.items():
                    name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if name in EXIF_IDENTITY_TAG_NAMES and str(value).strip():
                        exif_identity_counts[name] += 1
            pixel_sha = canonical_pixel_sha256(oriented)
            perceptual_hash = dhash64(oriented)
            row.update(
                {
                    "image_decode_ok": True,
                    "image_format": image_format,
                    "image_mode": image_mode,
                    "raw_width": raw_size[0],
                    "raw_height": raw_size[1],
                    "exif_orientation": orientation,
                    "oriented_width": oriented_size[0],
                    "oriented_height": oriented_size[1],
                    "pixel_sha256_after_exif": pixel_sha,
                    "dhash64_hex_after_exif": f"{perceptual_hash:016x}",
                }
            )
            image_format_counts[image_format] += 1
            image_mode_counts[image_mode] += 1
            exif_orientation_counts[str(orientation)] += 1
        except Exception as exc:
            image_decode_failure_count += 1
            row.update(
                {
                    "image_decode_ok": False,
                    "image_decode_error": f"{type(exc).__name__}: {exc}",
                }
            )
            inventory_rows.append(row)
            continue

        value = masks.get(record.mask_id)
        if value is None:
            missing_mask_count += 1
            row.update({"mask_present": False, "geometry_status": "missing_mask"})
            inventory_rows.append(row)
            file_hash_groups[file_sha].append(len(inventory_rows) - 1)
            pixel_hash_groups[pixel_sha].append(len(inventory_rows) - 1)
            dhashes.append(perceptual_hash)
            continue
        row["mask_present"] = True
        try:
            context = bool_mask(value)
        except Exception as exc:
            mask_decode_failure_count += 1
            row.update(
                {
                    "mask_decode_ok": False,
                    "mask_decode_error": f"{type(exc).__name__}: {exc}",
                    "geometry_status": "mask_decode_failure",
                }
            )
            inventory_rows.append(row)
            file_hash_groups[file_sha].append(len(inventory_rows) - 1)
            pixel_hash_groups[pixel_sha].append(len(inventory_rows) - 1)
            dhashes.append(perceptual_hash)
            continue

        mask_height, mask_width = context.shape
        raw_shape = (raw_size[1], raw_size[0])
        oriented_shape = (oriented_size[1], oriented_size[0])
        if context.shape == raw_shape:
            geometry_status = "matches_raw_decoded_image"
        elif context.shape == oriented_shape:
            geometry_status = "matches_exif_transposed_image_only"
        elif context.shape == raw_shape[::-1]:
            geometry_status = "matches_raw_dimensions_swapped"
        else:
            geometry_status = "geometry_mismatch"
        geometry_counts[geometry_status] += 1
        mask_shape_counts[str(tuple(context.shape))] += 1
        packed_mask_bytes += int((context.size + 7) // 8)
        uncompressed_bool_bytes += int(context.nbytes)

        core = ~context
        context_pixels = int(context.sum())
        core_pixels = int(core.sum())
        context_fraction = float(context_pixels / context.size)
        core_fraction = float(core_pixels / core.size)
        context_empty = context_pixels == 0
        core_empty = core_pixels == 0
        context_full = context_pixels == context.size
        core_full = core_pixels == core.size
        core_touches_border = bool(
            core[0, :].any()
            or core[-1, :].any()
            or core[:, 0].any()
            or core[:, -1].any()
        )
        component_stats = connected_component_stats(
            core,
            meaningful_fraction=args.meaningful_component_fraction,
        )
        fragmented = component_stats["meaningful_component_count"] > 1
        implausible_area = (
            core_fraction < args.minimum_core_area_fraction
            or core_fraction > args.maximum_core_area_fraction
        )
        empty_context_count += int(context_empty)
        empty_core_count += int(core_empty)
        full_context_count += int(context_full)
        full_core_count += int(core_full)
        core_border_count += int(core_touches_border)
        fragmented_count += int(fragmented)
        implausible_area_count += int(implausible_area)

        row.update(
            {
                "mask_decode_ok": True,
                "mask_dtype": str(context.dtype),
                "mask_height": mask_height,
                "mask_width": mask_width,
                "geometry_status": geometry_status,
                "context_area_fraction": context_fraction,
                "core_area_fraction": core_fraction,
                "context_empty": context_empty,
                "core_empty": core_empty,
                "context_full": context_full,
                "core_full": core_full,
                "core_touches_border": core_touches_border,
                "implausible_core_area_warning": implausible_area,
                "fragmented_core_warning": fragmented,
                **component_stats,
            }
        )
        inventory_rows.append(row)
        row_index = len(inventory_rows) - 1
        file_hash_groups[file_sha].append(row_index)
        pixel_hash_groups[pixel_sha].append(row_index)
        dhashes.append(perceptual_hash)
        if (position + 1) % 1_000 == 0:
            print(
                f"[PROGRESS] audited {position + 1}/{len(records)} images",
                flush=True,
            )

    inventory_csv = output_dir / "image_mask_inventory.csv"
    inventory_fieldnames = sorted(
        {key for row in inventory_rows for key in row.keys()}
    )
    write_csv(inventory_csv, inventory_fieldnames, inventory_rows)

    exact_rows = cross_split_duplicate_rows(
        file_hash_groups,
        inventory_rows,
        "exact_file_sha256",
    )
    exact_rows.extend(
        cross_split_duplicate_rows(
            pixel_hash_groups,
            inventory_rows,
            "decoded_pixel_sha256_after_exif",
        )
    )
    exact_duplicates_csv = output_dir / "cross_split_exact_duplicates.csv"
    exact_fields = (
        ["hash_kind", "digest", "image_count", "splits", "relative_paths"]
        if exact_rows
        else ["hash_kind", "digest", "image_count", "splits", "relative_paths"]
    )
    write_csv(exact_duplicates_csv, exact_fields, exact_rows)

    perceptual_rows: list[dict[str, Any]] = []
    perceptual_truncated = False
    tree = BKTree()
    # dhashes has one entry per inventory row only when image decoding succeeded.
    hash_by_row = {
        index: int(str(row["dhash64_hex_after_exif"]), 16)
        for index, row in enumerate(inventory_rows)
        if row.get("image_decode_ok")
    }
    for index, value in hash_by_row.items():
        for distance, prior_index in tree.query(
            value,
            args.perceptual_hamming_threshold,
        ):
            prior = inventory_rows[prior_index]
            current = inventory_rows[index]
            if prior["split"] == current["split"]:
                continue
            perceptual_rows.append(
                {
                    "hamming_distance": distance,
                    "left_split": prior["split"],
                    "left_relative_path": prior["relative_path"],
                    "right_split": current["split"],
                    "right_relative_path": current["relative_path"],
                    "left_dhash64": prior["dhash64_hex_after_exif"],
                    "right_dhash64": current["dhash64_hex_after_exif"],
                    "interpretation": (
                        "screening candidate only; visually verify before "
                        "calling this a duplicate"
                    ),
                }
            )
            if len(perceptual_rows) >= args.max_perceptual_pairs:
                perceptual_truncated = True
                break
        if perceptual_truncated:
            break
        tree.add(value, index)
    perceptual_csv = output_dir / "cross_split_perceptual_duplicate_candidates.csv"
    perceptual_fields = [
        "hamming_distance",
        "left_split",
        "left_relative_path",
        "right_split",
        "right_relative_path",
        "left_dhash64",
        "right_dhash64",
        "interpretation",
    ]
    write_csv(perceptual_csv, perceptual_fields, perceptual_rows)

    quality_indices = candidate_quality_indices(
        inventory_rows,
        args.quality_preview_count,
    )
    quality_gallery = output_dir / "mask_quality_review.jpg"
    make_quality_gallery(
        quality_gallery,
        image_root,
        masks,
        inventory_rows,
        quality_indices,
    )

    core_fractions = [
        float(row["core_area_fraction"])
        for row in inventory_rows
        if row.get("mask_decode_ok")
    ]
    component_counts = [
        float(row["meaningful_component_count"])
        for row in inventory_rows
        if row.get("mask_decode_ok")
    ]
    report["all_sample_audit"] = {
        "inventory_csv": str(inventory_csv),
        "image_decode_failures": image_decode_failure_count,
        "missing_masks": missing_mask_count,
        "mask_decode_failures": mask_decode_failure_count,
        "mask_shape_count": len(mask_shape_counts),
        "mask_shape_counts": json_counter(mask_shape_counts),
        "geometry_counts": json_counter(geometry_counts),
        "all_masks_match_raw_decoded_image_geometry": (
            geometry_counts["matches_raw_decoded_image"] == len(records)
            and image_decode_failure_count == 0
            and missing_mask_count == 0
            and mask_decode_failure_count == 0
        ),
        "exif_orientation_counts": json_counter(exif_orientation_counts),
        "image_format_counts": json_counter(image_format_counts),
        "image_mode_counts": json_counter(image_mode_counts),
    }
    report["mask_quality_screen"] = {
        "quality_gallery": str(quality_gallery),
        "quality_gallery_row_indices": quality_indices,
        "empty_context_masks": empty_context_count,
        "empty_core_masks": empty_core_count,
        "full_context_masks": full_context_count,
        "full_core_masks": full_core_count,
        "core_touches_border_count": core_border_count,
        "fragmented_core_warning_count": fragmented_count,
        "implausible_core_area_warning_count": implausible_area_count,
        "core_area_fraction_summary": summarize_numeric(core_fractions),
        "meaningful_component_count_summary": summarize_numeric(component_counts),
        "visual_omission_limit": (
            "Whether a mask visibly omits part of a dog cannot be established "
            "from the mask alone because no independent dog segmentation or box "
            "is present. Review the stratified/extreme-case gallery and retain "
            "that review as a human audit, not an automated ground-truth claim."
        ),
    }
    report["split_leakage_screen"] = {
        "canonical_image_manifest_sha256": manifest_digest.hexdigest(),
        "cross_split_exact_duplicate_group_count": len(exact_rows),
        "cross_split_exact_duplicates_csv": str(exact_duplicates_csv),
        "perceptual_hash": (
            "64-bit dHash after EXIF transpose; this is a high-recall review "
            "screen and not proof of duplicate identity"
        ),
        "perceptual_hamming_threshold": args.perceptual_hamming_threshold,
        "cross_split_perceptual_candidate_count": len(perceptual_rows),
        "perceptual_candidates_truncated": perceptual_truncated,
        "cross_split_perceptual_candidates_csv": str(perceptual_csv),
    }
    report["identity_and_source_sequence"] = {
        "exif_identity_field_counts": json_counter(exif_identity_counts),
        "numeric_filename_semantics": (
            "The integer stem is the global mask lookup ID. It is not documented "
            "by the official loader as dog identity, breed, capture sequence, or "
            "collection identity."
        ),
        "leakage_resistant_identity_split_available": False,
        "conclusion": (
            "The official tree exposes split, dog-size target, indoor/outdoor "
            "environment, and numeric mask ID. Unless the audit finds a separate "
            "metadata/source manifest, these artifacts do not provide a defensible "
            "dog-identity or source-sequence key. Building an identity-grouped "
            "split would require authenticated upstream ImageNet/source metadata, "
            "not filename adjacency or perceptual similarity guesses."
        ),
    }

    repository_inventory = inspect_repository(repo_root)
    repo_candidates_path = output_dir / "repository_reuse_candidates.json"
    repo_candidates_path.write_text(
        json.dumps(repository_inventory, indent=2, sort_keys=True) + "\n"
    )
    report["repository_reuse_inventory"] = {
        **repository_inventory,
        "saved_json": str(repo_candidates_path),
    }

    disk = shutil.disk_usage(image_root)
    reserve_bytes = int(args.storage_reserve_gib * 1024**3)
    relative_path_bytes = sum(
        len(record.relative_path.encode("utf-8")) + 1 for record in records
    )
    compact_payload_estimate = (
        packed_mask_bytes
        + relative_path_bytes
        + len(records) * 64
    )
    validation_products_estimate = 2 * 1024**3
    two_copy_staging_estimate = (
        2 * compact_payload_estimate + validation_products_estimate
    )
    report["storage_projection"] = {
        "filesystem_total_bytes": disk.total,
        "filesystem_used_bytes": disk.used,
        "filesystem_free_bytes": disk.free,
        "image_tree_bytes": image_tree_bytes,
        "mask_pickle_bytes": mask_pickle.stat().st_size,
        "dog_masks_uncompressed_bool_bytes": uncompressed_bool_bytes,
        "dog_masks_bitpacked_payload_bytes": packed_mask_bytes,
        "relative_path_bytes": relative_path_bytes,
        "compact_artifact_payload_estimate_bytes": compact_payload_estimate,
        "temporary_validation_products_estimate_bytes": validation_products_estimate,
        "two_copy_staging_increment_estimate_bytes": two_copy_staging_estimate,
        "reserved_free_space_bytes": reserve_bytes,
        "safe_for_compaction_and_temporary_validation": (
            disk.free - reserve_bytes >= two_copy_staging_estimate
        ),
        "projection_note": (
            "This is a conservative planning estimate, not an archive-size "
            "guarantee. It budgets two compact copies plus 2 GiB of reports/tests "
            "while retaining the source pickle and the requested free-space reserve."
        ),
    }

    report["acceptance_summary"] = {
        "mask_receipt_matches": mask_receipt["matches"],
        "archive_receipt_reverified": archive_receipt.get("matches"),
        "all_official_counts_match": report["hierarchy_and_counts"]["all_12_counts_match"],
        "all_images_schema_valid": len(records) == len(all_paths) and not hierarchy_errors,
        "all_dog_numeric_ids_unique": not duplicate_image_mask_ids,
        "all_images_have_decodable_masks": (
            missing_mask_count == 0 and mask_decode_failure_count == 0
        ),
        "all_images_decode": image_decode_failure_count == 0,
        "all_masks_match_raw_geometry": report["all_sample_audit"][
            "all_masks_match_raw_decoded_image_geometry"
        ],
        "no_cross_split_exact_duplicates": len(exact_rows) == 0,
        "perceptual_duplicate_review_required": len(perceptual_rows) > 0,
        "mask_quality_gallery_review_required": True,
        "safe_storage_projection": report["storage_projection"][
            "safe_for_compaction_and_temporary_validation"
        ],
    }
    report["max_rss_gib"] = max_rss_gib()
    report["elapsed_seconds"] = time.time() - started

    report_path = output_dir / "spucodogs_deep_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[DONE] report={report_path}")
    print(f"[DONE] inventory={inventory_csv}")
    print(f"[DONE] group_counts={group_counts_csv}")
    print(f"[DONE] quality_gallery={quality_gallery}")
    print(
        "[RESULT] "
        f"images={len(records)} count_match="
        f"{report['hierarchy_and_counts']['all_12_counts_match']} "
        f"raw_geometry_match="
        f"{report['all_sample_audit']['all_masks_match_raw_decoded_image_geometry']}"
    )
    print(
        "[RESULT] "
        f"cross_split_exact_groups={len(exact_rows)} "
        f"perceptual_candidates={len(perceptual_rows)} "
        f"storage_safe={report['storage_projection']['safe_for_compaction_and_temporary_validation']}"
    )
    print(
        f"[RESULT] max_rss_gib={report['max_rss_gib']:.3f} "
        f"seconds={report['elapsed_seconds']:.1f}"
    )
    print(f"[REVIEW] Open {quality_gallery}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
