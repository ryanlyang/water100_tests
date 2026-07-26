#!/usr/bin/env python3
"""Read-only structural and alignment audit for the SpuCoAnimals mask pickle.

This script intentionally does not rewrite, compact, or delete the source
artifact. Pickle loading can execute code, so the input must be the trusted
author-provided file whose checksum was recorded at download time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IDENTIFIER_KEYS = {
    "filename",
    "file_name",
    "filepath",
    "file_path",
    "path",
    "image",
    "image_path",
    "img",
    "img_path",
    "name",
    "id",
    "sample_id",
}
MASK_KEYS = {
    "mask",
    "masks",
    "segmentation",
    "segmentation_mask",
    "foreground",
    "foreground_mask",
    "core",
    "core_mask",
}


@dataclass
class MaskRecord:
    record_index: int
    identifier: str | None
    object_path: tuple[str, ...]
    mask: Any
    mask_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the trusted SpuCoAnimals mask pickle against the downloaded "
            "SpuCoDogs image tree without modifying either input."
        )
    )
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checksum-file", type=Path)
    parser.add_argument(
        "--compute-sha256",
        action="store_true",
        help="Hash the full pickle and compare it with --checksum-file.",
    )
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--structure-depth", type=int, default=4)
    parser.add_argument("--structure-items", type=int, default=8)
    return parser.parse_args()


def max_rss_gib() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sha256(checksum_file: Path | None) -> str | None:
    if checksum_file is None or not checksum_file.is_file():
        return None
    for token in checksum_file.read_text(errors="replace").split():
        candidate = token.strip().lower()
        if len(candidate) == 64 and all(ch in "0123456789abcdef" for ch in candidate):
            return candidate
    return None


def type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def array_metadata(value: Any) -> tuple[tuple[int, ...], str] | None:
    if isinstance(value, np.ndarray):
        return tuple(int(v) for v in value.shape), str(value.dtype)
    if isinstance(value, Image.Image):
        return (int(value.height), int(value.width)), f"PIL:{value.mode}"

    # Avoid requiring torch merely to inspect numpy/PIL artifacts, while still
    # supporting tensor-valued trusted pickles in the FCV environment.
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return tuple(int(v) for v in value.shape), str(value.dtype)
    except ImportError:
        pass
    return None


def is_rle_mapping(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and "counts" in value
        and "size" in value
        and isinstance(value.get("size"), Sequence)
    )


def mask_kind(value: Any) -> str | None:
    if array_metadata(value) is not None:
        return "array"
    if is_rle_mapping(value):
        return "coco_rle"
    return None


def scalar_identifier(value: Any) -> str | None:
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return None


def local_identifier(mapping: Mapping[Any, Any]) -> str | None:
    lowered = {str(key).lower(): key for key in mapping.keys()}
    for name in sorted(IDENTIFIER_KEYS):
        original = lowered.get(name)
        if original is None:
            continue
        candidate = scalar_identifier(mapping[original])
        if candidate:
            return candidate
    return None


def discover_mask_records(root: Any) -> list[MaskRecord]:
    records: list[MaskRecord] = []
    visited_containers: set[int] = set()

    def add(identifier: str | None, path: tuple[str, ...], value: Any, kind: str) -> None:
        records.append(
            MaskRecord(
                record_index=len(records),
                identifier=identifier,
                object_path=path,
                mask=value,
                mask_kind=kind,
            )
        )

    def walk(value: Any, path: tuple[str, ...], inherited_id: str | None) -> None:
        kind = mask_kind(value)
        if kind is not None:
            add(inherited_id, path, value, kind)
            return

        if isinstance(value, Mapping):
            object_id = id(value)
            if object_id in visited_containers:
                return
            visited_containers.add(object_id)

            here_id = local_identifier(value) or inherited_id
            handled: set[Any] = set()

            # Common record form: {"filename": ..., "mask": ...}
            for key, child in value.items():
                child_kind = mask_kind(child)
                if str(key).lower() in MASK_KEYS and child_kind is not None:
                    add(here_id, path + (str(key),), child, child_kind)
                    handled.add(key)

            for key, child in value.items():
                if key in handled:
                    continue
                child_kind = mask_kind(child)
                key_id = scalar_identifier(key)
                if child_kind is not None:
                    # Common mapping form: {"relative/image.jpg": ndarray}
                    add(key_id or here_id, path + (str(key),), child, child_kind)
                else:
                    walk(child, path + (str(key),), here_id)
            return

        if isinstance(value, (list, tuple)):
            object_id = id(value)
            if object_id in visited_containers:
                return
            visited_containers.add(object_id)

            # Common tuple form: ("relative/image.jpg", mask)
            sibling_id = inherited_id
            if len(value) <= 8:
                for item in value:
                    candidate = scalar_identifier(item)
                    if candidate and not candidate.isdigit():
                        sibling_id = candidate
                        break
            for index, child in enumerate(value):
                walk(child, path + (str(index),), sibling_id)

    walk(root, tuple(), None)
    return records


def summarize_structure(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    seen: set[int] | None = None,
) -> dict[str, Any]:
    if seen is None:
        seen = set()

    summary: dict[str, Any] = {"type": type_name(value)}
    metadata = array_metadata(value)
    if metadata is not None:
        shape, dtype = metadata
        summary.update({"shape": list(shape), "dtype": dtype})
        return summary
    if is_rle_mapping(value):
        size = value.get("size")
        summary.update({"kind": "coco_rle", "size": list(size)})
        return summary
    if depth >= max_depth:
        try:
            summary["length"] = len(value)
        except (TypeError, AttributeError):
            pass
        return summary

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in seen:
            summary["cycle"] = True
            return summary
        seen.add(object_id)
        summary["length"] = len(value)
        items = list(value.items())[:max_items]
        summary["sample_items"] = [
            {
                "key": repr(key)[:500],
                "value": summarize_structure(
                    child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    seen=seen,
                ),
            }
            for key, child in items
        ]
        return summary

    if isinstance(value, (list, tuple)):
        object_id = id(value)
        if object_id in seen:
            summary["cycle"] = True
            return summary
        seen.add(object_id)
        summary["length"] = len(value)
        summary["sample_items"] = [
            summarize_structure(
                child,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                seen=seen,
            )
            for child in list(value[:max_items])
        ]
        return summary

    scalar = repr(value)
    summary["repr"] = scalar[:1000]
    return summary


def normalized_path_text(value: str) -> str:
    text = value.replace("\\", "/").strip().lower()
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def path_aliases(value: str) -> set[str]:
    text = normalized_path_text(value)
    if not text:
        return set()
    path = PurePosixPath(text)
    parts = [part for part in path.parts if part not in {"", "."}]
    aliases: set[str] = {text}

    max_suffix_parts = min(6, len(parts))
    for count in range(1, max_suffix_parts + 1):
        suffix = "/".join(parts[-count:])
        aliases.add(suffix)
        aliases.add(str(PurePosixPath(suffix).with_suffix("")))

    aliases.add(path.name)
    aliases.add(path.stem)
    return {alias for alias in aliases if alias}


def record_aliases(record: MaskRecord) -> set[str]:
    values: list[str] = []
    if record.identifier:
        values.append(record.identifier)
    # The official SpuCoAnimals loader derives mask_index from the integer image
    # filename and then accesses self.masks[mask_index]. For a top-level mask
    # list, the numeric object-path token is therefore the canonical join key.
    values.extend(token for token in record.object_path if token)
    if record.object_path:
        values.append("/".join(record.object_path))

    aliases: set[str] = set()
    for value in values:
        aliases.update(path_aliases(value))
    return aliases


def enumerate_images(image_root: Path) -> list[Path]:
    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_image_alias_index(
    image_root: Path, images: Sequence[Path]
) -> tuple[dict[str, set[int]], list[str]]:
    alias_index: dict[str, set[int]] = defaultdict(set)
    relative_paths: list[str] = []
    for index, image_path in enumerate(images):
        relative = image_path.relative_to(image_root).as_posix()
        relative_paths.append(relative)
        for alias in path_aliases(relative):
            alias_index[alias].add(index)
    return alias_index, relative_paths


def align_records(
    records: Sequence[MaskRecord],
    image_alias_index: Mapping[str, set[int]],
) -> tuple[list[set[int]], list[set[int]]]:
    record_to_images: list[set[int]] = []
    image_to_records: dict[int, set[int]] = defaultdict(set)

    for record in records:
        matched_images: set[int] = set()
        for alias in record_aliases(record):
            matched_images.update(image_alias_index.get(alias, set()))
        record_to_images.append(matched_images)
        for image_index in matched_images:
            image_to_records[image_index].add(record.record_index)

    max_image_index = max(image_to_records.keys(), default=-1)
    image_count = max(
        max_image_index + 1,
        max(
            (max(indices) + 1 for indices in image_alias_index.values() if indices),
            default=0,
        ),
    )
    image_to_record_list = [image_to_records.get(index, set()) for index in range(image_count)]
    return record_to_images, image_to_record_list


def mask_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, Image.Image):
        array = np.asarray(value)
    else:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().numpy()
            else:
                raise TypeError(f"Unsupported mask type: {type_name(value)}")
        except ImportError as exc:
            raise TypeError(f"Unsupported mask type: {type_name(value)}") from exc

    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 3:
        # Prefer a singleton/channel dimension; otherwise use the first plane
        # while preserving the full original shape in the audit.
        if array.shape[0] <= 4 and array.shape[1] > 4 and array.shape[2] > 4:
            array = array[0]
        elif array.shape[-1] <= 4 and array.shape[0] > 4 and array.shape[1] > 4:
            array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask after squeezing, found {array.shape}")
    return array


def detailed_mask_stats(value: Any) -> dict[str, Any]:
    metadata = array_metadata(value)
    stats: dict[str, Any] = {"type": type_name(value)}
    if metadata is None:
        if is_rle_mapping(value):
            stats.update({"kind": "coco_rle", "size": list(value["size"])})
        return stats

    shape, dtype = metadata
    stats.update({"shape": list(shape), "dtype": dtype})
    try:
        array = mask_to_numpy(value)
    except (TypeError, ValueError) as exc:
        stats["decode_error"] = str(exc)
        return stats

    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else array
    if finite.size == 0:
        stats["finite_values"] = 0
        return stats

    unique = np.unique(finite)
    stats.update(
        {
            "decoded_shape": list(array.shape),
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "unique_count": int(unique.size),
            "unique_preview": [
                float(item) if isinstance(item, (float, np.floating, int, np.integer)) else repr(item)
                for item in unique[:20]
            ],
        }
    )
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    threshold = minimum if math.isclose(minimum, maximum) else (minimum + maximum) / 2.0
    high = np.asarray(array > threshold)
    stats["midpoint_threshold"] = threshold
    stats["high_value_fraction"] = float(np.mean(high))
    return stats


def binary_high_mask(value: Any) -> np.ndarray:
    array = mask_to_numpy(value).astype(np.float32, copy=False)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Mask has no finite values")
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if math.isclose(minimum, maximum):
        return np.zeros(array.shape, dtype=bool)
    threshold = (minimum + maximum) / 2.0
    return np.asarray(array > threshold)


def overlay(image: Image.Image, binary: np.ndarray, *, invert: bool) -> Image.Image:
    image = image.convert("RGB")
    if invert:
        binary = ~binary
        color = np.array([0, 120, 255], dtype=np.float32)
    else:
        color = np.array([255, 30, 30], dtype=np.float32)

    mask_image = Image.fromarray((binary.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.resize(image.size, resample=Image.Resampling.NEAREST)
    mask = np.asarray(mask_image, dtype=np.float32)[..., None] / 255.0
    base = np.asarray(image, dtype=np.float32)
    blended = base * (1.0 - 0.45 * mask) + color * (0.45 * mask)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def make_previews(
    *,
    output_dir: Path,
    image_root: Path,
    relative_paths: Sequence[str],
    records: Sequence[MaskRecord],
    image_to_records: Sequence[set[int]],
    preview_count: int,
) -> tuple[list[dict[str, Any]], str | None]:
    uniquely_matched = [
        index for index, record_indices in enumerate(image_to_records) if len(record_indices) == 1
    ]
    if not uniquely_matched or preview_count <= 0:
        return [], None

    selected_count = min(preview_count, len(uniquely_matched))
    if selected_count == 1:
        selected = [uniquely_matched[0]]
    else:
        selected = [
            uniquely_matched[
                round(position * (len(uniquely_matched) - 1) / (selected_count - 1))
            ]
            for position in range(selected_count)
        ]

    tile_width = 224
    label_height = 42
    tile_height = 224 + label_height
    columns = 3
    sheet = Image.new("RGB", (columns * tile_width, selected_count * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    preview_rows: list[dict[str, Any]] = []

    for row, image_index in enumerate(selected):
        record_index = next(iter(image_to_records[image_index]))
        record = records[record_index]
        relative_path = relative_paths[image_index]
        image_path = image_root / relative_path

        preview_info: dict[str, Any] = {
            "image_relative_path": relative_path,
            "record_index": record_index,
            "record_identifier": record.identifier,
            "record_object_path": "/".join(record.object_path),
        }
        try:
            with Image.open(image_path) as opened:
                image = ImageOps.fit(opened.convert("RGB"), (224, 224))
            binary = binary_high_mask(record.mask)
            high_overlay = overlay(image, binary, invert=False)
            low_overlay = overlay(image, binary, invert=True)
            preview_info["high_value_fraction"] = float(np.mean(binary))
            preview_info["mask_stats"] = detailed_mask_stats(record.mask)
        except Exception as exc:  # Preserve diagnostics for unfamiliar formats.
            preview_info["preview_error"] = f"{type(exc).__name__}: {exc}"
            preview_rows.append(preview_info)
            continue

        y = row * tile_height
        sheet.paste(image, (0, y + label_height))
        sheet.paste(high_overlay, (tile_width, y + label_height))
        sheet.paste(low_overlay, (2 * tile_width, y + label_height))
        label = relative_path
        if len(label) > 80:
            label = "..." + label[-77:]
        draw.text((4, y + 2), label, fill="black")
        draw.text((4, y + 20), "Original", fill="black")
        draw.text((tile_width + 4, y + 20), "RED = high mask values", fill="black")
        draw.text((2 * tile_width + 4, y + 20), "BLUE = low mask values", fill="black")
        preview_rows.append(preview_info)

    preview_path = output_dir / "mask_polarity_overlays.jpg"
    sheet.save(preview_path, quality=92, optimize=True)
    return preview_rows, str(preview_path)


def write_alignment_csv(
    path: Path,
    relative_paths: Sequence[str],
    image_to_records: Sequence[set[int]],
    records: Sequence[MaskRecord],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_relative_path",
                "match_status",
                "matched_record_count",
                "matched_record_indices",
                "matched_record_identifiers",
                "matched_record_object_paths",
            ],
        )
        writer.writeheader()
        for image_index, relative_path in enumerate(relative_paths):
            record_indices = (
                sorted(image_to_records[image_index])
                if image_index < len(image_to_records)
                else []
            )
            if len(record_indices) == 1:
                status = "unique"
            elif len(record_indices) == 0:
                status = "unmatched"
            else:
                status = "ambiguous"
            writer.writerow(
                {
                    "image_relative_path": relative_path,
                    "match_status": status,
                    "matched_record_count": len(record_indices),
                    "matched_record_indices": ";".join(map(str, record_indices)),
                    "matched_record_identifiers": ";".join(
                        records[index].identifier or "" for index in record_indices
                    ),
                    "matched_record_object_paths": ";".join(
                        "/".join(records[index].object_path) for index in record_indices
                    ),
                }
            )


def json_safe_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda x: str(x[0]))}


def main() -> None:
    args = parse_args()
    mask_pickle = args.mask_pickle.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checksum_file = args.checksum_file.expanduser().resolve() if args.checksum_file else None

    if not mask_pickle.is_file():
        raise FileNotFoundError(f"Missing mask pickle: {mask_pickle}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing SpuCoDogs image root: {image_root}")
    if args.preview_count < 0:
        raise ValueError("--preview-count must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()

    report: dict[str, Any] = {
        "audit_version": 1,
        "read_only": True,
        "source_artifact_must_not_be_deleted": True,
        "mask_pickle": str(mask_pickle),
        "mask_pickle_size_bytes": mask_pickle.stat().st_size,
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "trusted_pickle_warning": (
            "Pickle loading can execute code. This audit assumes the input is "
            "the trusted author-provided artifact downloaded for this project."
        ),
        "provenance_limit": (
            "Artifact inspection can verify structure, alignment, and visual "
            "semantics, but cannot establish whether masks were human-annotated "
            "or model-generated. That requires source documentation."
        ),
        "official_loader_semantics": {
            "join_rule": (
                "mask_index = int(image filename stem); mask = masks[mask_index]"
            ),
            "high_true_mask_values": "spurious/background region",
            "low_false_mask_values": "core/animal region",
            "source": (
                "https://github.com/BigML-CS-UCLA/SpuCo/blob/master/"
                "src/spuco/datasets/spuco_animals.py"
            ),
        },
    }

    expected = expected_sha256(checksum_file)
    report["checksum_file"] = str(checksum_file) if checksum_file else None
    report["expected_sha256"] = expected
    if args.compute_sha256:
        hash_started = time.time()
        observed = sha256_file(mask_pickle)
        report["observed_sha256"] = observed
        report["sha256_seconds"] = time.time() - hash_started
        report["sha256_matches_expected"] = expected is None or observed == expected
        if expected is not None and observed != expected:
            raise RuntimeError(
                f"Checksum mismatch for {mask_pickle}: expected {expected}, observed {observed}"
            )

    print(f"[INFO] Loading trusted pickle: {mask_pickle}", flush=True)
    load_started = time.time()
    with mask_pickle.open("rb") as handle:
        root = pickle.load(handle)
    report["pickle_load_seconds"] = time.time() - load_started
    report["max_rss_gib_after_load"] = max_rss_gib()
    report["root_type"] = type_name(root)
    try:
        report["root_length"] = len(root)
    except (TypeError, AttributeError):
        report["root_length"] = None

    report["structure"] = summarize_structure(
        root,
        depth=0,
        max_depth=args.structure_depth,
        max_items=args.structure_items,
    )
    records = discover_mask_records(root)
    report["mask_record_count"] = len(records)
    report["mask_kind_counts"] = json_safe_counter(
        Counter(record.mask_kind for record in records)
    )
    report["mask_record_type_counts"] = json_safe_counter(
        Counter(type_name(record.mask) for record in records)
    )

    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    for record in records:
        metadata = array_metadata(record.mask)
        if metadata is None:
            continue
        shape, dtype = metadata
        shape_counts[str(tuple(shape))] += 1
        dtype_counts[dtype] += 1
    report["mask_shape_counts"] = json_safe_counter(shape_counts)
    report["mask_dtype_counts"] = json_safe_counter(dtype_counts)

    images = enumerate_images(image_root)
    image_alias_index, relative_paths = build_image_alias_index(image_root, images)
    report["spucodogs_image_count"] = len(images)
    report["image_counts_by_top_directory"] = json_safe_counter(
        Counter(PurePosixPath(path).parts[0] for path in relative_paths if path)
    )
    report["image_basename_collision_count"] = sum(
        1
        for alias, indices in image_alias_index.items()
        if "/" not in alias and "." in alias and len(indices) > 1
    )

    record_to_images, image_to_records = align_records(records, image_alias_index)
    # align_records infers image_count from aliases; pad defensively to the exact
    # source inventory length.
    if len(image_to_records) < len(images):
        image_to_records.extend(set() for _ in range(len(images) - len(image_to_records)))
    elif len(image_to_records) > len(images):
        image_to_records = image_to_records[: len(images)]

    unique_images = sum(len(indices) == 1 for indices in image_to_records)
    unmatched_images = sum(len(indices) == 0 for indices in image_to_records)
    ambiguous_images = sum(len(indices) > 1 for indices in image_to_records)
    uniquely_matched_records = sum(len(indices) == 1 for indices in record_to_images)
    unmatched_records = sum(len(indices) == 0 for indices in record_to_images)
    ambiguous_records = sum(len(indices) > 1 for indices in record_to_images)

    report["alignment"] = {
        "unique_image_matches": unique_images,
        "unmatched_images": unmatched_images,
        "ambiguous_images": ambiguous_images,
        "uniquely_matched_records": uniquely_matched_records,
        "unmatched_records": unmatched_records,
        "ambiguous_records": ambiguous_records,
        "all_spucodogs_images_uniquely_matched": (
            len(images) > 0 and unique_images == len(images)
        ),
        "possible_positional_alignment_only": (
            unique_images == 0 and len(records) == len(images) and len(images) > 0
        ),
    }

    alignment_csv = output_dir / "image_mask_alignment.csv"
    write_alignment_csv(alignment_csv, relative_paths, image_to_records, records)
    report["alignment_csv"] = str(alignment_csv)

    preview_rows, preview_path = make_previews(
        output_dir=output_dir,
        image_root=image_root,
        relative_paths=relative_paths,
        records=records,
        image_to_records=image_to_records,
        preview_count=args.preview_count,
    )
    report["preview_path"] = preview_path
    report["preview_records"] = preview_rows

    # Include a compact sample even if filename alignment failed.
    report["record_sample"] = [
        {
            "record_index": record.record_index,
            "identifier": record.identifier,
            "object_path": list(record.object_path),
            "mask_kind": record.mask_kind,
            "mask_stats": detailed_mask_stats(record.mask),
            "matched_image_indices": sorted(record_to_images[record.record_index])[:20],
        }
        for record in records[: min(20, len(records))]
    ]

    report["max_rss_gib_final"] = max_rss_gib()
    report["total_seconds"] = time.time() - started
    report["next_decision"] = {
        "safe_to_compact_by_filename": (
            len(images) > 0
            and unique_images == len(images)
            and ambiguous_images == 0
        ),
        "expected_foreground_polarity": (
            "The official loader treats low/false values as the core animal. "
            "Confirm this visually in the blue overlays before compaction."
        ),
        "foreground_polarity_requires_overlay_review": preview_path is not None,
        "safe_to_delete_original_pickle": False,
        "required_before_deletion": [
            "Create a dog-only compact artifact.",
            "Verify exact sample coverage and one-to-one alignment.",
            "Verify mask polarity and geometry from overlays.",
            "Hash the compact artifact and test its loader.",
            "Retain provenance and checksum receipts.",
        ],
    }

    report_path = output_dir / "mask_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"[DONE] report={report_path}")
    print(f"[DONE] alignment={alignment_csv}")
    print(f"[RESULT] root_type={report['root_type']} root_length={report['root_length']}")
    print(
        "[RESULT] "
        f"records={len(records)} images={len(images)} "
        f"unique={unique_images} unmatched={unmatched_images} ambiguous={ambiguous_images}"
    )
    print(
        "[RESULT] "
        f"max_rss_gib={report['max_rss_gib_final']:.3f} "
        f"seconds={report['total_seconds']:.1f}"
    )
    if preview_path:
        print(f"[REVIEW] Open both mask-polarity assumptions: {preview_path}")
    if not report["alignment"]["all_spucodogs_images_uniquely_matched"]:
        print(
            "[REVIEW] Filename alignment is not yet complete. Inspect structure, "
            "record_sample, and image_mask_alignment.csv before writing an extractor."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
