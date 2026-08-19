#!/usr/bin/env python3
"""Manifest utilities for ImageNet-9 class-conditional map corruption."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


CLASS_NAMES = (
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
CLASS_COUNT = 5045
TRAIN_COUNT = CLASS_COUNT * len(CLASS_NAMES)
RANDOM_CONDITION = "random_matched"
CONDITIONS = tuple(f"class_{name}" for name in CLASS_NAMES) + (RANDOM_CONDITION,)
PROTOCOL_VERSION = 1


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def read_training_rows(manifest_path: Path) -> List[Dict[str, object]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        source = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    rows = [
        {
            "split_index": index,
            "sample_id": row["sample_id"],
            "label": int(row["label"]),
            "class_name": row["class_name"].lower(),
            "source_path": row["source_path"],
        }
        for index, row in enumerate(source)
    ]
    if len(rows) != TRAIN_COUNT:
        raise RuntimeError(f"Expected {TRAIN_COUNT} training rows, found {len(rows)}")
    counts = Counter(str(row["class_name"]) for row in rows)
    expected = {name: CLASS_COUNT for name in CLASS_NAMES}
    if dict(counts) != expected:
        raise RuntimeError(f"Unexpected ImageNet-9 training class counts: {dict(counts)}")
    if [str(row["class_name"]) for row in rows] and any(
        int(row["label"]) != CLASS_NAMES.index(str(row["class_name"])) for row in rows
    ):
        raise RuntimeError("ImageNet-9 training labels do not match the canonical class order")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("ImageNet-9 training manifest contains duplicate sample IDs")
    return rows


def dataset_fingerprint(rows: Sequence[Mapping[str, object]]) -> str:
    payload = [
        {
            "split_index": int(row["split_index"]),
            "sample_id": str(row["sample_id"]),
            "label": int(row["label"]),
            "class_name": str(row["class_name"]),
            "source_path": str(row["source_path"]),
        }
        for row in rows
    ]
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def select_indices(
    condition: str,
    labels: Sequence[int],
    corruption_seed: int,
) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.shape != (TRAIN_COUNT,):
        raise RuntimeError(f"Expected {TRAIN_COUNT} labels, got {label_array.shape}")
    if condition.startswith("class_"):
        target_name = condition[len("class_") :]
        target_label = CLASS_NAMES.index(target_name)
        selected = np.flatnonzero(label_array == target_label)
    else:
        rng = np.random.default_rng(int(corruption_seed))
        selected = rng.choice(TRAIN_COUNT, size=CLASS_COUNT, replace=False)
    selected = np.asarray(sorted(int(value) for value in selected), dtype=np.int64)
    if selected.shape != (CLASS_COUNT,) or np.unique(selected).size != CLASS_COUNT:
        raise RuntimeError(
            f"Condition {condition} must select {CLASS_COUNT} unique rows; got {selected.shape}"
        )
    return selected


def build_manifest(
    condition: str,
    rows: Sequence[Mapping[str, object]],
    corruption_seed: int,
) -> Tuple[Dict[str, object], np.ndarray, List[Dict[str, object]]]:
    selected = select_indices(
        condition,
        [int(row["label"]) for row in rows],
        corruption_seed,
    )
    selected_rows = [dict(rows[int(index)]) for index in selected]
    counts = Counter(str(row["class_name"]) for row in selected_rows)
    condition_type = "class_conditional" if condition.startswith("class_") else "random"
    target_class = condition[len("class_") :] if condition.startswith("class_") else ""
    if target_class and counts != Counter({target_class: CLASS_COUNT}):
        raise RuntimeError(f"Systematic condition is not class-pure: {dict(counts)}")
    manifest: Dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "imagenet9",
        "condition": condition,
        "condition_type": condition_type,
        "target_class": target_class,
        "class_names": list(CLASS_NAMES),
        "training_example_count": TRAIN_COUNT,
        "training_class_counts": {name: CLASS_COUNT for name in CLASS_NAMES},
        "corruption_seed": int(corruption_seed),
        "corrupted_example_count": CLASS_COUNT,
        "corrupted_fraction_of_training": CLASS_COUNT / TRAIN_COUNT,
        "corrupted_class_counts": {name: int(counts.get(name, 0)) for name in CLASS_NAMES},
        "corruption_operation": "one_minus_then_sum_renormalize",
        "selection_is_fixed_across_training_seeds": True,
        "validation_corrupted": False,
        "official_test_corrupted": False,
        "dataset_fingerprint": dataset_fingerprint(rows),
        "selected_indices_sha256": sha256_bytes(selected.tobytes()),
        "selected_sample_ids_sha256": sha256_bytes(
            "\n".join(str(row["sample_id"]) for row in selected_rows).encode("utf-8")
        ),
    }
    return manifest, selected, selected_rows


def prepare_manifest(
    condition: str,
    manifest_path: Path,
    output_dir: Path,
    corruption_seed: int,
) -> Tuple[Dict[str, object], Path, str]:
    rows = read_training_rows(manifest_path)
    expected, selected, selected_rows = build_manifest(condition, rows, corruption_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "manifest.json"
    indices_path = output_dir / "indices.npy"
    rows_path = output_dir / "selected_samples.csv"
    if json_path.is_file() or indices_path.is_file() or rows_path.is_file():
        if not (json_path.is_file() and indices_path.is_file() and rows_path.is_file()):
            raise RuntimeError(f"Incomplete persisted corruption manifest: {output_dir}")
        stored = json.loads(json_path.read_text(encoding="utf-8"))
        stored_indices = np.asarray(
            np.load(indices_path, allow_pickle=False), dtype=np.int64
        ).reshape(-1)
        with rows_path.open(newline="", encoding="utf-8") as handle:
            stored_rows = list(csv.DictReader(handle))
        stored_ids = [row["sample_id"] for row in stored_rows]
        expected_ids = [str(row["sample_id"]) for row in selected_rows]
        if stored != expected or not np.array_equal(stored_indices, selected):
            raise RuntimeError(f"Persisted corruption selection changed: {output_dir}")
        if stored_ids != expected_ids:
            raise RuntimeError(f"Persisted selected sample rows changed: {rows_path}")
    else:
        atomic_npy(indices_path, selected)
        atomic_csv(rows_path, selected_rows)
        atomic_json(json_path, expected)
    return expected, indices_path, sha256_file(json_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--corruption-seed", type=int, default=0)
    args = parser.parse_args()
    unknown = sorted(set(args.conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    for condition in args.conditions:
        manifest, indices_path, digest = prepare_manifest(
            condition,
            args.manifest,
            args.output_root / condition,
            args.corruption_seed,
        )
        print(
            f"[MANIFEST] condition={condition} "
            f"count={manifest['corrupted_example_count']} "
            f"indices={indices_path} sha256={digest}",
            flush=True,
        )
    print(f"[DONE] prepared {len(args.conditions)} corruption manifests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
