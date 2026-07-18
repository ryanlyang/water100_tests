"""Build immutable, leakage-separated DecoyMNIST full-campaign manifests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from decoy_full_config import canonical_config_sha256
from decoy_manifest_provenance import (
    MANIFEST_SPECS,
    PUBLIC_COLUMNS,
    atomic_json,
    image_set_sha256,
    sha256_values,
    write_manifest_bundle,
)
from decoy_data import (
    NUM_CLASSES,
    discover_samples,
    locate_decoy_patch,
)


class ManifestError(ValueError):
    """Raised when source data or a proposed split violates the protocol."""


def largest_remainder_counts(
    class_sizes: Mapping[int, int], target_count: int
) -> Dict[int, int]:
    """Hamilton apportionment with ascending-label deterministic tie breaks."""

    labels = sorted(int(label) for label in class_sizes)
    if labels != list(range(NUM_CLASSES)):
        raise ManifestError(f"Expected labels 0..9, found {labels}.")
    sizes = {label: int(class_sizes[label]) for label in labels}
    if any(size <= 0 for size in sizes.values()):
        raise ManifestError("Every class must be nonempty.")
    total = sum(sizes.values())
    if not 0 < int(target_count) < total:
        raise ManifestError("target_count must be positive and below the source total.")
    exact = {label: sizes[label] * int(target_count) / total for label in labels}
    allocated = {label: int(math.floor(exact[label])) for label in labels}
    remainder = int(target_count) - sum(allocated.values())
    order = sorted(
        labels,
        key=lambda label: (-(exact[label] - allocated[label]), label),
    )
    for label in order[:remainder]:
        allocated[label] += 1
    if sum(allocated.values()) != int(target_count):
        raise AssertionError("Largest-remainder allocation failed to reach target.")
    return allocated


def stratified_three_way_split(
    by_label: Mapping[int, Sequence[Path]],
    *,
    candidate_train_count: int,
    biased_validation_count: int,
    oracle_validation_count: int,
    split_seed: int,
) -> Dict[str, List[Tuple[Path, int]]]:
    """Partition all source training paths into exact, deterministic split sizes."""

    class_sizes = {label: len(by_label[label]) for label in range(NUM_CLASSES)}
    total = sum(class_sizes.values())
    if candidate_train_count + biased_validation_count + oracle_validation_count != total:
        raise ManifestError("Requested three-way split does not consume source train.")
    biased_counts = largest_remainder_counts(class_sizes, biased_validation_count)
    after_biased = {
        label: class_sizes[label] - biased_counts[label] for label in range(NUM_CLASSES)
    }
    oracle_counts = largest_remainder_counts(after_biased, oracle_validation_count)
    train_counts = {
        label: class_sizes[label] - biased_counts[label] - oracle_counts[label]
        for label in range(NUM_CLASSES)
    }

    rng = np.random.default_rng(int(split_seed))
    result: Dict[str, List[Tuple[Path, int]]] = {
        "candidate_train": [],
        "biased_validation": [],
        "oracle_validation": [],
    }
    for label in range(NUM_CLASSES):
        paths = sorted(Path(path).resolve() for path in by_label[label])
        order = rng.permutation(len(paths)).tolist()
        shuffled = [paths[int(index)] for index in order]
        n_biased = biased_counts[label]
        n_oracle = oracle_counts[label]
        biased = shuffled[:n_biased]
        oracle = shuffled[n_biased : n_biased + n_oracle]
        train = shuffled[n_biased + n_oracle :]
        if len(train) != train_counts[label]:
            raise AssertionError("Per-class split allocation is inconsistent.")
        result["candidate_train"].extend((path, label) for path in train)
        result["biased_validation"].extend((path, label) for path in biased)
        result["oracle_validation"].extend((path, label) for path in oracle)

    for samples in result.values():
        samples.sort(key=lambda item: (item[1], item[0].as_posix()))
    expected = {
        "candidate_train": int(candidate_train_count),
        "biased_validation": int(biased_validation_count),
        "oracle_validation": int(oracle_validation_count),
    }
    for role, count in expected.items():
        if len(result[role]) != count:
            raise AssertionError(f"{role} has {len(result[role])}, expected {count}.")
    return result


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_")


def sample_id(path: Path, label: int, source_split: str) -> str:
    return f"{source_split}_{int(label)}_{_safe_stem(path)}"


def _audit_one(
    task: Tuple[Path, int, str, Path, float]
) -> Tuple[Path, Dict[str, Any]]:
    path, label, source_split, data_root, tolerance = task
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode != "L":
            raise ManifestError(
                f"Expected a grayscale mode-L PNG, found mode={image.mode!r} at {path}."
            )
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    if grayscale.shape != (28, 28):
        raise ManifestError(f"Expected 28x28 PNG, found {grayscale.shape} at {path}.")
    locate_decoy_patch(grayscale, int(label), source_split, tolerance=tolerance)
    relative = path.resolve().relative_to(data_root.resolve()).as_posix()
    return path.resolve(), {
        "sample_id": sample_id(path, label, source_split),
        "image_rel_path": relative,
        "label": int(label),
        "source_split": source_split,
        "image_sha256": digest,
    }


def _audit_samples(
    samples: Sequence[Tuple[Path, int]],
    source_split: str,
    data_root: Path,
    tolerance: float,
    workers: int,
) -> Dict[Path, Dict[str, Any]]:
    tasks = [
        (Path(path).resolve(), int(label), source_split, data_root, tolerance)
        for path, label in samples
    ]
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    if workers == 0:
        audited = map(_audit_one, tasks)
        return dict(audited)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(_audit_one, tasks))


def _frame_for_role(
    samples: Sequence[Tuple[Path, int]],
    records: Mapping[Path, Mapping[str, Any]],
    role: str,
) -> pd.DataFrame:
    study_split = MANIFEST_SPECS[role]["study_split"]
    rows = []
    for path, _label in samples:
        row = dict(records[Path(path).resolve()])
        row["study_split"] = study_split
        rows.append(row)
    frame = pd.DataFrame(rows, columns=PUBLIC_COLUMNS)
    return frame.sort_values(["label", "image_rel_path"]).reset_index(drop=True)


def _class_counts(frame: pd.DataFrame) -> Dict[str, int]:
    counts = frame["label"].astype(int).value_counts().sort_index()
    return {str(label): int(counts.get(label, 0)) for label in range(NUM_CLASSES)}


def _source_inventory_sha256(frames: Iterable[pd.DataFrame]) -> str:
    combined = pd.concat(list(frames), ignore_index=True)
    records = combined[
        ["sample_id", "image_rel_path", "label", "image_sha256"]
    ].sort_values("sample_id")
    encoded = json.dumps(
        records.to_dict("records"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def _ensure_writable(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to replace frozen manifest artifacts without --overwrite: "
            + ", ".join(existing)
        )


def prepare_decoymnist_manifests(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    workers: int = 12,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Audit every source PNG and write the exact Step-2 manifest bundle."""

    data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing DecoyMNIST data root: {data_root}")
    source_counts = config["data"]["source_counts"]
    partition = config["data"]["partition"]
    tolerance = float(config["data"]["shortcut_encoding"]["tolerance"])

    train_by_label = discover_samples(data_root, "train")
    test_by_label = discover_samples(data_root, "test")
    train_count = sum(len(paths) for paths in train_by_label.values())
    test_count = sum(len(paths) for paths in test_by_label.values())
    if train_count != int(source_counts["train"]):
        raise ManifestError(
            f"Expected {source_counts['train']} train PNGs, found {train_count}."
        )
    if test_count != int(source_counts["test"]):
        raise ManifestError(
            f"Expected {source_counts['test']} test PNGs, found {test_count}."
        )

    split = stratified_three_way_split(
        train_by_label,
        candidate_train_count=int(partition["candidate_train_count"]),
        biased_validation_count=int(partition["biased_validation_count"]),
        oracle_validation_count=int(partition["oracle_validation_source_count"]),
        split_seed=int(partition["split_seed"]),
    )
    all_train = sorted(
        (
            (Path(path).resolve(), label)
            for label in range(NUM_CLASSES)
            for path in train_by_label[label]
        ),
        key=lambda item: (item[1], item[0].as_posix()),
    )
    all_test = sorted(
        (
            (Path(path).resolve(), label)
            for label in range(NUM_CLASSES)
            for path in test_by_label[label]
        ),
        key=lambda item: (item[1], item[0].as_posix()),
    )
    train_records = _audit_samples(
        all_train, "train", data_root, tolerance, int(workers)
    )
    test_records = _audit_samples(
        all_test, "test", data_root, tolerance, int(workers)
    )

    frames = {
        "candidate_train": _frame_for_role(
            split["candidate_train"], train_records, "candidate_train"
        ),
        "biased_validation": _frame_for_role(
            split["biased_validation"], train_records, "biased_validation"
        ),
        "oracle_validation": _frame_for_role(
            split["oracle_validation"], train_records, "oracle_validation"
        ),
        "test": _frame_for_role(all_test, test_records, "test"),
    }
    for role, frame in frames.items():
        if frame["sample_id"].astype(str).duplicated().any():
            raise ManifestError(f"Stable sample IDs collide inside {role}.")

    train_sets = {
        role: set(frames[role]["sample_id"].astype(str))
        for role in ("candidate_train", "biased_validation", "oracle_validation")
    }
    if any(
        train_sets[left].intersection(train_sets[right])
        for index, left in enumerate(train_sets)
        for right in list(train_sets)[index + 1 :]
    ):
        raise ManifestError("Train-derived study splits overlap.")
    if len(set().union(*train_sets.values())) != train_count:
        raise ManifestError("Train-derived study splits do not partition train.")
    if set().union(*train_sets.values()).intersection(
        set(frames["test"]["sample_id"].astype(str))
    ):
        raise ManifestError("Official test overlaps train-derived study data.")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        role: destination / spec["filename"] for role, spec in MANIFEST_SPECS.items()
    }
    artifact_paths.update(
        {
            "assignments": destination / "split_assignments.json",
            "summary": destination / "split_summary.json",
            "bundle": destination / "manifest_bundle.json",
        }
    )
    _ensure_writable(list(artifact_paths.values()), overwrite)

    source_inventory = {
        "data_root": str(data_root),
        "train_count": train_count,
        "test_count": test_count,
        "train_sha256": _source_inventory_sha256(
            [
                frames["candidate_train"],
                frames["biased_validation"],
                frames["oracle_validation"],
            ]
        ),
        "test_sha256": _source_inventory_sha256([frames["test"]]),
    }
    source_inventory["all_sha256"] = hashlib.sha256(
        (source_inventory["train_sha256"] + source_inventory["test_sha256"]).encode(
            "ascii"
        )
    ).hexdigest()

    assignments = {
        "artifact_type": "fcv_vit_decoymnist_split_assignments",
        "artifact_version": 1,
        "algorithm": config["reproducibility"]["split_algorithm_version"],
        "split_seed": int(partition["split_seed"]),
        "config_sha256": canonical_config_sha256(config),
        "source_inventory": source_inventory,
        "splits": {
            role: {
                "count": int(len(frame)),
                "class_counts": _class_counts(frame),
                "sample_ids": sorted(frame["sample_id"].astype(str).tolist()),
                "sample_ids_sha256": sha256_values(
                    sorted(frame["sample_id"].astype(str).tolist())
                ),
            }
            for role, frame in frames.items()
        },
    }
    summary = {
        "artifact_type": "fcv_vit_decoymnist_split_summary",
        "artifact_version": 1,
        "protocol": {
            "candidate_training": "48k biased source-train images only",
            "vanilla_visibility": "biased validation only",
            "fcv_visibility": "biased validation and FCV views only",
            "oracle_visibility": "disjoint Oracle source; reversed in memory only",
            "test_visibility": "official reversed test; analysis only",
            "source_images_mutated": False,
        },
        "config_sha256": canonical_config_sha256(config),
        "source_inventory": source_inventory,
        "splits": {
            role: {
                "count": int(len(frame)),
                "class_counts": _class_counts(frame),
                "sample_ids_sha256": sha256_values(
                    sorted(frame["sample_id"].astype(str).tolist())
                ),
                "image_set_sha256": image_set_sha256(frame),
                "source_split": MANIFEST_SPECS[role]["source_split"],
                "study_split": MANIFEST_SPECS[role]["study_split"],
            }
            for role, frame in frames.items()
        },
        "manifest_columns": PUBLIC_COLUMNS,
    }

    for role, frame in frames.items():
        _atomic_csv(frame, artifact_paths[role])
    atomic_json(assignments, artifact_paths["assignments"])
    atomic_json(summary, artifact_paths["summary"])
    write_manifest_bundle(
        config,
        destination,
        artifact_paths,
        source_inventory=source_inventory,
    )

    return {
        "output_dir": str(destination),
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "summary": summary,
    }
