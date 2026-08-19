#!/usr/bin/env python3
"""Run one Waterbirds-95 R4RR systematic teacher corruption condition.

Only manifest-selected training teacher maps are changed. The canonical
ResNet-50 R4RR model, training schedule, validation selector, and test protocol
are imported from the existing Waterbirds implementation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms


SCRIPT_PATH = Path(__file__).resolve()
R4RR_ROOT = SCRIPT_PATH.parents[1]
TRAIN_DIR = R4RR_ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import r4rr_waterbirds as waterbirds_core  # noqa: E402


PROTOCOL_VERSION = 1
GROUP_NAMES = (
    "Land_on_Land",
    "Land_on_Water",
    "Water_on_Land",
    "Water_on_Water",
)
GROUP_KEYS = tuple(name.lower() for name in GROUP_NAMES)
GROUP_TO_INDEX = {key: index for index, key in enumerate(GROUP_KEYS)}
SYSTEMATIC_CONDITIONS = tuple(f"group_{key}" for key in GROUP_KEYS)
RANDOM_CONDITIONS = tuple(f"random_matched_{key}" for key in GROUP_KEYS)
CONDITIONS = tuple(
    condition
    for key in GROUP_KEYS
    for condition in (f"group_{key}", f"random_matched_{key}")
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
FIXED_EPOCHS = 200
FIXED_BATCH_SIZE = 96
FIXED_MOMENTUM = 0.9
FIXED_WEIGHT_DECAY = 1e-5
FIXED_KL_INCREMENT = 0.0
FIXED_ALIGNMENT_LOSS = "forward_kl"
EXPECTED_SPLIT_COUNTS = {"train": 4_795, "val": 1_199, "test": 5_794}
EXPECTED_GROUP_COUNTS = {
    "train": (3_498, 184, 56, 1_057),
    "val": (467, 466, 133, 133),
    "test": (2_255, 2_255, 642, 642),
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(str(temporary), str(path))


def parse_seeds(raw: str) -> Tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError("At least one training seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Training seeds must be unique, got {seeds}")
    return seeds


def invert_and_renormalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Apply the original R4RR inversion stress test exactly."""
    inverted = torch.clamp(1.0 - mask, min=0.0)
    total = float(inverted.sum().item())
    if total <= 1e-12:
        return torch.full_like(inverted, 1.0 / float(inverted.numel()))
    return inverted / total


class ManifestCorruptedWaterbirdsDataset(waterbirds_core.WaterbirdsMetadataDataset):
    """Canonical Waterbirds dataset with lazy inversion for manifest indices."""

    def __init__(self, *args, corrupted_indices: Iterable[int], **kwargs):
        super().__init__(*args, **kwargs)
        if not self.return_mask:
            raise ValueError("Manifest corruption requires return_mask=True")
        self.corrupted_indices = frozenset(int(index) for index in corrupted_indices)

    def __getitem__(self, index: int):
        values = list(super().__getitem__(index))
        if index in self.corrupted_indices:
            values[2] = invert_and_renormalize_mask(values[2])
        return tuple(values)


def _groups(dataset: waterbirds_core.WaterbirdsMetadataDataset) -> np.ndarray:
    return np.asarray(dataset.labels, dtype=np.int64) * 2 + np.asarray(
        dataset.places, dtype=np.int64
    )


def _target_group_key(condition: str) -> str:
    if condition.startswith("group_"):
        key = condition[len("group_") :]
    elif condition.startswith("random_matched_"):
        key = condition[len("random_matched_") :]
    else:
        raise ValueError(f"Unsupported condition prefix: {condition}")
    if key not in GROUP_TO_INDEX:
        raise ValueError(f"Unknown target group key {key!r}")
    return key


def select_corruption_indices(
    condition: str,
    groups: Sequence[int],
    corruption_seed: int,
    matched_group_counts: Sequence[int],
) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
    target_key = _target_group_key(condition)
    target_index = GROUP_TO_INDEX[target_key]
    groups_array = np.asarray(groups, dtype=np.int64)
    expected_count = int(matched_group_counts[target_index])

    if condition.startswith("group_"):
        selected = np.flatnonzero(groups_array == target_index)
    else:
        rng = np.random.default_rng(int(corruption_seed))
        selected = rng.choice(len(groups_array), size=expected_count, replace=False)

    selected_array = np.asarray(sorted(int(index) for index in selected), dtype=np.int64)
    if selected_array.size != expected_count:
        raise RuntimeError(
            f"Condition {condition} must select {expected_count} examples; got {selected_array.size}"
        )
    if np.unique(selected_array).size != selected_array.size:
        raise RuntimeError(f"Condition {condition} produced duplicate indices")
    return selected_array


def _stable_sample_id(path_raw: str, data_root: Path) -> str:
    path = Path(path_raw).expanduser().resolve()
    try:
        return path.relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _dataset_rows(
    dataset: waterbirds_core.WaterbirdsMetadataDataset, split: str, data_root: Path
) -> List[Dict[str, object]]:
    groups = _groups(dataset)
    return [
        {
            "split_index": int(index),
            "split": split,
            "sample_id": _stable_sample_id(path, data_root),
            "label": int(dataset.labels[index]),
            "place": int(dataset.places[index]),
            "group": int(groups[index]),
            "group_name": GROUP_NAMES[int(groups[index])],
        }
        for index, path in enumerate(dataset.paths)
    ]


def _dataset_fingerprint(
    datasets: Mapping[str, waterbirds_core.WaterbirdsMetadataDataset], data_root: Path
) -> str:
    payload: List[Dict[str, object]] = []
    for split in ("train", "val", "test"):
        payload.extend(_dataset_rows(datasets[split], split, data_root))
    return _sha256_bytes(_canonical_json_bytes(payload))


def _group_count_dict(groups: Sequence[int]) -> Dict[str, int]:
    values = np.bincount(np.asarray(groups, dtype=np.int64), minlength=len(GROUP_NAMES))
    return {name: int(values[index]) for index, name in enumerate(GROUP_NAMES)}


def validate_dataset_contract(
    datasets: Mapping[str, waterbirds_core.WaterbirdsMetadataDataset]
) -> Dict[str, Dict[str, int]]:
    observed: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        dataset = datasets[split]
        if len(dataset) != EXPECTED_SPLIT_COUNTS[split]:
            raise RuntimeError(
                f"Expected {EXPECTED_SPLIT_COUNTS[split]} {split} examples, got {len(dataset)}"
            )
        labels = np.asarray(dataset.labels, dtype=np.int64)
        places = np.asarray(dataset.places, dtype=np.int64)
        if not set(np.unique(labels)).issubset({0, 1}):
            raise RuntimeError(f"Unexpected labels in {split}: {np.unique(labels).tolist()}")
        if not set(np.unique(places)).issubset({0, 1}):
            raise RuntimeError(f"Unexpected places in {split}: {np.unique(places).tolist()}")
        observed[split] = _group_count_dict(_groups(dataset))
        expected = {
            name: int(EXPECTED_GROUP_COUNTS[split][index])
            for index, name in enumerate(GROUP_NAMES)
        }
        if observed[split] != expected:
            raise RuntimeError(
                f"Unexpected {split} group counts: {observed[split]}; expected {expected}"
            )
    return observed


def build_manifest(
    condition: str,
    datasets: Mapping[str, waterbirds_core.WaterbirdsMetadataDataset],
    data_root: Path,
    corruption_seed: int,
) -> Tuple[Dict[str, object], np.ndarray, List[str]]:
    split_group_counts = validate_dataset_contract(datasets)
    train_dataset = datasets["train"]
    train_groups = _groups(train_dataset)
    selected = select_corruption_indices(
        condition=condition,
        groups=train_groups,
        corruption_seed=corruption_seed,
        matched_group_counts=EXPECTED_GROUP_COUNTS["train"],
    )
    target_key = _target_group_key(condition)
    target_index = GROUP_TO_INDEX[target_key]
    target_name = GROUP_NAMES[target_index]
    condition_type = (
        "systematic_group" if condition.startswith("group_") else "matched_random_control"
    )
    expected_count = EXPECTED_GROUP_COUNTS["train"][target_index]
    if int(selected.size) != int(expected_count):
        raise RuntimeError(f"Count matching failed for {condition}")

    rows = _dataset_rows(train_dataset, "train", data_root)
    selected_ids = [str(rows[index]["sample_id"]) for index in selected.tolist()]
    selected_group_counts = _group_count_dict(train_groups[selected])
    if condition_type == "systematic_group":
        expected_selected = {
            name: (int(expected_count) if index == target_index else 0)
            for index, name in enumerate(GROUP_NAMES)
        }
        if selected_group_counts != expected_selected:
            raise RuntimeError(
                f"Systematic selection is not group-pure: {selected_group_counts}"
            )

    manifest: Dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "waterbirds95",
        "condition": condition,
        "condition_type": condition_type,
        "target_group_index": int(target_index),
        "target_group_key": target_key,
        "target_group_name": target_name,
        "matched_group_count": int(expected_count),
        "corruption_operation": "one_minus_then_sum_renormalize",
        "corruption_seed": int(corruption_seed),
        "split_example_counts": {split: int(len(datasets[split])) for split in datasets},
        "split_group_counts": split_group_counts,
        "training_example_count": int(len(train_dataset)),
        "corrupted_example_count": int(selected.size),
        "corrupted_fraction_of_training": float(selected.size / len(train_dataset)),
        "group_order": list(GROUP_NAMES),
        "corrupted_group_counts": selected_group_counts,
        "dataset_split_sha256": _dataset_fingerprint(datasets, data_root),
        "selected_sample_ids_sha256": _sha256_bytes(
            ("\n".join(selected_ids) + "\n").encode("utf-8")
        ),
    }
    return manifest, selected, selected_ids


def persist_or_validate_manifest(
    manifest_dir: Path,
    expected_manifest: Mapping[str, object],
    expected_indices: np.ndarray,
    expected_sample_ids: Sequence[str],
) -> Tuple[Dict[str, object], np.ndarray]:
    manifest_path = manifest_dir / "manifest.json"
    indices_path = manifest_dir / "selected_indices.npy"
    sample_ids_path = manifest_dir / "selected_sample_ids.txt"
    expected = dict(expected_manifest)

    if manifest_path.exists() or indices_path.exists() or sample_ids_path.exists():
        if not (manifest_path.is_file() and indices_path.is_file() and sample_ids_path.is_file()):
            raise RuntimeError(f"Incomplete persisted corruption manifest under {manifest_dir}")
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_indices = np.load(indices_path, allow_pickle=False)
        observed_sample_ids = sample_ids_path.read_text(encoding="utf-8").splitlines()
        observed_core = {
            key: value
            for key, value in observed.items()
            if key not in {"indices_file", "indices_sha256", "sample_ids_file", "manifest_sha256"}
        }
        if observed_core != expected:
            raise RuntimeError(f"Persisted manifest contract differs: {manifest_path}")
        if not np.array_equal(observed_indices, expected_indices):
            raise RuntimeError(f"Persisted corruption indices differ: {indices_path}")
        if observed_sample_ids != list(expected_sample_ids):
            raise RuntimeError(f"Persisted sample identities differ: {sample_ids_path}")
        if observed.get("indices_sha256") != _sha256_file(indices_path):
            raise RuntimeError(f"Persisted index checksum is invalid: {indices_path}")
        manifest_sha = str(observed.get("manifest_sha256", ""))
        checksum_payload = dict(observed)
        checksum_payload.pop("manifest_sha256", None)
        if manifest_sha != _sha256_bytes(_canonical_json_bytes(checksum_payload)):
            raise RuntimeError(f"Persisted manifest checksum is invalid: {manifest_path}")
        return observed, observed_indices.astype(np.int64, copy=False)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(indices_path, expected_indices.astype(np.int64, copy=False))
    _atomic_write_text(sample_ids_path, "\n".join(expected_sample_ids) + "\n")
    persisted = dict(expected)
    persisted.update(
        {
            "indices_file": str(indices_path.resolve()),
            "indices_sha256": _sha256_file(indices_path),
            "sample_ids_file": str(sample_ids_path.resolve()),
        }
    )
    persisted["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(persisted))
    _atomic_write_json(manifest_path, persisted)
    print(f"[MANIFEST] wrote {manifest_path}")
    return persisted, expected_indices


def load_optimized_hparams(config_path: Path) -> Dict[str, float]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = config["r4rr_optimized_hparams"]["waterbirds95"]
    hparams = {
        "attention_epoch": int(raw["attention_epoch"]),
        "kl_lambda": float(raw["kl_lambda"]),
        "base_lr": float(raw["base_lr"]),
        "classifier_lr": float(raw["classifier_lr"]),
        "lr2_mult": float(raw["lr2_mult"]),
    }
    expected = {
        "attention_epoch": 109,
        "kl_lambda": 295.30,
        "base_lr": 4.82e-5,
        "classifier_lr": 2.93e-3,
        "lr2_mult": 0.409,
    }
    for key, expected_value in expected.items():
        if abs(float(hparams[key]) - float(expected_value)) > 1e-12:
            raise RuntimeError(
                f"Expected optimized Waterbirds-95 {key}={expected_value}, got {hparams[key]}"
            )
    return hparams


def build_run_contract(
    args: argparse.Namespace,
    hparams: Mapping[str, float],
    manifest: Mapping[str, object],
) -> Dict[str, object]:
    hparams_path = Path(args.hparams_config).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    contract = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "waterbirds95",
        "condition": args.condition,
        "manifest_sha256": manifest["manifest_sha256"],
        "data_root": str(data_root),
        "metadata_sha256": _sha256_file(data_root / "metadata.csv"),
        "teacher_map_path": str(Path(args.teacher_map_path).expanduser().resolve()),
        "hparams_config": str(hparams_path),
        "hparams_config_sha256": _sha256_file(hparams_path),
        "model": "torchvision_resnet50_imagenet_pretrained",
        "tune_mode": "full",
        "epochs": FIXED_EPOCHS,
        "optimizer": "SGD",
        "momentum": FIXED_MOMENTUM,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "scheduler": "none",
        "base_lr": float(hparams["base_lr"]),
        "classifier_lr": float(hparams["classifier_lr"]),
        "attention_epoch": int(hparams["attention_epoch"]),
        "kl_lambda": float(hparams["kl_lambda"]),
        "kl_increment": FIXED_KL_INCREMENT,
        "lr2_mult": float(hparams["lr2_mult"]),
        "alignment_loss": FIXED_ALIGNMENT_LOSS,
        "optimizer_reset_at_attention_epoch": True,
        "validation_selector": "best_balanced_class_accuracy_at_or_after_attention_epoch",
        "group_order": list(GROUP_NAMES),
        "batch_size": int(args.batch_size),
    }
    contract["contract_sha256"] = _sha256_bytes(_canonical_json_bytes(contract))
    return contract


def audit_corruption_dataset(
    clean_dataset: waterbirds_core.WaterbirdsMetadataDataset,
    corrupted_dataset: ManifestCorruptedWaterbirdsDataset,
    selected_indices: np.ndarray,
    output_path: Path,
    audit_samples: int,
) -> Dict[str, object]:
    selected_preview = selected_indices[: max(1, audit_samples)].tolist()
    selected_set = set(int(index) for index in selected_indices.tolist())
    unselected_preview = [
        index for index in range(len(clean_dataset)) if index not in selected_set
    ][: max(1, audit_samples)]

    selected_checks = []
    for index in selected_preview:
        _, target_clean, clean_mask, path_clean = clean_dataset[int(index)]
        _, target_corrupted, corrupted_mask, path_corrupted = corrupted_dataset[int(index)]
        expected = invert_and_renormalize_mask(clean_mask)
        if (
            int(target_clean) != int(target_corrupted)
            or path_clean != path_corrupted
            or not torch.allclose(expected, corrupted_mask, atol=1e-7, rtol=1e-6)
        ):
            raise RuntimeError(f"Inversion audit failed for selected train index {index}")
        group = int(clean_dataset.labels[int(index)] * 2 + clean_dataset.places[int(index)])
        selected_checks.append(
            {
                "train_index": int(index),
                "label": int(target_clean),
                "place": int(clean_dataset.places[int(index)]),
                "group": group,
                "group_name": GROUP_NAMES[group],
                "sample_id": path_clean,
                "clean_sum": float(clean_mask.sum().item()),
                "corrupted_sum": float(corrupted_mask.sum().item()),
                "corrupted_min": float(corrupted_mask.min().item()),
                "corrupted_max": float(corrupted_mask.max().item()),
            }
        )

    unselected_checks = []
    for index in unselected_preview:
        _, target_clean, clean_mask, path_clean = clean_dataset[int(index)]
        _, target_corrupted, corrupted_mask, path_corrupted = corrupted_dataset[int(index)]
        if (
            int(target_clean) != int(target_corrupted)
            or path_clean != path_corrupted
            or not torch.equal(clean_mask, corrupted_mask)
        ):
            raise RuntimeError(f"Non-target map changed at train index {index}")
        unselected_checks.append({"train_index": int(index), "sample_id": path_clean})

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_checks": selected_checks,
        "unselected_checks": unselected_checks,
        "status": "passed",
    }
    _atomic_write_json(output_path, audit)
    print(f"[AUDIT] passed; wrote {output_path}")
    return audit


def _seed_result_path(condition_dir: Path, seed: int) -> Path:
    return condition_dir / f"seed_{seed}" / "metrics.json"


def _load_valid_seed_result(
    path: Path, seed: int, contract_sha256: str
) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        valid = (
            result.get("dataset") == "waterbirds95"
            and int(result.get("seed", -1)) == int(seed)
            and result.get("contract_sha256") == contract_sha256
            and result.get("group_order") == list(GROUP_NAMES)
            and len(result.get("test_group_acc", [])) == len(GROUP_NAMES)
        )
    except Exception:
        valid = False
        result = None
    if not valid:
        raise RuntimeError(f"Existing seed result does not match run contract: {path}")
    return result


def _group_metric_name(group_key: str) -> str:
    return f"test_group_{group_key}_acc"


def _summary_metric_fields() -> List[str]:
    metrics = [
        "best_balanced_val_acc",
        "test_acc",
        "test_mean_group_acc",
        "test_worst_group_acc",
    ]
    metrics.extend(_group_metric_name(group_key) for group_key in GROUP_KEYS)
    return metrics


def write_condition_summaries(
    condition_dir: Path,
    condition: str,
    seeds: Sequence[int],
    contract: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    results = []
    for seed in seeds:
        result = _load_valid_seed_result(
            _seed_result_path(condition_dir, seed), seed, str(contract["contract_sha256"])
        )
        if result is not None:
            results.append(result)

    per_seed_fields = [
        "dataset",
        "condition",
        "condition_type",
        "target_group_name",
        "seed",
        "corruption_seed",
        "corrupted_example_count",
        "corrupted_fraction_of_training",
        "best_epoch",
        "test_loss",
        *_summary_metric_fields(),
        "manifest_sha256",
        "contract_sha256",
        "checkpoint",
    ]
    rows = [
        {key: result.get(key, "") for key in per_seed_fields}
        for result in sorted(results, key=lambda item: int(item["seed"]))
    ]
    _atomic_write_csv(condition_dir / "per_seed_metrics.csv", per_seed_fields, rows)

    summary: Dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "waterbirds95",
        "condition": condition,
        "condition_type": manifest["condition_type"],
        "target_group_index": manifest["target_group_index"],
        "target_group_key": manifest["target_group_key"],
        "target_group_name": manifest["target_group_name"],
        "requested_seeds": [int(seed) for seed in seeds],
        "completed_seeds": sorted(int(result["seed"]) for result in results),
        "n_completed": len(results),
        "corruption_seed": int(manifest["corruption_seed"]),
        "corrupted_example_count": int(manifest["corrupted_example_count"]),
        "corrupted_fraction_of_training": float(manifest["corrupted_fraction_of_training"]),
        "corrupted_group_counts": manifest["corrupted_group_counts"],
        "group_order": list(GROUP_NAMES),
        "manifest_sha256": manifest["manifest_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "metrics": {},
    }
    for metric in _summary_metric_fields():
        values = np.asarray([float(result[metric]) for result in results], dtype=np.float64)
        summary["metrics"][metric] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std(ddof=0)) if values.size else None,
            "n": int(values.size),
        }
    _atomic_write_json(condition_dir / "summary.json", summary)

    if results:
        summary_row: Dict[str, object] = {
            "dataset": "waterbirds95",
            "condition": condition,
            "condition_type": manifest["condition_type"],
            "target_group_name": manifest["target_group_name"],
            "n_completed": len(results),
            "corruption_seed": manifest["corruption_seed"],
            "corrupted_example_count": manifest["corrupted_example_count"],
            "corrupted_fraction_of_training": manifest["corrupted_fraction_of_training"],
            "manifest_sha256": manifest["manifest_sha256"],
            "contract_sha256": contract["contract_sha256"],
        }
        for metric in _summary_metric_fields():
            summary_row[f"{metric}_mean"] = summary["metrics"][metric]["mean"]
            summary_row[f"{metric}_std"] = summary["metrics"][metric]["std"]
        _atomic_write_csv(condition_dir / "summary.csv", list(summary_row), [summary_row])


@torch.no_grad()
def evaluate_test(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Dict[str, object]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    total_loss = 0.0
    group_correct = np.zeros(len(GROUP_NAMES), dtype=np.int64)
    group_total = np.zeros(len(GROUP_NAMES), dtype=np.int64)

    for images, labels, _paths, groups in test_loader:
        images = images.to(device)
        labels = labels.to(device).long()
        groups = groups.to(device).long()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)
        total_loss += float(loss.item()) * images.size(0)
        correct += int((predictions == labels).sum().item())
        total += int(images.size(0))
        for group_index in range(len(GROUP_NAMES)):
            group_mask = groups == group_index
            if torch.any(group_mask):
                group_correct[group_index] += int(
                    (predictions[group_mask] == labels[group_mask]).sum().item()
                )
                group_total[group_index] += int(group_mask.sum().item())

    if total != EXPECTED_SPLIT_COUNTS["test"]:
        raise RuntimeError(f"Test evaluator saw {total} examples, expected 5794")
    expected_totals = np.asarray(EXPECTED_GROUP_COUNTS["test"], dtype=np.int64)
    if not np.array_equal(group_total, expected_totals):
        raise RuntimeError(
            f"Unexpected test group totals: {group_total.tolist()}; expected {expected_totals.tolist()}"
        )
    group_acc = 100.0 * group_correct / group_total
    return {
        "test_loss": total_loss / total,
        "test_acc": 100.0 * correct / total,
        "test_group_acc": group_acc,
        "test_mean_group_acc": float(group_acc.mean()),
        "test_worst_group_acc": float(group_acc.min()),
    }


def build_parser() -> argparse.ArgumentParser:
    repo_root = SCRIPT_PATH.parents[3]
    parser = argparse.ArgumentParser(
        description="Five-seed Waterbirds-95 R4RR systematic teacher corruption condition"
    )
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--teacher-map-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-root", default="")
    parser.add_argument(
        "--hparams-config",
        default=str(repo_root / "configs" / "r4rr_optimized_hparams.yaml"),
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=FIXED_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--audit-samples", type=int, default=8)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_seeds(args.seeds)
    if int(args.batch_size) != FIXED_BATCH_SIZE:
        raise ValueError("The locked Waterbirds-95 protocol uses batch_size=96")

    data_root = Path(args.data_root).expanduser().resolve()
    teacher_map_path = Path(args.teacher_map_path).expanduser().resolve()
    hparams_path = Path(args.hparams_config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    condition_dir = output_root / args.condition
    manifest_root = (
        Path(args.manifest_root).expanduser().resolve()
        if args.manifest_root
        else output_root / "corruption_manifests"
    )
    manifest_dir = manifest_root / args.condition

    if not (data_root / "metadata.csv").is_file():
        raise FileNotFoundError(f"Missing Waterbirds metadata: {data_root / 'metadata.csv'}")
    if not teacher_map_path.is_dir():
        raise FileNotFoundError(f"Missing teacher-map directory: {teacher_map_path}")
    if not hparams_path.is_file():
        raise FileNotFoundError(f"Missing optimized hyperparameters: {hparams_path}")

    hparams = load_optimized_hparams(hparams_path)
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            waterbirds_core.Brighten(8.0),
        ]
    )
    manifest_datasets = {
        split: waterbirds_core.WaterbirdsMetadataDataset(
            data_root=str(data_root),
            split=split,
            image_transform=image_transform,
            return_mask=False,
            return_path=True,
            return_group=(split == "test"),
        )
        for split in ("train", "val", "test")
    }
    expected_manifest, expected_indices, expected_ids = build_manifest(
        condition=args.condition,
        datasets=manifest_datasets,
        data_root=data_root,
        corruption_seed=args.corruption_seed,
    )
    manifest, selected_indices = persist_or_validate_manifest(
        manifest_dir, expected_manifest, expected_indices, expected_ids
    )

    clean_guided = waterbirds_core.WaterbirdsMetadataDataset(
        data_root=str(data_root),
        split="train",
        image_transform=image_transform,
        mask_root=str(teacher_map_path),
        mask_transform=mask_transform,
        return_mask=True,
        return_path=True,
        return_group=False,
    )
    corrupted_guided = ManifestCorruptedWaterbirdsDataset(
        data_root=str(data_root),
        split="train",
        image_transform=image_transform,
        mask_root=str(teacher_map_path),
        mask_transform=mask_transform,
        return_mask=True,
        return_path=True,
        return_group=False,
        corrupted_indices=selected_indices.tolist(),
    )
    audit_corruption_dataset(
        clean_dataset=clean_guided,
        corrupted_dataset=corrupted_guided,
        selected_indices=selected_indices,
        output_path=manifest_dir / "audit.json",
        audit_samples=args.audit_samples,
    )

    contract = build_run_contract(args, hparams, manifest)
    condition_dir.mkdir(parents=True, exist_ok=True)
    contract_path = condition_dir / "run_contract.json"
    if contract_path.is_file():
        observed_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed_contract != contract:
            raise RuntimeError(f"Existing run contract differs: {contract_path}")
    else:
        _atomic_write_json(contract_path, contract)

    print(f"[SETUP] condition={args.condition} type={manifest['condition_type']}")
    print(
        f"[SETUP] matched_group={manifest['target_group_name']} "
        f"corrupted={manifest['corrupted_example_count']}/{manifest['training_example_count']} "
        f"({100.0 * float(manifest['corrupted_fraction_of_training']):.4f}%)"
    )
    print(f"[SETUP] corrupted_group_counts={manifest['corrupted_group_counts']}")
    print(f"[SETUP] manifest={manifest_dir / 'manifest.json'}")
    print(f"[SETUP] output={condition_dir}")
    print(
        "[HPARAMS] "
        f"epochs={FIXED_EPOCHS} base_lr={hparams['base_lr']} "
        f"classifier_lr={hparams['classifier_lr']} attention_epoch={hparams['attention_epoch']} "
        f"kl_lambda={hparams['kl_lambda']} lr2_mult={hparams['lr2_mult']} "
        f"kl_increment={FIXED_KL_INCREMENT} alignment_loss={FIXED_ALIGNMENT_LOSS}"
    )
    if args.prepare_only:
        print("[DONE] manifest preparation and corruption audit completed")
        return

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda and not args.no_cuda:
        raise RuntimeError("CUDA is unavailable; pass --no-cuda only for a deliberate CPU run")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    waterbirds_core.device = device
    waterbirds_core.momentum = FIXED_MOMENTUM
    waterbirds_core.weight_decay = FIXED_WEIGHT_DECAY

    val_dataset = manifest_datasets["val"]
    test_dataset = manifest_datasets["test"]
    pin_memory = bool(use_cuda)
    for seed in seeds:
        result_path = _seed_result_path(condition_dir, seed)
        existing = _load_valid_seed_result(
            result_path, seed, str(contract["contract_sha256"])
        )
        if existing is not None:
            print(f"[RESUME] condition={args.condition} seed={seed} already complete")
            continue

        waterbirds_core.seed_everything(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {
            "batch_size": FIXED_BATCH_SIZE,
            "num_workers": int(args.num_workers),
            "worker_init_fn": waterbirds_core.seed_worker,
            "generator": generator,
            "pin_memory": pin_memory,
        }
        dataloaders = {
            "train": DataLoader(corrupted_guided, shuffle=True, **loader_kwargs),
            "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        }
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        dataset_sizes = {"train": len(corrupted_guided), "val": len(val_dataset)}

        print(f"[TRAIN] condition={args.condition} seed={seed} device={device}")
        model = waterbirds_core.make_cam_model(
            num_classes=2, model_name="resnet50", pretrained=True
        ).to(device)
        best_model, best_score, best_epoch = waterbirds_core.train_model(
            model,
            dataloaders,
            dataset_sizes,
            int(hparams["attention_epoch"]),
            float(hparams["kl_lambda"]),
            FIXED_EPOCHS,
            base_lr=float(hparams["base_lr"]),
            classifier_lr=float(hparams["classifier_lr"]),
            lr2_mult=float(hparams["lr2_mult"]),
            kl_incr=FIXED_KL_INCREMENT,
            use_attention=True,
            num_classes=2,
            alignment_loss=FIXED_ALIGNMENT_LOSS,
        )
        test_metrics = evaluate_test(best_model, test_loader, device)

        checkpoint = "NONE"
        if args.save_checkpoints:
            checkpoint_path = condition_dir / f"seed_{seed}" / "best_validation_checkpoint.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
            torch.save(
                {
                    "model_state_dict": best_model.state_dict(),
                    "seed": int(seed),
                    "condition": args.condition,
                    "group_order": list(GROUP_NAMES),
                    "run_contract": contract,
                    "metrics": {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in test_metrics.items()
                    },
                },
                temporary,
            )
            os.replace(str(temporary), str(checkpoint_path))
            checkpoint = str(checkpoint_path.resolve())

        group_acc = np.asarray(test_metrics["test_group_acc"], dtype=np.float64)
        result: Dict[str, object] = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": "waterbirds95",
            "condition": args.condition,
            "condition_type": manifest["condition_type"],
            "target_group_name": manifest["target_group_name"],
            "seed": int(seed),
            "corruption_seed": int(manifest["corruption_seed"]),
            "corrupted_example_count": int(manifest["corrupted_example_count"]),
            "corrupted_fraction_of_training": float(manifest["corrupted_fraction_of_training"]),
            "best_epoch": int(best_epoch),
            "best_balanced_val_acc": float(best_score),
            "test_loss": float(test_metrics["test_loss"]),
            "test_acc": float(test_metrics["test_acc"]),
            "test_mean_group_acc": float(test_metrics["test_mean_group_acc"]),
            "test_worst_group_acc": float(test_metrics["test_worst_group_acc"]),
            "test_group_acc": [float(value) for value in group_acc.tolist()],
            "group_order": list(GROUP_NAMES),
            "manifest_sha256": manifest["manifest_sha256"],
            "contract_sha256": contract["contract_sha256"],
            "checkpoint": checkpoint,
        }
        for group_key, value in zip(GROUP_KEYS, group_acc.tolist()):
            result[_group_metric_name(group_key)] = float(value)
        _atomic_write_json(result_path, result)
        write_condition_summaries(condition_dir, args.condition, seeds, contract, manifest)
        print(
            f"[SEED DONE] condition={args.condition} seed={seed} "
            f"best_val={result['best_balanced_val_acc']:.4f} test={result['test_acc']:.2f}% "
            f"mean_group={result['test_mean_group_acc']:.2f}% "
            f"worst_group={result['test_worst_group_acc']:.2f}%"
        )
        del model, best_model, dataloaders, test_loader
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

    write_condition_summaries(condition_dir, args.condition, seeds, contract, manifest)
    summary = json.loads((condition_dir / "summary.json").read_text(encoding="utf-8"))
    if int(summary["n_completed"]) != len(seeds):
        raise RuntimeError(
            f"Condition ended with {summary['n_completed']}/{len(seeds)} completed seeds"
        )
    metrics = summary["metrics"]
    print(
        f"[SUMMARY] condition={args.condition} n={summary['n_completed']} "
        f"mean_group={metrics['test_mean_group_acc']['mean']:.2f}+/-"
        f"{metrics['test_mean_group_acc']['std']:.2f} "
        f"worst_group={metrics['test_worst_group_acc']['mean']:.2f}+/-"
        f"{metrics['test_worst_group_acc']['std']:.2f}"
    )
    print(f"[DONE] {condition_dir}")


if __name__ == "__main__":
    main()
