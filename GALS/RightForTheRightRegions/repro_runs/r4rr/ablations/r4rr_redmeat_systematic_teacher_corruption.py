#!/usr/bin/env python3
"""Run one RedMeat R4RR systematic teacher-map corruption condition.

The canonical RedMeat ResNet-50 R4RR implementation is reused unchanged. This
module changes only the training teacher maps selected by a persisted manifest.
Validation and test examples never load teacher maps and cannot be corrupted.
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

import r4rr_redmeat as redmeat_core  # noqa: E402


PROTOCOL_VERSION = 1
CLASS_ORDER = (
    "prime_rib",
    "pork_chop",
    "steak",
    "baby_back_ribs",
    "filet_mignon",
)
SYSTEMATIC_CLASS_ORDER = tuple(sorted(CLASS_ORDER))
CONDITIONS = tuple([f"class_{name}" for name in SYSTEMATIC_CLASS_ORDER] + ["random_20pct"])
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
FIXED_EPOCHS = 150
FIXED_BATCH_SIZE = 96
FIXED_MOMENTUM = 0.9
FIXED_WEIGHT_DECAY = 1e-5
FIXED_KL_INCREMENT = 0.0
FIXED_ALIGNMENT_LOSS = "forward_kl"
FIXED_RANDOM_COUNT = 500
EXPECTED_SPLIT_COUNTS = {"train": 2_500, "val": 1_250, "test": 1_250}
EXPECTED_CLASS_COUNTS = {"train": 500, "val": 250, "test": 250}


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


class ManifestCorruptedRedMeatDataset(redmeat_core.RedMeatMetadataDataset):
    """Canonical RedMeat dataset with lazy inversion for persisted indices."""

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


def _stable_sample_id(path_raw: str, data_root: Path) -> str:
    path = Path(path_raw).expanduser().resolve()
    try:
        return path.relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _dataset_rows(
    dataset: redmeat_core.RedMeatMetadataDataset, split: str, data_root: Path
) -> List[Dict[str, object]]:
    return [
        {
            "split_index": int(index),
            "split": split,
            "sample_id": _stable_sample_id(path, data_root),
            "label_name": str(dataset.label_names[index]),
            "target": int(dataset.labels[index]),
        }
        for index, path in enumerate(dataset.paths)
    ]


def _dataset_fingerprint(
    datasets: Mapping[str, redmeat_core.RedMeatMetadataDataset], data_root: Path
) -> str:
    payload: List[Dict[str, object]] = []
    for split in ("train", "val", "test"):
        payload.extend(_dataset_rows(datasets[split], split, data_root))
    return _sha256_bytes(_canonical_json_bytes(payload))


def select_corruption_indices(
    condition: str,
    label_names: Sequence[str],
    corruption_seed: int,
    random_count: int = FIXED_RANDOM_COUNT,
) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")

    if condition.startswith("class_"):
        target_class = condition[len("class_") :]
        selected = [index for index, label in enumerate(label_names) if label == target_class]
    else:
        if random_count <= 0 or random_count > len(label_names):
            raise ValueError(
                f"random_count must be in [1, {len(label_names)}], got {random_count}"
            )
        rng = np.random.default_rng(int(corruption_seed))
        selected = rng.choice(len(label_names), size=int(random_count), replace=False).tolist()

    selected_array = np.asarray(sorted(int(index) for index in selected), dtype=np.int64)
    if selected_array.size == 0:
        raise RuntimeError(f"Condition {condition} selected no training examples")
    if np.unique(selected_array).size != selected_array.size:
        raise RuntimeError(f"Condition {condition} produced duplicate indices")
    return selected_array


def _count_classes(dataset: redmeat_core.RedMeatMetadataDataset) -> Dict[str, int]:
    return {
        class_name: int(sum(label == class_name for label in dataset.label_names))
        for class_name in CLASS_ORDER
    }


def validate_dataset_contract(
    datasets: Mapping[str, redmeat_core.RedMeatMetadataDataset]
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        dataset = datasets[split]
        if tuple(dataset.classes) != CLASS_ORDER:
            raise RuntimeError(
                f"Unexpected class order for {split}: {dataset.classes}; expected {CLASS_ORDER}"
            )
        if len(dataset) != EXPECTED_SPLIT_COUNTS[split]:
            raise RuntimeError(
                f"Expected {EXPECTED_SPLIT_COUNTS[split]} {split} examples, got {len(dataset)}"
            )
        counts[split] = _count_classes(dataset)
        expected = {name: EXPECTED_CLASS_COUNTS[split] for name in CLASS_ORDER}
        if counts[split] != expected:
            raise RuntimeError(
                f"Unexpected {split} class counts: {counts[split]}; expected {expected}"
            )
    return counts


def build_manifest(
    condition: str,
    datasets: Mapping[str, redmeat_core.RedMeatMetadataDataset],
    data_root: Path,
    corruption_seed: int,
    random_count: int,
) -> Tuple[Dict[str, object], np.ndarray, List[str]]:
    class_counts = validate_dataset_contract(datasets)
    train_dataset = datasets["train"]
    selected = select_corruption_indices(
        condition=condition,
        label_names=train_dataset.label_names,
        corruption_seed=corruption_seed,
        random_count=random_count,
    )
    target_class = condition[len("class_") :] if condition.startswith("class_") else None
    expected_count = EXPECTED_CLASS_COUNTS["train"] if target_class else FIXED_RANDOM_COUNT
    if int(selected.size) != expected_count:
        raise RuntimeError(
            f"Condition {condition} must select exactly {expected_count} examples; got {selected.size}"
        )

    rows = _dataset_rows(train_dataset, "train", data_root)
    selected_ids = [str(rows[index]["sample_id"]) for index in selected.tolist()]
    selected_class_counts = {
        name: int(sum(train_dataset.label_names[index] == name for index in selected.tolist()))
        for name in CLASS_ORDER
    }
    if target_class is not None:
        expected_selected = {name: (500 if name == target_class else 0) for name in CLASS_ORDER}
        if selected_class_counts != expected_selected:
            raise RuntimeError(
                f"Systematic selection is not class-pure: {selected_class_counts}"
            )

    manifest: Dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "redmeat",
        "condition": condition,
        "condition_type": "systematic_class" if target_class else "random_control",
        "target_class": target_class,
        "corruption_operation": "one_minus_then_sum_renormalize",
        "corruption_seed": int(corruption_seed),
        "random_count_requested": int(random_count) if target_class is None else None,
        "split_example_counts": {split: int(len(datasets[split])) for split in datasets},
        "split_class_counts": class_counts,
        "training_example_count": int(len(train_dataset)),
        "corrupted_example_count": int(selected.size),
        "corrupted_fraction_of_training": float(selected.size / len(train_dataset)),
        "class_order": list(CLASS_ORDER),
        "class_to_idx": {key: int(value) for key, value in train_dataset.class_to_idx.items()},
        "corrupted_class_counts": selected_class_counts,
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
    raw = config["r4rr_optimized_hparams"]["redmeat"]
    hparams = {
        "attention_epoch": int(raw["attention_epoch"]),
        "kl_lambda": float(raw["kl_lambda"]),
        "base_lr": float(raw["base_lr"]),
        "classifier_lr": float(raw["classifier_lr"]),
        "lr2_mult": float(raw["lr2_mult"]),
    }
    expected = {
        "attention_epoch": 2,
        "kl_lambda": 11.44,
        "base_lr": 2.40e-3,
        "classifier_lr": 2.33e-4,
        "lr2_mult": 1.567,
    }
    for key, expected_value in expected.items():
        observed = hparams[key]
        if abs(float(observed) - float(expected_value)) > 1e-12:
            raise RuntimeError(
                f"Expected optimized RedMeat {key}={expected_value}, got {observed}"
            )
    return hparams


def build_run_contract(
    args: argparse.Namespace,
    hparams: Mapping[str, float],
    manifest: Mapping[str, object],
) -> Dict[str, object]:
    hparams_path = Path(args.hparams_config).expanduser().resolve()
    contract = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "redmeat",
        "condition": args.condition,
        "manifest_sha256": manifest["manifest_sha256"],
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "metadata_sha256": _sha256_file(Path(args.data_root).expanduser().resolve() / "all_images.csv"),
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
        "class_order": list(CLASS_ORDER),
        "batch_size": int(args.batch_size),
    }
    contract["contract_sha256"] = _sha256_bytes(_canonical_json_bytes(contract))
    return contract


def audit_corruption_dataset(
    clean_dataset: redmeat_core.RedMeatMetadataDataset,
    corrupted_dataset: ManifestCorruptedRedMeatDataset,
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
        selected_checks.append(
            {
                "train_index": int(index),
                "target": int(target_clean),
                "label_name": clean_dataset.label_names[int(index)],
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
        unselected_checks.append(
            {
                "train_index": int(index),
                "target": int(target_clean),
                "label_name": clean_dataset.label_names[int(index)],
                "sample_id": path_clean,
            }
        )

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
            result.get("dataset") == "redmeat"
            and int(result.get("seed", -1)) == int(seed)
            and result.get("contract_sha256") == contract_sha256
            and result.get("class_order") == list(CLASS_ORDER)
            and len(result.get("test_class_acc", [])) == len(CLASS_ORDER)
        )
    except Exception:
        valid = False
        result = None
    if not valid:
        raise RuntimeError(f"Existing seed result does not match run contract: {path}")
    return result


def _class_metric_name(class_name: str) -> str:
    return f"test_class_{class_name}_acc"


def _summary_metric_fields() -> List[str]:
    metrics = [
        "best_balanced_val_acc",
        "test_acc",
        "test_mean_class_acc",
        "test_worst_class_acc",
    ]
    metrics.extend(_class_metric_name(class_name) for class_name in CLASS_ORDER)
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
        "target_class",
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
        "dataset": "redmeat",
        "condition": condition,
        "condition_type": manifest["condition_type"],
        "target_class": manifest["target_class"],
        "requested_seeds": [int(seed) for seed in seeds],
        "completed_seeds": sorted(int(result["seed"]) for result in results),
        "n_completed": len(results),
        "corruption_seed": int(manifest["corruption_seed"]),
        "corrupted_example_count": int(manifest["corrupted_example_count"]),
        "corrupted_fraction_of_training": float(manifest["corrupted_fraction_of_training"]),
        "corrupted_class_counts": manifest["corrupted_class_counts"],
        "class_order": list(CLASS_ORDER),
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
            "dataset": "redmeat",
            "condition": condition,
            "condition_type": manifest["condition_type"],
            "target_class": manifest["target_class"] or "",
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
    class_correct = np.zeros(len(CLASS_ORDER), dtype=np.int64)
    class_total = np.zeros(len(CLASS_ORDER), dtype=np.int64)

    for images, labels, _paths in test_loader:
        images = images.to(device)
        labels = labels.to(device).long()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)
        total_loss += float(loss.item()) * images.size(0)
        correct += int((predictions == labels).sum().item())
        total += int(images.size(0))
        labels_cpu = labels.detach().cpu().numpy()
        predictions_cpu = predictions.detach().cpu().numpy()
        for class_index in range(len(CLASS_ORDER)):
            class_mask = labels_cpu == class_index
            if np.any(class_mask):
                class_correct[class_index] += int(
                    np.sum(predictions_cpu[class_mask] == labels_cpu[class_mask])
                )
                class_total[class_index] += int(np.sum(class_mask))

    if total != EXPECTED_SPLIT_COUNTS["test"]:
        raise RuntimeError(f"Test evaluator saw {total} examples, expected 1250")
    if not np.all(class_total == EXPECTED_CLASS_COUNTS["test"]):
        raise RuntimeError(f"Unexpected test class totals: {class_total.tolist()}")
    class_acc = 100.0 * class_correct / class_total
    return {
        "test_loss": total_loss / total,
        "test_acc": 100.0 * correct / total,
        "test_class_acc": class_acc,
        "test_mean_class_acc": float(class_acc.mean()),
        "test_worst_class_acc": float(class_acc.min()),
    }


def build_parser() -> argparse.ArgumentParser:
    repo_root = SCRIPT_PATH.parents[3]
    parser = argparse.ArgumentParser(
        description="Five-seed RedMeat R4RR systematic teacher corruption condition"
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
    parser.add_argument("--random-count", type=int, default=FIXED_RANDOM_COUNT)
    parser.add_argument("--batch-size", type=int, default=FIXED_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--audit-samples", type=int, default=8)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--path-col", default="abs_file_path")
    return parser


def _dataset_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "classes": list(CLASS_ORDER),
        "split_col": args.split_col,
        "label_col": args.label_col,
        "path_col": args.path_col,
    }


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_seeds(args.seeds)
    if int(args.random_count) != FIXED_RANDOM_COUNT:
        raise ValueError("The locked RedMeat random control contains exactly 500 examples")
    if int(args.batch_size) != FIXED_BATCH_SIZE:
        raise ValueError("The locked RedMeat protocol uses batch_size=96")

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

    if not (data_root / "all_images.csv").is_file():
        raise FileNotFoundError(f"Missing RedMeat metadata: {data_root / 'all_images.csv'}")
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
            redmeat_core.base.Brighten(8.0),
        ]
    )
    common = _dataset_kwargs(args)
    manifest_datasets = {
        split: redmeat_core.RedMeatMetadataDataset(
            split=split,
            image_transform=image_transform,
            return_mask=False,
            return_path=True,
            **common,
        )
        for split in ("train", "val", "test")
    }
    expected_manifest, expected_indices, expected_ids = build_manifest(
        condition=args.condition,
        datasets=manifest_datasets,
        data_root=data_root,
        corruption_seed=args.corruption_seed,
        random_count=args.random_count,
    )
    manifest, selected_indices = persist_or_validate_manifest(
        manifest_dir, expected_manifest, expected_indices, expected_ids
    )

    clean_guided = redmeat_core.RedMeatMetadataDataset(
        split="train",
        image_transform=image_transform,
        mask_root=str(teacher_map_path),
        mask_transform=mask_transform,
        return_mask=True,
        return_path=True,
        **common,
    )
    corrupted_guided = ManifestCorruptedRedMeatDataset(
        split="train",
        image_transform=image_transform,
        mask_root=str(teacher_map_path),
        mask_transform=mask_transform,
        return_mask=True,
        return_path=True,
        corrupted_indices=selected_indices.tolist(),
        **common,
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
        f"[SETUP] corrupted={manifest['corrupted_example_count']}/"
        f"{manifest['training_example_count']} "
        f"({100.0 * float(manifest['corrupted_fraction_of_training']):.2f}%)"
    )
    print(f"[SETUP] corrupted_class_counts={manifest['corrupted_class_counts']}")
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
    redmeat_core.device = device
    redmeat_core.base.device = device
    redmeat_core.base.momentum = FIXED_MOMENTUM
    redmeat_core.base.weight_decay = FIXED_WEIGHT_DECAY

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

        redmeat_core.base.seed_everything(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {
            "batch_size": FIXED_BATCH_SIZE,
            "num_workers": int(args.num_workers),
            "worker_init_fn": redmeat_core.base.seed_worker,
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
        model = redmeat_core.make_redmeat_cam_model(
            num_classes=len(CLASS_ORDER),
            model_name="resnet50",
            pretrained=True,
            clip_model="RN50",
        ).to(device)
        redmeat_core.configure_tune_mode(model, tune_mode="full")
        best_model, best_score, best_epoch = redmeat_core.base.train_model(
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
            num_classes=len(CLASS_ORDER),
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
                    "class_order": list(CLASS_ORDER),
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

        class_acc = np.asarray(test_metrics["test_class_acc"], dtype=np.float64)
        result: Dict[str, object] = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": "redmeat",
            "condition": args.condition,
            "condition_type": manifest["condition_type"],
            "target_class": manifest["target_class"],
            "seed": int(seed),
            "corruption_seed": int(manifest["corruption_seed"]),
            "corrupted_example_count": int(manifest["corrupted_example_count"]),
            "corrupted_fraction_of_training": float(manifest["corrupted_fraction_of_training"]),
            "best_epoch": int(best_epoch),
            "best_balanced_val_acc": float(best_score),
            "test_loss": float(test_metrics["test_loss"]),
            "test_acc": float(test_metrics["test_acc"]),
            "test_mean_class_acc": float(test_metrics["test_mean_class_acc"]),
            "test_worst_class_acc": float(test_metrics["test_worst_class_acc"]),
            "test_class_acc": [float(value) for value in class_acc.tolist()],
            "class_order": list(CLASS_ORDER),
            "manifest_sha256": manifest["manifest_sha256"],
            "contract_sha256": contract["contract_sha256"],
            "checkpoint": checkpoint,
        }
        for class_name, value in zip(CLASS_ORDER, class_acc.tolist()):
            result[_class_metric_name(class_name)] = float(value)
        _atomic_write_json(result_path, result)
        write_condition_summaries(condition_dir, args.condition, seeds, contract, manifest)
        print(
            f"[SEED DONE] condition={args.condition} seed={seed} "
            f"best_val={result['best_balanced_val_acc']:.4f} "
            f"test={result['test_acc']:.2f}% mean_class={result['test_mean_class_acc']:.2f}% "
            f"worst_class={result['test_worst_class_acc']:.2f}%"
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
        f"mean_class={metrics['test_mean_class_acc']['mean']:.2f}+/-"
        f"{metrics['test_mean_class_acc']['std']:.2f} "
        f"worst_class={metrics['test_worst_class_acc']['mean']:.2f}+/-"
        f"{metrics['test_worst_class_acc']['std']:.2f}"
    )
    print(f"[DONE] {condition_dir}")


if __name__ == "__main__":
    main()
