#!/usr/bin/env python3
"""Run the DecoyMNIST systematic teacher-map corruption study.

This runner preserves the canonical DecoyMNIST R4RR training implementation
and changes only which *training* teacher maps are inverted. Corruption
membership is persisted before training and reused across model seeds.
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
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor


SCRIPT_PATH = Path(__file__).resolve()
R4RR_ROOT = SCRIPT_PATH.parents[1]
TRAIN_DIR = R4RR_ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import r4rr_decoy_fixed as decoy_core  # noqa: E402


PROTOCOL_VERSION = 1
CONDITIONS = tuple([f"digit_{digit}" for digit in range(10)] + ["random_10pct"])
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
FIXED_EPOCHS = 19
FIXED_WEIGHT_DECAY = 1e-4
FIXED_KL_INCREMENT = 0.0
FIXED_ALIGNMENT_LOSS = "forward_kl"
EXPECTED_SOURCE_TRAIN_EXAMPLES = 60_000
EXPECTED_TRAIN_EXAMPLES = 54_000
EXPECTED_VALIDATION_EXAMPLES = 6_000
EXPECTED_TEST_EXAMPLES = 10_000


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


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
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
    """Apply the original R4RR stress-test inversion exactly."""
    inverted = torch.clamp(1.0 - mask, min=0.0)
    total = float(inverted.sum().item())
    if total <= 1e-12:
        return torch.full_like(inverted, 1.0 / float(inverted.numel()))
    return inverted / total


class ManifestCorruptedGuidedImageFolder(decoy_core.GuidedImageFolder):
    """Canonical guided dataset with lazy inversion for persisted indices."""

    def __init__(self, *args, corrupted_indices: Iterable[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.corrupted_indices = frozenset(int(index) for index in corrupted_indices)

    def __getitem__(self, index: int):
        image, target, teacher_mask = super().__getitem__(index)
        if index in self.corrupted_indices:
            teacher_mask = invert_and_renormalize_mask(teacher_mask)
        return image, target, teacher_mask


def _sample_rows(dataset: ImageFolder, image_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, (path_raw, target_raw) in enumerate(dataset.samples):
        path = Path(path_raw).resolve()
        try:
            sample_id = path.relative_to(image_root.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"Training sample is outside image root: {path}") from exc
        rows.append(
            {
                "base_index": int(index),
                "sample_id": sample_id,
                "target": int(target_raw),
            }
        )
    return rows


def _dataset_fingerprint(
    rows: Sequence[Mapping[str, object]], train_indices: Sequence[int], val_indices: Sequence[int]
) -> str:
    split_by_index = {int(index): "train" for index in train_indices}
    split_by_index.update({int(index): "val" for index in val_indices})
    payload = [
        {
            "base_index": int(row["base_index"]),
            "sample_id": str(row["sample_id"]),
            "target": int(row["target"]),
            "split": split_by_index[int(row["base_index"])],
        }
        for row in rows
    ]
    return _sha256_bytes(_canonical_json_bytes(payload))


def select_corruption_indices(
    condition: str,
    sample_targets: Sequence[int],
    train_indices: Sequence[int],
    class_to_idx: Mapping[str, int],
    corruption_seed: int,
    random_fraction: float = 0.10,
) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")

    train_indices_array = np.asarray(train_indices, dtype=np.int64)
    if condition.startswith("digit_"):
        digit = int(condition.split("_", 1)[1])
        class_key = str(digit)
        if class_key not in class_to_idx:
            raise RuntimeError(
                f"DecoyMNIST class folder {class_key!r} is missing; class_to_idx={dict(class_to_idx)}"
            )
        target_index = int(class_to_idx[class_key])
        selected = [
            int(index)
            for index in train_indices
            if int(sample_targets[int(index)]) == target_index
        ]
    else:
        if not 0.0 < random_fraction <= 1.0:
            raise ValueError(f"random_fraction must be in (0, 1], got {random_fraction}")
        sample_count = int(round(len(train_indices) * random_fraction))
        rng = np.random.default_rng(int(corruption_seed))
        selected = rng.choice(train_indices_array, size=sample_count, replace=False).tolist()

    selected_array = np.asarray(sorted(int(index) for index in selected), dtype=np.int64)
    if selected_array.size == 0:
        raise RuntimeError(f"Condition {condition} selected no training examples")
    if np.unique(selected_array).size != selected_array.size:
        raise RuntimeError(f"Condition {condition} produced duplicate indices")
    return selected_array


def build_manifest(
    condition: str,
    dataset: ImageFolder,
    image_root: Path,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    corruption_seed: int,
    split_seed: int,
    val_fraction: float,
    random_fraction: float,
) -> Tuple[Dict[str, object], np.ndarray, List[str]]:
    rows = _sample_rows(dataset, image_root)
    targets = [int(row["target"]) for row in rows]
    selected = select_corruption_indices(
        condition=condition,
        sample_targets=targets,
        train_indices=train_indices,
        class_to_idx=dataset.class_to_idx,
        corruption_seed=corruption_seed,
        random_fraction=random_fraction,
    )
    selected_ids = [str(rows[int(index)]["sample_id"]) for index in selected.tolist()]

    train_set = set(int(index) for index in train_indices)
    val_set = set(int(index) for index in val_indices)
    selected_set = set(int(index) for index in selected.tolist())
    if not selected_set.issubset(train_set):
        raise RuntimeError("Corruption selection includes samples outside the training split")
    if selected_set.intersection(val_set):
        raise RuntimeError("Corruption selection overlaps the validation split")

    class_train_counts: Dict[str, int] = {}
    class_selected_counts: Dict[str, int] = {}
    idx_to_class = {int(index): name for name, index in dataset.class_to_idx.items()}
    for class_index, class_name in sorted(idx_to_class.items()):
        class_train_counts[class_name] = sum(
            int(targets[int(index)] == class_index) for index in train_indices
        )
        class_selected_counts[class_name] = sum(
            int(targets[int(index)] == class_index) for index in selected.tolist()
        )

    target_digit = int(condition.split("_", 1)[1]) if condition.startswith("digit_") else None
    manifest: Dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "decoymnist",
        "condition": condition,
        "condition_type": "systematic_class" if target_digit is not None else "random_control",
        "target_digit": target_digit,
        "corruption_operation": "one_minus_then_sum_renormalize",
        "corruption_seed": int(corruption_seed),
        "random_fraction_requested": float(random_fraction) if target_digit is None else None,
        "split_seed": int(split_seed),
        "validation_fraction": float(val_fraction),
        "source_example_count": len(rows),
        "training_example_count": len(train_indices),
        "validation_example_count": len(val_indices),
        "corrupted_example_count": int(selected.size),
        "corrupted_fraction_of_training": float(selected.size / len(train_indices)),
        "class_to_idx": {str(key): int(value) for key, value in dataset.class_to_idx.items()},
        "training_class_counts": class_train_counts,
        "corrupted_class_counts": class_selected_counts,
        "dataset_split_sha256": _dataset_fingerprint(rows, train_indices, val_indices),
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
        observed_without_files = {
            key: value
            for key, value in observed.items()
            if key not in {"indices_file", "indices_sha256", "sample_ids_file", "manifest_sha256"}
        }
        if observed_without_files != expected:
            raise RuntimeError(
                f"Persisted manifest contract differs from the requested condition: {manifest_path}"
            )
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
    raw = config["r4rr_optimized_hparams"]["decoymnist"]
    hparams = {
        "attention_epoch": int(raw["attention_epoch"]),
        "kl_lambda": float(raw["kl_lambda"]),
        "base_lr": float(raw["base_lr"]),
        "classifier_lr": float(raw["classifier_lr"]),
        "lr2_mult": float(raw["lr2_mult"]),
    }
    if hparams["attention_epoch"] != 7:
        raise RuntimeError(f"Expected optimized Decoy attention_epoch=7, got {hparams['attention_epoch']}")
    if abs(hparams["base_lr"] - hparams["classifier_lr"]) > 1e-12:
        raise RuntimeError("Decoy's single-LR LeNet requires base_lr == classifier_lr")
    return hparams


def build_run_contract(
    args: argparse.Namespace,
    hparams: Mapping[str, float],
    manifest: Mapping[str, object],
) -> Dict[str, object]:
    contract = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "decoymnist",
        "condition": args.condition,
        "manifest_sha256": manifest["manifest_sha256"],
        "teacher_map_path": str(Path(args.teacher_map_path).expanduser().resolve()),
        "png_root": str(Path(args.png_root).expanduser().resolve()),
        "epochs": FIXED_EPOCHS,
        "optimizer": "Adam",
        "learning_rate": float(hparams["base_lr"]),
        "weight_decay": FIXED_WEIGHT_DECAY,
        "attention_epoch": int(hparams["attention_epoch"]),
        "kl_lambda": float(hparams["kl_lambda"]),
        "kl_increment": FIXED_KL_INCREMENT,
        "alignment_loss": FIXED_ALIGNMENT_LOSS,
        "lr2_mult_recorded_but_not_applicable_to_single_lr_lenet": float(hparams["lr2_mult"]),
        "validation_selector": "best_validation_accuracy_after_attention_epoch",
        "split_seed": int(args.split_seed),
        "validation_fraction": float(args.val_fraction),
        "batch_size": int(args.batch_size),
        "test_batch_size": int(args.test_batch_size),
    }
    contract["contract_sha256"] = _sha256_bytes(_canonical_json_bytes(contract))
    return contract


def audit_corruption_dataset(
    clean_dataset: decoy_core.GuidedImageFolder,
    corrupted_dataset: ManifestCorruptedGuidedImageFolder,
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
        _, target_clean, clean_mask = clean_dataset[int(index)]
        _, target_corrupted, corrupted_mask = corrupted_dataset[int(index)]
        expected = invert_and_renormalize_mask(clean_mask)
        if int(target_clean) != int(target_corrupted) or not torch.allclose(
            expected, corrupted_mask, atol=1e-7, rtol=1e-6
        ):
            raise RuntimeError(f"Inversion audit failed for selected base index {index}")
        selected_checks.append(
            {
                "base_index": int(index),
                "target": int(target_clean),
                "clean_sum": float(clean_mask.sum().item()),
                "corrupted_sum": float(corrupted_mask.sum().item()),
                "corrupted_min": float(corrupted_mask.min().item()),
                "corrupted_max": float(corrupted_mask.max().item()),
            }
        )

    unselected_checks = []
    for index in unselected_preview:
        _, target_clean, clean_mask = clean_dataset[int(index)]
        _, target_corrupted, corrupted_mask = corrupted_dataset[int(index)]
        if int(target_clean) != int(target_corrupted) or not torch.equal(clean_mask, corrupted_mask):
            raise RuntimeError(f"Non-target map changed at base index {index}")
        unselected_checks.append({"base_index": int(index), "target": int(target_clean)})

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


def _load_valid_seed_result(path: Path, seed: int, contract_sha256: str) -> Dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        valid = (
            result.get("dataset") == "decoymnist"
            and int(result.get("seed", -1)) == int(seed)
            and result.get("contract_sha256") == contract_sha256
            and len(result.get("test_class_acc", [])) == 10
        )
    except Exception:
        valid = False
        result = None
    if not valid:
        raise RuntimeError(
            f"Existing seed result does not match the current run contract: {path}"
        )
    return result


def _summary_metric_fields() -> List[str]:
    metrics = [
        "best_val_acc",
        "test_acc",
        "test_balanced_class_acc",
        "test_worst_class_acc",
    ]
    metrics.extend(f"test_class_{digit}_acc" for digit in range(10))
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
        "target_digit",
        "seed",
        "corruption_seed",
        "corrupted_example_count",
        "corrupted_fraction_of_training",
        "best_epoch",
        *_summary_metric_fields(),
        "contract_sha256",
        "checkpoint",
    ]
    per_seed_rows = []
    for result in sorted(results, key=lambda item: int(item["seed"])):
        row = {key: result.get(key, "") for key in per_seed_fields}
        per_seed_rows.append(row)
    _atomic_write_csv(condition_dir / "per_seed_metrics.csv", per_seed_fields, per_seed_rows)

    summary: Dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "decoymnist",
        "condition": condition,
        "condition_type": manifest["condition_type"],
        "target_digit": manifest["target_digit"],
        "requested_seeds": [int(seed) for seed in seeds],
        "completed_seeds": sorted(int(result["seed"]) for result in results),
        "n_completed": len(results),
        "corruption_seed": int(manifest["corruption_seed"]),
        "corrupted_example_count": int(manifest["corrupted_example_count"]),
        "corrupted_fraction_of_training": float(manifest["corrupted_fraction_of_training"]),
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
            "dataset": "decoymnist",
            "condition": condition,
            "condition_type": manifest["condition_type"],
            "target_digit": manifest["target_digit"] if manifest["target_digit"] is not None else "",
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
        _atomic_write_csv(
            condition_dir / "summary.csv", list(summary_row.keys()), [summary_row]
        )


def build_parser() -> argparse.ArgumentParser:
    repo_root = SCRIPT_PATH.parents[3]
    parser = argparse.ArgumentParser(
        description="Five-seed DecoyMNIST R4RR systematic teacher corruption condition"
    )
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--png-root", required=True)
    parser.add_argument("--teacher-map-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-root", default="")
    parser.add_argument(
        "--hparams-config",
        default=str(repo_root / "configs" / "r4rr_optimized_hparams.yaml"),
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--random-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--audit-samples", type=int, default=8)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_seeds(args.seeds)
    if abs(float(args.random_fraction) - 0.10) > 1e-12:
        raise ValueError("The locked DecoyMNIST Round 2 random control is exactly 10%")
    if int(args.split_seed) != 0 or abs(float(args.val_fraction) - 0.10) > 1e-12:
        raise ValueError("The locked DecoyMNIST protocol uses split_seed=0 and val_fraction=0.10")

    png_root = Path(args.png_root).expanduser().resolve()
    train_root = png_root / "train"
    test_root = png_root / "test"
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

    for required in (train_root, test_root, teacher_map_path):
        if not required.is_dir():
            raise FileNotFoundError(f"Required directory does not exist: {required}")
    if not hparams_path.is_file():
        raise FileNotFoundError(f"Hyperparameter config does not exist: {hparams_path}")

    hparams = load_optimized_hparams(hparams_path)
    image_transform = Compose(
        [Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda image: image * 2.0 - 1.0)]
    )
    mask_transform = transforms.Compose(
        [
            decoy_core.ExpandWhite(thr=10, radius=3),
            decoy_core.EdgeExtract(thr=10, edge_width=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            decoy_core.Brighten(8.0),
        ]
    )

    split_dataset = ImageFolder(str(train_root))
    test_dataset_for_count = ImageFolder(str(test_root))
    expected_classes = {str(digit) for digit in range(10)}
    if set(split_dataset.class_to_idx) != expected_classes:
        raise RuntimeError(
            "Expected DecoyMNIST class folders 0 through 9; got "
            f"{sorted(split_dataset.class_to_idx)}"
        )
    if len(split_dataset) != EXPECTED_SOURCE_TRAIN_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_TRAIN_EXAMPLES} source training examples, "
            f"got {len(split_dataset)}"
        )
    if len(test_dataset_for_count) != EXPECTED_TEST_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_EXAMPLES} test examples, got {len(test_dataset_for_count)}"
        )
    train_indices, val_indices = decoy_core.fixed_split_indices(
        len(split_dataset), args.val_fraction, args.split_seed
    )
    if len(train_indices) != EXPECTED_TRAIN_EXAMPLES or len(val_indices) != EXPECTED_VALIDATION_EXAMPLES:
        raise RuntimeError(
            "Unexpected fixed split sizes: "
            f"train={len(train_indices)} validation={len(val_indices)}"
        )
    expected_manifest, expected_indices, expected_sample_ids = build_manifest(
        condition=args.condition,
        dataset=split_dataset,
        image_root=train_root,
        train_indices=train_indices,
        val_indices=val_indices,
        corruption_seed=args.corruption_seed,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        random_fraction=args.random_fraction,
    )
    manifest, selected_indices = persist_or_validate_manifest(
        manifest_dir, expected_manifest, expected_indices, expected_sample_ids
    )
    if args.condition == "random_10pct" and int(manifest["corrupted_example_count"]) != 5_400:
        raise RuntimeError(
            "The locked random_10pct control must contain exactly 5,400 training examples"
        )

    clean_guided = decoy_core.GuidedImageFolder(
        image_root=str(train_root),
        mask_root=str(teacher_map_path),
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    corrupted_guided = ManifestCorruptedGuidedImageFolder(
        image_root=str(train_root),
        mask_root=str(teacher_map_path),
        image_transform=image_transform,
        mask_transform=mask_transform,
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
        f"[SETUP] corrupted={manifest['corrupted_example_count']}/"
        f"{manifest['training_example_count']} "
        f"({100.0 * float(manifest['corrupted_fraction_of_training']):.4f}%)"
    )
    print(f"[SETUP] manifest={manifest_dir / 'manifest.json'}")
    print(f"[SETUP] output={condition_dir}")
    print(
        "[HPARAMS] "
        f"epochs={FIXED_EPOCHS} lr={hparams['base_lr']} weight_decay={FIXED_WEIGHT_DECAY} "
        f"attention_epoch={hparams['attention_epoch']} kl_lambda={hparams['kl_lambda']} "
        f"kl_increment={FIXED_KL_INCREMENT} alignment_loss={FIXED_ALIGNMENT_LOSS}"
    )
    if args.prepare_only:
        print("[DONE] manifest preparation and corruption audit completed")
        return

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda and not args.no_cuda:
        raise RuntimeError("CUDA is unavailable; pass --no-cuda only for a deliberate CPU check")
    device = torch.device("cuda" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": int(args.num_workers), "pin_memory": use_cuda}
    full_train_plain = ImageFolder(str(train_root), transform=image_transform)
    true_test = ImageFolder(str(test_root), transform=image_transform)

    core_args = argparse.Namespace(
        val_frac=float(args.val_fraction),
        split_seed=int(args.split_seed),
        batch_size=int(args.batch_size),
        test_batch_size=int(args.test_batch_size),
        lr=float(hparams["base_lr"]),
        weight_decay=FIXED_WEIGHT_DECAY,
        epochs=FIXED_EPOCHS,
        attention_epoch=int(hparams["attention_epoch"]),
        kl_lambda=float(hparams["kl_lambda"]),
        kl_incr=FIXED_KL_INCREMENT,
        alignment_loss=FIXED_ALIGNMENT_LOSS,
        print_every=int(args.print_every),
        epoch_checkpoint_dir="",
    )

    for seed in seeds:
        result_path = _seed_result_path(condition_dir, seed)
        existing = _load_valid_seed_result(
            result_path, seed, str(contract["contract_sha256"])
        )
        if existing is not None:
            print(f"[RESUME] condition={args.condition} seed={seed} already complete")
            continue

        print(f"[TRAIN] condition={args.condition} seed={seed} device={device}")
        row = decoy_core.train_one_seed(
            args=core_args,
            seed=seed,
            full_train_guided=corrupted_guided,
            full_train_plain=full_train_plain,
            true_test=true_test,
            device=device,
            loader_kwargs=loader_kwargs,
        )
        state_dict = row.pop("state_dict")
        checkpoint = "NONE"
        if args.save_checkpoints:
            checkpoint_path = condition_dir / f"seed_{seed}" / "best_validation_checkpoint.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
            torch.save(
                {
                    "model_state_dict": state_dict,
                    "seed": int(seed),
                    "condition": args.condition,
                    "run_contract": contract,
                    "metrics": {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in row.items()
                    },
                },
                temporary,
            )
            os.replace(str(temporary), str(checkpoint_path))
            checkpoint = str(checkpoint_path.resolve())

        class_acc = np.asarray(row["test_class_acc"], dtype=np.float64)
        result: Dict[str, object] = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": "decoymnist",
            "condition": args.condition,
            "condition_type": manifest["condition_type"],
            "target_digit": manifest["target_digit"],
            "seed": int(seed),
            "corruption_seed": int(manifest["corruption_seed"]),
            "corrupted_example_count": int(manifest["corrupted_example_count"]),
            "corrupted_fraction_of_training": float(manifest["corrupted_fraction_of_training"]),
            "best_epoch": int(row["best_epoch"]),
            "best_val_acc": float(row["best_val_acc"]),
            "best_val_loss": float(row["best_val_loss"]),
            "test_acc": float(row["test_acc"]),
            "test_balanced_class_acc": float(row["test_balanced_class_acc"]),
            "test_worst_class_acc": float(row["test_worst_class_acc"]),
            "test_loss": float(row["test_loss"]),
            "test_class_acc": [float(value) for value in class_acc.tolist()],
            "manifest_sha256": manifest["manifest_sha256"],
            "contract_sha256": contract["contract_sha256"],
            "checkpoint": checkpoint,
        }
        for digit, value in enumerate(class_acc.tolist()):
            result[f"test_class_{digit}_acc"] = float(value)
        _atomic_write_json(result_path, result)
        write_condition_summaries(
            condition_dir=condition_dir,
            condition=args.condition,
            seeds=seeds,
            contract=contract,
            manifest=manifest,
        )
        print(
            f"[SEED DONE] condition={args.condition} seed={seed} "
            f"best_val={result['best_val_acc']:.2f}% test={result['test_acc']:.2f}% "
            f"mean_class={result['test_balanced_class_acc']:.2f}% "
            f"worst_class={result['test_worst_class_acc']:.2f}%"
        )
        del state_dict, row
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

    write_condition_summaries(
        condition_dir=condition_dir,
        condition=args.condition,
        seeds=seeds,
        contract=contract,
        manifest=manifest,
    )
    summary = json.loads((condition_dir / "summary.json").read_text(encoding="utf-8"))
    if int(summary["n_completed"]) != len(seeds):
        raise RuntimeError(
            f"Condition ended with {summary['n_completed']}/{len(seeds)} completed seeds"
        )
    metrics = summary["metrics"]
    print(
        f"[SUMMARY] condition={args.condition} n={summary['n_completed']} "
        f"mean_class={metrics['test_balanced_class_acc']['mean']:.2f}+/-"
        f"{metrics['test_balanced_class_acc']['std']:.2f} "
        f"worst_class={metrics['test_worst_class_acc']['mean']:.2f}+/-"
        f"{metrics['test_worst_class_acc']['std']:.2f}"
    )
    print(f"[DONE] {condition_dir}")


if __name__ == "__main__":
    main()
