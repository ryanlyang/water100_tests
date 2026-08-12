#!/usr/bin/env python3
"""Cache OpenAI CLIP RN50 features and tune only logistic-regression C on IN-9."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


OBJECTIVE_NAME = "val_macro_class_accuracy"
MODEL_NAME = "RN50"
FEATURE_MODE = "l2"
SOLVER = "lbfgs"
PENALTY = "l2"
FIT_INTERCEPT = True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--study-db", type=Path, required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--target-complete-trials", type=int, default=50)
    parser.add_argument("--sweep-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--c-min", type=float, default=1e-2)
    parser.add_argument("--c-max", type=float, default=1e2)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--max-hours", type=float, default=94.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_clip():
    try:
        import clip  # type: ignore

        return clip
    except Exception:
        repo = Path(__file__).resolve().parents[1]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from CLIP.clip import clip  # type: ignore

        return clip


def _l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denominator = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), eps)
    return features / denominator


def _cache_contract(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "model": "openai_clip_RN50",
        "feature_mode": FEATURE_MODE,
        "train_split": "train",
        "validation_split": "val",
        "official_variants_used": False,
    }


def _extract_features(args: argparse.Namespace) -> Tuple[np.ndarray, ...]:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from imagenet9_data import load_original_samples

    class ClipDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            sample = self.samples[index]
            with Image.open(sample.path) as image_file:
                image = self.transform(image_file.convert("RGB"))
            return image, sample.label

    clip = _load_clip()
    model, preprocess = clip.load(MODEL_NAME, device=args.device, jit=False)
    model.eval()

    def encode(split: str) -> Tuple[np.ndarray, np.ndarray]:
        samples = load_original_samples(args.manifest, split, verify_files=True)
        loader = DataLoader(
            ClipDataset(samples, preprocess),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        feature_batches: List[np.ndarray] = []
        label_batches: List[np.ndarray] = []
        with torch.no_grad():
            for images, labels in loader:
                encoded = model.encode_image(images.to(args.device, non_blocking=True)).float()
                feature_batches.append(encoded.cpu().numpy())
                label_batches.append(labels.numpy())
        features = np.concatenate(feature_batches).astype(np.float32, copy=False)
        labels = np.concatenate(label_batches).astype(np.int64, copy=False)
        return _l2_normalize(features).astype(np.float32, copy=False), labels

    train_features, train_labels = encode("train")
    val_features, val_labels = encode("val")
    del model
    torch.cuda.empty_cache()
    return train_features, train_labels, val_features, val_labels


def _load_or_build_cache(args: argparse.Namespace) -> Tuple[np.ndarray, ...]:
    contract = _cache_contract(args)
    metadata_path = args.feature_cache.with_suffix(args.feature_cache.suffix + ".json")
    if args.feature_cache.is_file() and metadata_path.is_file():
        stored = json.loads(metadata_path.read_text())
        if stored.get("contract") != contract:
            raise RuntimeError("Refusing to use a CLIP feature cache with a changed contract")
        with np.load(args.feature_cache) as payload:
            arrays = tuple(
                payload[name]
                for name in ("train_features", "train_labels", "val_features", "val_labels")
            )
        print(f"[CACHE] loaded {args.feature_cache}", flush=True)
        return arrays

    print("[CACHE] extracting OpenAI CLIP RN50 train/validation features", flush=True)
    arrays = _extract_features(args)
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.feature_cache.with_suffix(args.feature_cache.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        train_features=arrays[0],
        train_labels=arrays[1],
        val_features=arrays[2],
        val_labels=arrays[3],
    )
    temporary.replace(args.feature_cache)
    metadata = {
        "contract": contract,
        "train_shape": list(arrays[0].shape),
        "val_shape": list(arrays[2].shape),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[CACHE] wrote {args.feature_cache}", flush=True)
    return arrays


def _macro_accuracy(labels: np.ndarray, predictions: np.ndarray, num_classes: int = 9) -> Tuple[float, List[float]]:
    per_class = []
    for label in range(num_classes):
        mask = labels == label
        if not np.any(mask):
            raise RuntimeError(f"Validation split is missing class {label}")
        per_class.append(float(np.mean(predictions[mask] == labels[mask])))
    return float(np.mean(per_class)), per_class


def _write_outputs(study, args: argparse.Namespace, contract: Dict[str, object]) -> None:
    import optuna

    rows = []
    for trial in study.get_trials(deepcopy=False):
        rows.append(
            {
                "trial": trial.number,
                "state": trial.state.name,
                "objective": trial.value if trial.state == optuna.trial.TrialState.COMPLETE else "",
                "C": trial.params.get("C", ""),
                "val_accuracy": trial.user_attrs.get("val_accuracy", ""),
                "val_per_class_accuracy": json.dumps(trial.user_attrs.get("val_per_class_accuracy", [])),
                "seconds": trial.user_attrs.get("seconds", ""),
            }
        )
    fieldnames = ["trial", "state", "objective", "C", "val_accuracy", "val_per_class_accuracy", "seconds"]
    temporary_csv = args.output_csv.with_suffix(args.output_csv.suffix + ".tmp")
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(args.output_csv)

    complete = [trial for trial in study.get_trials(deepcopy=False) if trial.state == optuna.trial.TrialState.COMPLETE]
    summary: Dict[str, object] = {
        "method": "clip_lr",
        "study_name": study.study_name,
        "target_complete_trials": args.target_complete_trials,
        "complete_trials": len(complete),
        "objective": OBJECTIVE_NAME,
        "contract": contract,
        "official_variants_used_for_selection": False,
    }
    if complete:
        summary.update(
            best_trial=study.best_trial.number,
            best_value=study.best_value,
            best_params=study.best_params,
            best_user_attrs=study.best_trial.user_attrs,
        )
    temporary_json = args.summary_json.with_suffix(args.summary_json.suffix + ".tmp")
    temporary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_json.replace(args.summary_json)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.c_min <= 0 or args.c_max <= args.c_min:
        raise ValueError("Require 0 < --c-min < --c-max")
    for path in (args.study_db.parent, args.output_csv.parent, args.summary_json.parent):
        path.mkdir(parents=True, exist_ok=True)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    _seed_everything(args.train_seed)
    train_features, train_labels, val_features, val_labels = _load_or_build_cache(args)

    import optuna
    from sklearn.linear_model import LogisticRegression

    contract = {
        **_cache_contract(args),
        "C": [args.c_min, args.c_max, "log"],
        "solver": SOLVER,
        "penalty": PENALTY,
        "fit_intercept": FIT_INTERCEPT,
        "tol": args.tol,
        "max_iter": args.max_iter,
        "objective": OBJECTIVE_NAME,
    }
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{args.study_db.resolve()}",
        engine_kwargs={"connect_args": {"timeout": 60}},
        heartbeat_interval=60,
        grace_period=600,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=args.sweep_seed),
        direction="maximize",
        load_if_exists=True,
    )
    if hasattr(optuna.storages, "fail_stale_trials"):
        optuna.storages.fail_stale_trials(study)
    stored_hash = study.user_attrs.get("contract_sha256")
    if stored_hash is not None and stored_hash != contract_hash:
        raise RuntimeError("Refusing to resume CLIP-LR with a changed experiment contract")
    if stored_hash is None:
        study.set_user_attr("contract_sha256", contract_hash)
        study.set_user_attr("contract", contract)
        if not study.get_trials(deepcopy=False):
            study.enqueue_trial({"C": 1.0})

    def objective(trial) -> float:
        start = time.time()
        c_value = float(trial.suggest_float("C", args.c_min, args.c_max, log=True))
        classifier = LogisticRegression(
            C=c_value,
            penalty=PENALTY,
            solver=SOLVER,
            fit_intercept=FIT_INTERCEPT,
            tol=args.tol,
            max_iter=args.max_iter,
            random_state=args.train_seed,
            multi_class="auto",
        )
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(val_features)
        accuracy = float(np.mean(predictions == val_labels))
        macro, per_class = _macro_accuracy(val_labels, predictions)
        trial.set_user_attr("val_accuracy", accuracy)
        trial.set_user_attr("val_per_class_accuracy", per_class)
        trial.set_user_attr("seconds", time.time() - start)
        print(
            f"[TRIAL {trial.number}] C={c_value:.9g} val_acc={accuracy:.6f} "
            f"val_macro={macro:.6f}",
            flush=True,
        )
        return macro

    def persist(current_study, _trial) -> None:
        _write_outputs(current_study, args, contract)

    complete_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.get_trials(deepcopy=False)
    )
    remaining = max(args.target_complete_trials - complete_before, 0)
    print(
        f"[STUDY] clip_lr complete={complete_before}/{args.target_complete_trials} "
        f"cache={args.feature_cache}",
        flush=True,
    )
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=args.max_hours * 3600,
            callbacks=[persist],
            gc_after_trial=True,
        )
    _write_outputs(study, args, contract)
    complete_after = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.get_trials(deepcopy=False)
    )
    print(f"[DONE] clip_lr complete={complete_after}/{args.target_complete_trials}", flush=True)
    if complete_after < args.target_complete_trials:
        print("[INCOMPLETE] Resubmit the same stable study to continue.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
