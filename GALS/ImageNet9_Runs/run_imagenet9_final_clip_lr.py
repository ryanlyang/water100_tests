#!/usr/bin/env python3
"""Evaluate the locked ImageNet-9 CLIP-RN50 logistic regressor over five seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from imagenet9_data import FORBIDDEN_SELECTION_VARIANTS, load_official_variant_samples
from imagenet9_final_utils import atomic_json, parse_seeds, write_method_tables
from sweep_imagenet9_clip_lr import _l2_normalize, _load_clip, _macro_accuracy


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument("--train-feature-cache", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-test-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["0,1,2,3,4"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_selection(args: argparse.Namespace) -> Mapping[str, object]:
    if not args.sweep_summary.is_file() or not args.train_feature_cache.is_file():
        raise FileNotFoundError(
            f"Missing CLIP-LR sweep artifacts: {args.sweep_summary}, {args.train_feature_cache}"
        )
    summary = json.loads(args.sweep_summary.read_text())
    if summary.get("method") != "clip_lr":
        raise RuntimeError(f"Unexpected CLIP-LR summary method: {summary.get('method')}")
    if int(summary.get("complete_trials", 0)) < int(summary.get("target_complete_trials", 50)):
        raise RuntimeError(f"CLIP-LR sweep is incomplete: {args.sweep_summary}")
    if summary.get("official_variants_used_for_selection") is not False:
        raise RuntimeError("CLIP-LR summary does not certify held-out official variants")
    return summary


def build_official_feature_cache(args: argparse.Namespace, cache_path: Path) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    contract = {
        "official_manifest": str(args.official_manifest.resolve()),
        "official_manifest_sha256": sha256(args.official_manifest),
        "official_test_root": str(args.official_test_root.resolve()),
        "model": "openai_clip_RN50",
        "feature_mode": "l2",
        "variants": list(FORBIDDEN_SELECTION_VARIANTS),
    }
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if cache_path.is_file() and metadata_path.is_file():
        if json.loads(metadata_path.read_text()).get("contract") != contract:
            raise RuntimeError("Refusing to use changed official CLIP feature cache")
        print(f"[CACHE] loaded {cache_path}", flush=True)
        return

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
    model, preprocess = clip.load("RN50", device=args.device, jit=False)
    model.eval()
    arrays: Dict[str, np.ndarray] = {}
    for variant in FORBIDDEN_SELECTION_VARIANTS:
        samples = load_official_variant_samples(
            args.official_manifest,
            args.official_test_root,
            variant,
            verify_files=True,
        )
        loader = DataLoader(
            ClipDataset(samples, preprocess),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        features: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        with torch.no_grad():
            for images, targets in loader:
                encoded = model.encode_image(
                    images.to(args.device, non_blocking=True)
                ).float()
                features.append(encoded.cpu().numpy())
                labels.append(targets.numpy())
        arrays[f"{variant}_features"] = _l2_normalize(
            np.concatenate(features)
        ).astype(np.float32, copy=False)
        arrays[f"{variant}_labels"] = np.concatenate(labels).astype(
            np.int64, copy=False
        )
        print(f"[CACHE] variant={variant} samples={len(samples)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(cache_path)
    atomic_json(metadata_path, {"contract": contract})
    print(f"[CACHE] wrote {cache_path}", flush=True)


def atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    selection = load_selection(args)
    c_value = float(selection["best_params"]["C"])
    contract = selection["contract"]
    final_contract = {
        "method": "clip_lr",
        "source_summary": str(args.sweep_summary.resolve()),
        "source_feature_cache": str(args.train_feature_cache.resolve()),
        "C": c_value,
        "solver": contract["solver"],
        "penalty": contract["penalty"],
        "fit_intercept": contract["fit_intercept"],
        "tol": contract["tol"],
        "max_iter": contract["max_iter"],
        "seeds": seeds,
        "official_variants_used_for_selection": False,
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.run_root / "run_contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != final_contract:
        raise RuntimeError(f"Refusing to resume changed CLIP-LR contract: {contract_path}")
    if not contract_path.is_file():
        atomic_json(contract_path, final_contract)

    with np.load(args.train_feature_cache) as payload:
        train_features = payload["train_features"]
        train_labels = payload["train_labels"]
    official_cache = args.run_root / "official_features.npz"
    build_official_feature_cache(args, official_cache)
    official = np.load(official_cache)

    from sklearn.linear_model import LogisticRegression

    evaluations: List[Path] = []
    for seed in seeds:
        seed_root = args.run_root / f"seed_{seed}"
        checkpoint = seed_root / "classifier.pkl"
        evaluation_json = seed_root / "official_evaluation.json"
        if not evaluation_json.is_file():
            seed_everything(seed)
            classifier = LogisticRegression(
                C=c_value,
                penalty=str(contract["penalty"]),
                solver=str(contract["solver"]),
                fit_intercept=bool(contract["fit_intercept"]),
                tol=float(contract["tol"]),
                max_iter=int(contract["max_iter"]),
                random_state=seed,
                multi_class="auto",
            )
            classifier.fit(train_features, train_labels)
            atomic_pickle(checkpoint, classifier)
            variant_results: Dict[str, object] = {}
            for variant in FORBIDDEN_SELECTION_VARIANTS:
                features = official[f"{variant}_features"]
                labels = official[f"{variant}_labels"]
                predictions = classifier.predict(features)
                macro, per_class = _macro_accuracy(labels, predictions)
                accuracy = float(np.mean(predictions == labels))
                variant_results[variant] = {
                    "accuracy": accuracy,
                    "macro_class_accuracy": macro,
                    "per_class_accuracy": per_class,
                    "class_support": [int(np.sum(labels == index)) for index in range(9)],
                    "samples": int(labels.shape[0]),
                }
                print(
                    f"[EVAL] method=clip_lr seed={seed} variant={variant} "
                    f"acc={100.0 * accuracy:.2f} macro={100.0 * macro:.2f}",
                    flush=True,
                )
            atomic_json(
                evaluation_json,
                {
                    "method": "clip_lr",
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "selection_objective": "val_macro_class_accuracy",
                    "selection_value": selection["best_value"],
                    "official_variants_used_for_selection": False,
                    "variant_results": variant_results,
                },
            )
        else:
            print(f"[RESUME] method=clip_lr seed={seed}", flush=True)
        evaluations.append(evaluation_json)
        write_method_tables("clip_lr", args.run_root, evaluations)
        print(f"[SEED DONE] method=clip_lr seed={seed}", flush=True)
    official.close()
    print(f"[DONE] {args.run_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
