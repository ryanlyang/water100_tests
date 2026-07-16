#!/usr/bin/env python3
"""Fixed-C CLIP ViT + LogisticRegression on DecoyMNIST (no sweep)."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _add_repo_to_syspath() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _try_import_clip():
    try:
        import clip  # type: ignore

        return clip
    except Exception:
        _add_repo_to_syspath()
        from CLIP.clip import clip  # type: ignore

        return clip


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _class_acc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.zeros((num_classes,), dtype=np.float64)
    for c in range(num_classes):
        idx = np.where(y_true == c)[0]
        if idx.size == 0:
            acc[c] = float("nan")
        else:
            acc[c] = float(np.mean((y_pred[idx] == y_true[idx]).astype(np.float64)) * 100.0)
    return acc


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def _nanmin(x: np.ndarray) -> float:
    return float(np.nanmin(x))


def _fmt_arr(arr: np.ndarray) -> str:
    return np.array2string(arr, precision=2, separator=",")


@dataclass(frozen=True)
class _FeatPack:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


class _ClipSubset(Dataset):
    def __init__(self, subset: Subset, preprocess):
        self.subset = subset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        x = self.preprocess(img)
        return x, int(label)


class _ClipFolder(Dataset):
    def __init__(self, folder: ImageFolder, preprocess):
        self.folder = folder
        self.preprocess = preprocess

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        img, label = self.folder[idx]
        x = self.preprocess(img)
        return x, int(label)


def _extract_features(ds: Dataset, model, device: str, batch_size: int, num_workers: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
        worker_init_fn=_seed_worker,
        generator=g,
    )
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    X = np.concatenate(feats, axis=0).astype(np.float64, copy=False)
    y = np.concatenate(labels, axis=0).astype(np.int64, copy=False)
    X = np.ascontiguousarray(X)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return X, y


def _safe_fit(clf, X_train: np.ndarray, y_train: np.ndarray) -> None:
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            clf.fit(X_train, y_train)
    except Exception:
        clf.fit(X_train, y_train)


def _eval_split(X: np.ndarray, y: np.ndarray, clf, num_classes: int) -> Dict[str, object]:
    pred = clf.predict(X)
    acc = float(np.mean((pred == y).astype(np.float64)) * 100.0)
    class_acc = _class_acc(y, pred, num_classes=num_classes)
    return {
        "acc": acc,
        "balanced_class_acc": _nanmean(class_acc),
        "worst_class_acc": _nanmin(class_acc),
        "class_accs": _fmt_arr(class_acc),
    }


def _write_rows(csv_path: str, rows: Iterable[Dict], header: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _prepare_splits(png_root: str, val_frac: float, split_seed: int):
    train_dir = os.path.join(png_root, "train")
    test_dir = os.path.join(png_root, "test")
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Missing train dir: {train_dir}")
    if not os.path.isdir(test_dir):
        raise RuntimeError(f"Missing test dir: {test_dir}")

    full_train = ImageFolder(train_dir)
    test_ds = ImageFolder(test_dir)
    if full_train.classes != test_ds.classes:
        raise RuntimeError(f"Class mismatch train={full_train.classes} test={test_ds.classes}")

    n_total = len(full_train)
    n_val = int(val_frac * n_total)
    n_train = n_total - n_val
    split_g = torch.Generator().manual_seed(split_seed)
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=split_g)
    return full_train.classes, train_subset, val_subset, test_ds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DecoyMNIST CLIP ViT + fixed-C LR (no sweep).")
    p.add_argument("--png-root", default="/workspace/Waterbird_Runs/MakeMNIST/data/DecoyMNIST_png")
    p.add_argument("--clip-model", default="ViT-B/32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--C", type=float, default=0.16412713188550926)
    p.add_argument("--fit-intercept", type=int, default=1, choices=[0, 1])
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--output-csv", default="decoy_clip_vit_lr_fixed.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clip = _try_import_clip()
    _seed_everything(args.seed_start)

    classes, train_subset, val_subset, test_ds = _prepare_splits(
        png_root=args.png_root,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
    )
    num_classes = len(classes)

    try:
        model, preprocess = clip.load(args.clip_model, device=args.device, jit=False)
    except TypeError:
        model, preprocess = clip.load(args.clip_model, device=args.device)

    print("Running DecoyMNIST CLIP ViT LR fixed-C")
    print(f"device={args.device}")
    print(f"png_root={args.png_root}")
    print(f"clip_model={args.clip_model}")
    print(
        f"train={len(train_subset)} val={len(val_subset)} test={len(test_ds)} "
        f"split={1.0 - args.val_frac:.2f}/{args.val_frac:.2f} split_seed={args.split_seed}"
    )
    print(f"LR fixed: C={args.C} penalty=l2 solver=lbfgs fit_intercept={bool(args.fit_intercept)} tol={args.tol}")

    t0 = time.time()
    train_clip = _ClipSubset(train_subset, preprocess)
    val_clip = _ClipSubset(val_subset, preprocess)
    test_clip = _ClipFolder(test_ds, preprocess)

    X_train, y_train = _extract_features(train_clip, model, args.device, args.batch_size, args.num_workers, args.seed_start)
    X_val, y_val = _extract_features(val_clip, model, args.device, args.batch_size, args.num_workers, args.seed_start)
    X_test, y_test = _extract_features(test_clip, model, args.device, args.batch_size, args.num_workers, args.seed_start)

    X_train = np.ascontiguousarray(_l2_normalize(X_train), dtype=np.float64)
    X_val = np.ascontiguousarray(_l2_normalize(X_val), dtype=np.float64)
    X_test = np.ascontiguousarray(_l2_normalize(X_test), dtype=np.float64)
    print(f"[CLIP] feature extraction done in {(time.time() - t0):.1f}s")

    from sklearn.linear_model import LogisticRegression

    rows: List[Dict] = []
    for i in range(args.n_seeds):
        seed = args.seed_start + i
        clf = LogisticRegression(
            random_state=seed,
            C=float(args.C),
            penalty="l2",
            solver="lbfgs",
            fit_intercept=bool(args.fit_intercept),
            max_iter=args.max_iter,
            tol=float(args.tol),
            n_jobs=1,
            verbose=0,
        )
        t_fit = time.time()
        _safe_fit(clf, X_train, y_train)
        val_metrics = _eval_split(X_val, y_val, clf, num_classes)
        test_metrics = _eval_split(X_test, y_test, clf, num_classes)
        row = {
            "seed": seed,
            "clip_model": args.clip_model,
            "C": float(args.C),
            "penalty": "l2",
            "solver": "lbfgs",
            "fit_intercept": bool(args.fit_intercept),
            "tol": float(args.tol),
            "val_acc": val_metrics["acc"],
            "val_balanced_class_acc": val_metrics["balanced_class_acc"],
            "val_worst_class_acc": val_metrics["worst_class_acc"],
            "val_class_accs": val_metrics["class_accs"],
            "test_acc": test_metrics["acc"],
            "test_balanced_class_acc": test_metrics["balanced_class_acc"],
            "test_worst_class_acc": test_metrics["worst_class_acc"],
            "test_class_accs": test_metrics["class_accs"],
            "seconds": int(time.time() - t_fit),
        }
        rows.append(row)
        print(
            f"seed={seed} val_acc={row['val_acc']:.2f}% val_worst={row['val_worst_class_acc']:.2f}% "
            f"test_acc={row['test_acc']:.2f}% test_worst={row['test_worst_class_acc']:.2f}%"
        )

    header = [
        "seed",
        "clip_model",
        "C",
        "penalty",
        "solver",
        "fit_intercept",
        "tol",
        "val_acc",
        "val_balanced_class_acc",
        "val_worst_class_acc",
        "val_class_accs",
        "test_acc",
        "test_balanced_class_acc",
        "test_worst_class_acc",
        "test_class_accs",
        "seconds",
    ]
    _write_rows(args.output_csv, rows, header)

    vals = np.array([float(r["val_acc"]) for r in rows], dtype=float)
    vals_w = np.array([float(r["val_worst_class_acc"]) for r in rows], dtype=float)
    tests = np.array([float(r["test_acc"]) for r in rows], dtype=float)
    tests_w = np.array([float(r["test_worst_class_acc"]) for r in rows], dtype=float)
    print("\nSummary over seeds")
    print(f"val_acc mean={np.mean(vals):.2f}% std={np.std(vals):.2f}%")
    print(f"val_worst_class_acc mean={np.mean(vals_w):.2f}% std={np.std(vals_w):.2f}%")
    print(f"test_acc mean={np.mean(tests):.2f}% std={np.std(tests):.2f}%")
    print(f"test_worst_class_acc mean={np.mean(tests_w):.2f}% std={np.std(tests_w):.2f}%")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
