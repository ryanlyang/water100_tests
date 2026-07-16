#!/usr/bin/env python3
"""Fixed-hyperparameter CLIP+LogReg run for DecoyMNIST (no sweep).

Uses a CLIP backbone to extract features, then trains LogisticRegression with a
single fixed hyperparameter setting (defaults copied from your Waterbirds-100
best trial):
  - clip_model=RN50
  - C=0.2515000498909345
  - penalty=l2
  - solver=lbfgs
  - fit_intercept=True

Because DecoyMNIST PNG layout has train/test folders (no explicit val split),
this script creates an internal val split from train using a fixed split seed
(default: 0), mirroring your cdep-style Decoy setup behavior.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for s in _parse_csv_list(text):
        out.append(int(s))
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _try_import_clip(clip_repo: str = ""):
    try:
        import clip  # type: ignore

        return clip
    except Exception:
        pass

    candidates: List[Path] = []
    if clip_repo:
        candidates.append(Path(clip_repo).expanduser().resolve())

    here = _repo_root()
    candidates.extend(
        [
            here / "GALS",
            here.parent / "GALS",
            Path.cwd() / "GALS",
            Path.cwd().parent / "GALS",
        ]
    )

    seen = set()
    for c in candidates:
        c = c.resolve()
        if str(c) in seen:
            continue
        seen.add(str(c))
        if (c / "CLIP" / "clip" / "clip.py").exists():
            sys.path.insert(0, str(c))
            from CLIP.clip import clip  # type: ignore

            return clip

    raise ImportError(
        "Could not import CLIP. Install `clip` package, or pass --clip-repo "
        "to a repo root containing CLIP/clip/clip.py"
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(denom, eps, out=denom)
    x /= denom
    return x


def _class_acc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.zeros((num_classes,), dtype=np.float64)
    for c in range(num_classes):
        idx = np.where(y_true == c)[0]
        if idx.size == 0:
            acc[c] = float("nan")
            continue
        acc[c] = float(np.mean((y_pred[idx] == y_true[idx]).astype(np.float64)) * 100.0)
    return acc


def _fmt_arr(arr: np.ndarray) -> str:
    return np.array2string(arr, precision=2, separator=",")


def _extract_features(
    dataset: ImageFolder,
    model,
    device: str,
    batch_size: int,
    num_workers: int,
    desc: str = "features",
) -> Tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
        persistent_workers=(num_workers > 0),
    )

    X: Optional[np.ndarray] = None
    y_all = np.empty((len(dataset),), dtype=np.int64)
    offset = 0

    model.eval()
    with torch.no_grad():
        for images, y in tqdm(loader, desc=f"[CLIP-LR] Extracting {desc}", leave=False):
            images = images.to(device, non_blocking=True)
            f = model.encode_image(images).float()
            f = f / f.norm(dim=-1, keepdim=True)
            f_np = f.cpu().numpy().astype(np.float32, copy=False)
            bsz = f_np.shape[0]
            if X is None:
                X = np.empty((len(dataset), f_np.shape[1]), dtype=np.float32)
            X[offset:offset + bsz] = f_np
            y_all[offset:offset + bsz] = y.numpy().astype(np.int64, copy=False)
            offset += bsz

    if X is None:
        raise RuntimeError("No features extracted; dataset is empty.")
    if offset != len(dataset):
        raise RuntimeError(f"Feature extraction size mismatch: offset={offset}, n={len(dataset)}")

    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return X, y_all


def _split_train_val(n_total: int, val_frac: float, split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n_val = int(val_frac * n_total)
    if n_val < 1 or n_val >= n_total:
        raise ValueError(f"Invalid val size from val_frac={val_frac} with n_total={n_total}")
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    C: float,
    penalty: str,
    solver: str,
    fit_intercept: bool,
    l1_ratio: Optional[float],
    max_iter: int,
) -> Dict[str, object]:
    from sklearn.linear_model import LogisticRegression

    clf_kwargs = dict(
        random_state=seed,
        C=float(C),
        penalty=str(penalty),
        solver=str(solver),
        fit_intercept=bool(fit_intercept),
        max_iter=int(max_iter),
        n_jobs=1,
        verbose=0,
    )
    if l1_ratio is not None and penalty == "elasticnet":
        clf_kwargs["l1_ratio"] = float(l1_ratio)

    clf = LogisticRegression(**clf_kwargs)
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            clf.fit(X_train, y_train)
    except Exception:
        clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    n_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    val_group = _class_acc(y_val, val_pred, n_classes)
    test_group = _class_acc(y_test, test_pred, n_classes)

    return {
        "val_acc": float(np.mean((val_pred == y_val).astype(np.float64)) * 100.0),
        "val_avg_group_acc": float(np.nanmean(val_group)),
        "val_worst_group_acc": float(np.nanmin(val_group)),
        "val_group_accs": _fmt_arr(val_group),
        "test_acc": float(np.mean((test_pred == y_test).astype(np.float64)) * 100.0),
        "test_avg_group_acc": float(np.nanmean(test_group)),
        "test_worst_group_acc": float(np.nanmin(test_group)),
        "test_group_accs": _fmt_arr(test_group),
    }


def _write_rows(csv_path: str, rows: Iterable[Dict[str, object]], header: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed-hparam CLIP+LR for DecoyMNIST")
    p.add_argument("--png-root", type=str, default=None, help="Path to DecoyMNIST_png root")
    p.add_argument("--clip-model", type=str, default="RN50")
    p.add_argument("--clip-repo", type=str, default="")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4", help="LogReg random_state seeds")

    # Fixed LR settings from Waterbirds-100 best trial.
    p.add_argument("--C", type=float, default=0.2515000498909345)
    p.add_argument("--penalty", type=str, default="l2")
    p.add_argument("--solver", type=str, default="lbfgs")
    p.add_argument("--fit-intercept", action="store_true", default=True)
    p.add_argument("--no-fit-intercept", action="store_false", dest="fit_intercept")
    p.add_argument("--l1-ratio", type=float, default=None)
    p.add_argument("--max-iter", type=int, default=5000)

    p.add_argument("--output-csv", type=str, default="")
    args = p.parse_args()

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds is empty")

    repo_root = Path(__file__).resolve().parent
    png_root = Path(args.png_root).expanduser().resolve() if args.png_root else (repo_root / "data" / "DecoyMNIST_png")
    train_dir = png_root / "train"
    test_dir = png_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected train/test under {png_root}")

    clip_module = _try_import_clip(args.clip_repo)

    print("[INFO] DecoyMNIST fixed CLIP+LR")
    print(f"[INFO] png_root={png_root}")
    print(f"[INFO] clip_model={args.clip_model}")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] seeds={seeds}")
    print(
        f"[INFO] fixed params: C={args.C} penalty={args.penalty} solver={args.solver} "
        f"fit_intercept={args.fit_intercept}"
    )

    model, preprocess = clip_module.load(args.clip_model, device=args.device, jit=False)

    train_ds = ImageFolder(str(train_dir), transform=preprocess)
    test_ds = ImageFolder(str(test_dir), transform=preprocess)
    print(f"[INFO] n_train_full={len(train_ds)} n_test={len(test_ds)} classes={train_ds.classes}")

    _seed_everything(args.split_seed)
    X_train_full, y_train_full = _extract_features(
        train_ds,
        model,
        args.device,
        args.batch_size,
        args.num_workers,
        desc="train",
    )
    X_test, y_test = _extract_features(
        test_ds,
        model,
        args.device,
        args.batch_size,
        args.num_workers,
        desc="test",
    )

    # Free CLIP model before sklearn fit to reduce peak memory.
    del model
    if "cuda" in args.device and torch.cuda.is_available():
        torch.cuda.empty_cache()

    X_train_full = _l2_normalize(X_train_full)
    X_test = _l2_normalize(X_test)

    train_idx, val_idx = _split_train_val(len(train_ds), args.val_frac, args.split_seed)
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]
    print(f"[INFO] split: train={len(train_idx)} val={len(val_idx)} (split_seed={args.split_seed})")

    rows: List[Dict[str, object]] = []
    for i, seed in enumerate(seeds):
        out = _evaluate(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            seed=seed,
            C=args.C,
            penalty=args.penalty,
            solver=args.solver,
            fit_intercept=args.fit_intercept,
            l1_ratio=args.l1_ratio,
            max_iter=args.max_iter,
        )
        row = {
            "run_id": i,
            "seed": seed,
            "clip_model": args.clip_model,
            "C": args.C,
            "penalty": args.penalty,
            "solver": args.solver,
            "l1_ratio": "" if args.l1_ratio is None else args.l1_ratio,
            "fit_intercept": bool(args.fit_intercept),
            "split_seed": args.split_seed,
            "val_frac": args.val_frac,
            **out,
        }
        rows.append(row)
        print(
            f"[RUN {i}] seed={seed} val_acc={row['val_acc']:.4f} "
            f"val_avg_group_acc={row['val_avg_group_acc']:.4f} test_acc={row['test_acc']:.4f} "
            f"test_avg_group_acc={row['test_avg_group_acc']:.4f}"
        )

    for key in ("val_acc", "val_avg_group_acc", "val_worst_group_acc", "test_acc", "test_avg_group_acc", "test_worst_group_acc"):
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        print(f"[SUMMARY] {key}: {vals.mean():.4f} +/- {vals.std():.4f} (n={len(vals)})")

    if args.output_csv:
        _write_rows(
            args.output_csv,
            rows,
            header=[
                "run_id",
                "seed",
                "clip_model",
                "C",
                "penalty",
                "solver",
                "l1_ratio",
                "fit_intercept",
                "split_seed",
                "val_frac",
                "val_acc",
                "val_avg_group_acc",
                "val_worst_group_acc",
                "val_group_accs",
                "test_acc",
                "test_avg_group_acc",
                "test_worst_group_acc",
                "test_group_accs",
            ],
        )
        print(f"[INFO] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
