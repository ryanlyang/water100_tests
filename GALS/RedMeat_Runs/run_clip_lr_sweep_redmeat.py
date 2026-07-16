#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


_PENALTY_SOLVER_BASE_CHOICES = [
    ("l2", "lbfgs", None),
    ("l2", "liblinear", None),
    ("l2", "saga", None),
    ("l1", "liblinear", None),
    ("l1", "saga", None),
    ("elasticnet", "saga", "suggest"),
]
_PENALTY_SOLVER_SPEC_DEFAULT = "l2:lbfgs,l2:liblinear,l2:saga,l1:liblinear,l1:saga,elasticnet:saga"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _add_repo_to_syspath():
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


def _write_row(csv_path: str, row: Dict, header: Iterable[str]) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _print_runtime_summary(tag: str, rows: Iterable[Dict], num_epochs: Optional[int] = None) -> None:
    secs = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
    if not secs:
        print(f"[TIME] {tag}: no successful trials to summarize.")
        return
    arr = np.array(secs, dtype=float)
    med_trial_min = float(np.median(arr) / 60.0)
    total_gpu_hours = float(np.sum(arr) / 3600.0)
    print(f"[TIME] {tag}: median min/trial={med_trial_min:.4f} | total tuning GPU-hours={total_gpu_hours:.4f}")
    if num_epochs is not None and num_epochs > 0:
        med_epoch_min = float(np.median(arr / float(num_epochs)) / 60.0)
        print(f"[TIME] {tag}: median min/epoch={med_epoch_min:.4f} (epochs/trial={int(num_epochs)})")
    else:
        print(f"[TIME] {tag}: median min/epoch=N/A (no epoch notion for CLIP+LR)")


def _print_metric_mean_std(tag: str, rows: Iterable[Dict], metric_keys: Iterable[str]) -> None:
    rows = list(rows)
    if not rows:
        return
    print(f"[{tag}] Mean/std over runs:")
    for key in metric_keys:
        vals = [r.get(key) for r in rows]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        print(f"  {key}: {float(np.mean(arr)):.4f} +/- {float(np.std(arr)):.4f} (n={arr.size})")


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


def _resolve_image_path(dataset_root: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    rel = str(rel_or_abs).lstrip("/")
    return os.path.join(dataset_root, rel)


@dataclass(frozen=True)
class _Sample:
    path: str
    label: int


class _ImageDataset:
    def __init__(self, samples: List[_Sample], preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.preprocess(img)
        return x, s.label


def _extract_features(samples: List[_Sample], model, preprocess, device: str, batch_size: int, num_workers: int):
    import torch
    from torch.utils.data import DataLoader

    ds = _ImageDataset(samples, preprocess)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
    )

    feats = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = model.encode_image(x)
            f = f.float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())

    X = np.concatenate(feats, axis=0).astype(np.float64, copy=False)
    X = np.ascontiguousarray(X, dtype=np.float64)
    # Guard sklearn/native solver calls from non-finite values that can trigger
    # low-level crashes in some BLAS/solver stacks.
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    return X, y


def _safe_fit(clf, X_train: np.ndarray, y_train: np.ndarray):
    # Force a single-threaded BLAS/OpenMP context during fit to reduce
    # segfault risk on shared cluster nodes.
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            clf.fit(X_train, y_train)
    except Exception:
        clf.fit(X_train, y_train)


def _build_splits(
    dataset_path: str,
    split_col: str,
    label_col: str,
    path_col: str,
    train_value: str,
    val_value: str,
    test_value: str,
    classes: Optional[List[str]],
):
    meta = os.path.join(dataset_path, "all_images.csv")
    if not os.path.exists(meta):
        raise FileNotFoundError(f"Missing all_images.csv at: {meta}")

    df = pd.read_csv(meta)
    for c in (split_col, label_col, path_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {meta}. Columns={list(df.columns)}")

    if classes is None:
        classes = sorted(df[label_col].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    unknown = sorted(set(df[label_col].astype(str).unique().tolist()) - set(class_to_idx.keys()))
    if unknown:
        raise ValueError(f"Found labels not in class list: {unknown}")

    def make_samples(split_value: str) -> List[_Sample]:
        d = df[df[split_col].astype(str) == str(split_value)]
        out = []
        for _, r in d.iterrows():
            p = _resolve_image_path(dataset_path, str(r[path_col]))
            out.append(_Sample(path=p, label=class_to_idx[str(r[label_col])]))
        return out

    train_samples = make_samples(train_value)
    val_samples = make_samples(val_value)
    test_samples = make_samples(test_value)

    if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        raise ValueError(
            "One or more splits are empty. "
            f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
        )

    return classes, train_samples, val_samples, test_samples


def _parse_penalty_solver_choices(spec: str) -> List[Tuple[str, str, Optional[str]]]:
    allowed_pairs = {(p, s) for p, s, _ in _PENALTY_SOLVER_BASE_CHOICES}
    out: List[Tuple[str, str, Optional[str]]] = []
    seen = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(
                f"Invalid penalty-solver token '{tok}'. Expected '<penalty>:<solver>' (e.g., l2:lbfgs)."
            )
        penalty, solver = [x.strip() for x in tok.split(":", 1)]
        key = (penalty, solver)
        if key not in allowed_pairs:
            raise ValueError(
                f"Unsupported penalty/solver pair '{penalty}:{solver}'. "
                "Allowed: l2:lbfgs, l2:liblinear, l2:saga, l1:liblinear, l1:saga, elasticnet:saga"
            )
        if key in seen:
            continue
        seen.add(key)
        if key == ("elasticnet", "saga"):
            out.append((penalty, solver, "suggest"))
        else:
            out.append((penalty, solver, None))
    if not out:
        raise ValueError("No valid penalty/solver choices parsed from --penalty-solvers.")
    return out


def _choice_id(choice: Tuple[str, str, Optional[str]]) -> str:
    penalty, solver, l1_ratio = choice
    ratio_tag = "suggest" if l1_ratio == "suggest" else "none"
    return f"{penalty}|{solver}|{ratio_tag}"


def _parse_feature_modes(spec: str) -> List[str]:
    allowed = {"l2", "raw", "zscore"}
    out: List[str] = []
    seen = set()
    for tok in str(spec).split(","):
        m = tok.strip().lower()
        if not m:
            continue
        if m not in allowed:
            raise ValueError(f"Unsupported feature mode '{m}'. Allowed: raw,l2,zscore")
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    if not out:
        raise ValueError("No valid feature modes parsed from --feature-modes.")
    return out


def _parse_class_weight_choices(spec: str) -> List[Optional[str]]:
    allowed = {"none", "balanced"}
    out: List[Optional[str]] = []
    seen = set()
    for tok in str(spec).split(","):
        v = tok.strip().lower()
        if not v:
            continue
        if v not in allowed:
            raise ValueError(f"Unsupported class weight option '{v}'. Allowed: none,balanced")
        if v in seen:
            continue
        seen.add(v)
        out.append(None if v == "none" else "balanced")
    if not out:
        raise ValueError("No valid class weight choices parsed from --class-weight-options.")
    return out


def _ensure_finite_contiguous(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return X


def _sample_penalty_solver_random(rng: np.random.Generator, choices) -> Tuple[str, str, Optional[float]]:
    penalty, solver, l1_ratio = choices[int(rng.integers(0, len(choices)))]
    if l1_ratio == "suggest":
        l1_ratio = float(rng.uniform(0.05, 0.95))
    return penalty, solver, l1_ratio


def _suggest_float_or_const(trial, name: str, low: float, high: float, *, log: bool = False) -> float:
    # Allow fixed-value "sweeps" where low == high.
    if float(low) == float(high):
        return float(low)
    return float(trial.suggest_float(name, low, high, log=log))


def _run_trial(
    trial_id: int,
    sampler: str,
    rng: np.random.Generator,
    args,
    num_classes: int,
    X_train_by_mode: Dict[str, np.ndarray],
    y_train: np.ndarray,
    X_val_by_mode: Dict[str, np.ndarray],
    y_val: np.ndarray,
    X_test_by_mode: Dict[str, np.ndarray],
    y_test: np.ndarray,
):
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    if sampler == "random":
        # Uniform random sampling over the configured search space.
        C = float(np.exp(rng.uniform(np.log(args.C_min), np.log(args.C_max))))
        fit_intercept = bool(rng.integers(0, 2))
        penalty, solver, l1_ratio = _sample_penalty_solver_random(rng, args.penalty_solver_choices)
        feature_mode = str(args.feature_modes[int(rng.integers(0, len(args.feature_modes)))])
        class_weight = args.class_weight_choices[int(rng.integers(0, len(args.class_weight_choices)))]
        tol = float(np.exp(rng.uniform(np.log(args.tol_min), np.log(args.tol_max))))
    else:
        trial = args.trial
        C = _suggest_float_or_const(trial, "C", args.C_min, args.C_max, log=True)
        fit_intercept = bool(trial.suggest_categorical("fit_intercept", [True, False]))
        choice_id = str(trial.suggest_categorical("penalty_solver", args.penalty_solver_ids))
        penalty, solver, l1_spec = args.penalty_solver_by_id[choice_id]
        if l1_spec == "suggest":
            l1_ratio = float(trial.suggest_float("l1_ratio", 0.05, 0.95))
        else:
            l1_ratio = None
        feature_mode = str(trial.suggest_categorical("feature_mode", args.feature_modes))
        class_weight_name = str(trial.suggest_categorical("class_weight", args.class_weight_option_names))
        class_weight = None if class_weight_name == "none" else "balanced"
        tol = _suggest_float_or_const(trial, "tol", args.tol_min, args.tol_max, log=True)

    X_train = X_train_by_mode[feature_mode]
    X_val = X_val_by_mode[feature_mode]
    X_test = X_test_by_mode[feature_mode]

    clf_kwargs = dict(
        random_state=args.seed,
        C=C,
        penalty=penalty,
        solver=solver,
        fit_intercept=fit_intercept,
        max_iter=args.max_iter,
        tol=tol,
        class_weight=class_weight,
        n_jobs=1,
        verbose=0,
    )
    if l1_ratio is not None and penalty == "elasticnet":
        clf_kwargs["l1_ratio"] = float(l1_ratio)

    clf = LogisticRegression(**clf_kwargs)
    _safe_fit(clf, X_train, y_train)

    val_pred = clf.predict(X_val)
    val_acc = float(np.mean((val_pred == y_val).astype(np.float64)) * 100.0)
    val_class = _class_acc(y_val, val_pred, num_classes=num_classes)
    val_avg_group = _nanmean(val_class)
    val_worst_group = _nanmin(val_class)

    test_pred = clf.predict(X_test)
    test_acc = float(np.mean((test_pred == y_test).astype(np.float64)) * 100.0)
    test_class = _class_acc(y_test, test_pred, num_classes=num_classes)
    test_avg_group = _nanmean(test_class)
    test_worst_group = _nanmin(test_class)

    row = {
        "trial": trial_id,
        "clip_model": args.clip_model,
        "C": C,
        "penalty": penalty,
        "solver": solver,
        "l1_ratio": ("" if l1_ratio is None else float(l1_ratio)),
        "fit_intercept": fit_intercept,
        "feature_mode": feature_mode,
        "tol": tol,
        "class_weight": ("none" if class_weight is None else str(class_weight)),
        "val_acc": val_acc,
        "val_avg_group_acc": val_avg_group,
        "val_worst_group_acc": val_worst_group,
        "val_group_accs": np.array2string(val_class, precision=2, separator=","),
        "test_acc": test_acc,
        "test_avg_group_acc": test_avg_group,
        "test_worst_group_acc": test_worst_group,
        "test_group_accs": np.array2string(test_class, precision=2, separator=","),
        "sampler": sampler,
        "seconds": int(time.time() - t0),
    }
    return row


def _run_fixed_params(
    *,
    run_id: int,
    run_label: str,
    seed: int,
    C: float,
    fit_intercept: bool,
    penalty: str,
    solver: str,
    l1_ratio: Optional[float],
    feature_mode: str,
    tol: float,
    class_weight: Optional[str],
    args,
    num_classes: int,
    X_train_by_mode: Dict[str, np.ndarray],
    y_train: np.ndarray,
    X_val_by_mode: Dict[str, np.ndarray],
    y_val: np.ndarray,
    X_test_by_mode: Dict[str, np.ndarray],
    y_test: np.ndarray,
):
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    X_train = X_train_by_mode[feature_mode]
    X_val = X_val_by_mode[feature_mode]
    X_test = X_test_by_mode[feature_mode]

    clf_kwargs = dict(
        random_state=seed,
        C=C,
        penalty=penalty,
        solver=solver,
        fit_intercept=fit_intercept,
        max_iter=args.max_iter,
        tol=tol,
        class_weight=class_weight,
        n_jobs=1,
        verbose=0,
    )
    if l1_ratio is not None and penalty == "elasticnet":
        clf_kwargs["l1_ratio"] = float(l1_ratio)

    clf = LogisticRegression(**clf_kwargs)
    _safe_fit(clf, X_train, y_train)

    val_pred = clf.predict(X_val)
    val_acc = float(np.mean((val_pred == y_val).astype(np.float64)) * 100.0)
    val_class = _class_acc(y_val, val_pred, num_classes=num_classes)
    val_avg_group = _nanmean(val_class)
    val_worst_group = _nanmin(val_class)

    test_pred = clf.predict(X_test)
    test_acc = float(np.mean((test_pred == y_test).astype(np.float64)) * 100.0)
    test_class = _class_acc(y_test, test_pred, num_classes=num_classes)
    test_avg_group = _nanmean(test_class)
    test_worst_group = _nanmin(test_class)

    return {
        "run_id": run_id,
        "run_label": run_label,
        "seed": seed,
        "clip_model": args.clip_model,
        "C": C,
        "penalty": penalty,
        "solver": solver,
        "l1_ratio": ("" if l1_ratio is None else float(l1_ratio)),
        "fit_intercept": fit_intercept,
        "feature_mode": feature_mode,
        "tol": tol,
        "class_weight": ("none" if class_weight is None else str(class_weight)),
        "val_acc": val_acc,
        "val_avg_group_acc": val_avg_group,
        "val_worst_group_acc": val_worst_group,
        "val_group_accs": np.array2string(val_class, precision=2, separator=","),
        "test_acc": test_acc,
        "test_avg_group_acc": test_avg_group,
        "test_worst_group_acc": test_worst_group,
        "test_group_accs": np.array2string(test_class, precision=2, separator=","),
        "seconds": int(time.time() - t0),
    }


def main():
    p = argparse.ArgumentParser(
        description="Random sweep for CLIP+LogReg on RedMeat (class-balanced metrics)."
    )
    p.add_argument("data_path", help="Path to food-101-redmeat directory containing all_images.csv")
    p.add_argument("--clip-model", default="RN50", help='CLIP model name (e.g. "RN50", "ViT-B/32").')
    p.add_argument("--device", default="cuda", help='Torch device for CLIP feature extraction (e.g. "cuda", "cpu").')
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", default="clip_lr_redmeat_sweep.csv")
    p.add_argument(
        "--sampler",
        choices=["random", "tpe"],
        default="random",
        help="Sampling strategy (tpe is mapped to random for backward compatibility).",
    )
    p.add_argument("--C-min", type=float, default=1e-6)
    p.add_argument("--C-max", type=float, default=1e6)
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--tol-min", type=float, default=1e-6)
    p.add_argument("--tol-max", type=float, default=1e-1)
    p.add_argument(
        "--penalty-solvers",
        default=_PENALTY_SOLVER_SPEC_DEFAULT,
        help=(
            "Comma-separated LogisticRegression penalty:solver pairs to search over. "
            "Allowed: l2:lbfgs,l2:liblinear,l2:saga,l1:liblinear,l1:saga,elasticnet:saga"
        ),
    )
    p.add_argument(
        "--feature-modes",
        default="l2,raw,zscore",
        help="Comma-separated feature transforms to sample from. Allowed: raw,l2,zscore",
    )
    p.add_argument(
        "--class-weight-options",
        default="none,balanced",
        help="Comma-separated class weight options to sample from. Allowed: none,balanced",
    )
    p.add_argument("--post-seeds", type=int, default=5)
    p.add_argument("--post-seed-start", type=int, default=0)
    p.add_argument("--post-output-csv", default=None)
    p.add_argument(
        "--objective",
        choices=["val_acc", "val_avg_group_acc", "val_worst_group_acc"],
        default="val_avg_group_acc",
        help="Validation metric to maximize.",
    )

    p.add_argument("--split-col", default="split")
    p.add_argument("--label-col", default="label")
    p.add_argument("--path-col", default="abs_file_path")
    p.add_argument("--train-value", default="train")
    p.add_argument("--val-value", default="val")
    p.add_argument("--test-value", default="test")
    p.add_argument(
        "--classes",
        default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon",
        help="Comma-separated class list. Empty string means infer from metadata.",
    )

    args = p.parse_args()
    args.penalty_solver_choices = _parse_penalty_solver_choices(args.penalty_solvers)
    args.penalty_solver_ids = [_choice_id(c) for c in args.penalty_solver_choices]
    args.penalty_solver_by_id = {
        cid: choice for cid, choice in zip(args.penalty_solver_ids, args.penalty_solver_choices)
    }
    args.feature_modes = _parse_feature_modes(args.feature_modes)
    args.class_weight_choices = _parse_class_weight_choices(args.class_weight_options)
    args.class_weight_option_names = [("none" if c is None else str(c)) for c in args.class_weight_choices]

    class_list = [c.strip() for c in str(args.classes).split(",") if c.strip()] if args.classes else None
    classes, train_samples, val_samples, test_samples = _build_splits(
        dataset_path=args.data_path,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
        train_value=args.train_value,
        val_value=args.val_value,
        test_value=args.test_value,
        classes=class_list,
    )
    num_classes = len(classes)

    header = [
        "trial",
        "clip_model",
        "C",
        "penalty",
        "solver",
        "l1_ratio",
        "fit_intercept",
        "feature_mode",
        "tol",
        "class_weight",
        "val_acc",
        "val_avg_group_acc",
        "val_worst_group_acc",
        "val_group_accs",
        "test_acc",
        "test_avg_group_acc",
        "test_worst_group_acc",
        "test_group_accs",
        "sampler",
        "seconds",
    ]

    rng = np.random.default_rng(args.seed)
    sweep_rows = []

    import torch

    clip = _try_import_clip()
    try:
        model, preprocess = clip.load(args.clip_model, device=args.device, jit=False)
    except TypeError:
        model, preprocess = clip.load(args.clip_model, device=args.device)

    print("[CLIP-LR] Extracting train features...")
    X_train, y_train = _extract_features(train_samples, model, preprocess, args.device, args.batch_size, args.num_workers)
    print("[CLIP-LR] Extracting val features...")
    X_val, y_val = _extract_features(val_samples, model, preprocess, args.device, args.batch_size, args.num_workers)
    print("[CLIP-LR] Extracting test features...")
    X_test, y_test = _extract_features(test_samples, model, preprocess, args.device, args.batch_size, args.num_workers)

    X_train_raw = _ensure_finite_contiguous(X_train)
    X_val_raw = _ensure_finite_contiguous(X_val)
    X_test_raw = _ensure_finite_contiguous(X_test)

    X_train_by_mode: Dict[str, np.ndarray] = {}
    X_val_by_mode: Dict[str, np.ndarray] = {}
    X_test_by_mode: Dict[str, np.ndarray] = {}

    if "raw" in args.feature_modes:
        X_train_by_mode["raw"] = X_train_raw
        X_val_by_mode["raw"] = X_val_raw
        X_test_by_mode["raw"] = X_test_raw

    if "l2" in args.feature_modes:
        X_train_by_mode["l2"] = _ensure_finite_contiguous(_l2_normalize(X_train_raw))
        X_val_by_mode["l2"] = _ensure_finite_contiguous(_l2_normalize(X_val_raw))
        X_test_by_mode["l2"] = _ensure_finite_contiguous(_l2_normalize(X_test_raw))

    if "zscore" in args.feature_modes:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_by_mode["zscore"] = _ensure_finite_contiguous(scaler.fit_transform(X_train_raw))
        X_val_by_mode["zscore"] = _ensure_finite_contiguous(scaler.transform(X_val_raw))
        X_test_by_mode["zscore"] = _ensure_finite_contiguous(scaler.transform(X_test_raw))

    print("[CLIP-LR] Feature extraction complete. Starting LR sweep...")
    if "cuda" in args.device:
        # Avoid explicit empty_cache(): this has caused sporadic process-level
        # crashes on some cluster nodes right after CLIP extraction.
        del model

    best_row = None

    def score(row: Dict) -> float:
        return float(row[args.objective])

    if args.sampler == "random":
        for trial_id in range(args.n_trials):
            try:
                row = _run_trial(
                    trial_id,
                    "random",
                    rng,
                    args,
                    num_classes,
                    X_train_by_mode,
                    y_train,
                    X_val_by_mode,
                    y_val,
                    X_test_by_mode,
                    y_test,
                )
            except Exception as exc:
                print(f"[SWEEP] Trial {trial_id} failed: {exc}", flush=True)
                continue
            _write_row(args.output_csv, row, header)
            sweep_rows.append(row)
            if best_row is None or score(row) > score(best_row):
                best_row = row
            print(
                f"[SWEEP] Trial {trial_id} done. "
                f"val_acc={row['val_acc']:.2f} val_avg_group_acc={row['val_avg_group_acc']:.2f} "
                f"test_acc={row['test_acc']:.2f} test_avg_group_acc={row['test_avg_group_acc']:.2f} "
                f"(objective={args.objective}:{row[args.objective]:.4f} "
                f"val_worst_group_acc={row['val_worst_group_acc']:.2f} "
                f"test_worst_group_acc={row['test_worst_group_acc']:.2f})"
            )
    else:
        try:
            import optuna
        except Exception as exc:
            print(f"[SWEEP] Optuna not available ({exc}); falling back to random search.")
            args.sampler = "random"
            for trial_id in range(args.n_trials):
                try:
                    row = _run_trial(
                        trial_id,
                        "random",
                        rng,
                        args,
                        num_classes,
                        X_train_by_mode,
                        y_train,
                        X_val_by_mode,
                        y_val,
                        X_test_by_mode,
                        y_test,
                    )
                except Exception as ex2:
                    print(f"[SWEEP] Trial {trial_id} failed: {ex2}", flush=True)
                    continue
                _write_row(args.output_csv, row, header)
                sweep_rows.append(row)
                if best_row is None or score(row) > score(best_row):
                    best_row = row
                print(
                    f"[SWEEP] Trial {trial_id} done. "
                    f"val_acc={row['val_acc']:.2f} val_avg_group_acc={row['val_avg_group_acc']:.2f} "
                    f"test_acc={row['test_acc']:.2f} test_avg_group_acc={row['test_avg_group_acc']:.2f} "
                    f"(objective={args.objective}:{row[args.objective]:.4f} "
                    f"val_worst_group_acc={row['val_worst_group_acc']:.2f} "
                    f"test_worst_group_acc={row['test_worst_group_acc']:.2f})"
                )
        else:
            sampler = optuna.samplers.TPESampler(seed=args.seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)

            def objective(trial):
                nonlocal best_row
                args.trial = trial
                try:
                    row = _run_trial(
                        trial.number,
                        "tpe",
                        rng,
                        args,
                        num_classes,
                        X_train_by_mode,
                        y_train,
                        X_val_by_mode,
                        y_val,
                        X_test_by_mode,
                        y_test,
                    )
                except Exception as exc:
                    print(f"[SWEEP] Trial {trial.number} failed: {exc}", flush=True)
                    return -1e12
                _write_row(args.output_csv, row, header)
                sweep_rows.append(row)
                if best_row is None or score(row) > score(best_row):
                    best_row = row
                print(
                    f"[SWEEP] Trial {trial.number} done. "
                    f"val_acc={row['val_acc']:.2f} val_avg_group_acc={row['val_avg_group_acc']:.2f} "
                    f"test_acc={row['test_acc']:.2f} test_avg_group_acc={row['test_avg_group_acc']:.2f} "
                    f"(objective={args.objective}:{row[args.objective]:.4f} "
                    f"val_worst_group_acc={row['val_worst_group_acc']:.2f} "
                    f"test_worst_group_acc={row['test_worst_group_acc']:.2f})"
                )
                return score(row)

            study.optimize(objective, n_trials=args.n_trials)

    if best_row is not None:
        print("[SWEEP] Best trial:")
        for k in header:
            print(f"  {k}: {best_row[k]}")

    if best_row is not None and args.post_seeds > 0:
        post_csv = args.post_output_csv
        if post_csv is None:
            root, ext = os.path.splitext(args.output_csv)
            post_csv = f"{root}_best_seeds{ext or '.csv'}"

        post_header = [
            "run_id",
            "run_label",
            "seed",
            "clip_model",
            "C",
            "penalty",
            "solver",
            "l1_ratio",
            "fit_intercept",
            "feature_mode",
            "tol",
            "class_weight",
            "val_acc",
            "val_avg_group_acc",
            "val_worst_group_acc",
            "val_group_accs",
            "test_acc",
            "test_avg_group_acc",
            "test_worst_group_acc",
            "test_group_accs",
            "sweep_best_trial",
            "sampler",
            "objective",
            "seconds",
        ]

        best_C = float(best_row["C"])
        best_penalty = str(best_row["penalty"])
        best_solver = str(best_row["solver"])
        best_fit_intercept = str(best_row["fit_intercept"]).lower() in ("1", "true", "yes")
        best_feature_mode = str(best_row["feature_mode"])
        best_tol = float(best_row["tol"])
        best_class_weight_raw = best_row["class_weight"]
        if best_class_weight_raw in ("", "none", "None", None):
            best_class_weight = None
        else:
            best_class_weight = str(best_class_weight_raw)
        best_l1_ratio = best_row["l1_ratio"]
        if best_l1_ratio in ("", None):
            best_l1_ratio = None
        else:
            best_l1_ratio = float(best_l1_ratio)

        seeds = list(range(args.post_seed_start, args.post_seed_start + args.post_seeds))
        print(f"[POST] Rerunning best hyperparameters for {len(seeds)} seeds: {seeds}")
        post_rows = []
        for idx, s in enumerate(seeds):
            out_row = _run_fixed_params(
                run_id=idx,
                run_label="best5",
                seed=s,
                C=best_C,
                fit_intercept=best_fit_intercept,
                penalty=best_penalty,
                solver=best_solver,
                l1_ratio=best_l1_ratio,
                feature_mode=best_feature_mode,
                tol=best_tol,
                class_weight=best_class_weight,
                args=args,
                num_classes=num_classes,
                X_train_by_mode=X_train_by_mode,
                y_train=y_train,
                X_val_by_mode=X_val_by_mode,
                y_val=y_val,
                X_test_by_mode=X_test_by_mode,
                y_test=y_test,
            )
            out_row["sweep_best_trial"] = int(best_row["trial"])
            out_row["sampler"] = args.sampler
            out_row["objective"] = args.objective
            _write_row(post_csv, out_row, post_header)
            post_rows.append(out_row)
            print(
                f"[POST] seed={s} "
                f"val_acc={out_row['val_acc']:.2f} val_avg_group_acc={out_row['val_avg_group_acc']:.2f} "
                f"test_acc={out_row['test_acc']:.2f} test_avg_group_acc={out_row['test_avg_group_acc']:.2f} "
                f"(objective={args.objective}:{out_row[args.objective]:.4f})"
            )

        _print_metric_mean_std(
            "POST",
            post_rows,
            ["test_acc", "test_avg_group_acc", "test_worst_group_acc", "val_acc", "val_avg_group_acc", "val_worst_group_acc"],
        )
        _print_runtime_summary("post_best_seeds", post_rows, num_epochs=None)

    _print_runtime_summary("sweep", sweep_rows, num_epochs=None)


if __name__ == "__main__":
    main()
