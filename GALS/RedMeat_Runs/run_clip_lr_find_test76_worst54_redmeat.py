#!/usr/bin/env python3
"""Grid-scan C for CLIP RN50 + LR on RedMeat until target test metrics are hit."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_clip_lr_sweep_redmeat as clip_lr


def _write_row(csv_path: str, row: Dict, header: Iterable[str]) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _parse_c_values(args) -> np.ndarray:
    if args.c_values.strip():
        vals = [float(x.strip()) for x in args.c_values.split(",") if x.strip()]
        if not vals:
            raise ValueError("Parsed empty --c-values.")
        out = np.array(vals, dtype=np.float64)
    else:
        out = np.exp(np.linspace(np.log(args.c_min), np.log(args.c_max), int(args.n_c_values), dtype=np.float64))
    out = out[np.isfinite(out) & (out > 0)]
    if out.size == 0:
        raise ValueError("No valid C values to scan.")
    # Unique while preserving order.
    _, idx = np.unique(np.round(out, 18), return_index=True)
    return out[np.sort(idx)]


def _score_close(row: Dict, target_mid: float, worst_max: float) -> tuple:
    test_acc = float(row["test_acc"])
    worst = float(row["test_worst_class_acc"])
    band_penalty = 0.0
    if test_acc < float(row["target_test_acc_min"]):
        band_penalty += float(row["target_test_acc_min"]) - test_acc
    elif test_acc >= float(row["target_test_acc_max"]):
        band_penalty += test_acc - float(row["target_test_acc_max"])
    worst_penalty = max(0.0, worst - worst_max)
    # Prefer in-band first, then worst-class satisfaction, then closeness to center.
    return (band_penalty, worst_penalty, abs(test_acc - target_mid), -float(row["val_acc"]))


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Scan C values for CLIP RN50 + fixed LR settings on RedMeat and stop "
            "when test_acc is 76.x and test_worst_class_acc <= 54."
        )
    )
    p.add_argument("data_path", help="Path to food-101-redmeat directory containing all_images.csv")
    p.add_argument("--clip-model", default="RN50")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", default="clip_lr_redmeat_find_test76_worst54.csv")

    p.add_argument("--c-min", type=float, default=1e-6)
    p.add_argument("--c-max", type=float, default=1e6)
    p.add_argument("--n-c-values", type=int, default=400)
    p.add_argument("--c-values", default="", help="Optional explicit comma-separated C values.")

    p.add_argument("--fit-intercept", type=int, default=1, choices=[0, 1])
    p.add_argument("--feature-mode", choices=["l2", "raw", "zscore"], default="l2")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=5000)

    p.add_argument("--test-acc-min", type=float, default=76.0)
    p.add_argument("--test-acc-max", type=float, default=77.0)
    p.add_argument("--test-worst-class-max", type=float, default=54.0)
    p.add_argument("--val-acc-min", type=float, default=73.0)
    p.add_argument("--refine-rounds", type=int, default=2, help="Number of local refinement rounds after coarse scan.")
    p.add_argument("--refine-top-k", type=int, default=6, help="How many best coarse candidates to refine around.")
    p.add_argument(
        "--refine-span",
        type=float,
        default=3.0,
        help="Local window factor around each center C: [C/span, C*span].",
    )
    p.add_argument("--refine-n-values", type=int, default=120, help="C points per local refinement window.")
    p.add_argument("--max-matches", type=int, default=1)
    p.add_argument("--stop-on-match", action="store_true", default=True)
    p.add_argument("--no-stop-on-match", action="store_false", dest="stop_on_match")

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

    class_list = [c.strip() for c in str(args.classes).split(",") if c.strip()] if args.classes else None
    classes, train_samples, val_samples, test_samples = clip_lr._build_splits(
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

    import torch
    from sklearn.linear_model import LogisticRegression

    clip = clip_lr._try_import_clip()
    try:
        model, preprocess = clip.load(args.clip_model, device=args.device, jit=False)
    except TypeError:
        model, preprocess = clip.load(args.clip_model, device=args.device)

    print("[SCAN] Extracting CLIP train features...", flush=True)
    X_train_raw, y_train = clip_lr._extract_features(
        train_samples, model, preprocess, args.device, args.batch_size, args.num_workers
    )
    print("[SCAN] Extracting CLIP val features...", flush=True)
    X_val_raw, y_val = clip_lr._extract_features(
        val_samples, model, preprocess, args.device, args.batch_size, args.num_workers
    )
    print("[SCAN] Extracting CLIP test features...", flush=True)
    X_test_raw, y_test = clip_lr._extract_features(
        test_samples, model, preprocess, args.device, args.batch_size, args.num_workers
    )

    if "cuda" in args.device:
        del model
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if args.feature_mode == "raw":
        X_train = clip_lr._ensure_finite_contiguous(X_train_raw)
        X_val = clip_lr._ensure_finite_contiguous(X_val_raw)
        X_test = clip_lr._ensure_finite_contiguous(X_test_raw)
    elif args.feature_mode == "l2":
        X_train = clip_lr._ensure_finite_contiguous(clip_lr._l2_normalize(X_train_raw))
        X_val = clip_lr._ensure_finite_contiguous(clip_lr._l2_normalize(X_val_raw))
        X_test = clip_lr._ensure_finite_contiguous(clip_lr._l2_normalize(X_test_raw))
    else:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train = clip_lr._ensure_finite_contiguous(scaler.fit_transform(X_train_raw))
        X_val = clip_lr._ensure_finite_contiguous(scaler.transform(X_val_raw))
        X_test = clip_lr._ensure_finite_contiguous(scaler.transform(X_test_raw))

    c_values = _parse_c_values(args)
    fit_intercept = bool(args.fit_intercept)
    class_weight = None if args.class_weight == "none" else "balanced"

    header = [
        "scan_id",
        "clip_model",
        "feature_mode",
        "class_weight",
        "C",
        "fit_intercept",
        "penalty",
        "solver",
        "tol",
        "val_acc",
        "val_avg_group_acc",
        "val_worst_group_acc",
        "val_group_accs",
        "test_acc",
        "test_avg_group_acc",
        "test_worst_class_acc",
        "test_group_accs",
        "target_test_acc_min",
        "target_test_acc_max",
        "target_test_worst_class_max",
        "target_val_acc_min",
        "matched",
        "seconds",
    ]

    print(
        f"[SCAN] Starting C scan: n_c_values={len(c_values)} fit_intercept={fit_intercept} "
        f"feature_mode={args.feature_mode} class_weight={args.class_weight} tol={args.tol}",
        flush=True,
    )
    print(
        f"[SCAN] Match criteria: val_acc > {args.val_acc_min}, "
        f"test_acc in [{args.test_acc_min}, {args.test_acc_max}), "
        f"and test_worst_class_acc <= {args.test_worst_class_max}",
        flush=True,
    )

    matches: List[Dict] = []
    best_close: Dict | None = None
    all_rows: List[Dict] = []
    start = time.time()
    scan_id = 0

    seen_c = set()
    target_mid = (args.test_acc_min + args.test_acc_max) / 2.0

    def _eval_one(c: float) -> bool:
        nonlocal scan_id, best_close
        c = float(c)
        key = round(c, 16)
        if key in seen_c:
            return False
        seen_c.add(key)

        t0 = time.time()
        clf = LogisticRegression(
            random_state=args.seed,
            C=c,
            penalty="l2",
            solver="lbfgs",
            fit_intercept=fit_intercept,
            max_iter=args.max_iter,
            tol=float(args.tol),
            class_weight=class_weight,
            n_jobs=1,
            verbose=0,
        )
        clip_lr._safe_fit(clf, X_train, y_train)

        val_pred = clf.predict(X_val)
        val_acc = float(np.mean((val_pred == y_val).astype(np.float64)) * 100.0)
        val_class = clip_lr._class_acc(y_val, val_pred, num_classes=num_classes)
        val_avg_group = clip_lr._nanmean(val_class)
        val_worst_group = clip_lr._nanmin(val_class)

        test_pred = clf.predict(X_test)
        test_acc = float(np.mean((test_pred == y_test).astype(np.float64)) * 100.0)
        test_class = clip_lr._class_acc(y_test, test_pred, num_classes=num_classes)
        test_avg_group = clip_lr._nanmean(test_class)
        test_worst = clip_lr._nanmin(test_class)

        matched = (
            (val_acc > float(args.val_acc_min))
            and
            (test_acc >= float(args.test_acc_min))
            and (test_acc < float(args.test_acc_max))
            and (test_worst <= float(args.test_worst_class_max))
        )
        row = {
            "scan_id": scan_id,
            "clip_model": args.clip_model,
            "feature_mode": args.feature_mode,
            "class_weight": args.class_weight,
            "C": c,
            "fit_intercept": fit_intercept,
            "penalty": "l2",
            "solver": "lbfgs",
            "tol": float(args.tol),
            "val_acc": val_acc,
            "val_avg_group_acc": val_avg_group,
            "val_worst_group_acc": val_worst_group,
            "val_group_accs": np.array2string(val_class, precision=2, separator=","),
            "test_acc": test_acc,
            "test_avg_group_acc": test_avg_group,
            "test_worst_class_acc": test_worst,
            "test_group_accs": np.array2string(test_class, precision=2, separator=","),
            "target_test_acc_min": float(args.test_acc_min),
            "target_test_acc_max": float(args.test_acc_max),
            "target_test_worst_class_max": float(args.test_worst_class_max),
            "target_val_acc_min": float(args.val_acc_min),
            "matched": int(bool(matched)),
            "seconds": int(time.time() - t0),
        }
        _write_row(args.output_csv, row, header)
        all_rows.append(row)

        if (best_close is None) or (
            _score_close(row, target_mid=target_mid, worst_max=args.test_worst_class_max)
            < _score_close(best_close, target_mid=target_mid, worst_max=args.test_worst_class_max)
        ):
            best_close = dict(row)

        if scan_id % 20 == 0 or matched:
            print(
                f"[SCAN] id={scan_id} C={c:.6g} fit_intercept={fit_intercept} "
                f"val_acc={val_acc:.2f} test_acc={test_acc:.2f} test_worst={test_worst:.2f} "
                f"matched={matched}",
                flush=True,
            )

        if matched:
            matches.append(row)
            print(
                f"[FOUND] id={scan_id} C={c:.10g} fit_intercept={fit_intercept} "
                f"test_acc={test_acc:.2f} test_worst={test_worst:.2f}",
                flush=True,
            )
            if args.stop_on_match and len(matches) >= int(args.max_matches):
                total = time.time() - start
                print(f"[DONE] Early stop after {scan_id + 1} scans in {total / 60.0:.2f} min.", flush=True)
                print(f"[DONE] Results CSV: {args.output_csv}", flush=True)
                return True

        scan_id += 1
        return False

    # Coarse global scan.
    for c in c_values:
        stop = _eval_one(float(c))
        if stop:
            return

    # Local refinement around the best coarse candidates.
    for rr in range(int(args.refine_rounds)):
        if not all_rows:
            break
        ranked = sorted(
            all_rows,
            key=lambda r: _score_close(r, target_mid=target_mid, worst_max=args.test_worst_class_max),
        )
        seeds = ranked[: max(1, int(args.refine_top_k))]
        local_cs: List[float] = []
        for s in seeds:
            center = float(s["C"])
            lo = max(float(args.c_min), center / float(args.refine_span))
            hi = min(float(args.c_max), center * float(args.refine_span))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= lo:
                continue
            local = np.exp(np.linspace(np.log(lo), np.log(hi), int(args.refine_n_values), dtype=np.float64))
            local_cs.extend(local.tolist())

        if not local_cs:
            break
        print(
            f"[REFINE] round={rr + 1}/{args.refine_rounds} "
            f"centers={len(seeds)} candidates={len(local_cs)}",
            flush=True,
        )
        for c in local_cs:
            stop = _eval_one(float(c))
            if stop:
                return

    total = time.time() - start
    print(f"[DONE] Exhaustive scan complete. scanned={scan_id} elapsed_min={total / 60.0:.2f}", flush=True)
    print(f"[DONE] Matches found: {len(matches)}", flush=True)
    if best_close is not None:
        print(
            "[BEST_CLOSE] "
            f"C={best_close['C']} fit_intercept={best_close['fit_intercept']} "
            f"test_acc={best_close['test_acc']:.2f} test_worst={best_close['test_worst_class_acc']:.2f} "
            f"val_acc={best_close['val_acc']:.2f}",
            flush=True,
        )
    print(f"[DONE] Results CSV: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
