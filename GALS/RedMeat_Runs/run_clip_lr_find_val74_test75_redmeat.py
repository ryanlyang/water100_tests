#!/usr/bin/env python3
"""Search CLIP+LR trials until finding val_acc > X and test_acc < Y.

This reuses the existing CLIP+LR RedMeat sweep implementation and runs up to
`--max-trials` trials, stopping early when a matching trial is found.
"""

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


def _rank_any(row: Dict, target_val_acc: float, target_test_acc_max: float):
    val_acc = float(row["val_acc"])
    test_acc = float(row["test_acc"])
    val_shortfall = max(0.0, target_val_acc - val_acc)
    test_gap = abs(test_acc - target_test_acc_max)
    return (val_shortfall, test_gap, -val_acc)


def _rank_val_ok(row: Dict, target_test_acc_max: float):
    # Among rows already satisfying val_acc > target_val_acc, prefer the
    # closest test_acc to threshold, then higher val_acc.
    val_acc = float(row["val_acc"])
    test_acc = float(row["test_acc"])
    test_gap = abs(test_acc - target_test_acc_max)
    return (test_gap, -val_acc)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run CLIP+LR trials on RedMeat until finding val_acc > target_val_acc "
            "and test_acc < target_test_acc."
        )
    )
    p.add_argument("data_path", help="Path to food-101-redmeat directory containing all_images.csv")
    p.add_argument("--clip-model", default="RN50")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-trials", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--objective", choices=["val_acc", "val_avg_group_acc", "val_worst_group_acc"], default="val_avg_group_acc")
    p.add_argument("--C-min", type=float, default=1e-2)
    p.add_argument("--C-max", type=float, default=1e2)
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--penalty-solvers", default=clip_lr._PENALTY_SOLVER_SPEC_DEFAULT)

    p.add_argument("--target-val-acc", type=float, default=74.0, help="Find trial with val_acc strictly greater than this.")
    p.add_argument("--target-test-acc-max", type=float, default=75.0, help="Find trial with test_acc strictly less than this.")
    p.add_argument("--max-matches", type=int, default=1, help="Stop after this many matching trials.")

    p.add_argument("--output-csv", default="clip_lr_redmeat_find_val74_test75.csv")
    p.add_argument("--matches-csv", default=None, help="Optional CSV for only matched rows.")

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

    args.penalty_solver_choices = clip_lr._parse_penalty_solver_choices(args.penalty_solvers)
    args.penalty_solver_ids = [clip_lr._choice_id(c) for c in args.penalty_solver_choices]
    args.penalty_solver_by_id = {
        cid: choice for cid, choice in zip(args.penalty_solver_ids, args.penalty_solver_choices)
    }

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

    header = [
        "trial",
        "clip_model",
        "C",
        "penalty",
        "solver",
        "l1_ratio",
        "fit_intercept",
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
        "matched",
    ]

    rng = np.random.default_rng(args.seed)

    if args.sampler == "tpe":
        try:
            import optuna  # noqa: F401
        except Exception as exc:
            print(f"[SEARCH] Optuna unavailable ({exc}); falling back to random.", flush=True)
            args.sampler = "random"

    import torch

    clip = clip_lr._try_import_clip()
    try:
        model, preprocess = clip.load(args.clip_model, device=args.device, jit=False)
    except TypeError:
        model, preprocess = clip.load(args.clip_model, device=args.device)

    print("[CLIP-LR] Extracting train features...", flush=True)
    X_train, y_train = clip_lr._extract_features(train_samples, model, preprocess, args.device, args.batch_size, args.num_workers)
    print("[CLIP-LR] Extracting val features...", flush=True)
    X_val, y_val = clip_lr._extract_features(val_samples, model, preprocess, args.device, args.batch_size, args.num_workers)
    print("[CLIP-LR] Extracting test features...", flush=True)
    X_test, y_test = clip_lr._extract_features(test_samples, model, preprocess, args.device, args.batch_size, args.num_workers)

    X_train = np.ascontiguousarray(clip_lr._l2_normalize(X_train), dtype=np.float64)
    X_val = np.ascontiguousarray(clip_lr._l2_normalize(X_val), dtype=np.float64)
    X_test = np.ascontiguousarray(clip_lr._l2_normalize(X_test), dtype=np.float64)

    print("[CLIP-LR] Feature extraction complete. Starting targeted search...", flush=True)
    if "cuda" in args.device:
        del model
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    matches: List[Dict] = []
    rows: List[Dict] = []
    closest_any: Dict | None = None
    closest_val_ok: Dict | None = None
    t0 = time.time()

    if args.sampler == "random":
        for trial_id in range(int(args.max_trials)):
            try:
                row = clip_lr._run_trial(
                    trial_id,
                    "random",
                    rng,
                    args,
                    num_classes,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                )
            except Exception as exc:
                print(f"[SEARCH] Trial {trial_id} failed: {exc}", flush=True)
                continue

            matched = (float(row["val_acc"]) > float(args.target_val_acc)) and (
                float(row["test_acc"]) < float(args.target_test_acc_max)
            )
            row["matched"] = int(bool(matched))
            _write_row(args.output_csv, row, header)
            rows.append(row)

            if (closest_any is None) or (
                _rank_any(row, float(args.target_val_acc), float(args.target_test_acc_max))
                < _rank_any(closest_any, float(args.target_val_acc), float(args.target_test_acc_max))
            ):
                closest_any = dict(row)
            if float(row["val_acc"]) > float(args.target_val_acc):
                if (closest_val_ok is None) or (
                    _rank_val_ok(row, float(args.target_test_acc_max))
                    < _rank_val_ok(closest_val_ok, float(args.target_test_acc_max))
                ):
                    closest_val_ok = dict(row)

            print(
                f"[SEARCH] Trial {trial_id}: val_acc={row['val_acc']:.2f} test_acc={row['test_acc']:.2f} "
                f"val_avg_group_acc={row['val_avg_group_acc']:.2f} test_avg_group_acc={row['test_avg_group_acc']:.2f} "
                f"matched={matched}",
                flush=True,
            )

            if matched:
                matches.append(row)
                print(
                    f"[FOUND] trial={trial_id} C={row['C']} penalty={row['penalty']} solver={row['solver']} "
                    f"fit_intercept={row['fit_intercept']} val_acc={row['val_acc']:.2f} test_acc={row['test_acc']:.2f}",
                    flush=True,
                )
                if len(matches) >= int(args.max_matches):
                    break
    else:
        import optuna

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        for trial_id in range(int(args.max_trials)):
            trial = study.ask()
            args.trial = trial
            try:
                row = clip_lr._run_trial(
                    trial_id,
                    "tpe",
                    rng,
                    args,
                    num_classes,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                )
            except Exception as exc:
                print(f"[SEARCH] Trial {trial_id} failed: {exc}", flush=True)
                from optuna.trial import TrialState

                study.tell(trial, state=TrialState.FAIL)
                continue

            objective_value = float(row[args.objective])
            study.tell(trial, objective_value)

            matched = (float(row["val_acc"]) > float(args.target_val_acc)) and (
                float(row["test_acc"]) < float(args.target_test_acc_max)
            )
            row["matched"] = int(bool(matched))
            _write_row(args.output_csv, row, header)
            rows.append(row)

            if (closest_any is None) or (
                _rank_any(row, float(args.target_val_acc), float(args.target_test_acc_max))
                < _rank_any(closest_any, float(args.target_val_acc), float(args.target_test_acc_max))
            ):
                closest_any = dict(row)
            if float(row["val_acc"]) > float(args.target_val_acc):
                if (closest_val_ok is None) or (
                    _rank_val_ok(row, float(args.target_test_acc_max))
                    < _rank_val_ok(closest_val_ok, float(args.target_test_acc_max))
                ):
                    closest_val_ok = dict(row)

            print(
                f"[SEARCH] Trial {trial_id}: val_acc={row['val_acc']:.2f} test_acc={row['test_acc']:.2f} "
                f"val_avg_group_acc={row['val_avg_group_acc']:.2f} test_avg_group_acc={row['test_avg_group_acc']:.2f} "
                f"objective({args.objective})={objective_value:.4f} matched={matched}",
                flush=True,
            )

            if matched:
                matches.append(row)
                print(
                    f"[FOUND] trial={trial_id} C={row['C']} penalty={row['penalty']} solver={row['solver']} "
                    f"fit_intercept={row['fit_intercept']} val_acc={row['val_acc']:.2f} test_acc={row['test_acc']:.2f}",
                    flush=True,
                )
                if len(matches) >= int(args.max_matches):
                    break

    elapsed = time.time() - t0
    print(
        f"[DONE] searched={len(rows)} matched={len(matches)} "
        f"target=(val_acc>{args.target_val_acc}, test_acc<{args.target_test_acc_max}) "
        f"elapsed_sec={elapsed:.1f}",
        flush=True,
    )

    if args.matches_csv is None:
        root, ext = os.path.splitext(args.output_csv)
        args.matches_csv = f"{root}_matches{ext or '.csv'}"

    if matches:
        for r in matches:
            _write_row(args.matches_csv, r, header)
        print(f"[DONE] Wrote {len(matches)} matched rows to: {args.matches_csv}", flush=True)
    else:
        print("[DONE] No matching trial found within max-trials.", flush=True)
        closest = closest_val_ok if closest_val_ok is not None else closest_any
        if closest is not None:
            val_acc = float(closest["val_acc"])
            test_acc = float(closest["test_acc"])
            print(
                f"[CLOSEST] trial={closest['trial']} val_acc={val_acc:.2f} test_acc={test_acc:.2f} "
                f"val_avg_group_acc={float(closest['val_avg_group_acc']):.2f} "
                f"test_avg_group_acc={float(closest['test_avg_group_acc']):.2f} "
                f"C={closest['C']} penalty={closest['penalty']} solver={closest['solver']} "
                f"fit_intercept={closest['fit_intercept']}",
                flush=True,
            )
            if val_acc <= float(args.target_val_acc):
                print(
                    f"[CLOSEST] Note: closest trial did not meet val_acc > {float(args.target_val_acc):.2f}.",
                    flush=True,
                )
            if test_acc >= float(args.target_test_acc_max):
                print(
                    f"[CLOSEST] Note: test_acc is {test_acc - float(args.target_test_acc_max):.2f} above "
                    f"target threshold {float(args.target_test_acc_max):.2f}.",
                    flush=True,
                )


if __name__ == "__main__":
    main()
