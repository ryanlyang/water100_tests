#!/usr/bin/env python3
"""Optuna/random sweep for Waterbirds NSF."""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List

import numpy as np

import run_nsf_waterbird as nsf


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _parse_int_choices(value: str) -> List[int]:
    choices = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not choices:
        raise ValueError("Choice list must not be empty")
    return choices


def _write_row(csv_path: str, row: Dict, header: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _fmt_group_acc(group_acc: np.ndarray) -> str:
    return ",".join(f"{float(x):.4f}" for x in group_acc.tolist())


def _print_runtime_summary(tag: str, rows: List[Dict], num_epochs: int) -> None:
    secs = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
    if not secs:
        print(f"[TIME] {tag}: no successful trials to summarize.", flush=True)
        return
    arr = np.array(secs, dtype=float)
    print(
        f"[TIME] {tag}: median min/trial={float(np.median(arr) / 60.0):.4f} "
        f"| total tuning GPU-hours={float(np.sum(arr) / 3600.0):.4f}",
        flush=True,
    )
    print(
        f"[TIME] {tag}: median min/epoch={float(np.median(arr / float(num_epochs)) / 60.0):.4f} "
        f"(epochs/trial={int(num_epochs)})",
        flush=True,
    )


def _build_run_args(
    args,
    seed: int,
    base_lr: float,
    classifier_lr: float,
    transform_steps: int,
    classifier_steps: int,
    transform_lr: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_path=args.data_path,
        seed=seed,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        num_workers=args.num_workers,
        transform_steps=transform_steps,
        classifier_steps=classifier_steps,
        transform_lr=transform_lr,
        nsf_classifier_lr=args.nsf_classifier_lr,
        beta_reg_weight=args.beta_reg_weight,
        checkpoint_dir=args.checkpoint_dir,
    )


def _run_trial(trial_id: int, args, rng: np.random.Generator, sampler_name: str) -> Dict:
    t0 = time.time()
    if sampler_name == "random":
        base_lr = _loguniform(rng, args.base_lr_min, args.base_lr_max)
        classifier_lr = _loguniform(rng, args.cls_lr_min, args.cls_lr_max)
        transform_lr = _loguniform(rng, args.transform_lr_min, args.transform_lr_max)
        transform_steps = int(rng.choice(args.transform_steps_choices))
        classifier_steps = int(rng.choice(args.classifier_steps_choices))
    else:
        base_lr = float(args.trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
        classifier_lr = float(args.trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
        transform_lr = float(
            args.trial.suggest_float("transform_lr", args.transform_lr_min, args.transform_lr_max, log=True)
        )
        transform_steps = int(args.trial.suggest_categorical("transform_steps", args.transform_steps_choices))
        classifier_steps = int(args.trial.suggest_categorical("classifier_steps", args.classifier_steps_choices))

    run_args = _build_run_args(
        args=args,
        seed=args.train_seed,
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        transform_steps=transform_steps,
        classifier_steps=classifier_steps,
        transform_lr=transform_lr,
    )
    result = nsf.run_single(run_args)
    group_acc = result["test_group_acc"]

    return {
        "trial": trial_id,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "nesterov": args.nesterov,
        "transform_steps": transform_steps,
        "classifier_steps": classifier_steps,
        "transform_lr": transform_lr,
        "nsf_classifier_lr": args.nsf_classifier_lr,
        "beta_reg_weight": args.beta_reg_weight,
        "erm_best_epoch": result["erm_best_epoch"],
        "erm_val_acc": result["erm_val_acc"],
        "erm_val_balanced_group": result["erm_val_balanced_group"],
        "erm_val_worst_group": result["erm_val_worst_group"],
        "nsf_val_acc": result["nsf_val_acc"],
        "nsf_val_balanced_group": result["nsf_val_balanced_group"],
        "nsf_val_worst_group": result["nsf_val_worst_group"],
        "test_acc": result["test_acc"],
        "test_balanced_group": result["test_balanced_group"],
        "test_worst_group": result["test_worst_group"],
        "test_group_acc": _fmt_group_acc(group_acc),
        "outlier_count": result["outlier_count"],
        "outlier_frac": result["outlier_frac"],
        "checkpoint": result["checkpoint"],
        "sampler": sampler_name,
        "seconds": int(time.time() - t0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna/random sweep for Waterbirds NSF.")
    p.add_argument("data_path", help="Waterbirds root containing metadata.csv")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="Sweep sampler seed")
    p.add_argument("--train-seed", type=int, default=0, help="Fixed training seed during hyperparameter search")
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")

    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-dir", default="NSF_Checkpoints")

    p.add_argument("--base-lr-min", type=float, default=1e-5)
    p.add_argument("--base-lr-max", type=float, default=5e-2)
    p.add_argument("--cls-lr-min", type=float, default=1e-5)
    p.add_argument("--cls-lr-max", type=float, default=5e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true", default=False)
    p.add_argument("--no-nesterov", action="store_false", dest="nesterov")

    p.add_argument("--transform-lr-min", type=float, default=1e-4)
    p.add_argument("--transform-lr-max", type=float, default=1e-2)
    p.add_argument("--transform-steps-choices", default="5,10,25,50,100")
    p.add_argument("--classifier-steps-choices", default="100,250,500,1000,2000")
    p.add_argument("--nsf-classifier-lr", type=float, default=1e-3)
    p.add_argument("--beta-reg-weight", type=float, default=10.0)

    p.add_argument("--output-csv", default="nsf_waterbird_sweep.csv")
    p.add_argument("--post-seeds", type=int, default=5)
    p.add_argument("--post-seed-start", type=int, default=0)
    p.add_argument("--post-output-csv", default="nsf_waterbird_best5.csv")

    args = p.parse_args()
    args.transform_steps_choices = _parse_int_choices(args.transform_steps_choices)
    args.classifier_steps_choices = _parse_int_choices(args.classifier_steps_choices)

    header = [
        "trial",
        "base_lr",
        "classifier_lr",
        "weight_decay",
        "momentum",
        "nesterov",
        "transform_steps",
        "classifier_steps",
        "transform_lr",
        "nsf_classifier_lr",
        "beta_reg_weight",
        "erm_best_epoch",
        "erm_val_acc",
        "erm_val_balanced_group",
        "erm_val_worst_group",
        "nsf_val_acc",
        "nsf_val_balanced_group",
        "nsf_val_worst_group",
        "test_acc",
        "test_balanced_group",
        "test_worst_group",
        "test_group_acc",
        "outlier_count",
        "outlier_frac",
        "checkpoint",
        "sampler",
        "seconds",
    ]

    rng = np.random.default_rng(args.seed)
    sweep_rows: List[Dict] = []
    best_row = None

    if args.sampler == "tpe":
        try:
            import optuna  # noqa: F401
        except Exception as exc:
            print(f"[SWEEP] Optuna not available ({exc}); falling back to random search.", flush=True)
            args.sampler = "random"

    if args.sampler == "random":
        for trial_id in range(args.n_trials):
            row = _run_trial(trial_id, args, rng, "random")
            _write_row(args.output_csv, row, header)
            sweep_rows.append(row)
            if best_row is None or row["nsf_val_balanced_group"] > best_row["nsf_val_balanced_group"]:
                best_row = row
            print(
                f"[SWEEP] Trial {trial_id} done. nsf_val_bal_group={row['nsf_val_balanced_group']:.4f} "
                f"nsf_val_worst={row['nsf_val_worst_group']:.4f} test_worst={row['test_worst_group']:.4f} "
                f"base_lr={row['base_lr']} cls_lr={row['classifier_lr']} "
                f"transform_steps={row['transform_steps']} classifier_steps={row['classifier_steps']} "
                f"transform_lr={row['transform_lr']}",
                flush=True,
            )
    else:
        import optuna

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            nonlocal best_row
            args.trial = trial
            row = _run_trial(trial.number, args, rng, "tpe")
            _write_row(args.output_csv, row, header)
            sweep_rows.append(row)
            if best_row is None or row["nsf_val_balanced_group"] > best_row["nsf_val_balanced_group"]:
                best_row = row
            print(
                f"[SWEEP] Trial {trial.number} done. nsf_val_bal_group={row['nsf_val_balanced_group']:.4f} "
                f"nsf_val_worst={row['nsf_val_worst_group']:.4f} test_worst={row['test_worst_group']:.4f} "
                f"base_lr={row['base_lr']} cls_lr={row['classifier_lr']} "
                f"transform_steps={row['transform_steps']} classifier_steps={row['classifier_steps']} "
                f"transform_lr={row['transform_lr']}",
                flush=True,
            )
            return float(row["nsf_val_balanced_group"])

        study.optimize(objective, n_trials=args.n_trials)

    if best_row is None:
        raise SystemExit("No trials completed")

    _print_runtime_summary("sweep", sweep_rows, args.num_epochs)
    print(f"[SWEEP] Best row: {best_row}", flush=True)

    post_header = ["seed"] + header[1:]
    post_rows = []
    for seed in range(args.post_seed_start, args.post_seed_start + args.post_seeds):
        run_args = _build_run_args(
            args=args,
            seed=seed,
            base_lr=float(best_row["base_lr"]),
            classifier_lr=float(best_row["classifier_lr"]),
            transform_steps=int(best_row["transform_steps"]),
            classifier_steps=int(best_row["classifier_steps"]),
            transform_lr=float(best_row["transform_lr"]),
        )
        t0 = time.time()
        result = nsf.run_single(run_args)
        post_row = {
            "seed": seed,
            "base_lr": best_row["base_lr"],
            "classifier_lr": best_row["classifier_lr"],
            "weight_decay": best_row["weight_decay"],
            "momentum": best_row["momentum"],
            "nesterov": best_row["nesterov"],
            "transform_steps": best_row["transform_steps"],
            "classifier_steps": best_row["classifier_steps"],
            "transform_lr": best_row["transform_lr"],
            "nsf_classifier_lr": best_row["nsf_classifier_lr"],
            "beta_reg_weight": best_row["beta_reg_weight"],
            "erm_best_epoch": result["erm_best_epoch"],
            "erm_val_acc": result["erm_val_acc"],
            "erm_val_balanced_group": result["erm_val_balanced_group"],
            "erm_val_worst_group": result["erm_val_worst_group"],
            "nsf_val_acc": result["nsf_val_acc"],
            "nsf_val_balanced_group": result["nsf_val_balanced_group"],
            "nsf_val_worst_group": result["nsf_val_worst_group"],
            "test_acc": result["test_acc"],
            "test_balanced_group": result["test_balanced_group"],
            "test_worst_group": result["test_worst_group"],
            "test_group_acc": _fmt_group_acc(result["test_group_acc"]),
            "outlier_count": result["outlier_count"],
            "outlier_frac": result["outlier_frac"],
            "checkpoint": result["checkpoint"],
            "sampler": "post_best",
            "seconds": int(time.time() - t0),
        }
        _write_row(args.post_output_csv, post_row, post_header)
        post_rows.append(post_row)
        print(
            f"[POST] seed={seed} nsf_val_bal_group={post_row['nsf_val_balanced_group']:.4f} "
            f"test_worst={post_row['test_worst_group']:.4f} "
            f"test_bal_group={post_row['test_balanced_group']:.4f}",
            flush=True,
        )

    for metric in ["test_acc", "test_balanced_group", "test_worst_group", "nsf_val_balanced_group"]:
        vals = np.array([float(r[metric]) for r in post_rows], dtype=float)
        print(f"[POST SUMMARY] {metric}: mean={vals.mean():.4f} std={vals.std(ddof=0):.4f}", flush=True)


if __name__ == "__main__":
    main()
