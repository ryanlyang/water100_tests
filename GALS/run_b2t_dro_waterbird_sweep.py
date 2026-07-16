#!/usr/bin/env python3
"""Optuna/random sweep for Waterbirds B2T-DRO."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

import run_b2t_dro_waterbird as b2t


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _write_row(csv_path: str, row: Dict, header: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _print_runtime_summary(tag: str, rows: List[Dict], num_epochs: int) -> None:
    secs = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
    if not secs:
        print(f"[TIME] {tag}: no successful trials to summarize.")
        return
    arr = np.array(secs, dtype=float)
    print(f"[TIME] {tag}: median min/trial={float(np.median(arr) / 60.0):.4f} | total tuning GPU-hours={float(np.sum(arr) / 3600.0):.4f}")
    print(f"[TIME] {tag}: median min/epoch={float(np.median(arr / float(num_epochs)) / 60.0):.4f} (epochs/trial={int(num_epochs)})")


def _fmt_group_acc(group_acc: np.ndarray) -> str:
    return ",".join(f"{float(x):.4f}" for x in group_acc.tolist())


def _build_run_args(
    args,
    seed: int,
    base_lr: float,
    classifier_lr: float,
    momentum: float,
    dro_step_size: float,
):
    return SimpleNamespace(
        data_path=args.data_path,
        pseudo_bias_path=args.pseudo_bias_path,
        overwrite_pseudo_bias=False,
        b2t_clip_model=args.b2t_clip_model,
        pseudo_batch_size=args.pseudo_batch_size,
        seed=seed,
        model="resnet50",
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        momentum=momentum,
        weight_decay=args.weight_decay,
        dro_step_size=dro_step_size,
        balanced_sampler=args.balanced_sampler,
        nesterov=args.nesterov,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
    )


def _run_trial(trial_id: int, args, rng: np.random.Generator, sampler_name: str) -> Dict:
    t0 = time.time()
    if sampler_name == "random":
        base_lr = _loguniform(rng, args.base_lr_min, args.base_lr_max)
        classifier_lr = _loguniform(rng, args.cls_lr_min, args.cls_lr_max)
        momentum = float(rng.uniform(args.momentum_min, args.momentum_max))
        dro_step_size = _loguniform(rng, args.dro_step_size_min, args.dro_step_size_max)
    else:
        base_lr = float(args.trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
        classifier_lr = float(args.trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
        momentum = float(args.trial.suggest_float("momentum", args.momentum_min, args.momentum_max))
        dro_step_size = float(
            args.trial.suggest_float("dro_step_size", args.dro_step_size_min, args.dro_step_size_max, log=True)
        )

    run_args = _build_run_args(
        args=args,
        seed=args.train_seed,
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        momentum=momentum,
        dro_step_size=dro_step_size,
    )
    result = b2t.run_single(run_args)
    group_acc = result["test_group_acc"]

    return {
        "trial": trial_id,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "weight_decay": args.weight_decay,
        "momentum": momentum,
        "nesterov": args.nesterov,
        "dro_step_size": dro_step_size,
        "balanced_sampler": args.balanced_sampler,
        "best_epoch": result["best_epoch"],
        "best_val_acc": result["best_val_acc"],
        "best_val_balanced_group": result["best_val_balanced_group"],
        "best_val_worst_group": result["best_val_worst_group"],
        "test_acc": result["test_acc"],
        "test_balanced_group": result["test_balanced_group"],
        "test_worst_group": result["test_worst_group"],
        "test_group_acc": _fmt_group_acc(group_acc),
        "checkpoint": result["checkpoint"],
        "pseudo_bias_path": result["pseudo_bias_path"],
        "sampler": sampler_name,
        "seconds": int(time.time() - t0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna/random sweep for Waterbirds B2T-DRO.")
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
    p.add_argument("--checkpoint-dir", default="B2T_DRO_Checkpoints")

    p.add_argument("--base-lr-min", type=float, default=1e-5)
    p.add_argument("--base-lr-max", type=float, default=5e-2)
    p.add_argument("--cls-lr-min", type=float, default=1e-5)
    p.add_argument("--cls-lr-max", type=float, default=5e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--momentum-min", type=float, default=0.85)
    p.add_argument("--momentum-max", type=float, default=0.95)
    p.add_argument("--dro-step-size-min", type=float, default=1e-3)
    p.add_argument("--dro-step-size-max", type=float, default=1e-1)
    p.add_argument("--balanced-sampler", action="store_true", default=True)
    p.add_argument("--no-balanced-sampler", action="store_false", dest="balanced_sampler")
    p.add_argument("--nesterov", action="store_true", default=False)
    p.add_argument("--no-nesterov", action="store_false", dest="nesterov")

    p.add_argument("--b2t-clip-model", default="RN50")
    p.add_argument("--pseudo-batch-size", type=int, default=256)
    p.add_argument("--pseudo-bias-path", default=None)
    p.add_argument("--overwrite-pseudo-bias", action="store_true", default=False)

    p.add_argument("--output-csv", default="b2t_dro_waterbird_sweep.csv")
    p.add_argument("--post-seeds", type=int, default=5)
    p.add_argument("--post-seed-start", type=int, default=0)
    p.add_argument("--post-output-csv", default="b2t_dro_waterbird_best5.csv")

    args = p.parse_args()
    if args.momentum_min > args.momentum_max:
        raise ValueError("--momentum-min must be <= --momentum-max")

    if args.pseudo_bias_path is None:
        stem = Path(args.output_csv).with_suffix("").name
        args.pseudo_bias_path = str(Path(args.output_csv).resolve().parent / f"{stem}_pseudo_bias.pt")

    b2t.ensure_b2t_pseudo_bias(
        data_path=args.data_path,
        pseudo_bias_path=args.pseudo_bias_path,
        clip_model_name=args.b2t_clip_model,
        batch_size=args.pseudo_batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite_pseudo_bias,
    )

    header = [
        "trial",
        "base_lr",
        "classifier_lr",
        "weight_decay",
        "momentum",
        "nesterov",
        "dro_step_size",
        "balanced_sampler",
        "best_epoch",
        "best_val_acc",
        "best_val_balanced_group",
        "best_val_worst_group",
        "test_acc",
        "test_balanced_group",
        "test_worst_group",
        "test_group_acc",
        "checkpoint",
        "pseudo_bias_path",
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
            if best_row is None or row["best_val_balanced_group"] > best_row["best_val_balanced_group"]:
                best_row = row
            print(
                f"[SWEEP] Trial {trial_id} done. val_bal_group={row['best_val_balanced_group']:.4f} "
                f"val_worst={row['best_val_worst_group']:.4f} test_worst={row['test_worst_group']:.4f} "
                f"base_lr={row['base_lr']} cls_lr={row['classifier_lr']} momentum={row['momentum']} "
                f"dro_step_size={row['dro_step_size']}",
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
            if best_row is None or row["best_val_balanced_group"] > best_row["best_val_balanced_group"]:
                best_row = row
            print(
                f"[SWEEP] Trial {trial.number} done. val_bal_group={row['best_val_balanced_group']:.4f} "
                f"val_worst={row['best_val_worst_group']:.4f} test_worst={row['test_worst_group']:.4f} "
                f"base_lr={row['base_lr']} cls_lr={row['classifier_lr']} momentum={row['momentum']} "
                f"dro_step_size={row['dro_step_size']}",
                flush=True,
            )
            return float(row["best_val_balanced_group"])

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
            momentum=float(best_row["momentum"]),
            dro_step_size=float(best_row["dro_step_size"]),
        )
        t0 = time.time()
        result = b2t.run_single(run_args)
        post_row = {
            "seed": seed,
            "base_lr": best_row["base_lr"],
            "classifier_lr": best_row["classifier_lr"],
            "weight_decay": best_row["weight_decay"],
            "momentum": best_row["momentum"],
            "nesterov": best_row["nesterov"],
            "dro_step_size": best_row["dro_step_size"],
            "balanced_sampler": best_row["balanced_sampler"],
            "best_epoch": result["best_epoch"],
            "best_val_acc": result["best_val_acc"],
            "best_val_balanced_group": result["best_val_balanced_group"],
            "best_val_worst_group": result["best_val_worst_group"],
            "test_acc": result["test_acc"],
            "test_balanced_group": result["test_balanced_group"],
            "test_worst_group": result["test_worst_group"],
            "test_group_acc": _fmt_group_acc(result["test_group_acc"]),
            "checkpoint": result["checkpoint"],
            "pseudo_bias_path": result["pseudo_bias_path"],
            "sampler": "post_best",
            "seconds": int(time.time() - t0),
        }
        _write_row(args.post_output_csv, post_row, post_header)
        post_rows.append(post_row)
        print(
            f"[POST] seed={seed} val_bal_group={post_row['best_val_balanced_group']:.4f} "
            f"test_worst={post_row['test_worst_group']:.4f} test_bal_group={post_row['test_balanced_group']:.4f}",
            flush=True,
        )

    for metric in ["test_acc", "test_balanced_group", "test_worst_group", "best_val_balanced_group"]:
        vals = np.array([float(r[metric]) for r in post_rows], dtype=float)
        print(f"[POST SUMMARY] {metric}: mean={vals.mean():.4f} std={vals.std(ddof=0):.4f}", flush=True)


if __name__ == "__main__":
    main()
