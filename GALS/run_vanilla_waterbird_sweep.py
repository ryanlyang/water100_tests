#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import run_vanilla_waterbird as rvw


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _write_row(csv_path: str, row: dict, header: list):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _print_runtime_summary(tag: str, rows: list, num_epochs: int):
    secs = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
    if not secs:
        print(f"[TIME] {tag}: no successful trials to summarize.")
        return
    arr = np.array(secs, dtype=float)
    med_trial_min = float(np.median(arr) / 60.0)
    med_epoch_min = float(np.median(arr / float(num_epochs)) / 60.0) if num_epochs > 0 else float("nan")
    total_gpu_hours = float(np.sum(arr) / 3600.0)
    print(f"[TIME] {tag}: median min/trial={med_trial_min:.4f} | total tuning GPU-hours={total_gpu_hours:.4f}")
    if num_epochs > 0:
        print(f"[TIME] {tag}: median min/epoch={med_epoch_min:.4f} (epochs/trial={int(num_epochs)})")
    else:
        print(f"[TIME] {tag}: median min/epoch=N/A (num_epochs <= 0)")


def _build_run_args(
    args,
    seed: int,
    base_lr: float,
    classifier_lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
):
    return SimpleNamespace(
        data_path=args.data_path,
        seed=seed,
        model=args.model,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=base_lr,
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
    )


def _run_trial(trial_id: int, args, rng: np.random.Generator, sampler_name: str) -> dict:
    t0 = time.time()
    if sampler_name == "random":
        base_lr = _loguniform(rng, args.base_lr_min, args.base_lr_max)
        classifier_lr = _loguniform(rng, args.cls_lr_min, args.cls_lr_max)
        momentum = float(rng.uniform(args.momentum_min, args.momentum_max))
    else:
        base_lr = float(args.trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
        classifier_lr = float(args.trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
        momentum = float(args.trial.suggest_float("momentum", args.momentum_min, args.momentum_max))

    run_args = _build_run_args(
        args=args,
        seed=args.train_seed,
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        weight_decay=args.weight_decay,
        momentum=momentum,
        nesterov=args.nesterov,
    )

    best_balanced_val, test_acc, per_group, worst_group, ckpt = rvw.run_single(run_args)

    return {
        "trial": trial_id,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "weight_decay": args.weight_decay,
        "momentum": momentum,
        "nesterov": args.nesterov,
        "best_balanced_val_acc": best_balanced_val,
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "checkpoint": ckpt,
        "sampler": sampler_name,
        "seconds": int(time.time() - t0),
    }


def main():
    p = argparse.ArgumentParser(description="Optuna/random sweep for vanilla Waterbirds CNN.")
    p.add_argument("data_path", help="Waterbirds dataset root containing metadata.csv")

    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="Sweep sampler seed")
    p.add_argument("--train-seed", type=int, default=0, help="Fixed training seed during hyperparameter search")
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")

    p.add_argument("--model", choices=["resnet50", "resnet18"], default="resnet50")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-dir", default="Vanilla_Checkpoints")

    # Vanilla sweep now mirrors GALS-style two-LR setup.
    p.add_argument("--base-lr-min", type=float, default=1e-5)
    p.add_argument("--base-lr-max", type=float, default=5e-2)
    p.add_argument("--cls-lr-min", type=float, default=1e-5)
    p.add_argument("--cls-lr-max", type=float, default=5e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--momentum-min", type=float, default=0.85)
    p.add_argument("--momentum-max", type=float, default=0.95)
    p.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Optional fixed momentum override (if set, momentum-min/max are ignored).",
    )
    p.add_argument("--nesterov", action="store_true", default=False)
    p.add_argument("--no-nesterov", action="store_false", dest="nesterov")

    p.add_argument("--output-csv", default="vanilla_waterbird_sweep.csv")

    p.add_argument("--post-seeds", type=int, default=5)
    p.add_argument("--post-seed-start", type=int, default=0)
    p.add_argument("--post-output-csv", default="vanilla_waterbird_best5.csv")

    args = p.parse_args()

    if args.momentum is not None:
        args.momentum_min = float(args.momentum)
        args.momentum_max = float(args.momentum)
    if args.momentum_min > args.momentum_max:
        raise ValueError("--momentum-min must be <= --momentum-max")

    header = [
        "trial",
        "base_lr",
        "classifier_lr",
        "weight_decay",
        "momentum",
        "nesterov",
        "best_balanced_val_acc",
        "test_acc",
        "per_group",
        "worst_group",
        "checkpoint",
        "sampler",
        "seconds",
    ]

    rng = np.random.default_rng(args.seed)

    if args.sampler == "tpe":
        try:
            import optuna  # noqa: F401
        except Exception as exc:
            print(f"[SWEEP] Optuna not available ({exc}); falling back to random search.")
            args.sampler = "random"

    best_row = None
    sweep_rows = []

    if args.sampler == "random":
        for trial_id in range(args.n_trials):
            row = _run_trial(trial_id, args, rng, "random")
            _write_row(args.output_csv, row, header)
            sweep_rows.append(row)
            if best_row is None or row["best_balanced_val_acc"] > best_row["best_balanced_val_acc"]:
                best_row = row
            print(f"[SWEEP] Trial {trial_id} done. best_balanced_val_acc={row['best_balanced_val_acc']:.4f}")
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
            if best_row is None or row["best_balanced_val_acc"] > best_row["best_balanced_val_acc"]:
                best_row = row
            print(f"[SWEEP] Trial {trial.number} done. best_balanced_val_acc={row['best_balanced_val_acc']:.4f}")
            return row["best_balanced_val_acc"]

        study.optimize(objective, n_trials=args.n_trials)

    if best_row is None:
        raise SystemExit("No trials completed")

    print("[SWEEP] Best trial:")
    for k in header:
        print(f"  {k}: {best_row[k]}")

    post_header = [
        "phase",
        "seed",
        "base_lr",
        "classifier_lr",
        "weight_decay",
        "momentum",
        "nesterov",
        "best_balanced_val_acc",
        "test_acc",
        "per_group",
        "worst_group",
        "checkpoint",
        "seconds",
    ]

    best_base_lr = float(best_row["base_lr"])
    best_classifier_lr = float(best_row["classifier_lr"])
    best_wd = float(best_row["weight_decay"])
    best_momentum = float(best_row["momentum"])
    best_nesterov = str(best_row["nesterov"]).lower() in ("1", "true", "yes")

    print(f"[POST] Running best hyperparameters on {args.post_seeds} seeds...")
    post_rows = []
    for i in range(args.post_seeds):
        seed = args.post_seed_start + i
        run_args = _build_run_args(
            args=args,
            seed=seed,
            base_lr=best_base_lr,
            classifier_lr=best_classifier_lr,
            weight_decay=best_wd,
            momentum=best_momentum,
            nesterov=best_nesterov,
        )
        t0 = time.time()
        best_balanced_val, test_acc, per_group, worst_group, ckpt = rvw.run_single(run_args)
        row = {
            "phase": "best5",
            "seed": seed,
            "base_lr": best_base_lr,
            "classifier_lr": best_classifier_lr,
            "weight_decay": best_wd,
            "momentum": best_momentum,
            "nesterov": best_nesterov,
            "best_balanced_val_acc": best_balanced_val,
            "test_acc": test_acc,
            "per_group": per_group,
            "worst_group": worst_group,
            "checkpoint": ckpt,
            "seconds": int(time.time() - t0),
        }
        _write_row(args.post_output_csv, row, post_header)
        post_rows.append(row)
        print(
            f"[POST] seed={seed} best_balanced_val_acc={best_balanced_val:.4f} test_acc={test_acc:.2f}%",
            flush=True,
        )

    _print_runtime_summary("sweep", sweep_rows, args.num_epochs)
    if post_rows:
        _print_runtime_summary("post_best_seeds", post_rows, args.num_epochs)


if __name__ == "__main__":
    main()
