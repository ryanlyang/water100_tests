#!/usr/bin/env python3
"""Optuna sweep for RedMeat MobileNetV2 CE-only baseline."""

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace


MOBILENET_ROOT = Path(__file__).resolve().parents[1]
if str(MOBILENET_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILENET_ROOT))
import common  # noqa: E402
from baseline.waterbirds_sweep import HEADER, _write_row  # noqa: E402


def _run_trial(trial, args):
    base_lr = float(trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
    classifier_lr = float(trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
    momentum = float(trial.suggest_float("momentum", args.momentum_min, args.momentum_max))
    run_args = SimpleNamespace(
        data_path=args.data_path,
        seed=int(args.seed),
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        momentum=momentum,
        weight_decay=float(args.weight_decay),
        nesterov=bool(args.nesterov),
        batch_size=int(args.batch_size),
        num_epochs=int(args.num_epochs),
        img_size=int(args.img_size),
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        pretrained=bool(args.pretrained),
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
        classes=args.classes,
    )
    result = common.run_baseline_redmeat(run_args)
    return {
        "trial": int(trial.number),
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "momentum": momentum,
        "weight_decay": float(args.weight_decay),
        "nesterov": bool(args.nesterov),
        "batch_size": int(args.batch_size),
        "num_epochs": int(args.num_epochs),
        "img_size": int(args.img_size),
        "best_balanced_val_acc": result.best_balanced_val_acc,
        "best_epoch": result.best_epoch,
        "test_acc": result.test_acc,
        "per_group": result.per_group,
        "worst_group": result.worst_group,
        "checkpoint": result.checkpoint,
        "seconds": result.seconds,
    }


def main():
    p = argparse.ArgumentParser(description="Optuna sweep for RedMeat MobileNetV2 CE baseline.")
    p.add_argument("data_path")
    p.add_argument("--n-trials", "--n_trials", dest="n_trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", "--output_csv", dest="output_csv", default="redmeat_mobilenetv2_ce_sweep.csv")
    p.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default="MobileNetV2_CE_RedMeat_Checkpoints")
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=96)
    p.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=150)
    p.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=None)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--nesterov", action="store_true", default=False)
    p.add_argument("--no-nesterov", action="store_false", dest="nesterov")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--split-col", "--split_col", dest="split_col", default="split")
    p.add_argument("--label-col", "--label_col", dest="label_col", default="label")
    p.add_argument("--path-col", "--path_col", dest="path_col", default="abs_file_path")
    p.add_argument("--classes", default=common.DEFAULT_REMEAT_CLASSES)
    p.add_argument("--base-lr-min", "--base_lr_min", dest="base_lr_min", type=float, default=1e-5)
    p.add_argument("--base-lr-max", "--base_lr_max", dest="base_lr_max", type=float, default=5e-2)
    p.add_argument("--cls-lr-min", "--cls_lr_min", dest="cls_lr_min", type=float, default=1e-5)
    p.add_argument("--cls-lr-max", "--cls_lr_max", dest="cls_lr_max", type=float, default=5e-2)
    p.add_argument("--momentum-min", "--momentum_min", dest="momentum_min", type=float, default=0.85)
    p.add_argument("--momentum-max", "--momentum_max", dest="momentum_max", type=float, default=0.95)
    args = p.parse_args()

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError(f"Optuna is required for this sweep: {exc}") from exc

    print(
        "[SWEEP CONFIG] RedMeat MobileNetV2 CE | "
        f"trials={args.n_trials} seed={args.seed} base_lr=[{args.base_lr_min},{args.base_lr_max}] "
        f"classifier_lr=[{args.cls_lr_min},{args.cls_lr_max}] momentum=[{args.momentum_min},{args.momentum_max}] "
        f"fixed: epochs={args.num_epochs}, batch={args.batch_size}, img={args.img_size}, "
        f"weight_decay={args.weight_decay}, nesterov={args.nesterov}",
        flush=True,
    )
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    best_row = None

    def objective(trial):
        nonlocal best_row
        row = _run_trial(trial, args)
        _write_row(args.output_csv, row)
        if best_row is None or float(row["best_balanced_val_acc"]) > float(best_row["best_balanced_val_acc"]):
            best_row = row
        print(
            f"[TRIAL {trial.number}] best_balanced_val_acc={row['best_balanced_val_acc']:.4f} "
            f"test_acc={row['test_acc']:.2f} per_class={row['per_group']:.2f} "
            f"worst_class={row['worst_group']:.2f} base_lr={row['base_lr']:.6g} "
            f"cls_lr={row['classifier_lr']:.6g} momentum={row['momentum']:.6g}",
            flush=True,
        )
        return float(row["best_balanced_val_acc"])

    study.optimize(objective, n_trials=int(args.n_trials), catch=(Exception,))

    print("\n[SWEEP DONE]", flush=True)
    if best_row:
        for key in HEADER:
            print(f"  {key}: {best_row[key]}", flush=True)


if __name__ == "__main__":
    main()
