#!/usr/bin/env python3
"""Optuna sweep for Waterbirds MobileNetV2 + R4RR."""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


MOBILENET_ROOT = Path(__file__).resolve().parents[1]
if str(MOBILENET_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILENET_ROOT))
import common  # noqa: E402


HEADER = [
    "trial",
    "attention_epoch",
    "kl_lambda",
    "kl_increment",
    "base_lr",
    "classifier_lr",
    "lr2_mult",
    "momentum",
    "weight_decay",
    "batch_size",
    "num_epochs",
    "img_size",
    "best_balanced_val_acc",
    "best_epoch",
    "test_acc",
    "per_group",
    "worst_group",
    "checkpoint",
    "seconds",
]


def _write_row(csv_path, row):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_existing(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row:
                continue
            try:
                row["trial"] = int(float(row["trial"]))
                for key in ("attention_epoch", "batch_size", "num_epochs", "img_size", "best_epoch", "seconds"):
                    row[key] = int(float(row[key]))
                for key in (
                    "kl_lambda",
                    "kl_increment",
                    "base_lr",
                    "classifier_lr",
                    "lr2_mult",
                    "momentum",
                    "weight_decay",
                    "best_balanced_val_acc",
                    "test_acc",
                    "per_group",
                    "worst_group",
                ):
                    row[key] = float(row[key])
                rows.append(row)
            except Exception:
                continue
    return rows


def _enqueue_existing_trials(study, rows, args):
    try:
        import optuna
    except Exception:
        return
    distributions = {
        "attention_epoch": optuna.distributions.IntDistribution(args.attn_min, args.attn_max),
        "kl_lambda": optuna.distributions.FloatDistribution(args.kl_min, args.kl_max, log=True),
        "base_lr": optuna.distributions.FloatDistribution(args.base_lr_min, args.base_lr_max, log=True),
        "classifier_lr": optuna.distributions.FloatDistribution(args.cls_lr_min, args.cls_lr_max, log=True),
        "lr2_mult": optuna.distributions.FloatDistribution(args.lr2_mult_min, args.lr2_mult_max, log=True),
    }
    for row in rows:
        try:
            trial = optuna.trial.create_trial(
                params={
                    "attention_epoch": int(row["attention_epoch"]),
                    "kl_lambda": float(row["kl_lambda"]),
                    "base_lr": float(row["base_lr"]),
                    "classifier_lr": float(row["classifier_lr"]),
                    "lr2_mult": float(row["lr2_mult"]),
                },
                distributions=distributions,
                value=float(row["best_balanced_val_acc"]),
            )
            study.add_trial(trial)
        except Exception as exc:
            print(f"[RESUME] Skipped prior row {row.get('trial')}: {exc}", flush=True)


def _run_trial(trial, args):
    attention_epoch = int(trial.suggest_int("attention_epoch", args.attn_min, args.attn_max))
    kl_lambda = float(trial.suggest_float("kl_lambda", args.kl_min, args.kl_max, log=True))
    base_lr = float(trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
    classifier_lr = float(trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
    lr2_mult = float(trial.suggest_float("lr2_mult", args.lr2_mult_min, args.lr2_mult_max, log=True))

    run_args = SimpleNamespace(
        data_path=args.data_path,
        teacher_map_path=args.teacher_map_path,
        seed=int(args.seed),
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        lr2_mult=lr2_mult,
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        num_epochs=int(args.num_epochs),
        img_size=int(args.img_size),
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        pretrained=bool(args.pretrained),
    )
    result = common.run_guided_waterbirds(run_args, attention_epoch, kl_lambda, args.kl_increment)
    return {
        "trial": int(trial.number),
        "attention_epoch": attention_epoch,
        "kl_lambda": kl_lambda,
        "kl_increment": float(args.kl_increment) if args.kl_increment is not None else "",
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "lr2_mult": lr2_mult,
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
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
    p = argparse.ArgumentParser(description="Optuna sweep for Waterbirds MobileNetV2 R4RR.")
    p.add_argument("data_path")
    p.add_argument("teacher_map_path")
    p.add_argument("--n-trials", "--n_trials", dest="n_trials", type=int, default=50)
    p.add_argument("--additional-trials", "--additional_trials", dest="additional_trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", "--output_csv", dest="output_csv", default="waterbirds_mobilenetv2_r4rr_sweep.csv")
    p.add_argument("--resume-csv", "--resume_csv", dest="resume_csv", default=None)
    p.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default="MobileNetV2_R4RR_Checkpoints")
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=96)
    p.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=200)
    p.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=None)
    p.add_argument("--kl-increment", "--kl_increment", dest="kl_increment", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--attn-min", "--attn_min", dest="attn_min", type=int, default=0)
    p.add_argument("--attn-max", "--attn_max", dest="attn_max", type=int, default=None)
    p.add_argument("--kl-min", "--kl_min", dest="kl_min", type=float, default=1.0)
    p.add_argument("--kl-max", "--kl_max", dest="kl_max", type=float, default=500.0)
    p.add_argument("--base-lr-min", "--base_lr_min", dest="base_lr_min", type=float, default=1e-5)
    p.add_argument("--base-lr-max", "--base_lr_max", dest="base_lr_max", type=float, default=5e-2)
    p.add_argument("--cls-lr-min", "--cls_lr_min", dest="cls_lr_min", type=float, default=1e-5)
    p.add_argument("--cls-lr-max", "--cls_lr_max", dest="cls_lr_max", type=float, default=5e-2)
    p.add_argument("--lr2-mult-min", "--lr2_mult_min", dest="lr2_mult_min", type=float, default=0.1)
    p.add_argument("--lr2-mult-max", "--lr2_mult_max", dest="lr2_mult_max", type=float, default=3.0)
    args = p.parse_args()

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError(f"Optuna is required for this sweep: {exc}") from exc

    args.attn_max = int(args.num_epochs) - 1 if args.attn_max is None else min(int(args.attn_max), int(args.num_epochs) - 1)
    print(
        "[SWEEP CONFIG] Waterbirds MobileNetV2 R4RR | "
        f"trials={args.n_trials} seed={args.seed} attn=[{args.attn_min},{args.attn_max}] "
        f"kl=[{args.kl_min},{args.kl_max}] base_lr=[{args.base_lr_min},{args.base_lr_max}] "
        f"classifier_lr=[{args.cls_lr_min},{args.cls_lr_max}] lr2_mult=[{args.lr2_mult_min},{args.lr2_mult_max}] "
        f"fixed: epochs={args.num_epochs}, batch={args.batch_size}, img={args.img_size}, "
        f"momentum={args.momentum}, weight_decay={args.weight_decay}",
        flush=True,
    )

    resume_path = args.resume_csv
    if resume_path is None and os.path.exists(args.output_csv):
        resume_path = args.output_csv
    existing_rows = _load_existing(resume_path)
    if resume_path:
        print(f"[RESUME] Loaded {len(existing_rows)} prior rows from {resume_path}", flush=True)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    _enqueue_existing_trials(study, existing_rows, args)
    best_row = max(existing_rows, key=lambda r: float(r["best_balanced_val_acc"])) if existing_rows else None

    target_trials = int(args.additional_trials) if args.additional_trials is not None else max(0, int(args.n_trials) - len(existing_rows))
    print(f"[SWEEP] Running {target_trials} new trials.", flush=True)

    def objective(trial):
        nonlocal best_row
        row = _run_trial(trial, args)
        _write_row(args.output_csv, row)
        if best_row is None or float(row["best_balanced_val_acc"]) > float(best_row["best_balanced_val_acc"]):
            best_row = row
        print(
            f"[TRIAL {trial.number}] best_balanced_val_acc={row['best_balanced_val_acc']:.4f} "
            f"test_acc={row['test_acc']:.2f} per_group={row['per_group']:.2f} "
            f"worst_group={row['worst_group']:.2f} attn={row['attention_epoch']} "
            f"kl={row['kl_lambda']:.6g} base_lr={row['base_lr']:.6g} "
            f"cls_lr={row['classifier_lr']:.6g} lr2_mult={row['lr2_mult']:.6g}",
            flush=True,
        )
        return float(row["best_balanced_val_acc"])

    if target_trials > 0:
        study.optimize(objective, n_trials=target_trials, catch=(Exception,))

    print("\n[SWEEP DONE]", flush=True)
    if best_row:
        for key in HEADER:
            print(f"  {key}: {best_row[key]}", flush=True)


if __name__ == "__main__":
    main()
