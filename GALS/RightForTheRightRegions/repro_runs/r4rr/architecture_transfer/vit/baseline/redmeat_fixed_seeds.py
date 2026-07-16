#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parent
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))
import redmeat_sweep as baseline  # noqa: E402


def write_row(csv_path, row, header):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Run RedMeat CE-only ViT ERM baseline with fixed hyperparameters across "
            "multiple seeds and report mean/std."
        )
    )
    p.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")
    p.add_argument("--seed-start", "--seed_start", dest="seed_start", type=int, default=0)
    p.add_argument("--n-seeds", "--n_seeds", dest="n_seeds", type=int, default=5)
    p.add_argument("--output-csv", "--output_csv", dest="output_csv", default="guided_redmeat_vit_baseline_fixed5seeds.csv")

    p.add_argument("--base-lr", "--base_lr", dest="base_lr", type=float, default=0.00010889412366851132)
    p.add_argument("--classifier-lr", "--classifier_lr", dest="classifier_lr", type=float, default=0.0006047233305652243)
    p.add_argument("--momentum", type=float, default=0.8815357624514035)

    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=150)
    p.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    p.add_argument("--vit-model", "--vit_model", dest="vit_model", choices=["vit_b_16"], default="vit_b_16")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--split-col", "--split_col", dest="split_col", default="split")
    p.add_argument("--label-col", "--label_col", dest="label_col", default="label")
    p.add_argument("--path-col", "--path_col", dest="path_col", default="abs_file_path")
    p.add_argument("--classes", default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon")
    args = p.parse_args()

    header = [
        "seed",
        "base_lr",
        "classifier_lr",
        "momentum",
        "weight_decay",
        "batch_size",
        "num_epochs",
        "img_size",
        "best_balanced_val_acc",
        "best_epoch",
        "test_acc",
        "test_loss",
        "per_group",
        "worst_group",
        "seconds",
    ]

    print(
        "[FIXED MULTI-SEED] RedMeat baseline CE-only (R4RR off) | "
        f"seeds={args.seed_start}..{args.seed_start + args.n_seeds - 1} | "
        f"base_lr={args.base_lr} classifier_lr={args.classifier_lr} momentum={args.momentum} "
        f"weight_decay={args.weight_decay} batch={args.batch_size} epochs={args.num_epochs} "
        f"img_size={args.img_size}",
        flush=True,
    )

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        run_args = SimpleNamespace(
            data_path=args.data_path,
            seed=seed,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            img_size=args.img_size,
            weight_decay=args.weight_decay,
            vit_model=args.vit_model,
            pretrained=args.pretrained,
            split_col=args.split_col,
            label_col=args.label_col,
            path_col=args.path_col,
            classes=args.classes,
        )
        row = baseline.run_single_trial(
            args=run_args,
            trial_number=seed,
            base_lr=args.base_lr,
            classifier_lr=args.classifier_lr,
            momentum=args.momentum,
        )
        out_row = {
            "seed": seed,
            "base_lr": row["base_lr"],
            "classifier_lr": row["classifier_lr"],
            "momentum": row["momentum"],
            "weight_decay": row["weight_decay"],
            "batch_size": row["batch_size"],
            "num_epochs": row["num_epochs"],
            "img_size": row["img_size"],
            "best_balanced_val_acc": row["best_balanced_val_acc"],
            "best_epoch": row["best_epoch"],
            "test_acc": row["test_acc"],
            "test_loss": row["test_loss"],
            "per_group": row["per_group"],
            "worst_group": row["worst_group"],
            "seconds": row["seconds"],
        }
        rows.append(out_row)
        write_row(args.output_csv, out_row, header)
        print(
            f"[SEED {seed}] best_balanced_val_acc={out_row['best_balanced_val_acc']:.4f} "
            f"test_acc={out_row['test_acc']:.2f}% per_group={out_row['per_group']:.2f}% "
            f"worst_group={out_row['worst_group']:.2f}%",
            flush=True,
        )

    def _mean_std(key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        arr = np.asarray(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    bal_mean, bal_std = _mean_std("best_balanced_val_acc")
    test_mean, test_std = _mean_std("test_acc")
    per_group_mean, per_group_std = _mean_std("per_group")
    worst_mean, worst_std = _mean_std("worst_group")

    print("\n[MULTI-SEED DONE]", flush=True)
    if bal_mean is not None:
        print(f"best_balanced_val_acc: mean={bal_mean:.4f}, std={bal_std:.4f}", flush=True)
    if test_mean is not None:
        print(f"test_acc: mean={test_mean:.2f}%, std={test_std:.2f}", flush=True)
    if per_group_mean is not None:
        print(f"test_per_class_mean: mean={per_group_mean:.2f}%, std={per_group_std:.2f}", flush=True)
    if worst_mean is not None:
        print(f"test_worst_class: mean={worst_mean:.2f}%, std={worst_std:.2f}", flush=True)
    print(f"csv: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
