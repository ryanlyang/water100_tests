#!/usr/bin/env python3
"""Fixed-hyperparameter multi-seed RedMeat MobileNetV2 CE baseline."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


MOBILENET_ROOT = Path(__file__).resolve().parents[1]
if str(MOBILENET_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILENET_ROOT))
import common  # noqa: E402
from baseline.waterbirds_sweep import HEADER  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Run RedMeat MobileNetV2 CE baseline over fixed seeds.")
    p.add_argument("data_path")
    p.add_argument("--seed-start", "--seed_start", dest="seed_start", type=int, default=0)
    p.add_argument("--n-seeds", "--n_seeds", dest="n_seeds", type=int, default=5)
    p.add_argument("--output-csv", "--output_csv", dest="output_csv", default="redmeat_mobilenetv2_ce_fixed5.csv")
    p.add_argument("--base-lr", "--base_lr", dest="base_lr", type=float, required=True)
    p.add_argument("--classifier-lr", "--classifier_lr", dest="classifier_lr", type=float, required=True)
    p.add_argument("--momentum", type=float, required=True)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--nesterov", action="store_true", default=False)
    p.add_argument("--no-nesterov", action="store_false", dest="nesterov")
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=96)
    p.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=150)
    p.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=None)
    p.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default="MobileNetV2_CE_RedMeat_Checkpoints")
    p.add_argument("--split-col", "--split_col", dest="split_col", default="split")
    p.add_argument("--label-col", "--label_col", dest="label_col", default="label")
    p.add_argument("--path-col", "--path_col", dest="path_col", default="abs_file_path")
    p.add_argument("--classes", default=common.DEFAULT_REMEAT_CLASSES)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    args = p.parse_args()

    print(
        "[FIXED MULTI-SEED] RedMeat MobileNetV2 CE | "
        f"seeds={args.seed_start}..{args.seed_start + args.n_seeds - 1} "
        f"base_lr={args.base_lr} classifier_lr={args.classifier_lr} momentum={args.momentum}",
        flush=True,
    )
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        run_args = SimpleNamespace(**vars(args))
        run_args.seed = int(seed)
        result = common.run_baseline_redmeat(run_args)
        row = {
            "trial": int(seed),
            "base_lr": float(args.base_lr),
            "classifier_lr": float(args.classifier_lr),
            "momentum": float(args.momentum),
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
        rows.append(row)
        common.write_csv_row(args.output_csv, row, HEADER)
        print(
            f"[SEED {seed}] val_bal={row['best_balanced_val_acc']:.4f} "
            f"test_acc={row['test_acc']:.2f} per_class={row['per_group']:.2f} "
            f"worst_class={row['worst_group']:.2f}",
            flush=True,
        )

    common.summarize_rows(rows, ["best_balanced_val_acc", "test_acc", "per_group", "worst_group"])
    print(f"csv: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
