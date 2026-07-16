#!/usr/bin/env python3
"""Fixed-hyperparameter multi-seed RedMeat MobileNetV2 + R4RR runner."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


MOBILENET_ROOT = Path(__file__).resolve().parents[1]
if str(MOBILENET_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILENET_ROOT))
import common  # noqa: E402
from r4rr.waterbirds_fixed_seeds import HEADER  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Run RedMeat MobileNetV2 R4RR over fixed seeds.")
    p.add_argument("data_path")
    p.add_argument("teacher_map_path")
    p.add_argument("--seed-start", "--seed_start", dest="seed_start", type=int, default=0)
    p.add_argument("--n-seeds", "--n_seeds", dest="n_seeds", type=int, default=5)
    p.add_argument("--output-csv", "--output_csv", dest="output_csv", default="redmeat_mobilenetv2_r4rr_fixed5.csv")
    p.add_argument("--attention-epoch", "--attention_epoch", dest="attention_epoch", type=int, required=True)
    p.add_argument("--kl-lambda", "--kl_lambda", dest="kl_lambda", type=float, required=True)
    p.add_argument("--base-lr", "--base_lr", dest="base_lr", type=float, required=True)
    p.add_argument("--classifier-lr", "--classifier_lr", dest="classifier_lr", type=float, required=True)
    p.add_argument("--lr2-mult", "--lr2_mult", dest="lr2_mult", type=float, required=True)
    p.add_argument("--kl-increment", "--kl_increment", dest="kl_increment", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=96)
    p.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=150)
    p.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=None)
    p.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default="MobileNetV2_R4RR_RedMeat_Checkpoints")
    p.add_argument("--split-col", "--split_col", dest="split_col", default="split")
    p.add_argument("--label-col", "--label_col", dest="label_col", default="label")
    p.add_argument("--path-col", "--path_col", dest="path_col", default="abs_file_path")
    p.add_argument("--classes", default=common.DEFAULT_REMEAT_CLASSES)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    args = p.parse_args()

    print(
        "[FIXED MULTI-SEED] RedMeat MobileNetV2 R4RR | "
        f"seeds={args.seed_start}..{args.seed_start + args.n_seeds - 1} "
        f"attn={args.attention_epoch} kl={args.kl_lambda} base_lr={args.base_lr} "
        f"classifier_lr={args.classifier_lr} lr2_mult={args.lr2_mult}",
        flush=True,
    )

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        run_args = SimpleNamespace(**vars(args))
        run_args.seed = int(seed)
        result = common.run_guided_redmeat(run_args, int(args.attention_epoch), float(args.kl_lambda), args.kl_increment)
        row = {
            "seed": int(seed),
            "attention_epoch": int(args.attention_epoch),
            "kl_lambda": float(args.kl_lambda),
            "kl_increment": float(args.kl_increment) if args.kl_increment is not None else "",
            "base_lr": float(args.base_lr),
            "classifier_lr": float(args.classifier_lr),
            "lr2_mult": float(args.lr2_mult),
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
