#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RUNNER_ROOT = Path(__file__).resolve().parent
if str(RUNNER_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNNER_ROOT))
import redmeat as runner  # noqa: E402


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
            "Run RedMeat LGM-style ViT R4RR with fixed hyperparameters across "
            "multiple seeds and report mean/std."
        )
    )
    p.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")
    p.add_argument("teacher_map_path", help="Teacher-map root")
    p.add_argument("--seed-start", "--seed_start", dest="seed_start", type=int, default=0)
    p.add_argument("--n-seeds", "--n_seeds", dest="n_seeds", type=int, default=5)
    p.add_argument("--output-csv", "--output_csv", dest="output_csv", default="guided_redmeat_vit_lgmstyle_fixed5seeds.csv")

    p.add_argument("--attention-epoch", "--attention_epoch", dest="attention_epoch", type=int, default=97)
    p.add_argument("--kl-lambda", "--kl_lambda", dest="kl_lambda", type=float, default=17.882370346219354)
    p.add_argument("--kl-increment", "--kl_increment", dest="kl_increment", type=float, default=0.0)
    p.add_argument("--base-lr", "--base_lr", dest="base_lr", type=float, default=0.00031577478613513036)
    p.add_argument("--classifier-lr", "--classifier_lr", dest="classifier_lr", type=float, default=0.012794789837853385)
    p.add_argument("--lr2-mult", "--lr2_mult", dest="lr2_mult", type=float, default=0.8401970664045666)

    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=150)
    p.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    p.add_argument("--kl-grid", "--kl_grid", dest="kl_grid", type=int, default=14)
    p.add_argument("--map-source", "--map_source", dest="map_source", choices=["attn", "embed", "fusion"], default="fusion")
    p.add_argument("--fusion-beta", "--fusion_beta", dest="fusion_beta", type=float, default=0.85)
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
        "kl_grid",
        "map_source",
        "fusion_beta",
        "best_balanced_val_acc",
        "test_acc",
        "per_group",
        "worst_group",
        "checkpoint",
    ]

    print(
        "[FIXED MULTI-SEED] RedMeat LGM-style ViT R4RR | "
        f"seeds={args.seed_start}..{args.seed_start + args.n_seeds - 1} | "
        f"attention_epoch={args.attention_epoch} kl_lambda={args.kl_lambda} "
        f"kl_increment={args.kl_increment} base_lr={args.base_lr} "
        f"classifier_lr={args.classifier_lr} lr2_mult={args.lr2_mult} "
        f"momentum={args.momentum} weight_decay={args.weight_decay} "
        f"batch={args.batch_size} epochs={args.num_epochs} img_size={args.img_size} "
        f"kl_grid={args.kl_grid} map_source={args.map_source} fusion_beta={args.fusion_beta}",
        flush=True,
    )

    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()] if args.classes else None
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        runner.SEED = int(seed)
        runner.base_lr = float(args.base_lr)
        runner.classifier_lr = float(args.classifier_lr)
        runner.lr2_mult = float(args.lr2_mult)
        runner.momentum = float(args.momentum)
        runner.weight_decay = float(args.weight_decay)
        runner.batch_size = int(args.batch_size)
        runner.num_epochs = int(args.num_epochs)
        runner.img_size = int(args.img_size)
        runner.kl_grid = int(args.kl_grid)
        runner.fusion_beta = float(args.fusion_beta)

        run_args = SimpleNamespace(
            data_path=args.data_path,
            teacher_map_path=args.teacher_map_path,
            split_col=args.split_col,
            label_col=args.label_col,
            path_col=args.path_col,
            classes=classes,
            vit_model=args.vit_model,
            pretrained=args.pretrained,
            map_source=args.map_source,
            fusion_beta=args.fusion_beta,
        )
        best_balanced_val, test_acc, per_group, worst_group, ckpt = runner.run_single(
            run_args,
            int(args.attention_epoch),
            float(args.kl_lambda),
            float(args.kl_increment),
        )
        row = {
            "seed": seed,
            "attention_epoch": int(args.attention_epoch),
            "kl_lambda": float(args.kl_lambda),
            "kl_increment": float(args.kl_increment),
            "base_lr": float(args.base_lr),
            "classifier_lr": float(args.classifier_lr),
            "lr2_mult": float(args.lr2_mult),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "num_epochs": int(args.num_epochs),
            "img_size": int(args.img_size),
            "kl_grid": int(args.kl_grid),
            "map_source": args.map_source,
            "fusion_beta": float(args.fusion_beta),
            "best_balanced_val_acc": float(best_balanced_val),
            "test_acc": float(test_acc),
            "per_group": float(per_group),
            "worst_group": float(worst_group),
            "checkpoint": ckpt,
        }
        rows.append(row)
        write_row(args.output_csv, row, header)
        print(
            f"[SEED {seed}] best_balanced_val_acc={row['best_balanced_val_acc']:.4f} "
            f"test_acc={row['test_acc']:.2f}% per_group={row['per_group']:.2f}% "
            f"worst_group={row['worst_group']:.2f}%",
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
