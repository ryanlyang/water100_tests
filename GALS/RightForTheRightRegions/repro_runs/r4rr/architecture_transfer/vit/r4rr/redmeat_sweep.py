#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


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


def run_trial(trial, args):
    t0 = time.time()
    attention_epoch = int(trial.suggest_int("attention_epoch", args.attn_min, args.attn_max))
    kl_lambda = float(trial.suggest_float("kl_lambda", args.kl_min, args.kl_max, log=True))
    base_lr = float(trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
    classifier_lr = float(trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
    lr2_mult = float(trial.suggest_float("lr2_mult", args.lr2_mult_min, args.lr2_mult_max, log=True))

    runner.SEED = int(args.seed)
    runner.base_lr = base_lr
    runner.classifier_lr = classifier_lr
    runner.lr2_mult = lr2_mult
    runner.momentum = float(args.momentum)
    runner.weight_decay = float(args.weight_decay)
    runner.batch_size = int(args.batch_size)
    runner.num_epochs = int(args.num_epochs)
    runner.img_size = int(args.img_size)
    runner.kl_grid = int(args.kl_grid)
    runner.fusion_beta = float(args.fusion_beta)

    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()] if args.classes else None
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
        attention_epoch,
        kl_lambda,
        args.kl_increment,
    )
    return {
        "trial": int(trial.number),
        "attention_epoch": attention_epoch,
        "kl_lambda": kl_lambda,
        "kl_increment": float(args.kl_increment),
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "lr2_mult": lr2_mult,
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
        "seconds": int(time.time() - t0),
    }


def main():
    p = argparse.ArgumentParser(description="Optuna sweep for RedMeat ViT LGM-style R4RR runner.")
    p.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")
    p.add_argument("teacher_map_path", help="Teacher-map root")
    p.add_argument("--n-trials", "--n_trials", dest="n_trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", "--output_csv", dest="output_csv", default="guided_redmeat_vit_lgmstyle_sgd_optuna.csv")
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=150)
    p.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    p.add_argument("--kl-grid", "--kl_grid", dest="kl_grid", type=int, default=14)
    p.add_argument("--kl-increment", "--kl_increment", dest="kl_increment", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--map-source", "--map_source", dest="map_source", choices=["attn", "embed", "fusion"], default="fusion")
    p.add_argument("--fusion-beta", "--fusion_beta", dest="fusion_beta", type=float, default=0.85)
    p.add_argument("--vit-model", "--vit_model", dest="vit_model", choices=["vit_b_16"], default="vit_b_16")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--split-col", "--split_col", dest="split_col", default="split")
    p.add_argument("--label-col", "--label_col", dest="label_col", default="label")
    p.add_argument("--path-col", "--path_col", dest="path_col", default="abs_file_path")
    p.add_argument("--classes", default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon")
    p.add_argument("--attn-min", "--attn_min", dest="attn_min", type=int, default=1)
    p.add_argument("--attn-max", "--attn_max", dest="attn_max", type=int, default=149)
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
        raise RuntimeError(f"Optuna is required but could not be imported: {exc}") from exc

    args.attn_max = min(int(args.attn_max), max(1, int(args.num_epochs) - 1))
    header = [
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
        "kl_grid",
        "map_source",
        "fusion_beta",
        "best_balanced_val_acc",
        "test_acc",
        "per_group",
        "worst_group",
        "checkpoint",
        "seconds",
    ]

    print(
        "[SWEEP CONFIG] RedMeat LGM-style ViT R4RR | "
        f"trials={args.n_trials} seed={args.seed} attn=[{args.attn_min},{args.attn_max}] "
        f"kl=[{args.kl_min},{args.kl_max}] base_lr=[{args.base_lr_min},{args.base_lr_max}] "
        f"classifier_lr=[{args.cls_lr_min},{args.cls_lr_max}] lr2_mult=[{args.lr2_mult_min},{args.lr2_mult_max}] "
        f"fixed: epochs={args.num_epochs}, batch={args.batch_size}, img={args.img_size}, "
        f"kl_grid={args.kl_grid}, map_source={args.map_source}, fusion_beta={args.fusion_beta}",
        flush=True,
    )

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    best_row = None

    def objective(trial):
        nonlocal best_row
        row = run_trial(trial, args)
        write_row(args.output_csv, row, header)
        if best_row is None or row["best_balanced_val_acc"] > best_row["best_balanced_val_acc"]:
            best_row = row
        print(
            f"[TRIAL {trial.number}] best_balanced_val_acc={row['best_balanced_val_acc']:.4f} "
            f"(attn={row['attention_epoch']}, kl={row['kl_lambda']:.6g}, "
            f"base_lr={row['base_lr']:.6g}, cls_lr={row['classifier_lr']:.6g}, "
            f"lr2_mult={row['lr2_mult']:.6g})",
            flush=True,
        )
        return row["best_balanced_val_acc"]

    study.optimize(objective, n_trials=int(args.n_trials), catch=(Exception,))

    print("\n[SWEEP DONE]", flush=True)
    if best_row is None:
        print("No successful trials.", flush=True)
        return
    print("Best trial row:", flush=True)
    for k in header:
        print(f"  {k}: {best_row[k]}", flush=True)


if __name__ == "__main__":
    main()
