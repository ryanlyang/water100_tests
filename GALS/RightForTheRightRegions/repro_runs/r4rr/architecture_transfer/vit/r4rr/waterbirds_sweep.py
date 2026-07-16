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
import waterbirds as lgm  # noqa: E402


def write_row(csv_path, row, header):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _coerce_existing_row(row):
    out = dict(row)
    int_keys = [
        "trial",
        "attention_epoch",
        "batch_size",
        "num_epochs",
        "img_size",
        "kl_grid",
        "seconds",
    ]
    float_keys = [
        "kl_lambda",
        "kl_increment",
        "base_lr",
        "classifier_lr",
        "lr2_mult",
        "momentum",
        "weight_decay",
        "fusion_beta",
        "best_balanced_val_acc",
        "test_acc",
        "per_group",
        "worst_group",
    ]
    for k in int_keys:
        if k in out and str(out[k]).strip() != "":
            out[k] = int(float(out[k]))
    for k in float_keys:
        if k in out and str(out[k]).strip() != "":
            out[k] = float(out[k])
    return out


def load_existing_rows(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [_coerce_existing_row(r) for r in reader]
    return rows


def run_trial(trial, args):
    t0 = time.time()

    attention_epoch = int(trial.suggest_int("attention_epoch", args.attn_min, args.attn_max))
    kl_lambda = float(trial.suggest_float("kl_lambda", args.kl_min, args.kl_max, log=True))
    base_lr = float(trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
    classifier_lr = float(trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
    lr2_mult = float(trial.suggest_float("lr2_mult", args.lr2_mult_min, args.lr2_mult_max, log=True))

    lgm.SEED = int(args.seed)
    lgm.base_lr = base_lr
    lgm.classifier_lr = classifier_lr
    lgm.lr2_mult = lr2_mult
    lgm.momentum = float(args.momentum)
    lgm.weight_decay = float(args.weight_decay)
    lgm.batch_size = int(args.batch_size)
    lgm.num_epochs = int(args.num_epochs)
    lgm.img_size = int(args.img_size)
    lgm.kl_grid = int(args.kl_grid)
    lgm.fusion_beta = float(args.fusion_beta)

    run_args = SimpleNamespace(
        data_path=args.data_path,
        teacher_map_path=args.teacher_map_path,
        vit_model=args.vit_model,
        pretrained=args.pretrained,
        map_source=args.map_source,
        fusion_beta=args.fusion_beta,
    )

    best_balanced_val, test_acc, per_group, worst_group, ckpt = lgm.run_single(
        run_args,
        attention_epoch,
        kl_lambda,
        args.kl_increment,
    )

    row = {
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
    return row


def main():
    p = argparse.ArgumentParser(
        description="Optuna sweep for Waterbirds ViT LGM-style SGD runner."
    )
    p.add_argument("data_path", help="Waterbirds dataset root")
    p.add_argument("teacher_map_path", help="Teacher-map root folder")

    p.add_argument("--n-trials", "--n_trials", dest="n_trials", type=int, default=50)
    p.add_argument(
        "--additional-trials",
        "--additional_trials",
        dest="additional_trials",
        type=int,
        default=None,
        help=(
            "If set, run this many NEW trials on top of any resumed rows "
            "(ignores --n-trials counting behavior)."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", "--output_csv", dest="output_csv", default="guided_waterbirds_vit_lgmstyle_sgd_640_optuna.csv")
    p.add_argument(
        "--resume-csv",
        "--resume_csv",
        dest="resume_csv",
        default=None,
        help=(
            "Optional existing sweep CSV to resume from. If omitted and --output-csv exists, "
            "that file is used automatically."
        ),
    )

    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=200)
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

    # User-specified search space.
    p.add_argument("--attn-min", "--attn_min", dest="attn_min", type=int, default=1)
    p.add_argument("--attn-max", "--attn_max", dest="attn_max", type=int, default=200)
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
        raise RuntimeError(
            f"Optuna is required for this sweep script but could not be imported: {exc}"
        ) from exc

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
        "[SWEEP CONFIG] "
        f"trials={args.n_trials} seed={args.seed} "
        f"attn=[{args.attn_min},{args.attn_max}] "
        f"kl=[{args.kl_min},{args.kl_max}] "
        f"base_lr=[{args.base_lr_min},{args.base_lr_max}] "
        f"classifier_lr=[{args.cls_lr_min},{args.cls_lr_max}] "
        f"lr2_mult=[{args.lr2_mult_min},{args.lr2_mult_max}] "
        f"| fixed: epochs={args.num_epochs}, batch={args.batch_size}, img={args.img_size}, "
        f"kl_grid={args.kl_grid}, map_source={args.map_source}, fusion_beta={args.fusion_beta}, "
        f"momentum={args.momentum}, weight_decay={args.weight_decay}, kl_increment={args.kl_increment}",
        flush=True,
    )

    resume_path = args.resume_csv
    if resume_path is None and os.path.exists(args.output_csv):
        resume_path = args.output_csv

    existing_rows = load_existing_rows(resume_path)
    if resume_path:
        if existing_rows:
            print(f"[RESUME] Loaded {len(existing_rows)} prior rows from: {resume_path}", flush=True)
        else:
            print(f"[RESUME] No usable prior rows found at: {resume_path}", flush=True)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    best_row = None
    if existing_rows:
        try:
            best_row = max(existing_rows, key=lambda r: float(r["best_balanced_val_acc"]))
        except Exception:
            best_row = None

    if existing_rows:
        distributions = {
            "attention_epoch": optuna.distributions.IntDistribution(args.attn_min, args.attn_max),
            "kl_lambda": optuna.distributions.FloatDistribution(args.kl_min, args.kl_max, log=True),
            "base_lr": optuna.distributions.FloatDistribution(args.base_lr_min, args.base_lr_max, log=True),
            "classifier_lr": optuna.distributions.FloatDistribution(args.cls_lr_min, args.cls_lr_max, log=True),
            "lr2_mult": optuna.distributions.FloatDistribution(args.lr2_mult_min, args.lr2_mult_max, log=True),
        }
        added = 0
        for r in existing_rows:
            try:
                params = {
                    "attention_epoch": int(r["attention_epoch"]),
                    "kl_lambda": float(r["kl_lambda"]),
                    "base_lr": float(r["base_lr"]),
                    "classifier_lr": float(r["classifier_lr"]),
                    "lr2_mult": float(r["lr2_mult"]),
                }
                value = float(r["best_balanced_val_acc"])
                trial = optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    value=value,
                )
                study.add_trial(trial)
                added += 1
            except Exception as exc:
                print(f"[RESUME] Skipping malformed row (trial={r.get('trial')}): {exc}", flush=True)
        print(f"[RESUME] Added {added} prior trials into Optuna study state.", flush=True)

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

    if args.additional_trials is not None:
        n_new = max(0, int(args.additional_trials))
        print(f"[RESUME] Running additional_trials={n_new} new trials.", flush=True)
    else:
        n_new = max(0, int(args.n_trials) - len(existing_rows))
        print(
            f"[RESUME] Running n_new={n_new} trials to reach total n_trials={args.n_trials}.",
            flush=True,
        )

    if n_new > 0:
        study.optimize(objective, n_trials=n_new, catch=(Exception,))
    else:
        print("[RESUME] No new trials requested; skipping optimization.", flush=True)

    print("\n[SWEEP DONE]", flush=True)
    if best_row is None:
        print("No successful trials.", flush=True)
        return

    print("Best trial row:", flush=True)
    for k in header:
        print(f"  {k}: {best_row[k]}", flush=True)


if __name__ == "__main__":
    main()
