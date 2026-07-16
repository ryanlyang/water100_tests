#!/usr/bin/env python3
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

# Ensure local package imports work no matter the cwd.
GALS_ROOT = Path(__file__).resolve().parent.parent
if str(GALS_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(GALS_ROOT))

from RedMeat_Runs import run_guided_redmeat_gals_vitatt as rgm  # noqa: E402
from RedMeat_Runs.optimnum_metric_redmeat import compute_guided_checkpoint_optimnum_pth  # noqa: E402


def loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def write_row(csv_path, row, header):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_runtime_summary(tag, rows, num_epochs):
    secs = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
    if not secs:
        print(f"[TIME] {tag}: no successful trials to summarize.")
        return

    arr = np.array(secs, dtype=float)
    med_trial_min = float(np.median(arr) / 60.0)
    total_gpu_hours = float(np.sum(arr) / 3600.0)
    print(f"[TIME] {tag}: median min/trial={med_trial_min:.4f} | total tuning GPU-hours={total_gpu_hours:.4f}")

    if num_epochs is not None and num_epochs > 0:
        med_epoch_min = float(np.median(arr / float(num_epochs)) / 60.0)
        print(f"[TIME] {tag}: median min/epoch={med_epoch_min:.4f} (epochs/trial={int(num_epochs)})")
    else:
        print(f"[TIME] {tag}: median min/epoch=N/A")


def _delete_checkpoint(path):
    if not path or str(path).upper() == "NONE":
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _run_single_with_params(args, seed, attn_epoch, kl_lambda, kl_incr, base_lr, classifier_lr, lr2_mult):
    rgm.SEED = int(seed)
    rgm.base_lr = float(base_lr)
    rgm.classifier_lr = float(classifier_lr)
    rgm.lr2_mult = float(lr2_mult)

    run_args = argparse.Namespace(
        data_path=args.data_path,
        att_path=args.att_path,
        att_key=args.att_key,
        att_combine=args.att_combine,
        att_norm01=args.att_norm01,
        att_brighten=args.att_brighten,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
        classes=args.class_list,
    )
    return rgm.run_single(run_args, int(attn_epoch), float(kl_lambda), float(kl_incr))


def run_trial(trial_id, args, rng, sampler_name):
    t0 = time.time()

    if sampler_name == "random":
        attn_epoch = int(rng.integers(args.attn_min, args.attn_max + 1))
        kl_lambda = loguniform(rng, args.kl_min, args.kl_max)
        kl_incr = 0.0
        base_lr = loguniform(rng, args.base_lr_min, args.base_lr_max)
        classifier_lr = loguniform(rng, args.cls_lr_min, args.cls_lr_max)
        lr2_mult = loguniform(rng, args.lr2_mult_min, args.lr2_mult_max)
    else:
        attn_epoch = int(args.trial.suggest_int("attention_epoch", args.attn_min, args.attn_max))
        kl_lambda = float(args.trial.suggest_float("kl_lambda", args.kl_min, args.kl_max, log=True))
        kl_incr = 0.0
        base_lr = float(args.trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
        classifier_lr = float(args.trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
        lr2_mult = float(args.trial.suggest_float("lr2_mult", args.lr2_mult_min, args.lr2_mult_max, log=True))

    best_balanced_val, test_acc, per_group, worst_group, ckpt = _run_single_with_params(
        args=args,
        seed=args.seed,
        attn_epoch=attn_epoch,
        kl_lambda=kl_lambda,
        kl_incr=kl_incr,
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        lr2_mult=lr2_mult,
    )

    if not ckpt or str(ckpt).upper() == "NONE":
        raise RuntimeError(
            "Guided optimnum scoring requires a saved checkpoint. "
            "Set SAVE_CHECKPOINTS=1 for this sweep."
        )

    optim_metrics = compute_guided_checkpoint_optimnum_pth(
        checkpoint_path=ckpt,
        data_path=args.data_path,
        att_path=args.att_path,
        classes=args.class_list,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
        att_key=args.att_key,
        att_combine=args.att_combine,
        att_norm01=bool(args.att_norm01),
        att_brighten=float(args.att_brighten),
        beta=float(args.optim_beta),
        batch_size=int(rgm.batch_size),
    )

    row = {
        "trial": trial_id,
        "attention_epoch": attn_epoch,
        "kl_lambda": kl_lambda,
        "kl_incr": kl_incr,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "lr2_mult": lr2_mult,
        "best_balanced_val_acc": best_balanced_val,
        "val_acc_for_optim": float(optim_metrics["val_acc"]),
        "val_ig_fwd_kl": float(optim_metrics["val_ig_fwd_kl"]),
        "log_optim_num": float(optim_metrics["log_optim_num"]),
        "optim_value": float(optim_metrics["optim_value"]),
        "optim_beta": float(args.optim_beta),
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "checkpoint": ckpt,
        "sampler": sampler_name,
        "seconds": int(time.time() - t0),
    }
    return row


def main():
    p = argparse.ArgumentParser(description="Optuna/random sweep for guided RedMeat (GALS ViT .pth guidance).")
    p.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")
    p.add_argument("att_path", help="Root containing GALS ViT attention .pth files")

    p.add_argument("--att-key", default="unnormalized_attentions", choices=["unnormalized_attentions", "attentions"])
    p.add_argument("--att-combine", default="mean", choices=["mean", "max"])
    p.add_argument("--att-norm01", action="store_true", default=True)
    p.add_argument("--att-brighten", type=float, default=1.0)

    p.add_argument("--split-col", default="split")
    p.add_argument("--label-col", default="label")
    p.add_argument("--path-col", default="abs_file_path")
    p.add_argument(
        "--classes",
        default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon",
        help="Comma-separated class list. Empty string = infer from metadata.",
    )

    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="Sweep sampler seed + trial training seed")
    p.add_argument("--output-csv", default="guided_redmeat_galsvit_sweep.csv")

    p.add_argument("--attn-min", type=int, default=0)
    p.add_argument("--attn-max", type=int, default=rgm.num_epochs - 1)
    p.add_argument("--kl-min", type=float, default=1.0)
    p.add_argument("--kl-max", type=float, default=500.0)
    p.add_argument("--base-lr-min", type=float, default=1e-5)
    p.add_argument("--base-lr-max", type=float, default=5e-2)
    p.add_argument("--cls-lr-min", type=float, default=1e-5)
    p.add_argument("--cls-lr-max", type=float, default=5e-2)
    p.add_argument("--lr2-mult-min", type=float, default=1e-1)
    p.add_argument("--lr2-mult-max", type=float, default=3.0)

    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--post-seeds", type=int, default=5)
    p.add_argument("--post-seed-start", type=int, default=0)
    p.add_argument("--post-output-csv", default=None)
    p.add_argument(
        "--optim-beta",
        type=float,
        default=10.0,
        help="Penalty strength in log_optim_num = log(val_acc) - beta * ig_fwd_kl.",
    )
    p.add_argument("--keep", choices=["best", "all", "none"], default="best")
    p.add_argument("--post-keep", choices=["all", "none"], default="none")
    p.add_argument("--num-epochs", type=int, default=rgm.num_epochs)

    args = p.parse_args()

    args.class_list = [c.strip() for c in str(args.classes).split(",") if c.strip()] if args.classes else None

    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be >= 1")
    rgm.num_epochs = int(args.num_epochs)
    args.attn_max = min(int(args.attn_max), max(0, rgm.num_epochs - 1))
    if args.attn_min > args.attn_max:
        args.attn_min = args.attn_max

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"Missing data_path: {args.data_path}")
    if not os.path.isdir(args.att_path):
        raise FileNotFoundError(f"Missing att_path: {args.att_path}")

    header = [
        "trial",
        "attention_epoch",
        "kl_lambda",
        "kl_incr",
        "base_lr",
        "classifier_lr",
        "lr2_mult",
        "best_balanced_val_acc",
        "val_acc_for_optim",
        "val_ig_fwd_kl",
        "log_optim_num",
        "optim_value",
        "optim_beta",
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
    best_ckpt = None
    sweep_rows = []

    def _record_trial_row(row):
        nonlocal best_row, best_ckpt
        write_row(args.output_csv, row, header)
        sweep_rows.append(row)

        is_new_best = best_row is None or row["optim_value"] > best_row["optim_value"]
        if args.keep == "none":
            _delete_checkpoint(row.get("checkpoint"))
        elif args.keep == "best":
            if is_new_best:
                _delete_checkpoint(best_ckpt)
                best_ckpt = row.get("checkpoint")
            else:
                _delete_checkpoint(row.get("checkpoint"))

        if is_new_best:
            best_row = row

    if args.sampler == "random":
        for trial_id in range(args.n_trials):
            row = run_trial(trial_id, args, rng, "random")
            _record_trial_row(row)
            print(
                f"[SWEEP] Trial {trial_id} done. log_optim_num={row['optim_value']:.6f} "
                f"(val_acc={row['val_acc_for_optim']:.4f}, ig_fwd_kl={row['val_ig_fwd_kl']:.6f})"
            )
    else:
        import optuna

        def objective(trial):
            nonlocal best_row
            args.trial = trial
            row = run_trial(trial.number, args, rng, "tpe")
            _record_trial_row(row)
            print(
                f"[SWEEP] Trial {trial.number} done. log_optim_num={row['optim_value']:.6f} "
                f"(val_acc={row['val_acc_for_optim']:.4f}, ig_fwd_kl={row['val_ig_fwd_kl']:.6f})"
            )
            return row["optim_value"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)

    if best_row is None:
        raise RuntimeError("No successful trials completed")

    print("[SWEEP] Best trial:")
    for k in header:
        print(f"  {k}: {best_row[k]}")

    post_rows = []
    if args.post_seeds > 0:
        post_csv = args.post_output_csv
        if post_csv is None:
            root, ext = os.path.splitext(args.output_csv)
            post_csv = f"{root}_best_seeds{ext or '.csv'}"

        post_header = [
            "phase",
            "seed",
            "attention_epoch",
            "kl_lambda",
            "kl_incr",
            "base_lr",
            "classifier_lr",
            "lr2_mult",
            "best_balanced_val_acc",
            "val_acc_for_optim",
            "val_ig_fwd_kl",
            "log_optim_num",
            "optim_value",
            "optim_beta",
            "test_acc",
            "per_group",
            "worst_group",
            "checkpoint",
            "sweep_best_trial",
            "sampler",
            "seconds",
        ]

        seeds = list(range(args.post_seed_start, args.post_seed_start + args.post_seeds))
        best_attn_epoch = int(best_row["attention_epoch"])
        best_kl_lambda = float(best_row["kl_lambda"])
        best_kl_incr = float(best_row["kl_incr"])
        best_base_lr = float(best_row["base_lr"])
        best_classifier_lr = float(best_row["classifier_lr"])
        best_lr2_mult = float(best_row["lr2_mult"])

        print(f"[POST] Rerunning best hyperparameters for {len(seeds)} seeds: {seeds}")
        for s in seeds:
            t0 = time.time()
            best_balanced_val, test_acc, per_group, worst_group, ckpt = _run_single_with_params(
                args=args,
                seed=s,
                attn_epoch=best_attn_epoch,
                kl_lambda=best_kl_lambda,
                kl_incr=best_kl_incr,
                base_lr=best_base_lr,
                classifier_lr=best_classifier_lr,
                lr2_mult=best_lr2_mult,
            )
            optim_metrics = compute_guided_checkpoint_optimnum_pth(
                checkpoint_path=ckpt,
                data_path=args.data_path,
                att_path=args.att_path,
                classes=args.class_list,
                split_col=args.split_col,
                label_col=args.label_col,
                path_col=args.path_col,
                att_key=args.att_key,
                att_combine=args.att_combine,
                att_norm01=bool(args.att_norm01),
                att_brighten=float(args.att_brighten),
                beta=float(args.optim_beta),
                batch_size=int(rgm.batch_size),
            )

            out_row = {
                "phase": "best5",
                "seed": s,
                "attention_epoch": best_attn_epoch,
                "kl_lambda": best_kl_lambda,
                "kl_incr": best_kl_incr,
                "base_lr": best_base_lr,
                "classifier_lr": best_classifier_lr,
                "lr2_mult": best_lr2_mult,
                "best_balanced_val_acc": best_balanced_val,
                "val_acc_for_optim": float(optim_metrics["val_acc"]),
                "val_ig_fwd_kl": float(optim_metrics["val_ig_fwd_kl"]),
                "log_optim_num": float(optim_metrics["log_optim_num"]),
                "optim_value": float(optim_metrics["optim_value"]),
                "optim_beta": float(args.optim_beta),
                "test_acc": test_acc,
                "per_group": per_group,
                "worst_group": worst_group,
                "checkpoint": ckpt,
                "sweep_best_trial": int(best_row["trial"]),
                "sampler": args.sampler,
                "seconds": int(time.time() - t0),
            }
            write_row(post_csv, out_row, post_header)
            post_rows.append(out_row)
            if args.post_keep == "none":
                _delete_checkpoint(out_row.get("checkpoint"))
            print(
                f"[POST] seed={s} log_optim_num={out_row['optim_value']:.6f} "
                f"test_acc={test_acc:.2f}% worst_group={worst_group:.2f}%",
                flush=True,
            )

    print_runtime_summary("sweep", sweep_rows, rgm.num_epochs)
    if post_rows:
        print_runtime_summary("post_best_seeds", post_rows, rgm.num_epochs)


if __name__ == "__main__":
    main()
