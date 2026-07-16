import argparse
import csv
import os
import time
from types import SimpleNamespace

import numpy as np

import run_guided_waterbird as rgw


def loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def write_row(csv_path, row, header):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_trial(trial_id, args, rng, sampler_name):
    attn_epoch = 0
    if sampler_name == "random":
        kl_lambda = loguniform(rng, args.kl_min, args.kl_max)
        kl_incr = loguniform(rng, args.kl_incr_min, args.kl_incr_max)
        base_lr = loguniform(rng, args.base_lr_min, args.base_lr_max)
        classifier_lr = loguniform(rng, args.cls_lr_min, args.cls_lr_max)
    else:
        kl_lambda = float(args.trial.suggest_float("kl_lambda", args.kl_min, args.kl_max, log=True))
        kl_incr = float(args.trial.suggest_float("kl_increment", args.kl_incr_min, args.kl_incr_max, log=True))
        base_lr = float(args.trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
        classifier_lr = float(args.trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))

    rgw.base_lr = base_lr
    rgw.classifier_lr = classifier_lr
    rgw.SEED = args.seed

    run_args = SimpleNamespace(data_path=args.data_path, gt_path=args.gt_path)
    best_balanced_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(
        run_args, attn_epoch, kl_lambda, kl_incr
    )

    row = {
        "trial": trial_id,
        "attention_epoch": attn_epoch,
        "kl_lambda": kl_lambda,
        "kl_increment": kl_incr,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "best_balanced_val_acc": best_balanced_val,
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "checkpoint": ckpt,
        "sampler": sampler_name,
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Waterbirds dataset root with metadata.csv")
    parser.add_argument("gt_path", help="Mask root (required by run_guided_waterbird.py)")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default="guided_waterbird_sweep_attn0.csv")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Stop launching new trials after this many hours")
    parser.add_argument("--post-seeds", default="",
                        help="Space-separated seeds to run after sweep with best params")
    parser.add_argument("--seeds-csv", default="guided_waterbird_sweep_attn0_seeds.csv")
    parser.add_argument("--kl-min", type=float, default=1.0)
    parser.add_argument("--kl-max", type=float, default=100000.0)
    parser.add_argument("--kl-incr-min", type=float, default=0.1)
    parser.add_argument("--kl-incr-max", type=float, default=10000.0)
    parser.add_argument("--base-lr-min", type=float, default=1e-6)
    parser.add_argument("--base-lr-max", type=float, default=1e-3)
    parser.add_argument("--cls-lr-min", type=float, default=1e-5)
    parser.add_argument("--cls-lr-max", type=float, default=1e-2)
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    args = parser.parse_args()

    header = [
        "trial",
        "attention_epoch",
        "kl_lambda",
        "kl_increment",
        "base_lr",
        "classifier_lr",
        "best_balanced_val_acc",
        "test_acc",
        "per_group",
        "worst_group",
        "checkpoint",
        "sampler",
    ]

    rng = np.random.default_rng(args.seed)
    start_ts = time.time()
    max_seconds = None if args.max_hours is None else args.max_hours * 3600.0

    if args.sampler == "tpe":
        try:
            import optuna
        except Exception as exc:
            print(f"[SWEEP] Optuna not available ({exc}); falling back to random search.")
            args.sampler = "random"

    best_row = None
    if args.sampler == "random":
        for trial_id in range(args.n_trials):
            if max_seconds is not None and (time.time() - start_ts) >= max_seconds:
                print("[SWEEP] Time limit reached; stopping new trials.")
                break
            row = run_trial(trial_id, args, rng, "random")
            write_row(args.output_csv, row, header)
            if best_row is None or row["best_balanced_val_acc"] > best_row["best_balanced_val_acc"]:
                best_row = row
            print(f"[SWEEP] Trial {trial_id} done. best_balanced_val_acc={row['best_balanced_val_acc']:.4f}")
    else:
        import optuna

        def objective(trial):
            nonlocal best_row
            args.trial = trial
            row = run_trial(trial.number, args, rng, "tpe")
            write_row(args.output_csv, row, header)
            if best_row is None or row["best_balanced_val_acc"] > best_row["best_balanced_val_acc"]:
                best_row = row
            print(f"[SWEEP] Trial {trial.number} done. best_balanced_val_acc={row['best_balanced_val_acc']:.4f}")
            return row["best_balanced_val_acc"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials, timeout=max_seconds)

    if best_row is not None:
        print("[SWEEP] Best trial:")
        for k in header:
            print(f"  {k}: {best_row[k]}")

    seeds = [s for s in args.post_seeds.split() if s.strip()]
    if best_row is not None and seeds:
        seeds_header = [
            "seed",
            "attention_epoch",
            "kl_lambda",
            "kl_increment",
            "base_lr",
            "classifier_lr",
            "best_balanced_val_acc",
            "test_acc",
            "per_group",
            "worst_group",
            "checkpoint",
        ]
        run_args = SimpleNamespace(data_path=args.data_path, gt_path=args.gt_path)
        for seed_str in seeds:
            seed = int(seed_str)
            rgw.SEED = seed
            rgw.base_lr = best_row["base_lr"]
            rgw.classifier_lr = best_row["classifier_lr"]
            best_balanced_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(
                run_args,
                best_row["attention_epoch"],
                best_row["kl_lambda"],
                best_row["kl_increment"],
            )
            seed_row = {
                "seed": seed,
                "attention_epoch": best_row["attention_epoch"],
                "kl_lambda": best_row["kl_lambda"],
                "kl_increment": best_row["kl_increment"],
                "base_lr": best_row["base_lr"],
                "classifier_lr": best_row["classifier_lr"],
                "best_balanced_val_acc": best_balanced_val,
                "test_acc": test_acc,
                "per_group": per_group,
                "worst_group": worst_group,
                "checkpoint": ckpt,
            }
            write_row(args.seeds_csv, seed_row, seeds_header)
        print(f"[SWEEP] Post-seed runs complete. Results in {args.seeds_csv}")


if __name__ == "__main__":
    main()
