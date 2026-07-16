import argparse
import csv
import os
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np

WATERBIRDS_TRAIN_ROOT = Path(__file__).resolve().parents[1] / "train"
if str(WATERBIRDS_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(WATERBIRDS_TRAIN_ROOT))
import r4rr_waterbirds as rgw


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


def _coerce_existing_row(row):
    out = dict(row)
    int_keys = ["trial", "attention_epoch", "seconds"]
    float_keys = [
        "kl_lambda",
        "kl_incr",
        "base_lr",
        "classifier_lr",
        "lr2_mult",
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

    rgw.base_lr = base_lr
    rgw.classifier_lr = classifier_lr
    rgw.lr2_mult = lr2_mult
    rgw.SEED = args.seed

    run_args = SimpleNamespace(data_path=args.data_path, teacher_map_path=args.teacher_map_path)
    best_balanced_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(
        run_args, attn_epoch, kl_lambda, kl_incr
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
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "checkpoint": ckpt,
        "sampler": sampler_name,
        "seconds": int(time.time() - t0),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Waterbirds dataset root with metadata.csv")
    parser.add_argument("teacher_map_path", help="Teacher-map root (required by r4rr_waterbirds.py)")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default="guided_waterbird_sweep.csv")
    parser.add_argument("--attn-min", type=int, default=0)
    parser.add_argument("--attn-max", type=int, default=rgw.num_epochs - 1)
    parser.add_argument("--kl-min", type=float, default=1.0)
    parser.add_argument("--kl-max", type=float, default=500.0)
    parser.add_argument("--base-lr-min", type=float, default=1e-5)
    parser.add_argument("--base-lr-max", type=float, default=5e-2)
    parser.add_argument("--cls-lr-min", type=float, default=1e-5)
    parser.add_argument("--cls-lr-max", type=float, default=5e-2)
    parser.add_argument("--lr2-mult-min", type=float, default=1e-1)
    parser.add_argument("--lr2-mult-max", type=float, default=3.0)
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument(
        "--resume-csv",
        default=None,
        help="Optional existing sweep CSV to resume from. Prior completed trials are loaded into Optuna.",
    )
    args = parser.parse_args()

    header = [
        "trial",
        "attention_epoch",
        "kl_lambda",
        "kl_incr",
        "base_lr",
        "classifier_lr",
        "lr2_mult",
        "best_balanced_val_acc",
        "test_acc",
        "per_group",
        "worst_group",
        "checkpoint",
        "sampler",
        "seconds",
    ]

    rng = np.random.default_rng(args.seed)
    existing_rows = []
    if args.resume_csv:
        existing_rows = load_existing_rows(args.resume_csv)
        if existing_rows:
            print(f"[RESUME] Loaded {len(existing_rows)} existing rows from: {args.resume_csv}")
        else:
            print(f"[RESUME] No existing rows found at: {args.resume_csv}. Starting fresh.")

    if args.sampler == "tpe":
        try:
            import optuna
        except Exception as exc:
            print(f"[SWEEP] Optuna not available ({exc}); falling back to random search.")
            args.sampler = "random"

    best_row = None
    sweep_rows = list(existing_rows)
    if existing_rows:
        best_row = max(existing_rows, key=lambda r: float(r["best_balanced_val_acc"]))

    next_trial_id = 0
    if existing_rows:
        next_trial_id = int(max(int(r["trial"]) for r in existing_rows) + 1)

    remaining_trials = max(0, int(args.n_trials) - len(existing_rows))
    if remaining_trials == 0:
        print(
            f"[RESUME] Existing completed rows ({len(existing_rows)}) already reach n_trials={args.n_trials}. "
            "Skipping sweep execution."
        )

    if args.sampler == "random":
        for i in range(remaining_trials):
            trial_id = next_trial_id + i
            try:
                row = run_trial(trial_id, args, rng, "random")
            except Exception as exc:
                print(f"[SWEEP] Trial {trial_id} failed: {exc}", flush=True)
                continue
            write_row(args.output_csv, row, header)
            sweep_rows.append(row)
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
            sweep_rows.append(row)
            if best_row is None or row["best_balanced_val_acc"] > best_row["best_balanced_val_acc"]:
                best_row = row
            print(f"[SWEEP] Trial {trial.number} done. best_balanced_val_acc={row['best_balanced_val_acc']:.4f}")
            return row["best_balanced_val_acc"]

        study = optuna.create_study(direction="maximize")

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
                    print(f"[RESUME] Skipping malformed resume row (trial={r.get('trial')}): {exc}")
            print(f"[RESUME] Added {added} prior trials into Optuna study state.")

        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, catch=(Exception,))

    if best_row is not None:
        print("[SWEEP] Best trial:")
        for k in header:
            print(f"  {k}: {best_row[k]}")
    print_runtime_summary("sweep", sweep_rows, rgw.num_epochs)


if __name__ == "__main__":
    main()
