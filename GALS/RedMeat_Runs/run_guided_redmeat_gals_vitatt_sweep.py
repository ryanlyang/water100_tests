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


def loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def write_row(csv_path, row, header):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _parse_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


def _parse_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def load_resume_rows(csv_path, max_trials):
    rows_by_trial = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            tid = _parse_int(raw.get("trial"), default=None)
            if tid is None or tid < 0 or tid >= int(max_trials):
                continue
            row = {
                "trial": tid,
                "attention_epoch": _parse_int(raw.get("attention_epoch"), default=None),
                "kl_lambda": _parse_float(raw.get("kl_lambda"), default=None),
                "kl_incr": _parse_float(raw.get("kl_incr"), default=0.0),
                "base_lr": _parse_float(raw.get("base_lr"), default=None),
                "classifier_lr": _parse_float(raw.get("classifier_lr"), default=None),
                "lr2_mult": _parse_float(raw.get("lr2_mult"), default=None),
                "best_balanced_val_acc": _parse_float(raw.get("best_balanced_val_acc"), default=None),
                "test_acc": _parse_float(raw.get("test_acc"), default=None),
                "per_group": _parse_float(raw.get("per_group"), default=None),
                "worst_group": _parse_float(raw.get("worst_group"), default=None),
                "checkpoint": raw.get("checkpoint"),
                "sampler": raw.get("sampler", "tpe"),
                "seconds": _parse_int(raw.get("seconds"), default=None),
            }
            rows_by_trial[tid] = row
    rows = [rows_by_trial[k] for k in sorted(rows_by_trial.keys())]
    completed = set(rows_by_trial.keys())
    return rows, completed


def summarize_post_rows(rows, summary_csv=None):
    metrics = ("best_balanced_val_acc", "test_acc", "per_group", "worst_group")
    out_rows = []
    for metric in metrics:
        vals = [row.get(metric) for row in rows]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        out_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": int(arr.size),
            }
        )
    if out_rows:
        print("[POST] Summary over seeds:")
        for row in out_rows:
            print(f"  {row['metric']}: {row['mean']:.4f} +/- {row['std']:.4f} (n={row['n']})")
    if summary_csv:
        header = ["metric", "mean", "std", "n"]
        file_exists = os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            for row in out_rows:
                writer.writerow(row)
    return out_rows


def add_completed_trials_to_study(study, rows, args):
    import optuna

    dists = {
        "attention_epoch": optuna.distributions.IntDistribution(args.attn_min, args.attn_max),
        "kl_lambda": optuna.distributions.FloatDistribution(args.kl_min, args.kl_max, log=True),
        "base_lr": optuna.distributions.FloatDistribution(args.base_lr_min, args.base_lr_max, log=True),
        "classifier_lr": optuna.distributions.FloatDistribution(args.cls_lr_min, args.cls_lr_max, log=True),
        "lr2_mult": optuna.distributions.FloatDistribution(args.lr2_mult_min, args.lr2_mult_max, log=True),
    }

    added = 0
    for row in rows:
        value = row.get("best_balanced_val_acc")
        if value is None:
            continue
        params = {
            "attention_epoch": row.get("attention_epoch"),
            "kl_lambda": row.get("kl_lambda"),
            "base_lr": row.get("base_lr"),
            "classifier_lr": row.get("classifier_lr"),
            "lr2_mult": row.get("lr2_mult"),
        }
        if any(v is None for v in params.values()):
            continue
        try:
            frozen = optuna.trial.create_trial(
                params=params,
                distributions=dists,
                value=float(value),
            )
            study.add_trial(frozen)
            added += 1
        except Exception as exc:
            print(f"[RESUME] Skipping study restore for trial {row.get('trial')}: {exc}")
    return added


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
    p.add_argument(
        "--resume-csv",
        default=None,
        help="Existing sweep CSV to resume from. Completed trial IDs are skipped.",
    )

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
        "--post-summary-csv",
        default=None,
        help="Optional CSV to store mean/std summary over post-seed reruns.",
    )
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
    if args.resume_csv and not os.path.isfile(args.resume_csv):
        raise FileNotFoundError(f"Missing resume CSV: {args.resume_csv}")
    if args.resume_csv and args.output_csv == "guided_redmeat_galsvit_sweep.csv":
        args.output_csv = args.resume_csv

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

    if args.sampler == "tpe":
        try:
            import optuna  # noqa: F401
        except Exception as exc:
            print(f"[SWEEP] Optuna not available ({exc}); falling back to random search.")
            args.sampler = "random"

    best_row = None
    sweep_rows = []
    completed_trial_ids = set()

    if args.resume_csv:
        resume_rows, completed_trial_ids = load_resume_rows(args.resume_csv, args.n_trials)
        sweep_rows.extend(resume_rows)
        for row in resume_rows:
            score = row.get("best_balanced_val_acc")
            if score is not None and (best_row is None or score > best_row["best_balanced_val_acc"]):
                best_row = row
        print(
            f"[RESUME] Loaded {len(resume_rows)} completed trials from {args.resume_csv}. "
            f"Remaining: {max(0, args.n_trials - len(completed_trial_ids))}"
        )

    if args.sampler == "random":
        for trial_id in range(args.n_trials):
            if trial_id in completed_trial_ids:
                print(f"[RESUME] Skipping completed trial {trial_id}")
                continue
            row = run_trial(trial_id, args, rng, "random")
            write_row(args.output_csv, row, header)
            sweep_rows.append(row)
            if best_row is None or row["best_balanced_val_acc"] > best_row["best_balanced_val_acc"]:
                best_row = row
            print(f"[SWEEP] Trial {trial_id} done. best_balanced_val_acc={row['best_balanced_val_acc']:.4f}")
    else:
        import optuna
        from optuna.trial import TrialState

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        if args.resume_csv:
            restored = add_completed_trials_to_study(study, sweep_rows, args)
            print(f"[RESUME] Restored {restored} prior trials into TPE history.")

        for trial_id in range(args.n_trials):
            if trial_id in completed_trial_ids:
                print(f"[RESUME] Skipping completed trial {trial_id}")
                continue
            trial = study.ask()
            args.trial = trial
            try:
                row = run_trial(trial_id, args, rng, "tpe")
            except Exception:
                study.tell(trial, state=TrialState.FAIL)
                raise
            value = row.get("best_balanced_val_acc")
            if value is None:
                study.tell(trial, state=TrialState.FAIL)
                raise RuntimeError(
                    f"Trial {trial_id} returned no best_balanced_val_acc; cannot continue TPE."
                )
            write_row(args.output_csv, row, header)
            sweep_rows.append(row)
            study.tell(trial, float(value))
            if best_row is None or value > best_row["best_balanced_val_acc"]:
                best_row = row
            print(f"[SWEEP] Trial {trial_id} done. best_balanced_val_acc={row['best_balanced_val_acc']:.4f}")

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
            print(
                f"[POST] seed={s} best_balanced_val_acc={best_balanced_val:.4f} "
                f"test_acc={test_acc:.2f}% worst_group={worst_group:.2f}%",
                flush=True,
            )

        summary_csv = args.post_summary_csv
        if summary_csv is None:
            root, ext = os.path.splitext(post_csv)
            summary_csv = f"{root}_summary{ext or '.csv'}"
        summarize_post_rows(post_rows, summary_csv=summary_csv)

    print_runtime_summary("sweep", sweep_rows, rgm.num_epochs)
    if post_rows:
        print_runtime_summary("post_best_seeds", post_rows, rgm.num_epochs)


if __name__ == "__main__":
    main()
