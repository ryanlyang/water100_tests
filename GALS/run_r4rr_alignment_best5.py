#!/usr/bin/env python3
"""Rerun the best R4RR alignment-loss sweep trial over fixed seeds."""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


LOSSES = ("reverse_kl", "jensen_shannon", "squared_l2", "cosine")
DATASETS = ("wb95", "wb100", "redmeat")
PARAM_FIELDS = (
    "attention_epoch",
    "kl_lambda",
    "kl_incr",
    "base_lr",
    "classifier_lr",
    "lr2_mult",
)
METRICS = ("best_balanced_val_acc", "test_acc", "per_group", "worst_group")


def parse_float(value, field, source):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field!r} in {source}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field!r} in {source}: {value!r}")
    return result


def valid_sweep_rows(path, alignment_loss):
    rows = []
    try:
        with open(path, newline="") as handle:
            for raw in csv.DictReader(handle):
                row_loss = str(raw.get("alignment_loss") or "forward_kl")
                if row_loss != alignment_loss:
                    continue
                try:
                    row = dict(raw)
                    row["trial"] = int(float(raw["trial"]))
                    row["attention_epoch"] = int(float(raw["attention_epoch"]))
                    for field in PARAM_FIELDS[1:] + ("best_balanced_val_acc",):
                        row[field] = parse_float(raw.get(field), field, path)
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append(row)
    except (OSError, csv.Error):
        return []
    return rows


def sweep_pattern(dataset, alignment_loss):
    prefix = {
        "wb95": "wb95",
        "wb100": "wb100",
        "redmeat": "redmeat",
    }[dataset]
    return f"{prefix}_r4rr_{alignment_loss}_sweep_*.csv"


def resolve_sweep_csv(log_dir, dataset, alignment_loss, min_trials):
    candidates = []
    for path in Path(log_dir).glob(sweep_pattern(dataset, alignment_loss)):
        if "best5" in path.name or "summary" in path.name:
            continue
        rows = valid_sweep_rows(str(path), alignment_loss)
        if len(rows) >= min_trials:
            candidates.append((path.stat().st_mtime, len(rows), path))
    if not candidates:
        pattern = Path(log_dir) / sweep_pattern(dataset, alignment_loss)
        raise FileNotFoundError(
            f"No completed sweep CSV with at least {min_trials} valid rows matched {pattern}"
        )
    _, _, selected = max(candidates)
    return str(selected)


def select_best_row(path, alignment_loss, min_trials):
    rows = valid_sweep_rows(path, alignment_loss)
    if len(rows) < min_trials:
        raise RuntimeError(
            f"Sweep CSV has {len(rows)} valid {alignment_loss} rows; expected at least {min_trials}: {path}"
        )
    best = max(rows, key=lambda row: row["best_balanced_val_acc"])
    return best, len(rows)


def load_completed_rows(path, expected):
    if not os.path.isfile(path):
        return {}
    completed = {}
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                seed = int(raw["seed"])
                same = (
                    raw.get("dataset") == expected["dataset"]
                    and raw.get("alignment_loss") == expected["alignment_loss"]
                    and os.path.realpath(raw.get("sweep_csv", ""))
                    == os.path.realpath(expected["sweep_csv"])
                    and int(float(raw.get("sweep_best_trial", -1))) == expected["sweep_best_trial"]
                )
                for field in PARAM_FIELDS:
                    same = same and math.isclose(
                        float(raw[field]), float(expected[field]), rel_tol=1e-12, abs_tol=1e-15
                    )
                for metric in METRICS:
                    parse_float(raw.get(metric), metric, path)
                if same:
                    completed[seed] = raw
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def append_row(path, row, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_summary(path, rows, metadata):
    import numpy as np

    fields = [
        "dataset",
        "alignment_loss",
        "sweep_csv",
        "sweep_best_trial",
        "metric",
        "mean",
        "std",
        "n",
    ]
    summary = []
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        summary.append(
            {
                "dataset": metadata["dataset"],
                "alignment_loss": metadata["alignment_loss"],
                "sweep_csv": metadata["sweep_csv"],
                "sweep_best_trial": metadata["sweep_best_trial"],
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "n": int(values.size),
            }
        )
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    os.replace(tmp, path)
    print("\n===== FIVE-SEED SUMMARY (population std) =====", flush=True)
    for row in summary:
        print(
            f"{row['metric']}: mean={row['mean']:.4f} std={row['std']:.4f} n={row['n']}",
            flush=True,
        )


def run_waterbirds(args, best, seed):
    train_dir = Path(__file__).resolve().parent / "RightForTheRightRegions/repro_runs/r4rr/train"
    sys.path.insert(0, str(train_dir))
    import r4rr_waterbirds as train

    train.SEED = seed
    train.base_lr = best["base_lr"]
    train.classifier_lr = best["classifier_lr"]
    train.lr2_mult = best["lr2_mult"]
    run_args = SimpleNamespace(
        data_path=args.data_path,
        teacher_map_path=args.teacher_map_path,
        alignment_loss=args.alignment_loss,
    )
    return train.run_single(
        run_args, best["attention_epoch"], best["kl_lambda"], best["kl_incr"]
    )


def run_redmeat(args, best, seed):
    train_dir = Path(__file__).resolve().parent / "RightForTheRightRegions/repro_runs/r4rr/train"
    sys.path.insert(0, str(train_dir))
    import r4rr_redmeat as train

    train.SEED = seed
    train.num_epochs = 150
    train.base_lr = best["base_lr"]
    train.classifier_lr = best["classifier_lr"]
    train.lr2_mult = best["lr2_mult"]
    run_args = SimpleNamespace(
        data_path=args.data_path,
        teacher_map_path=args.teacher_map_path,
        alignment_loss=args.alignment_loss,
        split_col="split",
        label_col="label",
        path_col="abs_file_path",
        classes=["prime_rib", "pork_chop", "steak", "baby_back_ribs", "filet_mignon"],
        model_name="resnet50",
        clip_model="RN50",
        tune_mode="full",
        pretrained=True,
    )
    return train.run_single(
        run_args, best["attention_epoch"], best["kl_lambda"], best["kl_incr"]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--alignment-loss", choices=LOSSES, required=True)
    parser.add_argument("--sweep-csv")
    parser.add_argument("--log-dir")
    parser.add_argument("--min-sweep-trials", type=int, default=50)
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--data-path")
    parser.add_argument("--teacher-map-path")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output-csv")
    parser.add_argument("--summary-csv")
    return parser.parse_args()


def main():
    args = parse_args()
    sweep_csv = args.sweep_csv
    if not sweep_csv:
        if not args.log_dir:
            raise ValueError("Provide --sweep-csv or --log-dir")
        sweep_csv = resolve_sweep_csv(
            args.log_dir, args.dataset, args.alignment_loss, args.min_sweep_trials
        )
    if args.resolve_only:
        print(sweep_csv)
        return
    for name in ("data_path", "teacher_map_path", "output_csv", "summary_csv"):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} is required")

    best, row_count = select_best_row(sweep_csv, args.alignment_loss, args.min_sweep_trials)
    best["kl_incr"] = float(best.get("kl_incr", 0.0))
    sweep_csv = os.path.realpath(sweep_csv)
    metadata = {
        "dataset": args.dataset,
        "alignment_loss": args.alignment_loss,
        "sweep_csv": sweep_csv,
        "sweep_best_trial": int(best["trial"]),
        **{field: best[field] for field in PARAM_FIELDS},
    }
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds requested: {seeds}")

    print(f"[SELECT] sweep_csv={sweep_csv} valid_rows={row_count}", flush=True)
    print(
        f"[SELECT] trial={best['trial']} val={best['best_balanced_val_acc']:.6f} "
        f"attention_epoch={best['attention_epoch']} alignment_weight={best['kl_lambda']:.9g} "
        f"base_lr={best['base_lr']:.9g} classifier_lr={best['classifier_lr']:.9g} "
        f"lr2_mult={best['lr2_mult']:.9g}",
        flush=True,
    )

    fields = [
        "dataset",
        "alignment_loss",
        "seed",
        "sweep_csv",
        "sweep_best_trial",
        *PARAM_FIELDS,
        *METRICS,
        "checkpoint",
        "seconds",
    ]
    completed = load_completed_rows(args.output_csv, metadata)
    for seed in seeds:
        if seed in completed:
            print(f"[RESUME] seed={seed} already complete; skipping", flush=True)
            continue
        started = time.time()
        if args.dataset == "redmeat":
            result = run_redmeat(args, best, seed)
        else:
            result = run_waterbirds(args, best, seed)
        best_val, test_acc, per_group, worst_group, checkpoint = result
        row = {
            **metadata,
            "seed": seed,
            "best_balanced_val_acc": best_val,
            "test_acc": test_acc,
            "per_group": per_group,
            "worst_group": worst_group,
            "checkpoint": checkpoint,
            "seconds": int(time.time() - started),
        }
        append_row(args.output_csv, row, fields)
        completed[seed] = row
        print(
            f"[SEED DONE] seed={seed} val={best_val:.4f} test={test_acc:.2f}% "
            f"mean_group={per_group:.2f}% worst_group={worst_group:.2f}%",
            flush=True,
        )

    missing = [seed for seed in seeds if seed not in completed]
    if missing:
        raise RuntimeError(f"Missing completed seeds: {missing}")
    ordered = [completed[seed] for seed in seeds]
    write_summary(args.summary_csv, ordered, metadata)
    print(f"[DONE] per-seed CSV: {args.output_csv}", flush=True)
    print(f"[DONE] summary CSV: {args.summary_csv}", flush=True)


if __name__ == "__main__":
    main()
