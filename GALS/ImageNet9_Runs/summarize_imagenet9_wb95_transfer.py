#!/usr/bin/env python3
"""Combine completed Waterbirds-95-to-ImageNet-9 transfer results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from imagenet9_final_utils import atomic_csv


METHODS = ("erm", "upweight", "abn", "elrep", "gals", "afr", "clip_lr", "r4rr")
METRICS = ("original", "mixed_same", "mixed_rand", "bg_gap", "mixed_next")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for method in METHODS:
        path = args.run_root / method / "main" / "summary.csv"
        if not path.is_file():
            if args.allow_incomplete:
                print(f"[SKIP] missing {method}: {path}")
                continue
            raise FileNotFoundError(path)
        by_metric = {row["metric"]: row for row in csv.DictReader(path.open())}
        missing = [metric for metric in METRICS if metric not in by_metric]
        if missing:
            raise RuntimeError(f"{path} is missing metrics: {missing}")
        counts = {int(by_metric[metric]["n"]) for metric in METRICS}
        if len(counts) != 1:
            raise RuntimeError(f"Inconsistent seed counts in {path}: {counts}")
        count = counts.pop()
        if count != 5 and not args.allow_incomplete:
            raise RuntimeError(f"Expected five seeds for {method}, found {count}")
        row = {"method": method, "n": count}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(by_metric[metric]["mean"])
            row[f"{metric}_std"] = float(by_metric[metric]["std"])
        rows.append(row)

    fieldnames = ["method", "n"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    output = args.run_root / "comparison.csv"
    atomic_csv(output, fieldnames, rows)
    print("\nWaterbirds-95 hyperparameter transfer to ImageNet-9")
    print(
        f"{'Method':<10} {'Original':>15} {'Mixed-Same':>15} "
        f"{'Mixed-Rand':>15} {'BG Gap':>15} {'Mixed-Next':>15}"
    )
    for row in rows:
        cells = []
        for metric in METRICS:
            cells.append(
                f"{row[f'{metric}_mean']:.2f} +/- {row[f'{metric}_std']:.2f}"
            )
        print(f"{row['method']:<10} " + " ".join(f"{cell:>15}" for cell in cells))
    print(f"\n[DONE] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
