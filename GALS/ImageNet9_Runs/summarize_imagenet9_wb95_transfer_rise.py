#!/usr/bin/env python3
"""Aggregate WB95-transfer ImageNet-9 RISE Pointing Game results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from imagenet9_final_utils import atomic_json
from imagenet9_pointing_game_utils import METHODS, PRIMARY_VARIANTS, write_csv


METRICS = (
    "pg_acc",
    "pg_macro_class_acc",
    "pg_worst_class_acc",
    "pg_random_acc",
    "classification_acc",
    "classification_macro_class_acc",
    "saliency_mass_in_foreground",
)


def parse_csv_values(text: str) -> List[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Values must be non-empty and unique: {values}")
    return values


def read_summary(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("explainer") != "rise" or int(payload.get("errors", 1)) != 0:
        raise RuntimeError(f"Invalid RISE summary: {path}")
    if int(payload.get("pg_total", 0)) <= 0:
        raise RuntimeError(f"Empty RISE summary: {path}")
    return payload


def summarize_method_variant(
    run_root: Path, method: str, variant: str, seeds: Sequence[int]
) -> Dict[str, object]:
    rows = [
        read_summary(
            run_root / method / variant / f"seed_{seed}" / "pointing_game_summary.json"
        )
        for seed in seeds
    ]
    for row, seed in zip(rows, seeds):
        if row.get("method") != method or row.get("variant") != variant:
            raise RuntimeError(f"Mixed method/variant result under {run_root}")
        if int(row.get("seed", -1)) != seed:
            raise RuntimeError(f"Seed mismatch for {method}/{variant}/seed_{seed}")
    summary: Dict[str, object] = {
        "dataset": "imagenet9",
        "transfer_source": rows[0].get("transfer_source", "waterbirds95"),
        "method": method,
        "variant": variant,
        "explainer": "rise",
        "target_mode": rows[0]["target_mode"],
        "mask_source": rows[0]["mask_source"],
        "n_seeds": len(rows),
        "seeds": ",".join(str(seed) for seed in seeds),
        "standard_deviation": "population",
        "rise_num_masks": rows[0]["rise_num_masks"],
        "rise_grid_size": rows[0]["rise_grid_size"],
        "rise_p1": rows[0]["rise_p1"],
        "rise_seed": rows[0]["rise_seed"],
        "rise_masks_sha256": rows[0]["rise_masks_sha256"],
    }
    for metric in METRICS:
        values = [100.0 * float(row[metric]) for row in rows]
        summary[f"{metric}_mean_pct"] = statistics.mean(values)
        summary[f"{metric}_std_pct"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    summary["zero_saliency_maps_total"] = sum(int(row["zero_saliency_maps"]) for row in rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--variants", default="original")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    methods = parse_csv_values(args.methods)
    variants = parse_csv_values(args.variants)
    seeds = [int(seed) for seed in parse_csv_values(args.seeds)]
    unknown_methods = sorted(set(methods) - set(METHODS))
    unknown_variants = sorted(set(variants) - set(PRIMARY_VARIANTS))
    if unknown_methods or unknown_variants:
        raise ValueError(f"Unknown methods={unknown_methods} variants={unknown_variants}")

    summaries = [
        summarize_method_variant(args.run_root, method, variant, seeds)
        for variant in variants
        for method in methods
    ]
    output_csv = args.run_root / "pointing_game_rise_5seed_comparison.csv"
    write_csv(output_csv, summaries)
    atomic_json(args.run_root / "pointing_game_rise_5seed_comparison.json", summaries)
    for row in summaries:
        print(
            f"[SUMMARY] {row['method']:8s} {row['variant']:10s} "
            f"PG={row['pg_acc_mean_pct']:.2f} +/- {row['pg_acc_std_pct']:.2f} "
            f"macro={row['pg_macro_class_acc_mean_pct']:.2f} +/- "
            f"{row['pg_macro_class_acc_std_pct']:.2f} "
            f"worst={row['pg_worst_class_acc_mean_pct']:.2f} +/- "
            f"{row['pg_worst_class_acc_std_pct']:.2f}",
            flush=True,
        )
    print(f"[DONE] {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
