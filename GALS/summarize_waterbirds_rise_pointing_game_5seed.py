#!/usr/bin/env python3
"""Summarize five-seed Waterbirds RISE Pointing Game evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


GROUPS = ("0_0", "1_0", "2_0", "3_0")
METRICS = (
    "pg_acc",
    "pg_macro_group_acc",
    "pg_worst_group_acc",
    "pg_random_acc",
    "classification_acc",
    "saliency_mass_in_bird",
)


def atomic_json(path: Path, obj: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_seeds(text: str) -> List[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def mean_std(values: Iterable[float]) -> Dict[str, object]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": statistics.mean(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "n": len(finite),
    }


def read_seed(method_dir: Path, seed: int) -> Dict[str, object]:
    path = method_dir / f"seed_{seed}" / "pointing_game" / "pointing_game_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing seed {seed} RISE summary: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(rows)}")
    row: Dict[str, object] = dict(rows[0])
    if row.get("explainer") != "rise" or row.get("split") != "test":
        raise RuntimeError(f"Not a test-set RISE result: {path}")
    if int(row.get("errors", 1)) != 0:
        raise RuntimeError(f"Result reports errors: {path}")
    row["seed"] = seed
    row["source_csv"] = str(path)
    for metric in METRICS:
        row[metric] = float(row[metric])
        row[f"{metric}_pct"] = 100.0 * float(row[metric])
    for group in GROUPS:
        key = f"group_{group}_pg_acc"
        row[key] = float(row[key])
        row[f"{key}_pct"] = 100.0 * float(row[key])
    for key in (
        "pg_hits",
        "pg_total",
        "zero_saliency_maps",
        "missing_images",
        "missing_masks",
        "errors",
    ):
        row[key] = int(row[key])
    return row


def summarize_method(method_dir: Path, seeds: Sequence[int]) -> Dict[str, object]:
    rows = [read_seed(method_dir, seed) for seed in seeds]
    dataset = str(rows[0]["dataset"])
    method = str(rows[0]["method"])
    for row in rows:
        if row["dataset"] != dataset or row["method"] != method:
            raise RuntimeError(f"Mixed dataset/method rows under {method_dir}")

    summary: Dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "split": "test",
        "target_mode": rows[0]["target_mode"],
        "explainer": "rise",
        "mask_source": rows[0]["mask_source"],
        "n_seeds": len(rows),
        "seeds": ",".join(str(seed) for seed in seeds),
        "evaluation_type": "deterministic_fixed" if method in ("clip_zs", "clip_lr") else "five_seed",
        "rise_num_masks": rows[0]["rise_num_masks"],
        "rise_grid_size": rows[0]["rise_grid_size"],
        "rise_p1": rows[0]["rise_p1"],
        "rise_seed": rows[0]["rise_seed"],
        "rise_masks_sha256": rows[0]["rise_masks_sha256"],
    }
    for metric in METRICS:
        stats = mean_std(float(row[f"{metric}_pct"]) for row in rows)
        summary[f"{metric}_mean_pct"] = stats["mean"]
        summary[f"{metric}_std_pct"] = stats["std"]
    zero_stats = mean_std(float(row["zero_saliency_maps"]) for row in rows)
    summary["zero_saliency_maps_mean"] = zero_stats["mean"]
    summary["zero_saliency_maps_std"] = zero_stats["std"]
    for group in GROUPS:
        source_key = f"group_{group}_pg_acc_pct"
        stats = mean_std(float(row[source_key]) for row in rows)
        summary[f"group_{group}_pg_acc_mean_pct"] = stats["mean"]
        summary[f"group_{group}_pg_acc_std_pct"] = stats["std"]

    write_csv(method_dir / "pointing_game_rise_per_seed.csv", rows)
    write_csv(method_dir / "pointing_game_rise_5seed_summary.csv", [summary])
    atomic_json(method_dir / "pointing_game_rise_5seed_summary.json", summary)
    print(
        f"[SUMMARY] {dataset} {method}: "
        f"overall={summary['pg_acc_mean_pct']:.2f} +/- {summary['pg_acc_std_pct']:.2f}, "
        f"macro_group={summary['pg_macro_group_acc_mean_pct']:.2f} +/- "
        f"{summary['pg_macro_group_acc_std_pct']:.2f}, "
        f"worst_group={summary['pg_worst_group_acc_mean_pct']:.2f} +/- "
        f"{summary['pg_worst_group_acc_std_pct']:.2f}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--method-dir", type=Path)
    group.add_argument("--run-root", type=Path)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.method_dir is not None:
        summarize_method(args.method_dir.expanduser().resolve(), seeds)
        return

    run_root = args.run_root.expanduser().resolve()
    summaries: List[Dict[str, object]] = []
    for method_dir in sorted(run_root.glob("waterbirds_*/*")):
        if method_dir.is_dir() and any(
            method_dir.glob("seed_*/pointing_game/pointing_game_summary.csv")
        ):
            method_seeds = [0] if method_dir.name in ("clip_zs", "clip_lr") else seeds
            summaries.append(summarize_method(method_dir, method_seeds))
    if not summaries:
        raise RuntimeError(f"No Waterbirds RISE results found under {run_root}")
    write_csv(run_root / "pointing_game_rise_all_methods_5seed_summary.csv", summaries)
    write_csv(run_root / "pointing_game_rise_all_methods_summary.csv", summaries)
    atomic_json(run_root / "pointing_game_rise_all_methods_5seed_summary.json", summaries)
    print(
        f"[DONE] {run_root / 'pointing_game_rise_all_methods_5seed_summary.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
