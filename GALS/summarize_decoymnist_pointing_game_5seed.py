#!/usr/bin/env python3
"""Summarize five-seed DecoyMNIST Pointing Game results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DIGITS = tuple(range(10))
RATE_KEYS = (
    "pg_acc",
    "pg_macro_class_acc",
    "pg_worst_class_acc",
    "classification_acc",
) + tuple(f"digit_{digit}_pg_acc" for digit in DIGITS)
REQUIRED_FINITE_RATE_KEYS = (
    "pg_acc",
    "pg_macro_class_acc",
    "pg_worst_class_acc",
    "classification_acc",
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_seeds(text: str) -> List[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return seeds


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
    return {
        "mean": statistics.mean(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "n": float(len(finite)),
    }


def read_seed(method_dir: Path, seed: int) -> Dict[str, object]:
    path = method_dir / f"seed_{seed}" / "pointing_game" / "pointing_game_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing seed {seed} summary: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(source_rows)}")
    row: Dict[str, object] = dict(source_rows[0])
    if str(row.get("dataset")) != "decoymnist" or int(row.get("seed", -1)) != seed:
        raise RuntimeError(f"Unexpected dataset/seed in {path}")
    if int(row.get("errors", 1)) != 0 or int(row.get("pg_total", 0)) <= 0:
        raise RuntimeError(f"Invalid evaluation counts in {path}")
    for key in RATE_KEYS:
        value = float(row[key])
        if key in REQUIRED_FINITE_RATE_KEYS and not math.isfinite(value):
            raise RuntimeError(f"Non-finite {key} in {path}: {value}")
        row[key] = value
        row[f"{key}_pct"] = 100.0 * value
    for key in ("pg_hits", "pg_total", "zero_saliency_maps", "errors"):
        row[key] = int(row[key])
    row["source_csv"] = str(path)
    return row


def summarize_method(method_dir: Path, seeds: Sequence[int]) -> Dict[str, object]:
    rows = [read_seed(method_dir, seed) for seed in seeds]
    method = str(rows[0]["method"])
    if any(str(row["method"]) != method for row in rows):
        raise RuntimeError(f"Mixed methods under {method_dir}")

    summary: Dict[str, object] = {
        "dataset": "decoymnist",
        "method": method,
        "split": rows[0]["split"],
        "target_mode": rows[0]["target_mode"],
        "mask_source": rows[0]["mask_source"],
        "n_seeds": len(rows),
        "seeds": ",".join(str(seed) for seed in seeds),
    }
    for key in RATE_KEYS:
        stats = mean_std(float(row[f"{key}_pct"]) for row in rows)
        summary[f"{key}_mean_pct"] = stats["mean"]
        summary[f"{key}_std_pct"] = stats["std"]

    zero_stats = mean_std(float(row["zero_saliency_maps"]) for row in rows)
    summary["zero_saliency_maps_mean"] = zero_stats["mean"]
    summary["zero_saliency_maps_std"] = zero_stats["std"]

    write_csv(method_dir / "pointing_game_per_seed.csv", rows)
    write_csv(method_dir / "pointing_game_5seed_summary.csv", [summary])
    atomic_json(method_dir / "pointing_game_5seed_summary.json", summary)
    print(
        f"[SUMMARY] decoymnist {method}: "
        f"overall={summary['pg_acc_mean_pct']:.2f} +/- {summary['pg_acc_std_pct']:.2f}, "
        f"macro={summary['pg_macro_class_acc_mean_pct']:.2f} +/- "
        f"{summary['pg_macro_class_acc_std_pct']:.2f}, "
        f"worst={summary['pg_worst_class_acc_mean_pct']:.2f} +/- "
        f"{summary['pg_worst_class_acc_std_pct']:.2f}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--method-dir", type=Path)
    source.add_argument("--run-root", type=Path)
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
    for method_dir in sorted(run_root.iterdir() if run_root.is_dir() else []):
        if method_dir.is_dir() and any(
            method_dir.glob("seed_*/pointing_game/pointing_game_summary.csv")
        ):
            summaries.append(summarize_method(method_dir, seeds))
    if not summaries:
        raise RuntimeError(f"No complete method results found under {run_root}")
    write_csv(run_root / "pointing_game_all_methods_5seed_summary.csv", summaries)
    atomic_json(run_root / "pointing_game_all_methods_5seed_summary.json", summaries)
    print(f"[DONE] {run_root / 'pointing_game_all_methods_5seed_summary.csv'}")


if __name__ == "__main__":
    main()
