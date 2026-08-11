#!/usr/bin/env python3
"""Summarize five-seed DecoyMNIST RISE Pointing Game results."""

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
MASK_PROTOCOL_VERSION = 1
PRIMARY_PG_PROTOCOL = "rise_pixel_argmax"
RATE_KEYS = (
    "pg_acc",
    "pg_macro_class_acc",
    "pg_worst_class_acc",
    "pg_random_acc",
    "classification_acc",
    "saliency_mass_in_digit",
) + tuple(
    f"digit_{digit}_{metric}"
    for digit in DIGITS
    for metric in ("pg_acc", "classification_acc", "saliency_mass_in_digit")
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
    if (
        row.get("explainer") != "rise"
        or int(row.get("mask_protocol_version", -1)) != MASK_PROTOCOL_VERSION
        or row.get("primary_pg_protocol") != PRIMARY_PG_PROTOCOL
    ):
        raise RuntimeError(f"Unexpected Pointing Game protocol in {path}")
    if (
        int(row.get("errors", 1)) != 0
        or int(row.get("pg_total", 0)) <= 0
        or (int(row.get("map_height", 0)), int(row.get("map_width", 0))) != (28, 28)
    ):
        raise RuntimeError(f"Invalid evaluation counts in {path}")
    for key in RATE_KEYS:
        value = float(row[key])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite {key} in {path}: {value}")
        row[key] = value
        row[f"{key}_pct"] = 100.0 * value
    for key in (
        "mask_protocol_version",
        "map_height",
        "map_width",
        "pg_hits",
        "pg_total",
        "zero_saliency_maps",
        "rise_num_masks",
        "rise_grid_size",
        "rise_seed",
        "errors",
    ):
        row[key] = int(row[key])
    row["rise_p1"] = float(row["rise_p1"])
    row["source_csv"] = str(path)
    return row


def summarize_method(method_dir: Path, seeds: Sequence[int]) -> Dict[str, object]:
    rows = [read_seed(method_dir, seed) for seed in seeds]
    method = str(rows[0]["method"])
    if any(str(row["method"]) != method for row in rows):
        raise RuntimeError(f"Mixed methods under {method_dir}")
    fixed_fields = (
        "split",
        "target_mode",
        "explainer",
        "mask_source",
        "mask_threshold",
        "max_samples",
        "sample_seed",
        "mask_protocol_version",
        "primary_pg_protocol",
        "map_height",
        "map_width",
        "rise_num_masks",
        "rise_grid_size",
        "rise_p1",
        "rise_seed",
        "rise_masks_sha256",
    )
    for field in fixed_fields:
        if any(row[field] != rows[0][field] for row in rows[1:]):
            raise RuntimeError(f"Mixed {field} values under {method_dir}")

    summary: Dict[str, object] = {
        "dataset": "decoymnist",
        "method": method,
        "split": rows[0]["split"],
        "target_mode": rows[0]["target_mode"],
        "explainer": "rise",
        "mask_source": rows[0]["mask_source"],
        "mask_threshold": rows[0]["mask_threshold"],
        "max_samples": rows[0]["max_samples"],
        "sample_seed": rows[0]["sample_seed"],
        "mask_protocol_version": MASK_PROTOCOL_VERSION,
        "primary_pg_protocol": PRIMARY_PG_PROTOCOL,
        "map_height": 28,
        "map_width": 28,
        "rise_num_masks": rows[0]["rise_num_masks"],
        "rise_grid_size": rows[0]["rise_grid_size"],
        "rise_p1": rows[0]["rise_p1"],
        "rise_seed": rows[0]["rise_seed"],
        "rise_masks_sha256": rows[0]["rise_masks_sha256"],
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
        f"[SUMMARY] decoymnist {method} RISE: "
        f"overall={summary['pg_acc_mean_pct']:.2f} +/- "
        f"{summary['pg_acc_std_pct']:.2f}, "
        f"macro={summary['pg_macro_class_acc_mean_pct']:.2f} +/- "
        f"{summary['pg_macro_class_acc_std_pct']:.2f}, "
        f"worst={summary['pg_worst_class_acc_mean_pct']:.2f} +/- "
        f"{summary['pg_worst_class_acc_std_pct']:.2f}, "
        f"random={summary['pg_random_acc_mean_pct']:.2f} +/- "
        f"{summary['pg_random_acc_std_pct']:.2f}, "
        f"mass_inside={summary['saliency_mass_in_digit_mean_pct']:.2f} +/- "
        f"{summary['saliency_mass_in_digit_std_pct']:.2f}, "
        f"classification={summary['classification_acc_mean_pct']:.2f} +/- "
        f"{summary['classification_acc_std_pct']:.2f}, "
        f"zero_maps={summary['zero_saliency_maps_mean']:.1f} +/- "
        f"{summary['zero_saliency_maps_std']:.1f}",
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
        raise RuntimeError(f"No complete RISE method results found under {run_root}")
    mask_hashes = {str(summary["rise_masks_sha256"]) for summary in summaries}
    if len(mask_hashes) != 1:
        raise RuntimeError(
            "Methods were evaluated with different RISE mask banks: "
            f"{sorted(mask_hashes)}"
        )
    write_csv(run_root / "pointing_game_all_methods_5seed_summary.csv", summaries)
    atomic_json(run_root / "pointing_game_all_methods_5seed_summary.json", summaries)
    print(f"[DONE] {run_root / 'pointing_game_all_methods_5seed_summary.csv'}")


if __name__ == "__main__":
    main()
