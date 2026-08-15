#!/usr/bin/env python3
"""Shared result handling for final ImageNet-9 five-seed evaluations."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


PRIMARY_VARIANTS = ("original", "mixed_same", "mixed_rand", "mixed_next")
DIAGNOSTIC_VARIANTS = ("only_fg", "only_bg_b", "only_bg_t", "no_fg")
ALL_VARIANTS = PRIMARY_VARIANTS + DIAGNOSTIC_VARIANTS


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluation_to_row(payload: Mapping[str, object]) -> Dict[str, object]:
    variants = payload["variant_results"]
    if not isinstance(variants, Mapping):
        raise TypeError("variant_results must be a mapping")
    missing = [name for name in ALL_VARIANTS if name not in variants]
    if missing:
        raise RuntimeError(f"Evaluation is missing official variants: {missing}")

    row: Dict[str, object] = {
        "method": payload["method"],
        "seed": int(payload["seed"]),
        "checkpoint": payload["checkpoint"],
        "selection_objective": payload.get("selection_objective", "val_macro_class_accuracy"),
        "selection_value": payload.get("selection_value", ""),
    }
    for name in ALL_VARIANTS:
        metrics = variants[name]
        if not isinstance(metrics, Mapping):
            raise TypeError(f"Metrics for {name} must be a mapping")
        row[name] = 100.0 * float(metrics["macro_class_accuracy"])
        row[f"{name}_accuracy"] = 100.0 * float(metrics["accuracy"])
    row["bg_gap"] = float(row["mixed_same"]) - float(row["mixed_rand"])
    row["bg_gap_accuracy"] = float(row["mixed_same_accuracy"]) - float(
        row["mixed_rand_accuracy"]
    )
    row["only_bg_average"] = 0.5 * (
        float(row["only_bg_b"]) + float(row["only_bg_t"])
    )
    row["only_bg_average_accuracy"] = 0.5 * (
        float(row["only_bg_b_accuracy"]) + float(row["only_bg_t_accuracy"])
    )
    return row


def write_method_tables(method: str, run_root: Path, evaluations: Sequence[Path]) -> None:
    payloads = [json.loads(path.read_text()) for path in evaluations]
    rows = sorted((evaluation_to_row(payload) for payload in payloads), key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError(f"Duplicate final seeds for {method}: {seeds}")

    metric_names = list(ALL_VARIANTS) + ["bg_gap", "only_bg_average"]
    accuracy_names = [f"{name}_accuracy" for name in ALL_VARIANTS] + [
        "bg_gap_accuracy",
        "only_bg_average_accuracy",
    ]
    fieldnames = [
        "method",
        "seed",
        "checkpoint",
        "selection_objective",
        "selection_value",
        *metric_names,
        *accuracy_names,
    ]
    atomic_csv(run_root / "per_seed.csv", fieldnames, rows)

    summary_rows: List[Dict[str, object]] = []
    for metric in metric_names + accuracy_names:
        values = [float(row[metric]) for row in rows]
        summary_rows.append(
            {
                "method": method,
                "metric": metric,
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    atomic_csv(
        run_root / "summary.csv",
        ("method", "metric", "mean", "std", "n"),
        summary_rows,
    )
    atomic_json(
        run_root / "summary.json",
        {
            "method": method,
            "seeds": seeds,
            "n": len(rows),
            "standard_deviation": "population",
            "official_variants_used_for_selection": False,
            "primary_metrics": [
                "original",
                "mixed_same",
                "mixed_rand",
                "bg_gap",
                "mixed_next",
            ],
            "per_seed_csv": str((run_root / "per_seed.csv").resolve()),
            "summary_csv": str((run_root / "summary.csv").resolve()),
        },
    )


def parse_seeds(values: Sequence[str]) -> List[int]:
    seeds: List[int] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                seeds.append(int(item))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"Seeds must be non-empty and unique, got {seeds}")
    return seeds
