#!/usr/bin/env python3
"""Aggregate ImageNet-9 class-conditional R4RR corruption results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping

from imagenet9_final_utils import atomic_csv
from imagenet9_systematic_corruption import CLASS_NAMES, CONDITIONS, CLASS_COUNT


PRIMARY = ("original", "mixed_same", "mixed_rand", "bg_gap", "mixed_next")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    combined: List[Dict[str, object]] = []
    per_seed: Dict[str, List[Dict[str, str]]] = {}
    missing = []
    for condition in CONDITIONS:
        condition_root = args.run_root / condition
        summary_path = condition_root / "corruption_summary.json"
        per_seed_path = condition_root / "per_seed.csv"
        if not summary_path.is_file() or not per_seed_path.is_file():
            missing.append(condition)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("dataset") != "imagenet9" or summary.get("condition") != condition:
            raise RuntimeError(f"Unexpected summary identity: {summary_path}")
        if int(summary.get("corrupted_example_count", -1)) != CLASS_COUNT:
            raise RuntimeError(f"Corruption count mismatch: {summary_path}")
        if summary.get("completed_seeds") != [0, 1, 2, 3, 4]:
            missing.append(f"{condition} ({summary.get('completed_seeds', [])})")
            continue
        row: Dict[str, object] = {
            "condition": condition,
            "condition_type": summary["condition_type"],
            "target_class": summary["target_class"],
            "corruption_seed": summary["corruption_seed"],
            "corrupted_example_count": summary["corrupted_example_count"],
            "corrupted_fraction_of_training": summary["corrupted_fraction_of_training"],
            "n_seeds": summary["n_completed"],
            "corruption_manifest_sha256": summary["corruption_manifest_sha256"],
        }
        for metric, stats in summary["metrics"].items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        combined.append(row)
        seed_rows = read_csv(per_seed_path)
        if [int(value["seed"]) for value in seed_rows] != [0, 1, 2, 3, 4]:
            raise RuntimeError(f"Incomplete per-seed rows: {per_seed_path}")
        per_seed[condition] = seed_rows

    if missing and not args.allow_incomplete:
        raise RuntimeError("Missing or incomplete conditions: " + ", ".join(missing))
    if not combined:
        raise RuntimeError(f"No complete conditions found under {args.run_root}")
    output = args.run_root / "imagenet9_systematic_corruption_all_conditions.csv"
    atomic_csv(output, list(combined[0]), combined)

    all_seed_rows = [row for condition in CONDITIONS for row in per_seed.get(condition, [])]
    if all_seed_rows:
        atomic_csv(
            args.run_root / "imagenet9_systematic_corruption_all_seeds.csv",
            list(all_seed_rows[0]),
            all_seed_rows,
        )

    random_rows = {int(row["seed"]): row for row in per_seed.get("random_matched", [])}
    paired: List[Mapping[str, object]] = []
    if random_rows:
        for class_name in CLASS_NAMES:
            condition = f"class_{class_name}"
            systematic_rows = {int(row["seed"]): row for row in per_seed.get(condition, [])}
            if set(systematic_rows) != set(range(5)) or set(random_rows) != set(range(5)):
                if args.allow_incomplete:
                    continue
                raise RuntimeError(f"Incomplete systematic/random pair for {class_name}")
            for seed in range(5):
                row: Dict[str, object] = {
                    "target_class": class_name,
                    "seed": seed,
                    "corrupted_example_count": CLASS_COUNT,
                    "systematic_condition": condition,
                    "random_condition": "random_matched",
                }
                for metric in PRIMARY:
                    systematic_value = float(systematic_rows[seed][metric])
                    random_value = float(random_rows[seed][metric])
                    row[f"systematic_{metric}"] = systematic_value
                    row[f"random_{metric}"] = random_value
                    row[f"systematic_minus_random_{metric}"] = (
                        systematic_value - random_value
                    )
                paired.append(row)
    if paired:
        atomic_csv(
            args.run_root / "imagenet9_systematic_vs_random_paired_seeds.csv",
            list(paired[0]),
            paired,
        )

    print("ImageNet-9 systematic R4RR teacher corruption")
    print(
        f"{'Condition':<24} {'N':>1} {'Original':>15} {'Mixed-Same':>15} "
        f"{'Mixed-Rand':>15} {'BG Gap':>15} {'Mixed-Next':>15}"
    )
    for row in combined:
        cells = [
            f"{float(row[f'{metric}_mean']):.2f} +/- {float(row[f'{metric}_std']):.2f}"
            for metric in PRIMARY
        ]
        print(
            f"{str(row['condition']):<24} {int(row['n_seeds']):>1d} "
            + " ".join(f"{cell:>15}" for cell in cells)
        )
    if missing:
        print("[INCOMPLETE] " + ", ".join(missing))
    print(f"[DONE] {output}")
    if paired:
        print(f"[DONE] {args.run_root / 'imagenet9_systematic_vs_random_paired_seeds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

