#!/usr/bin/env python3
"""Combine RedMeat R4RR corruption-condition summaries."""

import argparse
import csv
import json
import os
from pathlib import Path


CLASS_ORDER = (
    "prime_rib",
    "pork_chop",
    "steak",
    "baby_back_ribs",
    "filet_mignon",
)
CONDITIONS = (
    "random_20pct",
    "class_baby_back_ribs",
    "class_filet_mignon",
    "class_pork_chop",
    "class_prime_rib",
    "class_steak",
)


def atomic_write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()

    rows = []
    missing = []
    for condition in CONDITIONS:
        path = run_root / condition / "summary.json"
        if not path.is_file():
            missing.append(condition)
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("dataset") != "redmeat" or summary.get("condition") != condition:
            raise RuntimeError(f"Unexpected summary identity: {path}")
        if summary.get("class_order") != list(CLASS_ORDER):
            raise RuntimeError(f"Unexpected class order: {path}")
        if int(summary.get("n_completed", -1)) != 5:
            missing.append(f"{condition} ({summary.get('n_completed', 0)}/5 seeds)")
            continue

        row = {
            "condition": condition,
            "condition_type": summary["condition_type"],
            "target_class": summary["target_class"] or "",
            "corruption_seed": summary["corruption_seed"],
            "corrupted_example_count": summary["corrupted_example_count"],
            "corrupted_fraction_of_training": summary["corrupted_fraction_of_training"],
            "n_seeds": summary["n_completed"],
            "manifest_sha256": summary["manifest_sha256"],
            "contract_sha256": summary["contract_sha256"],
        }
        for metric, stats in summary["metrics"].items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)

    if missing and not args.allow_incomplete:
        raise RuntimeError("Missing or incomplete conditions: " + ", ".join(missing))
    if not rows:
        raise RuntimeError(f"No complete condition summaries found under {run_root}")

    output_csv = run_root / "redmeat_systematic_corruption_all_conditions.csv"
    atomic_write_csv(output_csv, list(rows[0]), rows)

    per_seed_rows = []
    for condition in CONDITIONS:
        path = run_root / condition / "per_seed_metrics.csv"
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                per_seed_rows.extend(csv.DictReader(handle))
    per_seed_output = run_root / "redmeat_systematic_corruption_all_seeds.csv"
    if per_seed_rows:
        atomic_write_csv(per_seed_output, list(per_seed_rows[0]), per_seed_rows)

    print("Condition                 N  Corrupt    MeanClass       WorstClass")
    print("-" * 76)
    for row in rows:
        print(
            f"{row['condition']:<25s} {int(row['n_seeds']):>1d} "
            f"{int(row['corrupted_example_count']):>7d} "
            f"{float(row['test_mean_class_acc_mean']):>7.2f} +/- "
            f"{float(row['test_mean_class_acc_std']):<6.2f} "
            f"{float(row['test_worst_class_acc_mean']):>7.2f} +/- "
            f"{float(row['test_worst_class_acc_std']):.2f}"
        )
    if missing:
        print("[INCOMPLETE] " + ", ".join(missing))
    print(f"[DONE] {output_csv}")
    if per_seed_rows:
        print(f"[DONE] {per_seed_output}")


if __name__ == "__main__":
    main()

