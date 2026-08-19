#!/usr/bin/env python3
"""Combine Waterbirds-95 R4RR corruption-condition summaries."""

import argparse
import csv
import json
import os
from pathlib import Path


GROUP_ORDER = (
    "Land_on_Land",
    "Land_on_Water",
    "Water_on_Land",
    "Water_on_Water",
)
GROUP_KEYS = tuple(name.lower() for name in GROUP_ORDER)
PAIRS = tuple(
    (f"group_{key}", f"random_matched_{key}", name)
    for key, name in zip(GROUP_KEYS, GROUP_ORDER)
)
CONDITIONS = tuple(condition for pair in PAIRS for condition in pair[:2])


def atomic_write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def load_per_seed(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()

    rows = []
    per_seed_by_condition = {}
    missing = []
    for condition in CONDITIONS:
        summary_path = run_root / condition / "summary.json"
        per_seed_path = run_root / condition / "per_seed_metrics.csv"
        if not summary_path.is_file() or not per_seed_path.is_file():
            missing.append(condition)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("dataset") != "waterbirds95" or summary.get("condition") != condition:
            raise RuntimeError(f"Unexpected summary identity: {summary_path}")
        if summary.get("group_order") != list(GROUP_ORDER):
            raise RuntimeError(f"Unexpected group order: {summary_path}")
        if int(summary.get("n_completed", -1)) != 5:
            missing.append(f"{condition} ({summary.get('n_completed', 0)}/5 seeds)")
            continue

        row = {
            "condition": condition,
            "condition_type": summary["condition_type"],
            "target_group": summary["target_group_name"],
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
        per_seed_by_condition[condition] = load_per_seed(per_seed_path)

    if missing and not args.allow_incomplete:
        raise RuntimeError("Missing or incomplete conditions: " + ", ".join(missing))
    if not rows:
        raise RuntimeError(f"No complete condition summaries found under {run_root}")

    output_csv = run_root / "waterbirds95_systematic_corruption_all_conditions.csv"
    atomic_write_csv(output_csv, list(rows[0]), rows)

    all_seed_rows = []
    for condition in CONDITIONS:
        all_seed_rows.extend(per_seed_by_condition.get(condition, []))
    all_seed_output = run_root / "waterbirds95_systematic_corruption_all_seeds.csv"
    if all_seed_rows:
        atomic_write_csv(all_seed_output, list(all_seed_rows[0]), all_seed_rows)

    pair_rows = []
    for systematic, random_control, target_group in PAIRS:
        if systematic not in per_seed_by_condition or random_control not in per_seed_by_condition:
            continue
        systematic_rows = {
            int(row["seed"]): row for row in per_seed_by_condition[systematic]
        }
        random_rows = {
            int(row["seed"]): row for row in per_seed_by_condition[random_control]
        }
        if set(systematic_rows) != set(range(5)) or set(random_rows) != set(range(5)):
            raise RuntimeError(f"Incomplete paired seed rows for {target_group}")
        for seed in range(5):
            systematic_row = systematic_rows[seed]
            random_row = random_rows[seed]
            if systematic_row["corrupted_example_count"] != random_row["corrupted_example_count"]:
                raise RuntimeError(f"Corruption counts are not matched for {target_group}")
            pair_rows.append(
                {
                    "target_group": target_group,
                    "seed": seed,
                    "corrupted_example_count": systematic_row["corrupted_example_count"],
                    "systematic_condition": systematic,
                    "random_condition": random_control,
                    "systematic_test_mean_group_acc": systematic_row["test_mean_group_acc"],
                    "random_test_mean_group_acc": random_row["test_mean_group_acc"],
                    "systematic_minus_random_mean_group": (
                        float(systematic_row["test_mean_group_acc"])
                        - float(random_row["test_mean_group_acc"])
                    ),
                    "systematic_test_worst_group_acc": systematic_row["test_worst_group_acc"],
                    "random_test_worst_group_acc": random_row["test_worst_group_acc"],
                    "systematic_minus_random_worst_group": (
                        float(systematic_row["test_worst_group_acc"])
                        - float(random_row["test_worst_group_acc"])
                    ),
                }
            )
    pair_output = run_root / "waterbirds95_systematic_vs_random_paired_seeds.csv"
    if pair_rows:
        atomic_write_csv(pair_output, list(pair_rows[0]), pair_rows)

    print("Condition                         N  Corrupt    MeanGroup        WorstGroup")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['condition']:<33s} {int(row['n_seeds']):>1d} "
            f"{int(row['corrupted_example_count']):>7d} "
            f"{float(row['test_mean_group_acc_mean']):>7.2f} +/- "
            f"{float(row['test_mean_group_acc_std']):<6.2f} "
            f"{float(row['test_worst_group_acc_mean']):>7.2f} +/- "
            f"{float(row['test_worst_group_acc_std']):.2f}"
        )
    if missing:
        print("[INCOMPLETE] " + ", ".join(missing))
    print(f"[DONE] {output_csv}")
    if all_seed_rows:
        print(f"[DONE] {all_seed_output}")
    if pair_rows:
        print(f"[DONE] {pair_output}")


if __name__ == "__main__":
    main()

