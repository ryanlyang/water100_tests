#!/usr/bin/env python3
"""Aggregate the RedMeat CLIP-LR light-unfreezing study over seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


METRICS = (
    "C",
    "val_acc",
    "val_avg_group_acc",
    "val_worst_group_acc",
    "test_acc",
    "test_avg_group_acc",
    "test_worst_group_acc",
)


def parse_seeds(text: str) -> List[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def stats(values: Iterable[float]):
    values = list(values)
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    seeds = parse_seeds(args.seeds)

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        path = root / f"seed_{seed}" / "results.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed result: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            seed_rows = list(csv.DictReader(handle))
        if not seed_rows:
            raise RuntimeError(f"Empty result file: {path}")
        for row in seed_rows:
            if int(row["seed"]) != seed:
                raise RuntimeError(f"Seed mismatch in {path}")
            converted: Dict[str, object] = dict(row)
            converted["seed"] = seed
            converted["finetune_epoch"] = int(row["finetune_epoch"])
            for metric in METRICS:
                converted[metric] = float(row[metric])
            rows.append(converted)

    expected_keys = {
        (int(row["finetune_epoch"]), str(row["protocol"])) for row in rows if int(row["seed"]) == seeds[0]
    }
    summary_rows: List[Dict[str, object]] = []
    for epoch, protocol in sorted(expected_keys):
        selected = [
            row
            for row in rows
            if int(row["finetune_epoch"]) == epoch and row["protocol"] == protocol
        ]
        if {int(row["seed"]) for row in selected} != set(seeds):
            raise RuntimeError(f"Incomplete seeds for epoch={epoch} protocol={protocol}")
        summary: Dict[str, object] = {
            "finetune_epoch": epoch,
            "protocol": protocol,
            "n_seeds": len(selected),
            "seeds": ",".join(str(seed) for seed in seeds),
            "clip_model": selected[0]["clip_model"],
            "unfreeze_scope": selected[0]["unfreeze_scope"],
            "encoder_lr": selected[0]["encoder_lr"],
            "head_lr": selected[0]["head_lr"],
            "weight_decay": selected[0]["weight_decay"],
        }
        for metric in METRICS:
            mean, std = stats(float(row[metric]) for row in selected)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summary_rows.append(summary)

    write_csv(root / "all_seeds.csv", rows)
    write_csv(root / "summary.csv", summary_rows)
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"{'Epoch':>5}  {'Protocol':<10}  {'C':>17}  {'Test mean class':>20}  {'Test worst class':>20}"
    )
    print("-" * 82)
    for row in summary_rows:
        print(
            f"{int(row['finetune_epoch']):5d}  {str(row['protocol']):<10}  "
            f"{row['C_mean']:8.4g} +/- {row['C_std']:<7.4g}  "
            f"{row['test_avg_group_acc_mean']:7.2f} +/- {row['test_avg_group_acc_std']:<6.2f}  "
            f"{row['test_worst_group_acc_mean']:7.2f} +/- {row['test_worst_group_acc_std']:<6.2f}"
        )
    print(f"[DONE] {root / 'summary.csv'}")


if __name__ == "__main__":
    main()

