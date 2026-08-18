#!/usr/bin/env python3
"""Compare forward KL with transferred WB95 alternative alignment losses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from imagenet9_final_utils import atomic_csv


LOSSES = ("forward_kl", "reverse_kl", "jensen_shannon", "squared_l2", "cosine")
METRICS = ("original", "mixed_same", "mixed_rand", "bg_gap", "mixed_next")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def result_root(transfer_root: Path, loss: str) -> Path:
    if loss == "forward_kl":
        return transfer_root / "waterbirds95" / "r4rr" / "main"
    return transfer_root / "waterbirds95_alignment" / loss / "main"


def main() -> int:
    args = parse_args()
    rows = []
    for loss in LOSSES:
        root = result_root(args.transfer_root, loss)
        summary_path = root / "summary.csv"
        contract_path = root / "run_contract.json"
        if not summary_path.is_file() or not contract_path.is_file():
            if args.allow_incomplete:
                print(f"[SKIP] incomplete {loss}: {root}")
                continue
            raise FileNotFoundError(f"Missing completed result files under {root}")
        by_metric = {row["metric"]: row for row in csv.DictReader(summary_path.open())}
        missing = [metric for metric in METRICS if metric not in by_metric]
        if missing:
            raise RuntimeError(f"{summary_path} is missing metrics: {missing}")
        counts = {int(by_metric[metric]["n"]) for metric in METRICS}
        if len(counts) != 1:
            raise RuntimeError(f"Inconsistent seed counts in {summary_path}: {counts}")
        count = counts.pop()
        if count != 5 and not args.allow_incomplete:
            raise RuntimeError(f"Expected five seeds for {loss}, found {count}")
        contract = json.loads(contract_path.read_text())
        if loss == "forward_kl":
            params = contract["config"]["params"]
            source_trial = "optimized_config"
            source_attention = int(params["source_attention_epoch"])
            target_attention = int(params["attention_epoch"])
        else:
            source_trial = int(contract["source_best_trial"])
            source_attention = int(contract["source_hparams"]["attention_epoch"])
            target_attention = int(contract["target_hparams"]["attention_epoch"])
        row = {
            "alignment_loss": loss,
            "source_best_trial": source_trial,
            "source_attention_epoch": source_attention,
            "target_attention_epoch": target_attention,
            "n": count,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(by_metric[metric]["mean"])
            row[f"{metric}_std"] = float(by_metric[metric]["std"])
        rows.append(row)

    fieldnames = [
        "alignment_loss", "source_best_trial", "source_attention_epoch",
        "target_attention_epoch", "n",
    ]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    output_root = args.transfer_root / "waterbirds95_alignment"
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "comparison.csv"
    atomic_csv(output, fieldnames, rows)

    print("\nWB95-optimized alignment losses transferred to ImageNet-9")
    print(
        f"{'Loss':<18} {'Original':>15} {'Mixed-Same':>15} "
        f"{'Mixed-Rand':>15} {'BG Gap':>15} {'Mixed-Next':>15}"
    )
    for row in rows:
        cells = [
            f"{row[f'{metric}_mean']:.2f} +/- {row[f'{metric}_std']:.2f}"
            for metric in METRICS
        ]
        print(f"{row['alignment_loss']:<18} " + " ".join(f"{cell:>15}" for cell in cells))
    print(f"\n[DONE] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
