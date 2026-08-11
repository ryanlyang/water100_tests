#!/usr/bin/env python3
"""Combine epochwise DecoyMNIST R4RR Pointing Game diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence

import torch


EPOCH_DIR_RE = re.compile(r"^epoch_(\d+)$")
PROTOCOL_VERSION = 2
PRIMARY_PROTOCOL = "native_resolution_overlap"


def torch_load_compat(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def read_single_csv(path: Path) -> Dict[str, str]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    epoch_root = run_dir / "epochs"

    best_paths = sorted(
        (run_dir / "best_checkpoint").glob("*.pth"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not best_paths:
        raise RuntimeError(
            f"No validation-selected checkpoint found under {run_dir / 'best_checkpoint'}"
        )
    best_payload = torch_load_compat(best_paths[-1])
    best_epoch = int(best_payload["best_epoch"])

    rows: List[Dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        checkpoint = checkpoint_dir / f"decoy_r4rr_seed{args.seed}_epoch{epoch:02d}.pth"
        summary_path = epoch_root / f"epoch_{epoch:02d}" / "pointing_game_summary.csv"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing epoch checkpoint: {checkpoint}")
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing epoch Pointing Game summary: {summary_path}")

        payload = torch_load_compat(checkpoint)
        summary = read_single_csv(summary_path)
        if int(payload.get("seed", -1)) != int(args.seed) or int(payload.get("epoch", -1)) != epoch:
            raise RuntimeError(f"Checkpoint metadata mismatch: {checkpoint}")
        if (
            int(summary.get("seed", -1)) != int(args.seed)
            or summary.get("method") != "r4rr"
            or int(summary.get("mask_protocol_version", -1)) != PROTOCOL_VERSION
            or summary.get("primary_pg_protocol") != PRIMARY_PROTOCOL
        ):
            raise RuntimeError(f"Pointing Game metadata mismatch: {summary_path}")

        native_acc = 100.0 * float(summary["pg_native_acc"])
        native_random = 100.0 * float(summary["pg_native_random_acc"])
        rows.append(
            {
                "seed": int(args.seed),
                "epoch": epoch,
                "phase": "align" if bool(payload["attention_active"]) else "classify",
                "attention_active": int(bool(payload["attention_active"])),
                "selected_by_val": int(epoch == best_epoch),
                "effective_kl_lambda": float(payload["effective_kl_lambda"]),
                "train_loss": float(payload["train_loss"]),
                "train_acc_pct": float(payload["train_acc"]),
                "val_loss": float(payload["val_loss"]),
                "val_acc_pct": float(payload["val_acc"]),
                "test_classification_acc_pct": 100.0 * float(summary["classification_acc"]),
                "pg_native_acc_pct": native_acc,
                "pg_native_macro_pct": 100.0 * float(summary["pg_native_macro_class_acc"]),
                "pg_native_worst_pct": 100.0 * float(summary["pg_native_worst_class_acc"]),
                "pg_native_random_pct": native_random,
                "pg_native_minus_random_pct": native_acc - native_random,
                "pg_pixel_diagnostic_pct": 100.0 * float(summary["pg_acc"]),
                "zero_saliency_maps": int(summary["zero_saliency_maps"]),
                "pg_total": int(summary["pg_native_total"]),
                "checkpoint": str(checkpoint),
                "pointing_summary": str(summary_path),
            }
        )

    output_csv = run_dir / "epochwise_pointing_game.csv"
    write_csv(output_csv, rows)
    with open(run_dir / "epochwise_pointing_game.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print(
        "epoch phase    val_acc test_acc native_pg native-random pixel_pg zero_maps selected",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['epoch']:>5} {row['phase']:<8} "
            f"{row['val_acc_pct']:>7.2f} {row['test_classification_acc_pct']:>8.2f} "
            f"{row['pg_native_acc_pct']:>9.2f} {row['pg_native_minus_random_pct']:>13.2f} "
            f"{row['pg_pixel_diagnostic_pct']:>8.2f} {row['zero_saliency_maps']:>9} "
            f"{'yes' if row['selected_by_val'] else 'no'}",
            flush=True,
        )
    print(f"[DONE] validation-selected epoch={best_epoch}")
    print(f"[DONE] {output_csv}")


if __name__ == "__main__":
    main()
