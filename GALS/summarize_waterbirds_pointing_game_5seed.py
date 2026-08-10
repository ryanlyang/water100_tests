#!/usr/bin/env python3
"""Summarize per-seed Waterbirds Pointing Game CSVs as mean +/- population std."""

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


def _atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _parse_seeds(text: str) -> List[int]:
    seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one expected seed is required.")
    return seeds


def _read_one(method_dir: Path, seed: int) -> Dict[str, object]:
    path = method_dir / f"seed_{seed}" / "pointing_game" / "pointing_game_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing seed {seed} Pointing Game summary: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}, found {len(rows)}")
    row: Dict[str, object] = dict(rows[0])
    row["seed"] = seed
    row["source_csv"] = str(path)
    for key in ("pg_acc",) + tuple(f"group_{g}_pg_acc" for g in GROUPS):
        value = float(row[key])
        if key == "pg_acc" and not math.isfinite(value):
            raise RuntimeError(f"Non-finite {key} for seed {seed}: {value}")
        row[key] = value
        row[f"{key}_pct"] = value * 100.0
    for key in ("pg_hits", "pg_total", "missing_images", "missing_masks", "errors"):
        row[key] = int(row[key])
    return row


def _mean_std(values: Iterable[float]) -> Dict[str, object]:
    vals = [value for value in values if math.isfinite(value)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def summarize_method(method_dir: Path, seeds: Sequence[int]) -> Dict[str, object]:
    rows = [_read_one(method_dir, seed) for seed in seeds]
    dataset = str(rows[0]["dataset"])
    method = str(rows[0]["method"])
    for row in rows:
        if row["dataset"] != dataset or row["method"] != method:
            raise RuntimeError(f"Mixed dataset/method rows under {method_dir}")

    overall = _mean_std(float(row["pg_acc_pct"]) for row in rows)
    summary: Dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "split": rows[0]["split"],
        "target_mode": rows[0]["target_mode"],
        "n_seeds": len(rows),
        "seeds": ",".join(str(x) for x in seeds),
        "pg_acc_mean_pct": overall["mean"],
        "pg_acc_std_pct": overall["std"],
    }
    for group in GROUPS:
        stats = _mean_std(float(row[f"group_{group}_pg_acc_pct"]) for row in rows)
        summary[f"group_{group}_pg_acc_mean_pct"] = stats["mean"]
        summary[f"group_{group}_pg_acc_std_pct"] = stats["std"]
        summary[f"group_{group}_n_seeds"] = stats["n"]

    _write_csv(method_dir / "pointing_game_per_seed.csv", rows)
    _write_csv(method_dir / "pointing_game_5seed_summary.csv", [summary])
    _atomic_json(method_dir / "pointing_game_5seed_summary.json", summary)
    print(
        f"[SUMMARY] {dataset} {method}: "
        f"Pointing Game={overall['mean']:.2f} +/- {overall['std']:.2f} (n={len(rows)})",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--method-dir", type=Path)
    group.add_argument("--run-root", type=Path)
    p.add_argument("--seeds", default="0,1,2,3,4")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    if args.method_dir is not None:
        method_dir = args.method_dir.expanduser().resolve()
        summarize_method(method_dir, seeds)
        return

    run_root = args.run_root.expanduser().resolve()
    summaries: List[Dict[str, object]] = []
    for method_dir in sorted(run_root.glob("waterbirds_*/*")):
        if method_dir.is_dir() and any(method_dir.glob("seed_*/pointing_game/pointing_game_summary.csv")):
            summaries.append(summarize_method(method_dir, seeds))
    if not summaries:
        raise RuntimeError(f"No complete Pointing Game method directories found under {run_root}")
    _write_csv(run_root / "pointing_game_all_methods_5seed_summary.csv", summaries)
    _atomic_json(run_root / "pointing_game_all_methods_5seed_summary.json", summaries)
    print(f"[DONE] combined summary: {run_root / 'pointing_game_all_methods_5seed_summary.csv'}")


if __name__ == "__main__":
    main()
