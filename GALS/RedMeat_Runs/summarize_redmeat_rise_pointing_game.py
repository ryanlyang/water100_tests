#!/usr/bin/env python3
"""Strictly summarize RedMeat RISE Pointing Game results across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


METHODS = (
    "vanilla",
    "elrep",
    "upweight",
    "abn",
    "gals",
    "afr",
    "r4rr",
    "clip_lr",
    "clip_zs",
)
CLIP_METHODS = frozenset(("clip_lr", "clip_zs"))
CLASS_NAMES = (
    "prime_rib",
    "pork_chop",
    "steak",
    "baby_back_ribs",
    "filet_mignon",
)
RATE_METRICS = (
    "pg_acc",
    "pg_macro_class_acc",
    "pg_worst_class_acc",
    "pg_correct_only_acc",
    "pg_random_acc",
    "classification_acc",
    "classification_balanced_class_acc",
    "classification_worst_class_acc",
    "saliency_mass_in_meat",
)
CLASS_RATE_SUFFIXES = (
    "pg_acc",
    "classification_acc",
    "saliency_mass_in_meat",
)
SHARED_FIELDS = (
    "dataset",
    "split",
    "target_mode",
    "explainer",
    "primary_pg_protocol",
    "mask_protocol_version",
    "map_height",
    "map_width",
    "mask_source",
    "mask_manifest_sha256",
    "mask_threshold",
    "max_samples",
    "sample_seed",
    "rise_num_masks",
    "rise_grid_size",
    "rise_p1",
    "rise_seed",
    "rise_masks_sha256",
)
METHOD_FIXED_FIELDS = SHARED_FIELDS + ("method", "preprocessing")


def parse_csv_list(text: str, cast=str) -> List:
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate values are not allowed: {text}")
    return values


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_single_csv(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(rows)}")
    return dict(rows[0])


def finite_float(row: Mapping[str, str], key: str, source: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing/invalid {key!r} in {source}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {key!r} in {source}: {value}")
    return value


def integer(row: Mapping[str, str], key: str, source: Path) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing/invalid integer {key!r} in {source}") from exc


def assert_equal(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str], context: str
) -> None:
    for field in fields:
        values = {str(row.get(field, "")) for row in rows}
        if len(values) != 1:
            raise RuntimeError(f"Mixed {field!r} values in {context}: {sorted(values)}")


def mean_std(values: Iterable[float]) -> Tuple[float, float, int]:
    items = list(values)
    if not items or not all(math.isfinite(value) for value in items):
        raise RuntimeError("Cannot summarize empty or non-finite values")
    return statistics.mean(items), statistics.pstdev(items), len(items)


def read_seed(method_dir: Path, seed: int) -> Dict[str, object]:
    path = method_dir / f"seed_{seed}" / "pointing_game" / "pointing_game_summary.csv"
    raw = read_single_csv(path)
    method = method_dir.name
    if raw.get("dataset") != "redmeat" or raw.get("method") != method:
        raise RuntimeError(
            f"Dataset/method mismatch in {path}: "
            f"{raw.get('dataset')}/{raw.get('method')}"
        )
    if integer(raw, "seed", path) != seed:
        raise RuntimeError(f"Seed mismatch in {path}")
    if raw.get("split") != "test":
        raise RuntimeError(f"Expected test split in {path}")
    if raw.get("primary_pg_protocol") != "rise_pixel_argmax":
        raise RuntimeError(f"Unexpected Pointing Game protocol in {path}")
    if integer(raw, "mask_protocol_version", path) != 1:
        raise RuntimeError(f"Unexpected mask protocol in {path}")
    if integer(raw, "errors", path) != 0:
        raise RuntimeError(f"Evaluator reported errors in {path}")
    pg_total = integer(raw, "pg_total", path)
    max_samples = integer(raw, "max_samples", path)
    expected_total = max_samples if max_samples > 0 else 1250
    if pg_total != expected_total:
        raise RuntimeError(f"Expected {expected_total} samples in {path}, found {pg_total}")
    for field in ("mask_manifest_sha256", "rise_masks_sha256"):
        if not raw.get(field):
            raise RuntimeError(f"Missing {field} in {path}")

    row: Dict[str, object] = dict(raw)
    row["seed"] = seed
    row["source_csv"] = str(path)
    for key in RATE_METRICS:
        value = finite_float(raw, key, path)
        row[key] = value
        row[f"{key}_pct"] = 100.0 * value
    for class_name in CLASS_NAMES:
        for suffix in CLASS_RATE_SUFFIXES:
            key = f"class_{class_name}_{suffix}"
            value = finite_float(raw, key, path)
            row[key] = value
            row[f"{key}_pct"] = 100.0 * value
    for key in (
        "pg_hits",
        "pg_total",
        "pg_correct_only_hits",
        "pg_correct_only_total",
        "zero_saliency_maps",
        "missing_images",
        "missing_masks",
        "errors",
        "seconds",
    ):
        row[key] = integer(raw, key, path)
    row["zero_saliency_maps_pct"] = (
        100.0 * int(row["zero_saliency_maps"]) / pg_total
    )
    return row


def add_stats(
    target: Dict[str, object],
    rows: Sequence[Mapping[str, object]],
    source_key: str,
    output_prefix: Optional[str] = None,
) -> None:
    prefix = output_prefix or source_key
    mean, std, count = mean_std(float(row[source_key]) for row in rows)
    target[f"{prefix}_mean"] = mean
    target[f"{prefix}_std"] = std
    target[f"{prefix}_n"] = count


def summarize_method(method_dir: Path, seeds: Sequence[int]) -> Dict[str, object]:
    method_dir = method_dir.expanduser().resolve()
    if method_dir.name not in METHODS:
        raise ValueError(f"Unsupported method directory: {method_dir}")
    rows = [read_seed(method_dir, seed) for seed in seeds]
    assert_equal(rows, METHOD_FIXED_FIELDS, str(method_dir))

    summary: Dict[str, object] = {
        "dataset": "redmeat",
        "method": method_dir.name,
        "split": "test",
        "target_mode": rows[0]["target_mode"],
        "explainer": rows[0]["explainer"],
        "primary_pg_protocol": rows[0]["primary_pg_protocol"],
        "n_seeds": len(rows),
        "seeds": ",".join(str(seed) for seed in seeds),
        "pg_total_per_seed": rows[0]["pg_total"],
        "mask_protocol_version": rows[0]["mask_protocol_version"],
        "map_height": rows[0]["map_height"],
        "map_width": rows[0]["map_width"],
        "mask_source": rows[0]["mask_source"],
        "mask_manifest_sha256": rows[0]["mask_manifest_sha256"],
        "mask_threshold": rows[0]["mask_threshold"],
        "max_samples": rows[0]["max_samples"],
        "sample_seed": rows[0]["sample_seed"],
        "rise_masks_sha256": rows[0]["rise_masks_sha256"],
        "rise_num_masks": rows[0]["rise_num_masks"],
        "rise_grid_size": rows[0]["rise_grid_size"],
        "rise_p1": rows[0]["rise_p1"],
        "rise_seed": rows[0]["rise_seed"],
    }
    for metric in RATE_METRICS:
        add_stats(summary, rows, f"{metric}_pct")
    add_stats(summary, rows, "zero_saliency_maps_pct")
    add_stats(summary, rows, "seconds")
    for class_name in CLASS_NAMES:
        for suffix in CLASS_RATE_SUFFIXES:
            add_stats(summary, rows, f"class_{class_name}_{suffix}_pct")

    write_csv(method_dir / "pointing_game_per_seed.csv", rows)
    write_csv(method_dir / "pointing_game_seed_summary.csv", [summary])
    atomic_json(method_dir / "pointing_game_seed_summary.json", summary)
    print(
        f"[SUMMARY] redmeat {method_dir.name}: "
        f"overall={summary['pg_acc_pct_mean']:.2f} +/- "
        f"{summary['pg_acc_pct_std']:.2f}, "
        f"macro={summary['pg_macro_class_acc_pct_mean']:.2f} +/- "
        f"{summary['pg_macro_class_acc_pct_std']:.2f}, "
        f"worst={summary['pg_worst_class_acc_pct_mean']:.2f} +/- "
        f"{summary['pg_worst_class_acc_pct_std']:.2f} (n={len(rows)})",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--method-dir", type=Path)
    group.add_argument("--run-root", type=Path)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--clip-seeds", default="0")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Summarize available methods instead of requiring every requested method.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_csv_list(args.seeds, int)
    clip_seeds = parse_csv_list(args.clip_seeds, int)
    if args.method_dir is not None:
        method_dir = args.method_dir.expanduser().resolve()
        selected = clip_seeds if method_dir.name in CLIP_METHODS else seeds
        summarize_method(method_dir, selected)
        return

    run_root = args.run_root.expanduser().resolve()
    methods = parse_csv_list(args.methods)
    unsupported = sorted(set(methods) - set(METHODS))
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")
    summaries: List[Dict[str, object]] = []
    missing: List[str] = []
    for method in methods:
        method_dir = run_root / method
        selected = clip_seeds if method in CLIP_METHODS else seeds
        try:
            summaries.append(summarize_method(method_dir, selected))
        except FileNotFoundError:
            if args.allow_partial:
                missing.append(method)
                continue
            raise
    if not summaries:
        raise RuntimeError(f"No complete method results found under {run_root}")
    assert_equal(summaries, SHARED_FIELDS, str(run_root))
    write_csv(run_root / "pointing_game_all_methods_summary.csv", summaries)
    atomic_json(run_root / "pointing_game_all_methods_summary.json", summaries)
    if missing:
        print(
            f"[PARTIAL] skipped methods with missing results: {','.join(missing)}",
            flush=True,
        )
    print(f"[DONE] combined summary: {run_root / 'pointing_game_all_methods_summary.csv'}")


if __name__ == "__main__":
    main()
