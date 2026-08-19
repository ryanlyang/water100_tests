#!/usr/bin/env python3
"""Summarize deterministic CLIP RN50 zero-shot RISE Pointing Game results."""

from __future__ import annotations

import argparse
from pathlib import Path

from imagenet9_final_utils import atomic_json
from imagenet9_pointing_game_utils import PRIMARY_VARIANTS, write_csv
from summarize_imagenet9_wb95_transfer_rise import summarize_method_variant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(PRIMARY_VARIANTS))
    args = parser.parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(PRIMARY_VARIANTS))
    if unknown or not variants:
        raise ValueError(f"Unsupported or empty variants: {unknown}")

    rows = [
        summarize_method_variant(args.run_root, "clip_zs_rn50", variant, [0])
        for variant in variants
    ]
    output_csv = args.run_root / "clip_rn50_zeroshot_rise_comparison.csv"
    write_csv(output_csv, rows)
    atomic_json(args.run_root / "clip_rn50_zeroshot_rise_comparison.json", rows)
    for row in rows:
        print(
            f"[SUMMARY] CLIP-ZS RN50 {row['variant']:10s} "
            f"PG={row['pg_acc_mean_pct']:.2f} "
            f"macro={row['pg_macro_class_acc_mean_pct']:.2f} "
            f"worst={row['pg_worst_class_acc_mean_pct']:.2f} "
            f"classification={row['classification_acc_mean_pct']:.2f}",
            flush=True,
        )
    print("[NOTE] Frozen zero-shot evaluation is deterministic; no seed standard deviation.")
    print(f"[DONE] {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
