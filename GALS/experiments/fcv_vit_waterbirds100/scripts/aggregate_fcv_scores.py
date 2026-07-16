#!/usr/bin/env python3
"""Validate and aggregate Step 7 candidate FCV summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.fcv_scoring import aggregate_fcv_score_summaries  # noqa: E402
from fcv.token_banks import prepare_token_bank_source  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Step 7 FCV scores.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patch-masks", type=Path, default=None)
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--donor-plan", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    manifest = args.manifest or (
        output_root / config["outputs"]["split_manifests"] / "metadata_val.csv"
    )
    patch_masks = args.patch_masks or (
        output_root / config["outputs"]["patch_masks"] / "patch_masks_val.pt"
    )
    score_dir = args.score_dir or (
        output_root / config["outputs"]["fcv_scores"]
    )
    donor_plan = args.donor_plan or Path(score_dir) / "opposite_donor_plan.pt"
    output_csv = args.output_csv or Path(score_dir) / "candidate_fcv_scores.csv"
    output_summary = args.output_summary or (
        Path(score_dir) / "candidate_fcv_scores_summary.json"
    )
    source = prepare_token_bank_source(
        config,
        manifest,
        patch_masks,
        check_images=False,
    )
    summary = aggregate_fcv_score_summaries(
        config,
        score_dir,
        output_csv,
        output_summary,
        source=source,
        donor_plan_path=donor_plan,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
