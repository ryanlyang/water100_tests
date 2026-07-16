#!/usr/bin/env python3
"""Validate and aggregate Step 8 control summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.controls import aggregate_control_summaries  # noqa: E402
from fcv.token_banks import prepare_token_bank_source  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Step 8 control scores.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patch-masks", type=Path, default=None)
    parser.add_argument("--control-score-dir", type=Path, default=None)
    parser.add_argument("--opposite-donor-plan", type=Path, default=None)
    parser.add_argument("--control-plan", type=Path, default=None)
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
    control_score_dir = args.control_score_dir or (
        output_root / config["outputs"]["control_scores"]
    )
    opposite_plan = args.opposite_donor_plan or (
        output_root / config["outputs"]["fcv_scores"] / "opposite_donor_plan.pt"
    )
    control_plan = args.control_plan or Path(control_score_dir) / "control_plan.pt"
    output_csv = args.output_csv or (
        Path(control_score_dir) / "candidate_control_scores.csv"
    )
    output_summary = args.output_summary or (
        Path(control_score_dir) / "candidate_control_scores_summary.json"
    )
    source = prepare_token_bank_source(
        config,
        manifest,
        patch_masks,
        check_images=False,
    )
    summary = aggregate_control_summaries(
        config,
        control_score_dir,
        output_csv,
        output_summary,
        source=source,
        opposite_plan_path=opposite_plan,
        control_plan_path=control_plan,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
