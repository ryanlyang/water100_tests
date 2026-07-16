#!/usr/bin/env python3
"""Validate and aggregate all Step 9 Oracle validation summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.selectors import (  # noqa: E402
    aggregate_oracle_summaries,
    prepare_oracle_validation_source,
)


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Step 9 Oracle metrics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--oracle-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    selection_root = output_root / config["outputs"]["selection_results"]
    manifest = args.manifest or (
        output_root
        / config["outputs"]["split_manifests"]
        / "metadata_oracle_val_analysis_only.csv"
    )
    oracle_dir = args.oracle_dir or selection_root / "oracle_scores"
    output_csv = args.output_csv or selection_root / "candidate_oracle_scores.csv"
    output_summary = args.output_summary or (
        selection_root / "candidate_oracle_scores_summary.json"
    )
    source = prepare_oracle_validation_source(
        config,
        manifest,
        check_images=False,
    )
    summary = aggregate_oracle_summaries(
        config,
        oracle_dir,
        output_csv,
        output_summary,
        source=source,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
