#!/usr/bin/env python3
"""Validate and aggregate Step 11 post-hoc candidate-pool test scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.gap_analysis import aggregate_pool_test_summaries  # noqa: E402
from fcv.test_evaluation import (  # noqa: E402
    load_frozen_selection,
    prepare_final_test_source,
)


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate all post-hoc candidate-pool test summaries."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--selection-table", type=Path, default=None)
    parser.add_argument("--selection-summary", type=Path, default=None)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--pool-test-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    selection_root = output_root / config["outputs"]["selection_results"]
    test_manifest = args.test_manifest or (
        output_root
        / config["outputs"]["split_manifests"]
        / "metadata_test_analysis_only.csv"
    )
    candidate_root = args.candidate_root or (
        output_root / config["outputs"]["candidate_models"]
    )
    pool_test_dir = args.pool_test_dir or (
        selection_root / "candidate_pool_test_scores"
    )
    output_csv = args.output_csv or (
        selection_root / "candidate_pool_test_scores.csv"
    )
    output_summary = args.output_summary or (
        selection_root / "candidate_pool_test_scores_summary.json"
    )
    selection_table = args.selection_table or selection_root / "selection_table.csv"
    selection_summary = args.selection_summary or (
        selection_root / "selection_table_summary.json"
    )
    frozen = load_frozen_selection(config, selection_table, selection_summary)
    source = prepare_final_test_source(
        config,
        test_manifest,
        check_images=False,
    )
    summary = aggregate_pool_test_summaries(
        config,
        candidate_root,
        pool_test_dir,
        output_csv,
        output_summary,
        source=source,
        frozen=frozen,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
