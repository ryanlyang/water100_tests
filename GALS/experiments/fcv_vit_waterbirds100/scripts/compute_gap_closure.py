#!/usr/bin/env python3
"""Compute Step 11 FCV-to-Oracle selection gap closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.gap_analysis import (  # noqa: E402
    compute_gap_closure_summary,
    validate_final_test_results,
)
from fcv.test_evaluation import (  # noqa: E402
    load_frozen_selection,
    prepare_final_test_source,
)


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Step 11 gap closure.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection-table", type=Path, default=None)
    parser.add_argument("--selection-summary", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--final-results-csv", type=Path, default=None)
    parser.add_argument("--final-results-summary", type=Path, default=None)
    parser.add_argument("--pool-csv", type=Path, default=None)
    parser.add_argument("--pool-summary", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument(
        "--validate-step10-only",
        action="store_true",
        help="Validate frozen Step 9 and Step 10 outputs without requiring pool scores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    selection_root = output_root / config["outputs"]["selection_results"]
    selection_table = args.selection_table or selection_root / "selection_table.csv"
    selection_summary = args.selection_summary or (
        selection_root / "selection_table_summary.json"
    )
    test_manifest = args.test_manifest or (
        output_root
        / config["outputs"]["split_manifests"]
        / "metadata_test_analysis_only.csv"
    )
    final_results_csv = args.final_results_csv or (
        selection_root / "final_test_results.csv"
    )
    final_results_summary = args.final_results_summary or (
        selection_root / "final_test_results_summary.json"
    )
    pool_csv = args.pool_csv or selection_root / "candidate_pool_test_scores.csv"
    pool_summary = args.pool_summary or (
        selection_root / "candidate_pool_test_scores_summary.json"
    )
    output_csv = args.output_csv or selection_root / "gap_closure_summary.csv"
    output_summary = args.output_summary or (
        selection_root / "gap_closure_summary.json"
    )
    frozen = load_frozen_selection(config, selection_table, selection_summary)
    source = prepare_final_test_source(
        config,
        test_manifest,
        check_images=False,
    )
    final = validate_final_test_results(
        config,
        frozen,
        source,
        final_results_csv,
        final_results_summary,
    )
    if args.validate_step10_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "selector_count": len(final),
                    "selection_frozen_before_test": True,
                    "test_metrics_affected_selection": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    summary = compute_gap_closure_summary(
        config,
        frozen,
        source,
        final_results_csv,
        final_results_summary,
        pool_csv,
        pool_summary,
        output_csv,
        output_summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
