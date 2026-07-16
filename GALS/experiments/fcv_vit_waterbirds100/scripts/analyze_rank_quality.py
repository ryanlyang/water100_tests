#!/usr/bin/env python3
"""Run Step 12 selector correlation, rank quality, and scatter analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.rank_analysis import analyze_rank_quality  # noqa: E402
from fcv.test_evaluation import (  # noqa: E402
    load_frozen_selection,
    prepare_final_test_source,
)


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Step 12 selector rank quality.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection-table", type=Path, default=None)
    parser.add_argument("--selection-summary", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--pool-csv", type=Path, default=None)
    parser.add_argument("--pool-summary", type=Path, default=None)
    parser.add_argument("--output-results-csv", type=Path, default=None)
    parser.add_argument("--output-candidates-csv", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None)
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
    pool_csv = args.pool_csv or selection_root / "candidate_pool_test_scores.csv"
    pool_summary = args.pool_summary or (
        selection_root / "candidate_pool_test_scores_summary.json"
    )
    output_results_csv = args.output_results_csv or (
        selection_root / "rank_correlation_results.csv"
    )
    output_candidates_csv = args.output_candidates_csv or (
        selection_root / "candidate_rank_analysis.csv"
    )
    output_summary = args.output_summary or (
        selection_root / "rank_correlation_results_summary.json"
    )
    plot_dir = args.plot_dir or selection_root / "selector_scatter_plots"
    frozen = load_frozen_selection(config, selection_table, selection_summary)
    source = prepare_final_test_source(
        config,
        test_manifest,
        check_images=False,
    )
    summary = analyze_rank_quality(
        config,
        frozen,
        source,
        pool_csv,
        pool_summary,
        output_results_csv,
        output_candidates_csv,
        output_summary,
        plot_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
