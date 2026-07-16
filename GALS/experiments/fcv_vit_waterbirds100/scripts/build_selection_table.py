#!/usr/bin/env python3
"""Build the Step 9 validation-only selector comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.selectors import build_selection_table  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join Step 4/7/8/9 validation metrics and choose checkpoints."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-metrics", type=Path, default=None)
    parser.add_argument("--fcv-scores", type=Path, default=None)
    parser.add_argument("--control-scores", type=Path, default=None)
    parser.add_argument("--oracle-scores", type=Path, default=None)
    parser.add_argument("--candidate-metrics-summary", type=Path, default=None)
    parser.add_argument("--fcv-scores-summary", type=Path, default=None)
    parser.add_argument("--control-scores-summary", type=Path, default=None)
    parser.add_argument("--oracle-scores-summary", type=Path, default=None)
    parser.add_argument("--output-table", type=Path, default=None)
    parser.add_argument("--output-matrix", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    selection_root = output_root / config["outputs"]["selection_results"]
    candidate_root = output_root / config["outputs"]["candidate_models"]
    fcv_root = output_root / config["outputs"]["fcv_scores"]
    control_root = output_root / config["outputs"]["control_scores"]
    summary = build_selection_table(
        config,
        args.candidate_metrics
        or candidate_root / "candidate_metrics_biased_val.csv",
        args.fcv_scores or fcv_root / "candidate_fcv_scores.csv",
        args.control_scores or control_root / "candidate_control_scores.csv",
        args.oracle_scores or selection_root / "candidate_oracle_scores.csv",
        args.output_table or selection_root / "selection_table.csv",
        args.output_matrix or selection_root / "candidate_selector_scores.csv",
        args.output_summary or selection_root / "selection_table_summary.json",
        candidate_metrics_summary_json=args.candidate_metrics_summary
        or candidate_root / "candidate_pool_summary.json",
        fcv_scores_summary_json=args.fcv_scores_summary
        or fcv_root / "candidate_fcv_scores_summary.json",
        control_scores_summary_json=args.control_scores_summary
        or control_root / "candidate_control_scores_summary.json",
        oracle_scores_summary_json=args.oracle_scores_summary
        or selection_root / "candidate_oracle_scores_summary.json",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
