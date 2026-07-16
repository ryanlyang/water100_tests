#!/usr/bin/env python3
"""Aggregate the 27 independent Step 4 metric files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.candidate_training import aggregate_candidate_metrics  # noqa: E402
from fcv.config import load_and_validate_config  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-run biased-validation metrics into the 540-candidate table."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a diagnostic partial table instead of failing on missing array tasks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    candidate_root = args.candidate_root or output_root / config["outputs"][
        "candidate_models"
    ]
    output_csv = args.output_csv or candidate_root / "candidate_metrics_biased_val.csv"
    summary = args.summary or candidate_root / "candidate_pool_summary.json"
    result = aggregate_candidate_metrics(
        config,
        candidate_root,
        output_csv,
        summary,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
