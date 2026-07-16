#!/usr/bin/env python3
"""Score one Step 4 candidate or run on test for Step 11 post-hoc analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.candidate_training import validate_runtime_software  # noqa: E402
from fcv.gap_analysis import evaluate_pool_candidate_test  # noqa: E402
from fcv.test_evaluation import (  # noqa: E402
    load_frozen_selection,
    prepare_final_test_source,
)
from fcv.token_banks import candidate_checkpoints_for_run  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc test scoring for the complete Step 4 candidate pool."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--checkpoint", type=Path)
    candidate.add_argument(
        "--run-index",
        type=int,
        help="Score all ordered epoch candidates from one Step 4 sweep run.",
    )
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--selection-table", type=Path, default=None)
    parser.add_argument("--selection-summary", type=Path, default=None)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    validate_runtime_software(config)
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
    output_dir = args.output_dir or selection_root / "candidate_pool_test_scores"
    selection_table = args.selection_table or selection_root / "selection_table.csv"
    selection_summary = args.selection_summary or (
        selection_root / "selection_table_summary.json"
    )
    # Freeze and cryptographically validate Step 9 before opening the test manifest.
    frozen = load_frozen_selection(config, selection_table, selection_summary)
    source = prepare_final_test_source(
        config,
        test_manifest,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    checkpoints = (
        [args.checkpoint]
        if args.checkpoint is not None
        else candidate_checkpoints_for_run(config, candidate_root, args.run_index)
    )
    print(
        json.dumps(
            {
                "checkpoint_count": len(checkpoints),
                "device": args.device,
                "run_index": args.run_index,
                "test_manifest": str(source.manifest_path),
                "test_sample_count": source.sample_count,
                "output_dir": str(Path(output_dir).resolve()),
                "precision": "float32",
                "posthoc_pool_analysis_only": True,
                "eligible_for_model_selection": False,
                "selection_table_sha256": frozen.selection_table_sha256,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    for index, checkpoint in enumerate(checkpoints, start=1):
        result = evaluate_pool_candidate_test(
            config,
            checkpoint,
            source,
            frozen,
            output_dir,
            device=args.device,
            overwrite=args.overwrite,
        )
        metrics = result["metrics"]
        print(
            f"[POOL TEST] {index}/{len(checkpoints)} "
            f"candidate={result['candidate_id']} "
            f"status={result['invocation_status']} "
            f"accuracy={metrics['accuracy']:.6f} "
            f"balanced_group={metrics['balanced_group_accuracy']:.6f} "
            f"worst_group={metrics['worst_group_accuracy']:.6f}",
            flush=True,
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
