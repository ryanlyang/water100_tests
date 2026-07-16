#!/usr/bin/env python3
"""Evaluate one Step 4 candidate or one complete sweep run on Oracle validation."""

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
from fcv.selectors import (  # noqa: E402
    evaluate_candidate_oracle,
    prepare_oracle_validation_source,
)
from fcv.token_banks import candidate_checkpoints_for_run  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score candidate checkpoints on analysis-only Oracle validation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--checkpoint", type=Path)
    candidate.add_argument(
        "--run-index",
        type=int,
        help="Score all 20 ordered epoch candidates from one Step 4 sweep run.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
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
    manifest = args.manifest or (
        output_root
        / config["outputs"]["split_manifests"]
        / "metadata_oracle_val_analysis_only.csv"
    )
    candidate_root = args.candidate_root or (
        output_root / config["outputs"]["candidate_models"]
    )
    output_dir = args.output_dir or selection_root / "oracle_scores"
    source = prepare_oracle_validation_source(
        config,
        manifest,
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
                "manifest": str(Path(manifest).resolve()),
                "oracle_sample_count": source.sample_count,
                "output_dir": str(Path(output_dir).resolve()),
                "precision": "float32",
                "run_index": args.run_index,
                "test_data_accessed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    results = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        summary = evaluate_candidate_oracle(
            config,
            checkpoint,
            source,
            output_dir,
            device=args.device,
            overwrite=args.overwrite,
        )
        metrics = summary["metrics"]
        results.append(
            {
                "candidate_id": summary["candidate_id"],
                "balanced_group_accuracy": metrics["balanced_group_accuracy"],
                "worst_group_accuracy": metrics["worst_group_accuracy"],
                "status": summary["invocation_status"],
            }
        )
        print(
            f"[ORACLE] {index}/{len(checkpoints)} "
            f"candidate={summary['candidate_id']} "
            f"status={summary['invocation_status']} "
            f"accuracy={metrics['accuracy']:.6f} "
            f"balanced_group={metrics['balanced_group_accuracy']:.6f} "
            f"worst_group={metrics['worst_group_accuracy']:.6f}",
            flush=True,
        )
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
