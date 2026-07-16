#!/usr/bin/env python3
"""Score all four Step 8 controls for one candidate or sweep run."""

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
from fcv.controls import score_candidate_controls  # noqa: E402
from fcv.token_banks import (  # noqa: E402
    candidate_checkpoints_for_run,
    prepare_token_bank_source,
)
from fcv.vit_counterfactual_forward import validate_reconstruction_gate  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four FCV control swaps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--checkpoint", type=Path)
    candidate.add_argument("--run-index", type=int)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patch-masks", type=Path, default=None)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--token-bank-dir", type=Path, default=None)
    parser.add_argument("--opposite-donor-plan", type=Path, default=None)
    parser.add_argument("--control-plan", type=Path, default=None)
    parser.add_argument("--step7-score-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--target-batch-size", type=int, default=None)
    parser.add_argument("--counterfactual-forward-batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    validate_runtime_software(config)
    execution = config["execution"]
    if args.batch_size is None:
        args.batch_size = int(config["training"]["batch_size"])
    if args.batch_size != int(config["training"]["batch_size"]):
        raise ValueError(
            "Step 8 ordinary validation must use the locked Step 4/7 batch size."
        )
    if args.target_batch_size is None:
        args.target_batch_size = int(execution["control_target_batch_size"])
    if args.counterfactual_forward_batch_size is None:
        args.counterfactual_forward_batch_size = int(
            execution["control_counterfactual_forward_batch_size"]
        )
    if args.target_batch_size != int(execution["control_target_batch_size"]):
        raise ValueError("Step 8 target batch size differs from the locked value.")
    if args.counterfactual_forward_batch_size != int(
        execution["control_counterfactual_forward_batch_size"]
    ):
        raise ValueError("Step 8 forward batch size differs from the locked value.")
    reconstruction_reports = validate_reconstruction_gate(config)
    output_root = Path(config["paths"]["output_root"])
    manifest = args.manifest or (
        output_root / config["outputs"]["split_manifests"] / "metadata_val.csv"
    )
    patch_masks = args.patch_masks or (
        output_root / config["outputs"]["patch_masks"] / "patch_masks_val.pt"
    )
    candidate_root = args.candidate_root or (
        output_root / config["outputs"]["candidate_models"]
    )
    token_bank_dir = args.token_bank_dir or (
        output_root / config["outputs"]["token_banks"]
    )
    step7_score_dir = args.step7_score_dir or (
        output_root / config["outputs"]["fcv_scores"]
    )
    opposite_plan = args.opposite_donor_plan or (
        Path(step7_score_dir) / "opposite_donor_plan.pt"
    )
    output_dir = args.output_dir or (
        output_root / config["outputs"]["control_scores"]
    )
    control_plan = args.control_plan or Path(output_dir) / "control_plan.pt"
    source = prepare_token_bank_source(
        config,
        manifest,
        patch_masks,
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
                "control_plan": str(Path(control_plan).resolve()),
                "device": args.device,
                "opposite_donor_plan": str(Path(opposite_plan).resolve()),
                "output_dir": str(Path(output_dir).resolve()),
                "run_index": args.run_index,
                "validation_sample_count": source.sample_count,
                "reconstruction_reports": reconstruction_reports,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    results = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        summary = score_candidate_controls(
            config,
            checkpoint,
            source,
            token_bank_dir,
            opposite_plan,
            control_plan,
            step7_score_dir,
            output_dir,
            reconstruction_reports=reconstruction_reports,
            device=args.device,
            target_batch_size=args.target_batch_size,
            counterfactual_forward_batch_size=args.counterfactual_forward_batch_size,
            overwrite=args.overwrite,
        )
        results.append(
            {
                "candidate_id": summary["candidate_id"],
                "controls": summary["controls"],
                "status": summary["status"],
            }
        )
        print(
            f"[CONTROLS] {index}/{len(checkpoints)} "
            f"candidate={summary['candidate_id']} status={summary['status']} "
            f"same-opposite={summary['same_minus_opposite_accuracy']:.6f} "
            f"evidence-bg-gap={summary['evidence_vs_background_sensitivity_gap']:.6f}",
            flush=True,
        )
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
