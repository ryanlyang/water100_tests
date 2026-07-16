#!/usr/bin/env python3
"""Build Step 6 background-token banks for one candidate or one sweep run."""

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
from fcv.token_banks import (  # noqa: E402
    build_background_token_banks,
    candidate_checkpoints_for_run,
    prepare_token_bank_source,
)
from fcv.vit_counterfactual_forward import validate_reconstruction_gate  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract model-specific land/water raw-patch token banks."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--checkpoint", type=Path)
    candidate.add_argument(
        "--run-index",
        type=int,
        help="Build the three fixed-epoch candidates for one Step 4 run index.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patch-masks", type=Path, default=None)
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
    reconstruction_reports = validate_reconstruction_gate(config)
    output_root = Path(config["paths"]["output_root"])
    manifest_root = output_root / config["outputs"]["split_manifests"]
    manifest = args.manifest or manifest_root / "metadata_val.csv"
    patch_masks = args.patch_masks or (
        output_root / config["outputs"]["patch_masks"] / "patch_masks_val.pt"
    )
    candidate_root = args.candidate_root or (
        output_root / config["outputs"]["candidate_models"]
    )
    output_dir = args.output_dir or (
        output_root / config["outputs"]["token_banks"]
    )
    source = prepare_token_bank_source(
        config,
        manifest,
        patch_masks,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    if args.checkpoint is not None:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = candidate_checkpoints_for_run(
            config,
            candidate_root,
            args.run_index,
        )

    print(
        json.dumps(
            {
                "checkpoint_count": len(checkpoints),
                "device": args.device,
                "eligible_counts_by_label": dict(source.eligible_counts_by_label),
                "manifest": str(source.manifest_path),
                "output_dir": str(Path(output_dir).resolve()),
                "patch_masks": str(source.patch_mask_path),
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
        result = build_background_token_banks(
            config,
            checkpoint,
            source,
            output_dir,
            reconstruction_reports=reconstruction_reports,
            device=args.device,
            overwrite=args.overwrite,
        )
        results.append(
            {
                "candidate_id": result["candidate_id"],
                "status": result["status"],
                "banks": result["banks"],
            }
        )
        print(
            f"[TOKEN BANK] {index}/{len(checkpoints)} "
            f"candidate={result['candidate_id']} status={result['status']}",
            flush=True,
        )
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
