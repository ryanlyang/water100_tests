#!/usr/bin/env python3
"""Cache the shared deterministic opposite-context donor draws for Step 7."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.candidate_training import get_sweep_run  # noqa: E402
from fcv.config import load_and_validate_config  # noqa: E402
from fcv.fcv_scoring import (  # noqa: E402
    load_background_bank,
    prepare_opposite_donor_plan,
)
from fcv.token_banks import CONTEXT_NAMES, prepare_token_bank_source  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the candidate-independent FCV donor-index plan."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patch-masks", type=Path, default=None)
    parser.add_argument("--token-bank-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference-run-index", type=int, default=0)
    parser.add_argument("--reference-epoch", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    manifest = args.manifest or (
        output_root / config["outputs"]["split_manifests"] / "metadata_val.csv"
    )
    patch_masks = args.patch_masks or (
        output_root / config["outputs"]["patch_masks"] / "patch_masks_val.pt"
    )
    token_bank_dir = args.token_bank_dir or (
        output_root / config["outputs"]["token_banks"]
    )
    output = args.output or (
        output_root / config["outputs"]["fcv_scores"] / "opposite_donor_plan.pt"
    )
    epochs = int(config["training"]["epochs"])
    if args.reference_epoch < 1 or args.reference_epoch > epochs:
        raise ValueError(f"reference_epoch must be in [1, {epochs}].")
    run = get_sweep_run(config, args.reference_run_index)
    candidate_id = run.candidate_id(args.reference_epoch)
    source = prepare_token_bank_source(
        config,
        manifest,
        patch_masks,
        check_images=False,
    )
    banks = {
        label: load_background_bank(
            config,
            Path(token_bank_dir) / f"{candidate_id}_{context_name}.pt",
            source,
            expected_label=label,
            expected_candidate_id=candidate_id,
        )
        for label, context_name in CONTEXT_NAMES.items()
    }
    plan = prepare_opposite_donor_plan(
        config,
        source,
        banks,
        output,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "donor_samples_per_image": plan["donor_samples_per_image"],
                "donor_sampling_seed": plan["donor_sampling_seed"],
                "eligible_sample_count": plan["eligible_sample_count"],
                "output": str(Path(output).expanduser().resolve()),
                "plan_content_sha256": plan["plan_content_sha256"],
                "reference_candidate_id": plan["reference_candidate_id"],
                "sample_count": plan["sample_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
