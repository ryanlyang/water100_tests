#!/usr/bin/env python3
"""Build one reference bank and the shared FCV/control draw plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.candidate_training import (  # noqa: E402
    aggregate_candidate_metrics,
    get_sweep_run,
    validate_runtime_software,
)
from fcv.config import candidate_epochs, load_and_validate_config  # noqa: E402
from fcv.controls import prepare_control_plan  # noqa: E402
from fcv.fcv_scoring import (  # noqa: E402
    load_background_bank,
    prepare_opposite_donor_plan,
)
from fcv.storage import assert_storage_budget  # noqa: E402
from fcv.token_banks import (  # noqa: E402
    CONTEXT_NAMES,
    TokenBankError,
    build_background_token_banks,
    candidate_checkpoints_for_run,
    prepare_token_bank_source,
)
from fcv.vit_counterfactual_forward import validate_reconstruction_gate  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-run-index", type=int, default=0)
    parser.add_argument("--reference-epoch", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_and_validate_config(args.config)
    validate_runtime_software(config)
    if args.reference_epoch not in candidate_epochs(config):
        raise ValueError(
            f"reference_epoch must be in {candidate_epochs(config)}, got "
            f"{args.reference_epoch}."
        )
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    candidate_root = output_root / config["outputs"]["candidate_models"]
    manifest = output_root / config["outputs"]["split_manifests"] / "metadata_val.csv"
    patch_masks = output_root / config["outputs"]["patch_masks"] / "patch_masks_val.pt"
    bank_dir = output_root / config["outputs"]["token_banks"]
    fcv_dir = output_root / config["outputs"]["fcv_scores"]
    control_dir = output_root / config["outputs"]["control_scores"]
    pool_csv = candidate_root / "candidate_metrics_biased_val.csv"
    pool_summary = candidate_root / "candidate_pool_summary.json"
    aggregate_candidate_metrics(
        config, candidate_root, pool_csv, pool_summary, allow_incomplete=False
    )
    assert_storage_budget(config, output_root, stage="prepare_streaming_plans")
    source = prepare_token_bank_source(config, manifest, patch_masks)
    reconstruction_reports = validate_reconstruction_gate(config)
    checkpoints = candidate_checkpoints_for_run(
        config, candidate_root, args.reference_run_index
    )
    reference_id = get_sweep_run(config, args.reference_run_index).candidate_id(
        args.reference_epoch
    )
    reference_checkpoint = next(
        path
        for path in checkpoints
        if path.name == f"epoch_{args.reference_epoch:03d}.pt"
    )
    try:
        bank_summary = build_background_token_banks(
            config,
            reference_checkpoint,
            source,
            bank_dir,
            reconstruction_reports=reconstruction_reports,
            device=args.device,
            overwrite=False,
        )
    except TokenBankError:
        receipt_path = bank_dir / "cleanup_receipts" / f"{reference_id}.json"
        if not receipt_path.is_file():
            raise
        bank_summary = build_background_token_banks(
            config,
            reference_checkpoint,
            source,
            bank_dir,
            reconstruction_reports=reconstruction_reports,
            device=args.device,
            overwrite=True,
        )
    banks = {
        label: load_background_bank(
            config,
            bank_dir / f"{reference_id}_{context_name}.pt",
            source,
            expected_label=label,
            expected_candidate_id=reference_id,
        )
        for label, context_name in CONTEXT_NAMES.items()
    }
    donor_plan_path = fcv_dir / "opposite_donor_plan.pt"
    donor_plan = prepare_opposite_donor_plan(
        config,
        source,
        banks,
        donor_plan_path,
        overwrite=donor_plan_path.exists(),
    )
    control_plan_path = control_dir / "control_plan.pt"
    control_plan = prepare_control_plan(
        config,
        source,
        banks,
        donor_plan,
        donor_plan_path,
        control_plan_path,
        overwrite=control_plan_path.exists(),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "reference_candidate_id": reference_id,
                "reference_checkpoint": str(reference_checkpoint.resolve()),
                "reference_bank_status": bank_summary["status"],
                "opposite_donor_plan": str(donor_plan_path.resolve()),
                "opposite_donor_plan_content_sha256": donor_plan[
                    "plan_content_sha256"
                ],
                "control_plan": str(control_plan_path.resolve()),
                "control_plan_content_sha256": control_plan[
                    "plan_content_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
