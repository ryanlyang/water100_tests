#!/usr/bin/env python3
"""Run one no-checkpoint DecoyMNIST FCV array task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_full_config import load_and_validate_config  # noqa: E402
from decoy_campaign_preflight import (  # noqa: E402
    validate_launch_gate,
    validate_preflight_receipt,
)
from decoy_online_study import run_online_study  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            EXPERIMENT_ROOT
            / "configs"
            / "decoymnist_vit_s16_fcv_full_online.yaml"
        ),
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")),
    )
    parser.add_argument("--train-manifest")
    parser.add_argument("--validation-manifest")
    parser.add_argument("--oracle-manifest")
    parser.add_argument("--test-manifest")
    parser.add_argument("--mask-artifact")
    parser.add_argument("--donor-plan")
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--restart-partial", action="store_true")
    parser.add_argument("--smoke-one-epoch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_index < 0:
        raise ValueError("Provide --run-index or launch as a Slurm array task.")
    config = load_and_validate_config(args.config)
    campaign_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    preflight_receipt = campaign_root / "preflight" / "preflight_receipt.json"
    if args.smoke_one_epoch:
        gate = validate_preflight_receipt(config, preflight_receipt)
        default_output = campaign_root / "preflight" / "online_smoke_workspace"
    else:
        validate_launch_gate(config, campaign_root / "preflight" / "launch_gate.json")
        gate = validate_preflight_receipt(config, preflight_receipt)
        default_output = campaign_root
    output_root = Path(args.output_root or default_output)
    manifest_root = Path(config["paths"]["split_manifest_dir"])
    result = run_online_study(
        config,
        args.run_index,
        train_manifest=args.train_manifest or manifest_root / "metadata_train.csv",
        validation_manifest=args.validation_manifest or manifest_root / "metadata_val.csv",
        oracle_manifest=args.oracle_manifest
        or manifest_root / "metadata_oracle_source_analysis_only.csv",
        test_manifest=args.test_manifest
        or manifest_root / "metadata_test_analysis_only.csv",
        mask_artifact=args.mask_artifact
        or campaign_root / "teacher_mask_audit" / "projected_teacher_masks.npz",
        donor_plan_path=args.donor_plan
        or campaign_root / "donor_plans" / "multiclass_same_corner_donors.json",
        output_root=output_root,
        device_name=args.device,
        restart_partial=bool(args.restart_partial),
        epoch_limit=1 if args.smoke_one_epoch else None,
        preflight_receipt_sha256=str(gate["artifact_sha256"]),
        expected_pretrained_backbone_sha256=str(
            gate["pretrained_backbone_sha256"]
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
