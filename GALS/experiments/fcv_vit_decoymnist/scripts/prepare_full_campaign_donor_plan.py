#!/usr/bin/env python3
"""Prepare the frozen Step-6 same-corner multiclass donor plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_donor_plans import prepare_donor_plan  # noqa: E402
from decoy_full_config import load_and_validate_config  # noqa: E402


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
    parser.add_argument("--manifest")
    parser.add_argument("--mask-artifact")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    manifest = Path(args.manifest) if args.manifest else (
        Path(config["paths"]["split_manifest_dir"]) / "metadata_val.csv"
    )
    mask_artifact = Path(args.mask_artifact) if args.mask_artifact else (
        output_root / "teacher_mask_audit" / "projected_teacher_masks.npz"
    )
    output = Path(args.output) if args.output else (
        output_root / "donor_plans" / "multiclass_same_corner_donors.json"
    )
    payload = prepare_donor_plan(
        config,
        manifest,
        mask_artifact,
        output,
        overwrite=bool(args.overwrite),
    )
    summary = {
        "output": str(output.expanduser().resolve()),
        "target_count": int(payload["target_count"]),
        "donors_per_target": int(payload["donors_per_target"]),
        "plan_seed": int(payload["plan_seed"]),
        "plan_content_sha256": str(payload["plan_content_sha256"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
