#!/usr/bin/env python3
"""Build, score, validate, and prune token banks for one sweep run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.candidate_training import validate_runtime_software  # noqa: E402
from fcv.config import load_and_validate_config  # noqa: E402
from fcv.streaming import stream_score_run  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patch-masks", type=Path, default=None)
    parser.add_argument("--token-bank-dir", type=Path, default=None)
    parser.add_argument("--donor-plan", type=Path, default=None)
    parser.add_argument("--control-plan", type=Path, default=None)
    parser.add_argument("--fcv-score-dir", type=Path, default=None)
    parser.add_argument("--control-score-dir", type=Path, default=None)
    parser.add_argument("--storage-root", type=Path, default=None)
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    validate_runtime_software(config)
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    result = stream_score_run(
        config,
        run_index=args.run_index,
        candidate_root=args.candidate_root
        or output_root / config["outputs"]["candidate_models"],
        manifest=args.manifest
        or output_root
        / config["outputs"]["split_manifests"]
        / "metadata_val.csv",
        patch_masks=args.patch_masks
        or output_root
        / config["outputs"]["patch_masks"]
        / "patch_masks_val.pt",
        token_bank_dir=args.token_bank_dir
        or output_root / config["outputs"]["token_banks"],
        donor_plan=args.donor_plan
        or output_root
        / config["outputs"]["fcv_scores"]
        / "opposite_donor_plan.pt",
        control_plan=args.control_plan
        or output_root
        / config["outputs"]["control_scores"]
        / "control_plan.pt",
        fcv_score_dir=args.fcv_score_dir
        or output_root / config["outputs"]["fcv_scores"],
        control_score_dir=args.control_score_dir
        or output_root / config["outputs"]["control_scores"],
        output_root=args.storage_root or output_root,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
