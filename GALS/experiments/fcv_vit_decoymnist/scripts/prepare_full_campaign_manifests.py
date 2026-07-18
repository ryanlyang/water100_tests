#!/usr/bin/env python3
"""Audit DecoyMNIST and create the frozen full-campaign split manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_full_config import load_and_validate_config  # noqa: E402
from decoy_manifest_provenance import validate_manifest_bundle  # noqa: E402
from decoy_manifests import prepare_decoymnist_manifests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            EXPERIMENT_ROOT
            / "configs"
            / "decoymnist_vit_s16_fcv_full_online.yaml"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_and_validate_config(args.config)
    output_dir = args.output_dir or config["paths"]["split_manifest_dir"]
    result = prepare_decoymnist_manifests(
        config,
        output_dir,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    bindings = {}
    for role in ("candidate_train", "biased_validation", "oracle_validation", "test"):
        binding = validate_manifest_bundle(config, result["artifacts"][role], role)
        bindings[role] = {
            "manifest_sha256": binding.manifest_sha256,
            "bundle_sha256": binding.bundle_sha256,
        }
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": result["output_dir"],
                "splits": result["summary"]["splits"],
                "bindings": bindings,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
