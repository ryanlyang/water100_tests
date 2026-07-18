#!/usr/bin/env python3
"""Resolve and project full-campaign DecoyMNIST teacher maps."""

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
from decoy_teacher_masks import prepare_teacher_masks  # noqa: E402


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
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    manifest = args.manifest or str(
        Path(config["paths"]["split_manifest_dir"]) / "metadata_val.csv"
    )
    output_dir = args.output_dir or str(
        Path(config["paths"]["output_root"]) / "teacher_mask_audit"
    )
    result = prepare_teacher_masks(
        config, manifest, output_dir, overwrite=args.overwrite
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
