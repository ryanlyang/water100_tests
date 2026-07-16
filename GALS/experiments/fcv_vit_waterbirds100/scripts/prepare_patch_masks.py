#!/usr/bin/env python3
"""Create ViT patch-level evidence/background masks for FCV validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.patch_masks import prepare_patch_masks  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert R4RR teacher maps for the fixed train-derived validation holdout "
            "into 14x14 ViT patch partitions."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override <output_root>/split_manifests/metadata_val.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override <output_root>/patch_masks.",
    )
    parser.add_argument(
        "--require-all-eligible",
        action="store_true",
        help="Fail if any image has no evidence or fewer than the configured safe backgrounds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Step 3 patch-mask artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    manifest = args.manifest
    if manifest is None:
        manifest = (
            output_root
            / config["outputs"]["split_manifests"]
            / "metadata_val.csv"
        )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = output_root / config["outputs"]["patch_masks"]

    result = prepare_patch_masks(
        config,
        manifest,
        output_dir,
        overwrite=args.overwrite,
        require_all_eligible=args.require_all_eligible,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
