#!/usr/bin/env python3
"""Validate the complete fail-closed Step 3 artifact and current map bytes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.token_banks import prepare_token_bank_source  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patch-masks", type=Path, default=None)
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    manifest = args.manifest or (
        output_root / config["outputs"]["split_manifests"] / "metadata_val.csv"
    )
    patch_masks = args.patch_masks or (
        output_root / config["outputs"]["patch_masks"] / "patch_masks_val.pt"
    )
    source = prepare_token_bank_source(
        config,
        manifest,
        patch_masks,
        check_images=False,
    )
    print(
        json.dumps(
            {
                "status": "complete_and_current",
                "sample_count": source.sample_count,
                "patch_mask_sha256": source.patch_mask_sha256,
                "patch_mask_summary_sha256": source.patch_mask_summary_sha256,
                "preprocessing_sha256": source.patch_mask_preprocessing_sha256,
                "teacher_maps_sha256": source.teacher_maps_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
