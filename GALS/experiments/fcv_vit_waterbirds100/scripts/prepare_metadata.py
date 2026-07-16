#!/usr/bin/env python3
"""Create deterministic Waterbirds100 manifests for the ViT-FCV first study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.waterbirds_metadata import prepare_waterbirds100_manifests  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the fully biased Waterbirds100 training data into fixed 80/20 "
            "candidate-training and FCV-validation manifests."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override <config output_root>/split_manifests.",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Do not verify every image path. Intended only for synthetic tests.",
    )
    parser.add_argument(
        "--allow-missing-teacher-maps",
        action="store_true",
        help="Generate manifests despite missing held-out maps for diagnosis only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previously generated manifests for this locked split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(config["paths"]["output_root"]) / config["outputs"][
            "split_manifests"
        ]
    result = prepare_waterbirds100_manifests(
        config,
        output_dir,
        check_images=not args.skip_image_check,
        require_holdout_teacher_maps=not args.allow_missing_teacher_maps,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
