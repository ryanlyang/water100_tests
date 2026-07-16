#!/usr/bin/env python3
"""Validate and summarize the locked ViT-FCV first-study configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import config_summary, load_and_validate_config  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the ViT-FCV first-study configuration."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Also verify Tigris dataset, metadata, map, environment, and output paths.",
    )
    return parser.parse_args()


def check_paths(config: dict) -> dict:
    paths = config["paths"]
    cluster = config["cluster"]
    checks = {
        "repository_root": Path(paths["repository_root"]).is_dir(),
        "data_root": Path(paths["data_root"]).is_dir(),
        "metadata": Path(paths["metadata"]).is_file(),
        "teacher_map_root": Path(paths["teacher_map_root"]).is_dir(),
        "conda_environment": Path(cluster["conda_environment"]).is_dir(),
        "python": Path(cluster["python"]).is_file(),
        "output_parent": Path(paths["output_root"]).parent.is_dir(),
    }
    failed = [name for name, exists in checks.items() if not exists]
    if failed:
        raise FileNotFoundError(
            "Missing required Tigris paths: " + ", ".join(sorted(failed))
        )
    return checks


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    result = {"config": str(args.config.resolve()), "summary": config_summary(config)}
    if args.check_paths:
        result["path_checks"] = check_paths(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
