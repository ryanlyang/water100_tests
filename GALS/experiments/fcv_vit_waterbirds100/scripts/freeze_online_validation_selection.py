#!/usr/bin/env python3
"""Freeze 540-candidate validation selectors without opening test artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.candidate_training import validate_runtime_software  # noqa: E402
from fcv.config import load_and_validate_config  # noqa: E402
from fcv.online_analysis import freeze_online_validation_selection  # noqa: E402


DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    validate_runtime_software(config)
    result = freeze_online_validation_selection(config, config["paths"]["output_root"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

