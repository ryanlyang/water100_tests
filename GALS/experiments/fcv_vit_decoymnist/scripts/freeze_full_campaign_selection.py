#!/usr/bin/env python3
"""Freeze Vanilla/FCV/Oracle selections without loading test values."""

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
from decoy_selection_analysis import freeze_selector_matrix  # noqa: E402


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
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    summary = freeze_selector_matrix(
        config, args.output_root or config["paths"]["output_root"]
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
