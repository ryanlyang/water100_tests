#!/usr/bin/env python3
"""Recompute and validate a paused real online-run prefix on Tigris."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.online_analysis import validate_online_run_prefix  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--require-resumed-from-epoch", type=int, default=None)
    parser.add_argument("--storage-baseline-path", type=Path, default=None)
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    result = validate_online_run_prefix(
        config,
        config["paths"]["output_root"],
        run_index=args.run_index,
        expected_epoch=args.expected_epoch,
        require_resumed_from_epoch=args.require_resumed_from_epoch,
        storage_baseline_path=args.storage_baseline_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
