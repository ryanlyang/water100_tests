#!/usr/bin/env python3
"""Strictly validate the 81 candidates, then remove completed resume states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.cleanup import prune_completed_resume_states  # noqa: E402
from fcv.config import load_and_validate_config  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    candidate_root = args.candidate_root or (
        output_root / config["outputs"]["candidate_models"]
    )
    receipt = args.receipt or (
        Path(candidate_root) / "resume_state_cleanup_receipt.json"
    )
    result = prune_completed_resume_states(config, candidate_root, receipt)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "deleted_resume_state_count": len(
                    result["deleted_resume_states"]
                ),
                "receipt": str(Path(receipt).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
