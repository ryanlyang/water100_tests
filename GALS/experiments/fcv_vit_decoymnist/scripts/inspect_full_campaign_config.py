#!/usr/bin/env python3
"""Validate and summarize the frozen full DecoyMNIST FCV configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_full_config import (  # noqa: E402
    candidate_epochs,
    enumerate_runs,
    load_and_validate_config,
)


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
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    runs = enumerate_runs(config)
    epochs = candidate_epochs(config)
    print(
        json.dumps(
            {
                "status": "PASS",
                "study_id": config["study"]["id"],
                "config_provenance": config["_provenance"],
                "training_runs": len(runs),
                "candidate_epochs": epochs,
                "candidate_states": len(runs) * len(epochs),
                "first_run_id": runs[0].run_id,
                "last_run_id": runs[-1].run_id,
                "checkpoint_persistence": config["candidate_pool"][
                    "persist_model_checkpoints"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
