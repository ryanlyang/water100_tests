#!/usr/bin/env python3
"""Print the frozen Step-5 candidate grid and transform invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_candidate_training import (  # noqa: E402
    evaluation_transform_spec,
    training_transform_spec,
)
from decoy_full_config import (  # noqa: E402
    candidate_epochs,
    enumerate_runs,
    load_and_validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            EXPERIMENT_ROOT
            / "configs"
            / "decoymnist_vit_s16_fcv_full_online.yaml"
        ),
    )
    parser.add_argument("--run-index", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    runs = enumerate_runs(config)
    selected = runs if args.run_index is None else [runs[int(args.run_index)]]
    payload = {
        "training_run_count": len(runs),
        "candidate_epoch_count": len(candidate_epochs(config)),
        "online_candidate_count": len(runs) * len(candidate_epochs(config)),
        "evaluation_transform": evaluation_transform_spec(config),
        "runs": [
            {
                "run_id": run.run_id,
                "run_index": run.run_index,
                "learning_rate": run.learning_rate,
                "weight_decay": run.weight_decay,
                "crop_scale_min": run.crop_scale_min,
                "seed": run.seed,
                "training_transform": training_transform_spec(config, run),
            }
            for run in selected
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
