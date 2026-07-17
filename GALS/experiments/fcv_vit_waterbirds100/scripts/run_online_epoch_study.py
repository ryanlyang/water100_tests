#!/usr/bin/env python3
"""Train and online-score one LR/WD/seed run across all 20 epochs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.candidate_training import (  # noqa: E402
    get_sweep_run,
    pretrained_provenance_path,
    validate_runtime_software,
)
from fcv.config import load_and_validate_config  # noqa: E402
from fcv.online_study import train_and_score_online_run  # noqa: E402


DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    validate_runtime_software(config)
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    manifests = output_root / config["outputs"]["split_manifests"]
    result = train_and_score_online_run(
        config,
        get_sweep_run(config, args.run_index),
        manifests / "metadata_train.csv",
        manifests / "metadata_val.csv",
        output_root / config["outputs"]["patch_masks"] / "patch_masks_val.pt",
        manifests / "metadata_oracle_val_analysis_only.csv",
        manifests / "metadata_test_analysis_only.csv",
        output_root,
        device_name=args.device,
        pretrained_provenance_path=pretrained_provenance_path(config),
        stop_after_epoch=args.stop_after_epoch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
