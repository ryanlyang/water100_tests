#!/usr/bin/env python3
"""Train one resumable vanilla ViT candidate-pool run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.candidate_training import (  # noqa: E402
    get_sweep_run,
    train_candidate_run,
    validate_runtime_software,
)
from fcv.config import load_and_validate_config  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one LR/weight-decay/seed run from the locked ViT pool."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Override <output_root>/split_manifests/metadata_train.csv.",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=None,
        help="Override <output_root>/split_manifests/metadata_val.csv.",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=None,
        help="Override <output_root>/candidate_models.",
    )
    parser.add_argument(
        "--pretrained-summary",
        type=Path,
        default=None,
        help="Override the required Step 4 pretrained provenance artifact.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved run and paths without importing timm or training.",
    )
    parser.add_argument(
        "--smoke-stop-after-epoch",
        type=int,
        default=None,
        help=(
            "Deliberately return after this completed epoch so the GH200 smoke "
            "workflow can exercise exact resume. Requires --candidate-root."
        ),
    )
    parser.add_argument(
        "--smoke-interrupt-after-resume-commit-epoch",
        type=int,
        default=None,
        help=(
            "Failure injection after resume-state commit but before metrics.csv "
            "commit. Requires --candidate-root."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    manifest_root = output_root / config["outputs"]["split_manifests"]
    train_manifest = args.train_manifest or manifest_root / "metadata_train.csv"
    validation_manifest = args.validation_manifest or manifest_root / "metadata_val.csv"
    candidate_root = args.candidate_root or output_root / config["outputs"][
        "candidate_models"
    ]
    pretrained_summary = args.pretrained_summary or (
        output_root / "preflight" / "pretrained_model_summary.json"
    )
    run = get_sweep_run(config, args.run_index)
    resolved = {
        "run": {
            "run_index": run.run_index,
            "run_id": run.run_id,
            "learning_rate": run.learning_rate,
            "weight_decay": run.weight_decay,
            "seed": run.seed,
        },
        "train_manifest": str(Path(train_manifest).resolve()),
        "validation_manifest": str(Path(validation_manifest).resolve()),
        "candidate_root": str(Path(candidate_root).resolve()),
        "pretrained_summary": str(Path(pretrained_summary).resolve()),
        "device": args.device,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    if (
        args.smoke_stop_after_epoch is not None
        or args.smoke_interrupt_after_resume_commit_epoch is not None
    ) and args.candidate_root is None:
        raise ValueError(
            "Smoke interruption options require an explicit smoke candidate root."
        )

    validate_runtime_software(config)
    print(json.dumps(resolved, indent=2, sort_keys=True), flush=True)
    result = train_candidate_run(
        config,
        run,
        train_manifest,
        validation_manifest,
        candidate_root,
        device_name=args.device,
        stop_after_epoch=args.smoke_stop_after_epoch,
        simulate_interruption_after_resume_epoch=(
            args.smoke_interrupt_after_resume_commit_epoch
        ),
        pretrained_provenance_path=pretrained_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
