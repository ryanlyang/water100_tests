#!/usr/bin/env python3
"""Verify exact logits reconstruction from raw ViT patch embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.candidate_training import (  # noqa: E402
    build_model,
    candidate_training_fingerprint,
    validate_runtime_software,
)
from fcv.config import load_and_validate_config  # noqa: E402
from fcv.vit_counterfactual_forward import (  # noqa: E402
    load_candidate_model,
    verify_reconstructed_forward,
)


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check normal ViT logits against the resumed raw-patch forward."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional Step 4 candidate checkpoint. Uses a fresh model otherwise.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load cached pretrained weights when --checkpoint is not supplied.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.checkpoint is not None and args.pretrained:
        raise ValueError("--pretrained cannot be combined with --checkpoint.")
    config = load_and_validate_config(args.config)
    validate_runtime_software(config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if args.checkpoint is None:
        model = build_model(config, pretrained=bool(args.pretrained))
        model.to(device)
        model.eval()
        source = "pretrained" if args.pretrained else "random_initialization"
        candidate_id = None
        checkpoint_path = None
        checkpoint_sha256 = None
    else:
        model, artifact = load_candidate_model(config, args.checkpoint, device=device)
        source = str(Path(args.checkpoint).expanduser().resolve())
        candidate_id = artifact.get("candidate_id")
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    image_size = int(config["model"]["image_size"])
    images = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
    report = verify_reconstructed_forward(
        model,
        images,
        tolerance=args.tolerance,
    )
    output = report.to_dict()
    output.update(
        {
            "schema_version": 1,
            "artifact_type": "fcv_vit_reconstruction_report",
            "candidate_id": candidate_id,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": checkpoint_sha256,
            "device": str(device),
            "model": config["model"]["name"],
            "training_fingerprint": candidate_training_fingerprint(config),
            "source": source,
            "status": "passed" if report.passed else "failed",
        }
    )
    args.output_report = args.output_report.expanduser().resolve()
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output_report)
    print(json.dumps(output, indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
