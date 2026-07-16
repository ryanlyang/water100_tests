#!/usr/bin/env python3
"""Validate and cache the locked timm pretrained ViT before GPU jobs start."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.candidate_training import (  # noqa: E402
    build_model,
    pretrained_backbone_sha256,
    software_fingerprint,
    source_tree_provenance,
    validate_runtime_software,
)
from fcv.config import load_and_validate_config  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/cache and validate the locked pretrained ViT-S/16."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-summary", type=Path, default=None)
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    observed_versions = validate_runtime_software(config)
    model = build_model(config, pretrained=True)
    output_root = Path(config["paths"]["output_root"])
    output_summary = args.output_summary or (
        output_root / "preflight" / "pretrained_model_summary.json"
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_pretrained_initialization",
        "model": config["model"]["name"],
        "model_config": dict(config["model"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "patch_count": int(model.patch_embed.num_patches),
        "runtime_versions": observed_versions,
        "software_fingerprint": software_fingerprint(observed_versions),
        "source_tree_sha256": source_tree_provenance()["source_tree_sha256"],
        "pretrained_backbone_sha256": pretrained_backbone_sha256(model),
        "status": "cached_and_validated",
    }
    output_summary = Path(output_summary).expanduser().resolve()
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_summary.with_suffix(output_summary.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output_summary)
    summary["output_summary"] = str(output_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
