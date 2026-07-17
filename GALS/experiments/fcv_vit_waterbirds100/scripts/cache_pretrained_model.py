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
    pretrained_provenance_path,
    software_fingerprint,
    source_tree_provenance,
    validate_runtime_software,
)
from fcv.campaign_provenance import (  # noqa: E402
    create_campaign_provenance_receipt,
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
    output_summary = args.output_summary or pretrained_provenance_path(config)
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
    campaign = create_campaign_provenance_receipt(
        config,
        pretrained_path=output_summary,
        verify_all_image_bytes=True,
    )
    summary["output_summary"] = str(output_summary)
    summary["campaign_provenance_path"] = campaign["artifact_path"]
    summary["campaign_provenance_sha256"] = campaign["artifact_sha256"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
