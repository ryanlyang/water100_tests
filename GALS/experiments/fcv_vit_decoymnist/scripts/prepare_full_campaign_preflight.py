#!/usr/bin/env python3
"""Prepare frozen inputs and issue the GH200 production preflight receipt."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_campaign_preflight import (  # noqa: E402
    manifest_paths,
    pretrained_backbone_sha256,
    runtime_versions,
    source_tree_provenance,
    validate_runtime,
    verify_all_manifest_image_bytes,
)
from decoy_candidate_training import (  # noqa: E402
    ManifestImageDataset,
    build_evaluation_transform,
    build_model,
)
from decoy_donor_plans import (  # noqa: E402
    load_and_validate_donor_plan,
    prepare_donor_plan,
)
from decoy_full_config import (  # noqa: E402
    canonical_config_sha256,
    load_and_validate_config,
    sha256_file,
)
from decoy_manifest_provenance import atomic_json, validate_manifest_bundle  # noqa: E402
from decoy_manifests import prepare_decoymnist_manifests  # noqa: E402
from decoy_teacher_masks import (  # noqa: E402
    load_projected_teacher_masks,
    prepare_teacher_masks,
)
from decoy_vit_intervention import RECONSTRUCTION_TOLERANCE, verify_identity_forward  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "decoymnist_vit_s16_fcv_full_online.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    import torch

    started = time.time()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    preflight_root = output_root / "preflight"
    preflight_root.mkdir(parents=True, exist_ok=True)
    paths = manifest_paths(config)

    if not (Path(config["paths"]["split_manifest_dir"]) / "manifest_bundle.json").is_file():
        prepare_decoymnist_manifests(
            config,
            config["paths"]["split_manifest_dir"],
            workers=int(config["training"]["num_workers"]),
            overwrite=False,
        )
    bindings = {
        role: validate_manifest_bundle(config, path, role)
        for role, path in paths.items()
    }
    image_audit = verify_all_manifest_image_bytes(config, paths)

    mask_path = output_root / "teacher_mask_audit" / "projected_teacher_masks.npz"
    if not mask_path.is_file():
        prepare_teacher_masks(
            config,
            paths["biased_validation"],
            mask_path.parent,
            overwrite=False,
        )
    masks, mask_binding = load_projected_teacher_masks(
        config, paths["biased_validation"], mask_path
    )

    donor_path = output_root / "donor_plans" / "multiclass_same_corner_donors.json"
    if not donor_path.is_file():
        prepare_donor_plan(
            config,
            paths["biased_validation"],
            mask_path,
            donor_path,
            overwrite=False,
        )
    donor = load_and_validate_donor_plan(
        config, paths["biased_validation"], mask_path, donor_path
    )
    with tempfile.TemporaryDirectory(prefix="decoy_fcv_donor_regen_") as temporary:
        regenerated_path = Path(temporary) / "donors.json"
        regenerated = prepare_donor_plan(
            config,
            paths["biased_validation"],
            mask_path,
            regenerated_path,
            overwrite=True,
        )
        if regenerated["plan_content_sha256"] != donor["plan_content_sha256"]:
            raise RuntimeError("Deterministic donor-plan regeneration changed its hash.")

    observed_runtime = validate_runtime(config)
    if not torch.cuda.is_available():
        raise RuntimeError("The production preflight requires a GH200 GPU.")
    device_name = torch.cuda.get_device_name(0)
    if "GH200" not in device_name.upper():
        raise RuntimeError(f"Expected an NVIDIA GH200, found {device_name!r}.")

    model = build_model(config, pretrained=True).cuda().eval()
    transform = build_evaluation_transform(config)
    validation = ManifestImageDataset(
        config, paths["biased_validation"], "biased_validation", transform
    )
    images = torch.stack([validation[index][0] for index in range(2)]).cuda()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(images)
    if tuple(logits.shape) != (2, int(config["data"]["num_classes"])):
        raise RuntimeError(f"Locked model produced the wrong output shape: {logits.shape}")
    identity = verify_identity_forward(
        model, images, tolerance=RECONSTRUCTION_TOLERANCE
    )
    backbone_hash = pretrained_backbone_sha256(model)
    del images, logits, model
    torch.cuda.empty_cache()

    source = source_tree_provenance()
    pretrained_path = preflight_root / "pretrained_initialization.json"
    pretrained = {
        "artifact_type": "fcv_vit_decoymnist_pretrained_initialization",
        "artifact_version": 1,
        "status": "cached_and_validated",
        "config_sha256": canonical_config_sha256(config),
        "source_tree_sha256": source["source_tree_sha256"],
        "model_name": config["model"]["name"],
        "runtime_versions": observed_runtime,
        "pretrained_backbone_sha256": backbone_hash,
    }
    atomic_json(pretrained, pretrained_path)

    receipt = {
        "artifact_type": "fcv_vit_decoymnist_preflight_receipt",
        "artifact_version": 1,
        "status": "PASS",
        "config_sha256": canonical_config_sha256(config),
        "source_tree_sha256": source["source_tree_sha256"],
        "source_file_count": source["source_file_count"],
        "runtime_versions": runtime_versions(),
        "device_name": device_name,
        "manifest_bundle_path": str(bindings["candidate_train"].bundle_path),
        "manifest_bundle_sha256": bindings["candidate_train"].bundle_sha256,
        "manifest_role_sha256": {
            role: binding.manifest_sha256 for role, binding in bindings.items()
        },
        "image_byte_audit": image_audit,
        "projected_teacher_masks_path": str(mask_path),
        "projected_teacher_masks_sha256": sha256_file(mask_path),
        "eligible_target_count": int(masks["fcv_eligible"].sum()),
        "donor_plan_path": str(donor_path),
        "donor_plan_sha256": sha256_file(donor_path),
        "donor_plan_content_sha256": donor["plan_content_sha256"],
        "donor_plan_regeneration_matched": True,
        "pretrained_initialization_path": str(pretrained_path),
        "pretrained_initialization_sha256": sha256_file(pretrained_path),
        "pretrained_backbone_sha256": backbone_hash,
        "model_output_shape": [2, int(config["data"]["num_classes"])],
        "identity_forward": identity,
        "seconds": float(time.time() - started),
    }
    receipt_path = preflight_root / "preflight_receipt.json"
    atomic_json(receipt, receipt_path)
    print(json.dumps({**receipt, "artifact_path": str(receipt_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
