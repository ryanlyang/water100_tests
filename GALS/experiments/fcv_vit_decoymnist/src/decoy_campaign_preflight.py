"""Campaign-bound provenance and launch gates for the DecoyMNIST FCV study."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from decoy_full_config import canonical_config_sha256, sha256_file
from decoy_manifest_provenance import MANIFEST_SPECS, validate_manifest_bundle


class CampaignPreflightError(RuntimeError):
    """Raised when production inputs or launch receipts are stale."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_tree_provenance() -> Dict[str, Any]:
    """Hash every campaign source and launch file in deterministic order."""

    experiment_root = Path(__file__).resolve().parents[1]
    paths = sorted(
        [
            *experiment_root.glob("src/**/*.py"),
            *experiment_root.glob("scripts/*.py"),
            *experiment_root.glob("scripts/*.sh"),
            *experiment_root.glob("slurm/*.sbatch"),
        ],
        key=lambda path: path.relative_to(experiment_root).as_posix(),
    )
    entries = [
        {
            "path": path.relative_to(experiment_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
        if "__pycache__" not in path.parts
    ]
    return {
        "source_tree_sha256": _canonical_sha256({"files": entries}),
        "source_file_count": len(entries),
        "files": entries,
    }


def runtime_versions() -> Dict[str, str]:
    import timm
    import torch
    import torchvision

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "torchvision": str(torchvision.__version__),
        "timm": str(timm.__version__),
    }


def validate_runtime(config: Mapping[str, Any]) -> Dict[str, str]:
    observed = runtime_versions()
    expected = {
        "torch": str(config["cluster"]["torch_version"]),
        "torchvision": str(config["cluster"]["torchvision_version"]),
        "timm": str(config["cluster"]["timm_version"]),
    }
    mismatches = {
        key: {"expected": expected_value, "observed": observed.get(key)}
        for key, expected_value in expected.items()
        if observed.get(key) != expected_value
    }
    if mismatches:
        raise CampaignPreflightError(f"Locked runtime mismatch: {mismatches}")
    return observed


def _classifier_state_keys(model: Any) -> set[str]:
    classifier = model.get_classifier()
    parameter_ids = {id(value) for value in classifier.parameters()}
    buffer_ids = {id(value) for value in classifier.buffers()}
    keys = {
        name
        for name, value in model.named_parameters()
        if id(value) in parameter_ids
    }
    keys.update(
        name for name, value in model.named_buffers() if id(value) in buffer_ids
    )
    if not keys:
        raise CampaignPreflightError("Could not identify the task classifier state.")
    return keys


def pretrained_backbone_sha256(model: Any) -> str:
    """Hash pretrained parameters while excluding the random ten-class head."""

    import torch

    excluded = _classifier_state_keys(model)
    digest = hashlib.sha256()
    for key in sorted(model.state_dict()):
        if key in excluded:
            continue
        tensor = model.state_dict()[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_all_manifest_image_bytes(
    config: Mapping[str, Any], manifest_paths: Mapping[str, Path]
) -> Dict[str, Any]:
    """Rehash all 70k source PNGs against the authenticated manifests."""

    data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
    count = 0
    digest = hashlib.sha256()
    for role in ("candidate_train", "biased_validation", "oracle_validation", "test"):
        path = Path(manifest_paths[role]).expanduser().resolve()
        validate_manifest_bundle(config, path, role)
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            source = (data_root / str(row.image_rel_path)).resolve()
            if not source.is_relative_to(data_root):
                raise CampaignPreflightError(
                    f"Manifest image escapes the data root: {row.image_rel_path}"
                )
            observed = sha256_file(source)
            if observed != str(row.image_sha256):
                raise CampaignPreflightError(
                    f"Source image changed after manifest creation: {row.sample_id}"
                )
            digest.update(str(row.sample_id).encode("utf-8"))
            digest.update(observed.encode("ascii"))
            count += 1
    expected = sum(int(value) for value in config["data"]["source_counts"].values())
    if count != expected:
        raise CampaignPreflightError(
            f"Manifest image-byte audit found {count} images; expected {expected}."
        )
    return {"verified_image_count": count, "verified_image_set_sha256": digest.hexdigest()}


def manifest_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    root = Path(config["paths"]["split_manifest_dir"]).expanduser().resolve()
    return {
        role: root / spec["filename"] for role, spec in MANIFEST_SPECS.items()
    }


def load_json_mapping(path: str | Path) -> Dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise CampaignPreflightError(f"Missing campaign receipt: {candidate}")
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignPreflightError(f"Expected a JSON mapping: {candidate}")
    return value


def validate_preflight_receipt(
    config: Mapping[str, Any], receipt_path: str | Path
) -> Dict[str, Any]:
    path = Path(receipt_path).expanduser().resolve()
    payload = load_json_mapping(path)
    source = source_tree_provenance()
    referenced = (
        ("manifest_bundle_path", "manifest_bundle_sha256"),
        ("projected_teacher_masks_path", "projected_teacher_masks_sha256"),
        ("donor_plan_path", "donor_plan_sha256"),
        ("pretrained_initialization_path", "pretrained_initialization_sha256"),
    )
    references_valid = True
    for path_key, hash_key in referenced:
        artifact_path = Path(str(payload.get(path_key, ""))).expanduser()
        if (
            not str(payload.get(path_key, ""))
            or not artifact_path.is_file()
            or sha256_file(artifact_path) != payload.get(hash_key)
        ):
            references_valid = False
    valid = (
        payload.get("artifact_type") == "fcv_vit_decoymnist_preflight_receipt"
        and payload.get("artifact_version") == 1
        and payload.get("status") == "PASS"
        and payload.get("config_sha256") == canonical_config_sha256(config)
        and payload.get("source_tree_sha256") == source["source_tree_sha256"]
        and payload.get("runtime_versions") == runtime_versions()
        and isinstance(payload.get("pretrained_backbone_sha256"), str)
        and len(payload["pretrained_backbone_sha256"]) == 64
        and references_valid
    )
    if not valid:
        raise CampaignPreflightError(f"Preflight receipt is stale: {path}")
    result = dict(payload)
    result["artifact_path"] = str(path)
    result["artifact_sha256"] = sha256_file(path)
    return result


def validate_launch_gate(
    config: Mapping[str, Any], gate_path: str | Path
) -> Dict[str, Any]:
    path = Path(gate_path).expanduser().resolve()
    payload = load_json_mapping(path)
    preflight_path = Path(str(payload.get("preflight_receipt_path", "")))
    preflight = validate_preflight_receipt(config, preflight_path)
    source = source_tree_provenance()
    smoke_path = Path(str(payload.get("smoke_summary_path", ""))).expanduser()
    smoke_valid = (
        bool(str(payload.get("smoke_summary_path", "")))
        and smoke_path.is_file()
        and sha256_file(smoke_path) == payload.get("smoke_summary_sha256")
    )
    valid = (
        payload.get("artifact_type") == "fcv_vit_decoymnist_launch_gate"
        and payload.get("artifact_version") == 1
        and payload.get("status") == "PASS"
        and payload.get("config_sha256") == canonical_config_sha256(config)
        and payload.get("source_tree_sha256") == source["source_tree_sha256"]
        and payload.get("preflight_receipt_sha256") == preflight["artifact_sha256"]
        and payload.get("pretrained_backbone_sha256")
        == preflight["pretrained_backbone_sha256"]
        and payload.get("runtime_projection", {}).get("within_task_limit") is True
        and payload.get("storage_projection", {}).get("within_budget") is True
        and payload.get("no_checkpoint_artifacts_verified") is True
        and smoke_valid
    )
    if not valid:
        raise CampaignPreflightError(f"Full launch gate is stale: {path}")
    result = dict(payload)
    result["artifact_path"] = str(path)
    result["artifact_sha256"] = sha256_file(path)
    return result
