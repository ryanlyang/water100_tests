"""One immutable provenance root shared by every online FCV campaign task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from .candidate_training import (
    PublicManifestDataset,
    build_model,
    candidate_training_fingerprint,
    load_pretrained_cache_provenance,
    pretrained_backbone_sha256,
    pretrained_provenance_path,
    seed_everything,
    software_fingerprint,
    software_versions,
    state_dict_sha256,
    source_tree_provenance,
)
from .manifest_provenance import MANIFEST_SPECS, validate_manifest_bundle
from .selectors import prepare_oracle_validation_source
from .token_banks import prepare_token_bank_source


CAMPAIGN_PROVENANCE_FILENAME = "online_campaign_provenance.json"


class CampaignProvenanceError(RuntimeError):
    """Raised when an online task no longer matches the campaign trust root."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def campaign_provenance_path(config: Mapping[str, Any]) -> Path:
    return (
        Path(str(config["paths"]["output_root"])).expanduser().resolve()
        / "preflight"
        / CAMPAIGN_PROVENANCE_FILENAME
    )


def campaign_input_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    output_root = Path(str(config["paths"]["output_root"])).expanduser().resolve()
    manifest_root = output_root / str(config["outputs"]["split_manifests"])
    return {
        "candidate_train": manifest_root / MANIFEST_SPECS["candidate_train"]["filename"],
        "biased_validation": manifest_root
        / MANIFEST_SPECS["biased_validation"]["filename"],
        "oracle_validation": manifest_root
        / MANIFEST_SPECS["oracle_validation"]["filename"],
        "test": manifest_root / MANIFEST_SPECS["test"]["filename"],
        "patch_masks": output_root
        / str(config["outputs"]["patch_masks"])
        / "patch_masks_val.pt",
    }


def _image_inventory_sha256(path: Path) -> str:
    frame = pd.read_csv(path)
    required = {"sample_id", "image_path", "image_sha256"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise CampaignProvenanceError(
            f"Manifest image inventory is incomplete for {path}: {missing}"
        )
    records = [
        {
            "sample_id": str(row.sample_id),
            "image_path": str(Path(str(row.image_path)).expanduser().resolve()),
            "image_sha256": str(row.image_sha256),
        }
        for row in frame.itertuples(index=False)
    ]
    return _sha256_json(records)


def _config_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "study",
        "model",
        "training",
        "candidate_pool",
        "data",
        "fcv",
        "evaluation",
        "execution",
        "storage",
        "reproducibility",
        "cluster",
    )
    return {key: config[key] for key in keys}


def _expected_initialization_bindings(
    config: Mapping[str, Any], expected_backbone_sha256: str
) -> Dict[str, Any]:
    """Reconstruct the exact seeded initialization shared across LR/WD runs."""

    reproducibility = config["reproducibility"]
    by_seed: Dict[str, str] = {}
    for raw_seed in config["training"]["seeds"]:
        seed = int(raw_seed)
        seed_everything(
            seed,
            deterministic_algorithms=bool(
                reproducibility["deterministic_algorithms"]
            ),
            cudnn_benchmark=bool(reproducibility["cudnn_benchmark"]),
        )
        model = build_model(config, pretrained=bool(config["model"]["pretrained"]))
        observed_backbone = pretrained_backbone_sha256(model)
        if observed_backbone != expected_backbone_sha256:
            raise CampaignProvenanceError(
                "Seeded initialization does not use the cached pretrained backbone."
            )
        by_seed[str(seed)] = state_dict_sha256(model.state_dict())
        del model
    return {
        "seeds": [int(value) for value in config["training"]["seeds"]],
        "initial_model_state_sha256_by_seed": by_seed,
        "same_initialization_required_across_learning_rate_and_weight_decay": True,
    }


def _current_bindings(
    config: Mapping[str, Any], pretrained_path: str | Path
) -> Dict[str, Any]:
    paths = campaign_input_paths(config)
    pretrained = load_pretrained_cache_provenance(config, pretrained_path)
    manifests: Dict[str, Any] = {}
    for role in MANIFEST_SPECS:
        binding = validate_manifest_bundle(config, paths[role], role)
        manifests[role] = {
            "path": str(binding.manifest_path),
            "sha256": binding.manifest_sha256,
            "image_inventory_sha256": _image_inventory_sha256(binding.manifest_path),
            "bundle_path": str(binding.bundle_path),
            "bundle_sha256": binding.bundle_sha256,
            "original_metadata_sha256": binding.original_metadata_sha256,
            "split_indices_sha256": binding.split_indices_sha256,
            "split_summary_sha256": binding.split_summary_sha256,
        }

    patch_path = paths["patch_masks"].resolve()
    summary_path = patch_path.with_name("patch_masks_val_summary.json")
    if not patch_path.is_file() or not summary_path.is_file():
        raise CampaignProvenanceError("Patch-mask artifact or summary is missing.")
    with summary_path.open("r", encoding="utf-8") as handle:
        patch_summary = json.load(handle)
    patch_sha256 = _sha256_file(patch_path)
    if (
        not isinstance(patch_summary, Mapping)
        or patch_summary.get("status") != "complete"
        or patch_summary.get("patch_mask_path") != str(patch_path)
        or patch_summary.get("patch_mask_sha256") != patch_sha256
    ):
        raise CampaignProvenanceError("Patch-mask summary is stale or incompatible.")

    versions = software_versions()
    config_contract = _config_contract(config)
    initialization = _expected_initialization_bindings(
        config, pretrained["pretrained_backbone_sha256"]
    )
    return {
        "training_fingerprint": candidate_training_fingerprint(config),
        "config_contract_sha256": _sha256_json(config_contract),
        "config_contract": config_contract,
        "software_versions": versions,
        "software_fingerprint": software_fingerprint(versions),
        "source_tree": source_tree_provenance(),
        "pretrained": {
            "path": pretrained["artifact_path"],
            "sha256": pretrained["artifact_sha256"],
            "backbone_sha256": pretrained["pretrained_backbone_sha256"],
        },
        "initialization": initialization,
        "manifests": manifests,
        "patch_masks": {
            "path": str(patch_path),
            "sha256": patch_sha256,
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256_file(summary_path),
            "manifest_sha256": patch_summary.get("manifest_sha256"),
            "manifest_bundle_sha256": patch_summary.get("manifest_bundle_sha256"),
            "preprocessing_config_sha256": patch_summary.get(
                "preprocessing_config_sha256"
            ),
            "teacher_maps_sha256": patch_summary.get("teacher_maps_sha256"),
        },
    }


def create_campaign_provenance_receipt(
    config: Mapping[str, Any],
    *,
    pretrained_path: str | Path | None = None,
    verify_all_image_bytes: bool = True,
) -> Dict[str, Any]:
    """Validate all frozen inputs once and commit the campaign trust root."""

    pretrained_path = pretrained_path or pretrained_provenance_path(config)
    paths = campaign_input_paths(config)
    if verify_all_image_bytes:
        verify_non_test_campaign_inputs(config)
        # Keep the test implementation out of the validation-only import graph.
        from .test_evaluation import prepare_final_test_source

        prepare_final_test_source(config, paths["test"], check_images=True)
    bindings = _current_bindings(config, pretrained_path)
    payload = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_campaign_provenance",
        "status": "complete",
        "bindings": bindings,
        "bindings_sha256": _sha256_json(bindings),
        "all_manifest_image_bytes_verified_at_creation": bool(
            verify_all_image_bytes
        ),
        "checkpoint_retention_expanded": False,
    }
    path = campaign_provenance_path(config)
    _atomic_json(payload, path)
    result = dict(payload)
    result["artifact_path"] = str(path)
    result["artifact_sha256"] = _sha256_file(path)
    return result


def verify_non_test_campaign_inputs(config: Mapping[str, Any]) -> None:
    """Hash current train/holdout/Oracle bytes without importing test evaluation."""

    paths = campaign_input_paths(config)
    PublicManifestDataset(
        paths["candidate_train"], "candidate_train", None, check_images=True
    )
    # This validates public holdout images, masks, every referenced teacher map,
    # preprocessing, and diagnostics as one source.
    prepare_token_bank_source(
        config,
        paths["biased_validation"],
        paths["patch_masks"],
        check_images=True,
    )
    prepare_oracle_validation_source(
        config, paths["oracle_validation"], check_images=True
    )


def load_campaign_provenance_receipt(
    config: Mapping[str, Any],
    *,
    pretrained_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Require the current campaign inputs to match the committed trust root."""

    pretrained_path = pretrained_path or pretrained_provenance_path(config)
    path = Path(
        receipt_path or campaign_provenance_path(config)
    ).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing online campaign provenance receipt: {path}. Run cache preflight."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    current = _current_bindings(config, pretrained_path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type")
        != "fcv_vit_online_campaign_provenance"
        or payload.get("status") != "complete"
        or payload.get("bindings") != current
        or payload.get("bindings_sha256") != _sha256_json(current)
        or payload.get("all_manifest_image_bytes_verified_at_creation") is not True
        or payload.get("checkpoint_retention_expanded") is not False
    ):
        raise CampaignProvenanceError(
            "Online campaign provenance differs from current inputs/runtime."
        )
    result = dict(payload)
    result["artifact_path"] = str(path)
    result["artifact_sha256"] = _sha256_file(path)
    return result
