"""Cryptographic Step-2 manifest-bundle creation and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd


MANIFEST_SPECS: Dict[str, Dict[str, str]] = {
    "candidate_train": {
        "filename": "metadata_train.csv",
        "source_split": "train",
        "study_split": "candidate_train",
        "split_value_key": "train",
    },
    "biased_validation": {
        "filename": "metadata_val.csv",
        "source_split": "train",
        "study_split": "biased_validation",
        "split_value_key": "train",
    },
    "oracle_validation": {
        "filename": "metadata_oracle_val_analysis_only.csv",
        "source_split": "original_validation",
        "study_split": "oracle_validation_analysis_only",
        "split_value_key": "original_validation",
    },
    "test": {
        "filename": "metadata_test_analysis_only.csv",
        "source_split": "test",
        "study_split": "test_analysis_only",
        "split_value_key": "test",
    },
}


class ManifestProvenanceError(ValueError):
    """Raised when a manifest is not authenticated by the locked Step-2 bundle."""


@dataclass(frozen=True)
class ManifestBinding:
    manifest_key: str
    manifest_path: Path
    manifest_sha256: str
    bundle_path: Path
    bundle_sha256: str
    original_metadata_sha256: str
    split_indices_sha256: str
    split_summary_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_values(values: list[int]) -> str:
    payload = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_manifest_bundle(
    config: Mapping[str, Any],
    destination: str | Path,
    artifact_paths: Mapping[str, Path],
) -> Path:
    """Write the trust-root artifact after all four Step-2 manifests exist."""

    destination = Path(destination).expanduser().resolve()
    metadata_path = Path(config["paths"]["metadata"]).expanduser().resolve()
    indices_path = Path(artifact_paths["indices"]).expanduser().resolve()
    summary_path = Path(artifact_paths["summary"]).expanduser().resolve()
    if not metadata_path.is_file() or not indices_path.is_file() or not summary_path.is_file():
        raise ManifestProvenanceError("Cannot build manifest bundle from missing inputs.")

    manifests: Dict[str, Dict[str, Any]] = {}
    for key, spec in MANIFEST_SPECS.items():
        path = Path(artifact_paths[key]).expanduser().resolve()
        if not path.is_file():
            raise ManifestProvenanceError(f"Missing Step-2 manifest: {path}")
        frame = pd.read_csv(path)
        manifests[key] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": int(len(frame)),
            "metadata_indices_sha256": _sha256_values(
                sorted(frame["metadata_index"].astype(int).tolist())
            ),
            "source_split": spec["source_split"],
            "study_split": spec["study_split"],
        }

    bundle = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_waterbirds100_manifest_bundle",
        "status": "complete",
        "study_id": str(config["study"]["id"]),
        "protocol_version": str(config["study"]["protocol_version"]),
        "original_metadata_path": str(metadata_path),
        "original_metadata_sha256": sha256_file(metadata_path),
        "metadata_columns": dict(config["data"]["metadata_columns"]),
        "split_values": dict(config["data"]["split_values"]),
        "holdout": dict(config["data"]["biased_train_holdout"]),
        "split_indices_path": str(indices_path),
        "split_indices_sha256": sha256_file(indices_path),
        "split_summary_path": str(summary_path),
        "split_summary_sha256": sha256_file(summary_path),
        "manifests": manifests,
    }
    bundle_path = destination / "manifest_bundle.json"
    _atomic_json(bundle, bundle_path)
    return bundle_path


def _validate_manifest_rows(
    config: Mapping[str, Any],
    key: str,
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    spec = MANIFEST_SPECS[key]
    required = {
        "sample_id",
        "metadata_index",
        "image_rel_path",
        "label",
        "source_split",
        "study_split",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ManifestProvenanceError(f"Manifest {key} is missing columns: {missing}")
    if frame.empty or frame["sample_id"].astype(str).duplicated().any():
        raise ManifestProvenanceError(f"Manifest {key} is empty or has duplicate IDs.")
    if set(frame["source_split"].astype(str)) != {spec["source_split"]}:
        raise ManifestProvenanceError(
            f"Manifest {key} must have source_split={spec['source_split']!r}."
        )
    if set(frame["study_split"].astype(str)) != {spec["study_split"]}:
        raise ManifestProvenanceError(
            f"Manifest {key} must have study_split={spec['study_split']!r}."
        )

    indices = frame["metadata_index"].astype(int).to_numpy()
    if len(set(indices.tolist())) != len(indices):
        raise ManifestProvenanceError(f"Manifest {key} has duplicate metadata indices.")
    if (indices < 0).any() or (indices >= len(metadata)).any():
        raise ManifestProvenanceError(f"Manifest {key} has out-of-range metadata indices.")
    original = metadata.iloc[indices].reset_index(drop=True)
    columns = config["data"]["metadata_columns"]
    split_value = int(config["data"]["split_values"][spec["split_value_key"]])
    if not (original[columns["split"]].astype(int).to_numpy() == split_value).all():
        raise ManifestProvenanceError(
            f"Manifest {key} contains rows from the wrong original split."
        )
    if not (
        original[columns["label"]].astype(int).to_numpy()
        == frame["label"].astype(int).to_numpy()
    ).all():
        raise ManifestProvenanceError(f"Manifest {key} labels differ from metadata.")
    if original[columns["image_path"]].astype(str).tolist() != frame[
        "image_rel_path"
    ].astype(str).tolist():
        raise ManifestProvenanceError(f"Manifest {key} image identities differ from metadata.")
    expected_ids = [f"wb100_{int(index):05d}" for index in indices]
    if frame["sample_id"].astype(str).tolist() != expected_ids:
        raise ManifestProvenanceError(f"Manifest {key} sample IDs do not reproduce.")


def validate_manifest_bundle(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    expected_key: str,
) -> ManifestBinding:
    """Authenticate one manifest against metadata, split indices, and all bundle hashes."""

    if expected_key not in MANIFEST_SPECS:
        raise ManifestProvenanceError(f"Unknown manifest role: {expected_key}")
    manifest_path = Path(manifest_path).expanduser().resolve()
    if config.get("_synthetic_test_mode") is True:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing synthetic manifest: {manifest_path}")
        return ManifestBinding(
            manifest_key=expected_key,
            manifest_path=manifest_path,
            manifest_sha256=sha256_file(manifest_path),
            bundle_path=manifest_path.parent / "SYNTHETIC_TEST_MANIFEST_BUNDLE",
            bundle_sha256="SYNTHETIC_TEST_MODE",
            original_metadata_sha256="SYNTHETIC_TEST_MODE",
            split_indices_sha256="SYNTHETIC_TEST_MODE",
            split_summary_sha256="SYNTHETIC_TEST_MODE",
        )

    bundle_path = manifest_path.parent / "manifest_bundle.json"
    if not bundle_path.is_file():
        raise ManifestProvenanceError(
            f"Manifest is not authenticated by Step 2: missing {bundle_path}"
        )
    with bundle_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    if (
        bundle.get("schema_version") != 1
        or bundle.get("artifact_type") != "fcv_vit_waterbirds100_manifest_bundle"
        or bundle.get("status") != "complete"
        or bundle.get("study_id") != str(config["study"]["id"])
        or bundle.get("protocol_version") != str(config["study"]["protocol_version"])
        or bundle.get("metadata_columns") != dict(config["data"]["metadata_columns"])
        or bundle.get("split_values") != dict(config["data"]["split_values"])
        or bundle.get("holdout") != dict(config["data"]["biased_train_holdout"])
    ):
        raise ManifestProvenanceError("Step-2 manifest bundle is stale or incompatible.")

    metadata_path = Path(str(bundle.get("original_metadata_path", ""))).resolve()
    configured_metadata = Path(config["paths"]["metadata"]).expanduser().resolve()
    indices_path = Path(str(bundle.get("split_indices_path", ""))).resolve()
    summary_path = Path(str(bundle.get("split_summary_path", ""))).resolve()
    if metadata_path != configured_metadata or not metadata_path.is_file():
        raise ManifestProvenanceError("Manifest bundle references the wrong metadata file.")
    for path, field in (
        (metadata_path, "original_metadata_sha256"),
        (indices_path, "split_indices_sha256"),
        (summary_path, "split_summary_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != bundle.get(field):
            raise ManifestProvenanceError(f"Manifest-bundle dependency changed: {field}")

    manifests = bundle.get("manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != set(MANIFEST_SPECS):
        raise ManifestProvenanceError("Manifest bundle does not contain all four splits.")
    metadata = pd.read_csv(metadata_path)
    frames: Dict[str, pd.DataFrame] = {}
    for key, spec in MANIFEST_SPECS.items():
        details = manifests.get(key)
        if not isinstance(details, Mapping):
            raise ManifestProvenanceError(f"Manifest bundle lacks details for {key}.")
        path = (bundle_path.parent / spec["filename"]).resolve()
        if Path(str(details.get("path", ""))).resolve() != path or not path.is_file():
            raise ManifestProvenanceError(f"Manifest-bundle path is invalid for {key}.")
        if sha256_file(path) != details.get("sha256"):
            raise ManifestProvenanceError(f"Manifest bytes changed after Step 2: {key}")
        frame = pd.read_csv(path)
        _validate_manifest_rows(config, key, frame, metadata)
        if int(details.get("row_count", -1)) != len(frame):
            raise ManifestProvenanceError(f"Manifest row count changed for {key}.")
        indices = sorted(frame["metadata_index"].astype(int).tolist())
        if details.get("metadata_indices_sha256") != _sha256_values(indices):
            raise ManifestProvenanceError(f"Manifest index hash changed for {key}.")
        if (
            details.get("source_split") != spec["source_split"]
            or details.get("study_split") != spec["study_split"]
        ):
            raise ManifestProvenanceError(f"Manifest role changed for {key}.")
        frames[key] = frame

    with indices_path.open("r", encoding="utf-8") as handle:
        split_indices = json.load(handle)
    expected_train = sorted(frames["candidate_train"]["metadata_index"].astype(int))
    expected_val = sorted(frames["biased_validation"]["metadata_index"].astype(int))
    if (
        split_indices.get("metadata_sha256") != bundle["original_metadata_sha256"]
        or int(split_indices.get("split_seed", -1))
        != int(config["data"]["biased_train_holdout"]["split_seed"])
        or split_indices.get("candidate_train_metadata_indices") != expected_train
        or split_indices.get("biased_validation_metadata_indices") != expected_val
        or split_indices.get("candidate_train_indices_sha256")
        != _sha256_values(expected_train)
        or split_indices.get("biased_validation_indices_sha256")
        != _sha256_values(expected_val)
    ):
        raise ManifestProvenanceError("Split indices do not reproduce the manifest bundle.")
    if manifest_path != Path(str(manifests[expected_key]["path"])).resolve():
        raise ManifestProvenanceError(
            f"Manifest path is not the bundled {expected_key} artifact."
        )
    return ManifestBinding(
        manifest_key=expected_key,
        manifest_path=manifest_path,
        manifest_sha256=str(manifests[expected_key]["sha256"]),
        bundle_path=bundle_path,
        bundle_sha256=sha256_file(bundle_path),
        original_metadata_sha256=str(bundle["original_metadata_sha256"]),
        split_indices_sha256=str(bundle["split_indices_sha256"]),
        split_summary_sha256=str(bundle["split_summary_sha256"]),
    )
