"""Cryptographic provenance for full-campaign DecoyMNIST manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

from decoy_full_config import canonical_config_sha256, sha256_file


PUBLIC_COLUMNS = [
    "sample_id",
    "image_rel_path",
    "label",
    "source_split",
    "study_split",
    "image_sha256",
]

MANIFEST_SPECS: Dict[str, Dict[str, str]] = {
    "candidate_train": {
        "filename": "metadata_train.csv",
        "source_split": "train",
        "study_split": "candidate_train",
        "count_key": "candidate_train_count",
    },
    "biased_validation": {
        "filename": "metadata_val.csv",
        "source_split": "train",
        "study_split": "biased_validation",
        "count_key": "biased_validation_count",
    },
    "oracle_validation": {
        "filename": "metadata_oracle_source_analysis_only.csv",
        "source_split": "train",
        "study_split": "oracle_validation_source_analysis_only",
        "count_key": "oracle_validation_source_count",
    },
    "test": {
        "filename": "metadata_test_analysis_only.csv",
        "source_split": "test",
        "study_split": "test_analysis_only",
        "count_key": "test",
    },
}


class ManifestProvenanceError(ValueError):
    """Raised when a manifest no longer matches its frozen trust bundle."""


@dataclass(frozen=True)
class ManifestBinding:
    role: str
    manifest_path: Path
    manifest_sha256: str
    bundle_path: Path
    bundle_sha256: str
    split_assignments_sha256: str
    split_summary_sha256: str
    config_sha256: str


def sha256_values(values: Iterable[str]) -> str:
    encoded = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def image_set_sha256(frame: pd.DataFrame) -> str:
    records = frame[["sample_id", "image_sha256"]].sort_values("sample_id")
    encoded = json.dumps(
        records.to_dict("records"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_manifest_bundle(
    config: Mapping[str, Any],
    destination: str | Path,
    artifact_paths: Mapping[str, Path],
    *,
    source_inventory: Mapping[str, Any],
) -> Path:
    """Write the trust root last, after every manifest artifact is durable."""

    destination = Path(destination).expanduser().resolve()
    assignments_path = Path(artifact_paths["assignments"]).resolve()
    summary_path = Path(artifact_paths["summary"]).resolve()
    if not assignments_path.is_file() or not summary_path.is_file():
        raise ManifestProvenanceError("Split assignments and summary must exist first.")

    manifest_records: Dict[str, Dict[str, Any]] = {}
    for role, spec in MANIFEST_SPECS.items():
        path = Path(artifact_paths[role]).resolve()
        if not path.is_file():
            raise ManifestProvenanceError(f"Missing manifest for {role}: {path}")
        frame = pd.read_csv(path)
        manifest_records[role] = {
            "filename": spec["filename"],
            "sha256": sha256_file(path),
            "row_count": int(len(frame)),
            "sample_ids_sha256": sha256_values(
                sorted(frame["sample_id"].astype(str).tolist())
            ),
            "image_set_sha256": image_set_sha256(frame),
        }

    provenance = config.get("_provenance", {})
    bundle = {
        "artifact_type": "fcv_vit_decoymnist_manifest_bundle",
        "artifact_version": 1,
        "study_id": config["study"]["id"],
        "protocol_version": int(config["study"]["protocol_version"]),
        "config": {
            "path": str(provenance.get("config_path", "UNKNOWN")),
            "file_sha256": str(provenance.get("config_file_sha256", "UNKNOWN")),
            "canonical_sha256": canonical_config_sha256(config),
        },
        "split_algorithm": config["reproducibility"]["split_algorithm_version"],
        "source_inventory": dict(source_inventory),
        "split_assignments": {
            "filename": assignments_path.name,
            "sha256": sha256_file(assignments_path),
        },
        "split_summary": {
            "filename": summary_path.name,
            "sha256": sha256_file(summary_path),
        },
        "manifests": manifest_records,
    }
    path = destination / "manifest_bundle.json"
    atomic_json(bundle, path)
    return path


def _expected_count(config: Mapping[str, Any], role: str) -> int:
    spec = MANIFEST_SPECS[role]
    if role == "test":
        return int(config["data"]["source_counts"]["test"])
    return int(config["data"]["partition"][spec["count_key"]])


def validate_manifest_bundle(
    config: Mapping[str, Any], manifest_path: str | Path, role: str
) -> ManifestBinding:
    """Authenticate one role and cross-check the complete split partition."""

    if role not in MANIFEST_SPECS:
        raise ManifestProvenanceError(f"Unknown manifest role: {role!r}")
    path = Path(manifest_path).expanduser().resolve()
    spec = MANIFEST_SPECS[role]
    if path.name != spec["filename"]:
        raise ManifestProvenanceError(
            f"{role} must use filename {spec['filename']!r}, found {path.name!r}."
        )
    bundle_path = path.parent / "manifest_bundle.json"
    if not path.is_file() or not bundle_path.is_file():
        raise ManifestProvenanceError("Manifest or manifest bundle is missing.")
    with bundle_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    if bundle.get("artifact_type") != "fcv_vit_decoymnist_manifest_bundle":
        raise ManifestProvenanceError("Wrong manifest-bundle artifact type.")
    if bundle.get("study_id") != config["study"]["id"]:
        raise ManifestProvenanceError("Manifest bundle belongs to another study.")
    expected_config_hash = canonical_config_sha256(config)
    if bundle.get("config", {}).get("canonical_sha256") != expected_config_hash:
        raise ManifestProvenanceError("Manifest bundle was built from another config.")
    expected_file_hash = config.get("_provenance", {}).get("config_file_sha256")
    if expected_file_hash and bundle.get("config", {}).get("file_sha256") != expected_file_hash:
        raise ManifestProvenanceError("Manifest bundle does not bind this YAML file.")

    assignments = bundle.get("split_assignments", {})
    summary = bundle.get("split_summary", {})
    assignments_path = path.parent / str(assignments.get("filename", ""))
    summary_path = path.parent / str(summary.get("filename", ""))
    for artifact_path, record, name in (
        (assignments_path, assignments, "split assignments"),
        (summary_path, summary, "split summary"),
    ):
        if not artifact_path.is_file() or sha256_file(artifact_path) != record.get(
            "sha256"
        ):
            raise ManifestProvenanceError(f"Authenticated {name} is missing or altered.")
    with assignments_path.open("r", encoding="utf-8") as handle:
        assignment_payload = json.load(handle)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary_payload = json.load(handle)
    if assignment_payload.get("config_sha256") != expected_config_hash:
        raise ManifestProvenanceError("Split assignments bind another config.")
    if summary_payload.get("config_sha256") != expected_config_hash:
        raise ManifestProvenanceError("Split summary binds another config.")

    all_frames: Dict[str, pd.DataFrame] = {}
    for current_role, current_spec in MANIFEST_SPECS.items():
        current_path = path.parent / current_spec["filename"]
        record = bundle.get("manifests", {}).get(current_role, {})
        if not current_path.is_file() or sha256_file(current_path) != record.get("sha256"):
            raise ManifestProvenanceError(
                f"Authenticated {current_role} manifest is missing or altered."
            )
        frame = pd.read_csv(current_path)
        if frame.columns.tolist() != PUBLIC_COLUMNS:
            raise ManifestProvenanceError(
                f"{current_role} manifest columns differ from the public schema."
            )
        if len(frame) != _expected_count(config, current_role):
            raise ManifestProvenanceError(f"{current_role} row count is incorrect.")
        if int(record.get("row_count", -1)) != len(frame):
            raise ManifestProvenanceError(f"{current_role} bundle count is stale.")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ManifestProvenanceError(f"{current_role} contains duplicate sample IDs.")
        if set(frame["source_split"].astype(str)) != {current_spec["source_split"]}:
            raise ManifestProvenanceError(f"{current_role} has the wrong source split.")
        if set(frame["study_split"].astype(str)) != {current_spec["study_split"]}:
            raise ManifestProvenanceError(f"{current_role} has the wrong study split.")
        labels = set(frame["label"].astype(int).tolist())
        if not labels.issubset(set(range(int(config["data"]["num_classes"])))):
            raise ManifestProvenanceError(f"{current_role} contains an invalid label.")
        if not frame["image_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ManifestProvenanceError(f"{current_role} has malformed image hashes.")
        if sha256_values(sorted(frame["sample_id"].astype(str))) != record.get(
            "sample_ids_sha256"
        ):
            raise ManifestProvenanceError(f"{current_role} sample-ID hash is stale.")
        if image_set_sha256(frame) != record.get("image_set_sha256"):
            raise ManifestProvenanceError(f"{current_role} image-set hash is stale.")
        assigned = assignment_payload.get("splits", {}).get(current_role, {})
        observed_ids = sorted(frame["sample_id"].astype(str).tolist())
        observed_counts = {
            str(label): int(
                (frame["label"].astype(int) == label).sum()
            )
            for label in range(int(config["data"]["num_classes"]))
        }
        if (
            int(assigned.get("count", -1)) != len(frame)
            or assigned.get("sample_ids") != observed_ids
            or assigned.get("sample_ids_sha256") != sha256_values(observed_ids)
            or assigned.get("class_counts") != observed_counts
        ):
            raise ManifestProvenanceError(
                f"{current_role} membership differs from split assignments."
            )
        summarized = summary_payload.get("splits", {}).get(current_role, {})
        if (
            int(summarized.get("count", -1)) != len(frame)
            or summarized.get("class_counts") != observed_counts
            or summarized.get("sample_ids_sha256") != sha256_values(observed_ids)
            or summarized.get("image_set_sha256") != image_set_sha256(frame)
            or summarized.get("source_split") != current_spec["source_split"]
            or summarized.get("study_split") != current_spec["study_split"]
        ):
            raise ManifestProvenanceError(
                f"{current_role} membership differs from split summary."
            )
        all_frames[current_role] = frame

    train_roles = ("candidate_train", "biased_validation", "oracle_validation")
    train_sets = {
        name: set(all_frames[name]["sample_id"].astype(str)) for name in train_roles
    }
    for index, left in enumerate(train_roles):
        for right in train_roles[index + 1 :]:
            if train_sets[left].intersection(train_sets[right]):
                raise ManifestProvenanceError(f"{left} and {right} overlap.")
    train_union = set().union(*(train_sets[name] for name in train_roles))
    if len(train_union) != int(config["data"]["source_counts"]["train"]):
        raise ManifestProvenanceError("Train-derived manifests do not partition train.")
    test_ids = set(all_frames["test"]["sample_id"].astype(str))
    if train_union.intersection(test_ids):
        raise ManifestProvenanceError("Official test overlaps a train-derived manifest.")

    return ManifestBinding(
        role=role,
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        bundle_path=bundle_path,
        bundle_sha256=sha256_file(bundle_path),
        split_assignments_sha256=str(assignments["sha256"]),
        split_summary_sha256=str(summary["sha256"]),
        config_sha256=expected_config_hash,
    )
