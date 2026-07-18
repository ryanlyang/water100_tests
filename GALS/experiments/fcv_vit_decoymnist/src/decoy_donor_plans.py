"""Deterministic Step-6 multiclass donor plans for DecoyMNIST FCV."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from decoy_data import locate_decoy_patch
from decoy_full_config import canonical_config_sha256, sha256_file
from decoy_manifest_provenance import atomic_json
from decoy_teacher_masks import load_projected_teacher_masks


CORNER_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")


class DonorPlanError(ValueError):
    """Raised when a donor plan violates the frozen FCV protocol."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_content(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "plan_content_sha256"}


def _safe_source_path(data_root: Path, image_rel_path: str) -> Path:
    path = (data_root / str(image_rel_path)).resolve()
    if not path.is_relative_to(data_root):
        raise DonorPlanError(f"Manifest path escapes data root: {image_rel_path}")
    return path


def detect_training_corner(
    image_path: str | Path,
    label: int,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Validate the source image and return its published training-patch corner."""

    path = Path(image_path).expanduser().resolve()
    if expected_sha256 is not None and sha256_file(path) != str(expected_sha256):
        raise DonorPlanError(f"Source image changed after manifest creation: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise DonorPlanError(f"Could not read donor-plan image {path}: {exc}") from exc
    rows, columns = locate_decoy_patch(grayscale, int(label), "train")
    top = int(rows.start or 0) == 0
    left = int(columns.start or 0) == 0
    index = 0 if top and left else 1 if top else 2 if left else 3
    return CORNER_NAMES[index]


def build_donor_records(
    frame: pd.DataFrame,
    corners_by_id: Mapping[str, str],
    eligible_by_id: Mapping[str, bool],
    *,
    seed: int,
    donors_per_target: int,
) -> List[Dict[str, Any]]:
    """Build one deterministic plan from IDs, labels, corners, and eligibility."""

    required = {"sample_id", "label"}
    if not required.issubset(frame.columns):
        raise DonorPlanError(f"Donor frame lacks columns: {sorted(required - set(frame.columns))}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise DonorPlanError("Biased-validation sample IDs must be unique.")
    if int(donors_per_target) <= 0:
        raise DonorPlanError("donors_per_target must be positive.")

    ordered = frame.copy()
    ordered["sample_id"] = ordered["sample_id"].astype(str)
    ordered["label"] = ordered["label"].astype(int)
    ordered = ordered.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    ids = set(ordered["sample_id"])
    if set(corners_by_id) != ids or set(eligible_by_id) != ids:
        raise DonorPlanError("Corner/eligibility mappings must cover the manifest exactly.")

    pools: Dict[Tuple[str, int], List[str]] = {}
    for row in ordered.itertuples(index=False):
        if not bool(eligible_by_id[str(row.sample_id)]):
            continue
        corner = str(corners_by_id[row.sample_id])
        if corner not in CORNER_NAMES:
            raise DonorPlanError(f"Unknown corner {corner!r} for {row.sample_id}.")
        pools.setdefault((corner, int(row.label)), []).append(str(row.sample_id))
    for pool in pools.values():
        pool.sort()

    rng = np.random.default_rng(int(seed))
    records: List[Dict[str, Any]] = []
    for row in ordered.itertuples(index=False):
        target_id = str(row.sample_id)
        if not bool(eligible_by_id[target_id]):
            continue
        target_label = int(row.label)
        corner = str(corners_by_id[target_id])
        labels = sorted(
            label
            for (pool_corner, label), pool in pools.items()
            if pool_corner == corner and label != target_label and pool
        )
        if len(labels) < int(donors_per_target):
            raise DonorPlanError(
                f"Target {target_id} has only {len(labels)} same-corner non-target "
                f"labels; requires {donors_per_target}."
            )
        selected_labels = sorted(
            int(value)
            for value in rng.choice(
                np.asarray(labels, dtype=np.int64),
                size=int(donors_per_target),
                replace=False,
            ).tolist()
        )
        donors = []
        for donor_label in selected_labels:
            pool = pools[(corner, donor_label)]
            donor_id = str(pool[int(rng.integers(0, len(pool)))])
            if donor_id == target_id:
                raise DonorPlanError("Internal error: target selected as its own donor.")
            donors.append(
                {
                    "sample_id": donor_id,
                    "label": int(donor_label),
                    "corner": corner,
                }
            )
        records.append(
            {
                "target_sample_id": target_id,
                "target_label": target_label,
                "corner": corner,
                "donors": donors,
            }
        )
    return records


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    frame: pd.DataFrame,
    eligible_ids: Sequence[str],
    donors_per_target: int,
) -> None:
    labels_by_id = dict(
        zip(frame["sample_id"].astype(str), frame["label"].astype(int))
    )
    expected_targets = sorted(str(value) for value in eligible_ids)
    eligible_set = set(expected_targets)
    observed_targets = [str(record.get("target_sample_id")) for record in records]
    if observed_targets != expected_targets:
        raise DonorPlanError("Donor-plan targets do not exactly match eligible targets.")
    for record in records:
        target_id = str(record["target_sample_id"])
        target_label = int(record["target_label"])
        corner = str(record["corner"])
        if target_id not in labels_by_id or labels_by_id[target_id] != target_label:
            raise DonorPlanError(f"Target label is stale for {target_id}.")
        if corner not in CORNER_NAMES:
            raise DonorPlanError(f"Invalid target corner for {target_id}.")
        donors = list(record.get("donors", []))
        if len(donors) != int(donors_per_target):
            raise DonorPlanError(f"Wrong donor count for {target_id}.")
        donor_labels = []
        donor_ids = []
        for donor in donors:
            donor_id = str(donor.get("sample_id"))
            donor_label = int(donor.get("label"))
            donor_corner = str(donor.get("corner"))
            if donor_id not in labels_by_id or labels_by_id[donor_id] != donor_label:
                raise DonorPlanError(f"Donor label is stale for {donor_id}.")
            if donor_id not in eligible_set:
                raise DonorPlanError(f"Donor mask is not FCV-eligible: {donor_id}.")
            if donor_id == target_id or donor_label == target_label:
                raise DonorPlanError(f"Non-target donor rule failed for {target_id}.")
            if donor_corner != corner:
                raise DonorPlanError(f"Same-corner donor rule failed for {target_id}.")
            donor_ids.append(donor_id)
            donor_labels.append(donor_label)
        if len(set(donor_ids)) != len(donor_ids):
            raise DonorPlanError(f"Repeated donor sample for {target_id}.")
        if len(set(donor_labels)) != len(donor_labels):
            raise DonorPlanError(f"Donor labels are not distinct for {target_id}.")


def prepare_donor_plan(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    mask_artifact_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create the compact donor-ID plan after authenticating masks and images."""

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".json":
        raise DonorPlanError("Donor plans must be non-pickle JSON artifacts.")
    if destination.exists() and not overwrite:
        return load_and_validate_donor_plan(
            config, manifest_path, mask_artifact_path, destination
        )

    arrays, binding = load_projected_teacher_masks(
        config, manifest_path, mask_artifact_path
    )
    frame = pd.read_csv(binding.manifest_path)
    sample_ids = frame["sample_id"].astype(str).tolist()
    labels = frame["label"].astype(int).tolist()
    if arrays["sample_ids"].astype(str).tolist() != sample_ids:
        raise DonorPlanError("Mask rows do not align with donor manifest IDs.")
    if arrays["labels"].astype(int).tolist() != labels:
        raise DonorPlanError("Mask labels do not align with donor manifest labels.")

    data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
    corners_by_id: Dict[str, str] = {}
    for row in frame.itertuples(index=False):
        path = _safe_source_path(data_root, str(row.image_rel_path))
        corners_by_id[str(row.sample_id)] = detect_training_corner(
            path, int(row.label), expected_sha256=str(row.image_sha256)
        )
    eligible_by_id = dict(
        zip(sample_ids, arrays["fcv_eligible"].astype(bool).tolist())
    )
    donor_count = int(config["fcv"]["donor_samples_per_target"])
    seed = int(config["fcv"]["donor_plan_seed"])
    records = build_donor_records(
        frame,
        corners_by_id,
        eligible_by_id,
        seed=seed,
        donors_per_target=donor_count,
    )
    eligible_ids = sorted(
        sample_id for sample_id, eligible in eligible_by_id.items() if eligible
    )
    _validate_records(records, frame, eligible_ids, donor_count)
    payload: Dict[str, Any] = {
        "artifact_type": "fcv_vit_decoymnist_multiclass_donor_plan",
        "artifact_version": 1,
        "study_id": str(config["study"]["id"]),
        "config_sha256": canonical_config_sha256(config),
        "manifest_sha256": binding.manifest_sha256,
        "manifest_bundle_sha256": binding.bundle_sha256,
        "projected_teacher_masks_sha256": sha256_file(mask_artifact_path),
        "plan_seed": seed,
        "donors_per_target": donor_count,
        "source": "biased_validation_only",
        "target_count": len(records),
        "records": records,
    }
    payload["plan_content_sha256"] = _canonical_sha256(payload)
    atomic_json(payload, destination)
    return payload


def load_and_validate_donor_plan(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    mask_artifact_path: str | Path,
    plan_path: str | Path,
) -> Dict[str, Any]:
    """Authenticate a persisted donor plan without loading any model features."""

    path = Path(plan_path).expanduser().resolve()
    if path.suffix.lower() != ".json" or not path.is_file():
        raise DonorPlanError(f"Donor plan is missing or not JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DonorPlanError(f"Could not read donor plan {path}: {exc}") from exc
    if payload.get("plan_content_sha256") != _canonical_sha256(_plan_content(payload)):
        raise DonorPlanError("Donor-plan content hash is invalid.")
    if payload.get("artifact_type") != "fcv_vit_decoymnist_multiclass_donor_plan":
        raise DonorPlanError("Unexpected donor-plan artifact type.")

    arrays, binding = load_projected_teacher_masks(
        config, manifest_path, mask_artifact_path
    )
    expected = {
        "study_id": str(config["study"]["id"]),
        "config_sha256": canonical_config_sha256(config),
        "manifest_sha256": binding.manifest_sha256,
        "manifest_bundle_sha256": binding.bundle_sha256,
        "projected_teacher_masks_sha256": sha256_file(mask_artifact_path),
        "plan_seed": int(config["fcv"]["donor_plan_seed"]),
        "donors_per_target": int(config["fcv"]["donor_samples_per_target"]),
        "source": "biased_validation_only",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise DonorPlanError(f"Donor-plan {key} is stale or mismatched.")
    frame = pd.read_csv(binding.manifest_path)
    sample_ids = frame["sample_id"].astype(str).tolist()
    if arrays["sample_ids"].astype(str).tolist() != sample_ids:
        raise DonorPlanError("Projected masks and donor manifest are misaligned.")
    eligible_ids = sorted(
        sample_id
        for sample_id, eligible in zip(
            sample_ids, arrays["fcv_eligible"].astype(bool).tolist()
        )
        if eligible
    )
    records = payload.get("records")
    if not isinstance(records, list) or int(payload.get("target_count", -1)) != len(records):
        raise DonorPlanError("Donor-plan record count is invalid.")
    _validate_records(
        records,
        frame,
        eligible_ids,
        int(config["fcv"]["donor_samples_per_target"]),
    )
    return payload
