"""Deterministic opposite-context FCV scoring for the first ViT study.

Step 7 has two deliberately separate artifacts:

* one candidate-independent donor-index plan, cached once and reused by every
  checkpoint; and
* one per-candidate score CSV plus a compact provenance/metric summary.

The shared plan prevents candidate ranking from being confounded by different
random donor draws. Donor tokens are model-specific, but their provenance
layout is required to be identical across all Step 6 banks.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .candidate_training import (
    candidate_training_fingerprint,
    enumerate_sweep_runs,
    software_fingerprint,
    software_versions,
)
from .config import candidate_epochs
from .storage import validate_cleanup_receipt
from .token_banks import CONTEXT_NAMES, TokenBankError, TokenBankSource
from .vit_counterfactual_forward import (
    extract_raw_patch_tokens,
    forward_from_patch_tokens,
    load_candidate_model,
)


OPPOSITE_CONTEXT = {0: 1, 1: 0}
PROVENANCE_TENSOR_KEYS = (
    "token_source_image_index",
    "token_source_class",
    "token_source_patch_idx",
    "token_source_patch_row",
    "token_source_patch_col",
)


class FCVScoringError(ValueError):
    """Raised when Step 7 data, bank, plan, or score invariants fail."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _canonical_probability_draw_statistics(
    probabilities: torch.Tensor,
) -> tuple[List[float], float, float]:
    """Derive persisted draw statistics from the persisted draw values.

    The JSON draw list is the canonical raw FCV artifact.  Computing its mean
    separately with a CUDA float32 reduction can differ from NumPy's float64
    recomputation by more than the strict artifact tolerance on GH200.  Convert
    the draws once, then derive every saved statistic from that exact list.
    """

    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise FCVScoringError(
            "FCV probability draws must be a non-empty one-dimensional tensor."
        )
    values = [
        float(value)
        for value in probabilities.detach().to(device="cpu").tolist()
    ]
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or ((array < 0.0) | (array > 1.0)).any():
        raise FCVScoringError("FCV probability draws are invalid.")
    return values, float(array.mean()), float(array.std(ddof=0))


def _load_trusted_torch_artifact(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise FCVScoringError(f"Torch artifact is not a mapping: {path}")
    return payload


def _source_frame(source: TokenBankSource) -> pd.DataFrame:
    dataset = getattr(source.loader, "dataset", None)
    frame = getattr(dataset, "frame", None)
    if not isinstance(frame, pd.DataFrame):
        raise FCVScoringError("TokenBankSource loader has no public manifest frame.")
    if len(frame) != source.sample_count:
        raise FCVScoringError("TokenBankSource sample count and manifest disagree.")
    return frame


def token_bank_layout_sha256(bank: Mapping[str, Any]) -> str:
    """Hash donor provenance while intentionally excluding model token values."""

    digest = hashlib.sha256()
    metadata = {
        "context_name": bank.get("context_name"),
        "context_label_proxy": bank.get("context_label_proxy"),
        "patch_grid_size": bank.get("patch_grid_size"),
        "token_count": bank.get("token_count"),
        "source_image_count": bank.get("source_image_count"),
        "source_images": bank.get("source_images"),
    }
    digest.update(_json_bytes(metadata))
    for key in PROVENANCE_TENSOR_KEYS:
        value = bank.get(key)
        if not isinstance(value, torch.Tensor):
            raise FCVScoringError(f"Token bank has no tensor {key!r}.")
        value = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_json_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_source_images(
    bank: Mapping[str, Any],
    label: int,
) -> tuple[List[Mapping[str, Any]], Dict[str, int]]:
    source_images = bank.get("source_images")
    if not isinstance(source_images, Sequence) or len(source_images) < 2:
        raise FCVScoringError(
            f"{CONTEXT_NAMES[label]} requires at least two donor source images."
        )
    normalized: List[Mapping[str, Any]] = []
    sample_to_index: Dict[str, int] = {}
    for expected_index, item in enumerate(source_images):
        if not isinstance(item, Mapping):
            raise FCVScoringError("Token-bank source image entries must be mappings.")
        source_index = int(item.get("source_image_index", -1))
        sample_id = str(item.get("sample_id", ""))
        source_label = int(item.get("label", -1))
        if source_index != expected_index or not sample_id or source_label != label:
            raise FCVScoringError(
                f"Invalid {CONTEXT_NAMES[label]} source image entry at index "
                f"{expected_index}."
            )
        if sample_id in sample_to_index:
            raise FCVScoringError(f"Duplicate donor source sample ID: {sample_id}")
        sample_to_index[sample_id] = source_index
        normalized.append(item)
    saved_map = bank.get("source_sample_id_to_index")
    if not isinstance(saved_map, Mapping) or {
        str(key): int(value) for key, value in saved_map.items()
    } != sample_to_index:
        raise FCVScoringError("Token-bank source sample lookup is inconsistent.")
    return normalized, sample_to_index


def load_background_bank(
    config: Mapping[str, Any],
    path: str | Path,
    source: TokenBankSource,
    *,
    expected_label: int,
    expected_candidate_id: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> Dict[str, Any]:
    """Strictly load one Step 6 bank and validate all sampling provenance."""

    if expected_label not in CONTEXT_NAMES:
        raise FCVScoringError(f"Unsupported context label: {expected_label}")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 6 token bank: {path}")
    bank = dict(_load_trusted_torch_artifact(path))
    expected_values = {
        "artifact_type": "fcv_vit_background_token_bank",
        "schema_version": 1,
        "training_fingerprint": candidate_training_fingerprint(config),
        "split": "biased_validation",
        "context_name": CONTEXT_NAMES[expected_label],
        "context_label_proxy": expected_label,
        "validation_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "patch_mask_sha256": source.patch_mask_sha256,
        "patch_mask_summary_sha256": source.patch_mask_summary_sha256,
        "patch_mask_preprocessing_sha256": source.patch_mask_preprocessing_sha256,
        "teacher_maps_sha256": source.teacher_maps_sha256,
        "patch_grid_size": int(config["model"]["patch_grid_size"]),
        "storage_dtype": "float32",
        "software_fingerprint": software_fingerprint(),
    }
    for key, expected in expected_values.items():
        if bank.get(key) != expected:
            raise FCVScoringError(
                f"Token-bank field {key!r} is stale in {path}: "
                f"{bank.get(key)!r} versus {expected!r}."
            )
    if expected_candidate_id is not None and bank.get("candidate_id") != expected_candidate_id:
        raise FCVScoringError(
            f"Token bank {path} belongs to {bank.get('candidate_id')!r}, not "
            f"{expected_candidate_id!r}."
        )
    if expected_checkpoint_sha256 is not None and bank.get(
        "checkpoint_sha256"
    ) != expected_checkpoint_sha256:
        raise FCVScoringError(f"Token bank {path} was built from a different checkpoint.")
    if bank.get("sampling_contract") != dict(config["fcv"]["donor_bank"]):
        raise FCVScoringError(f"Token bank {path} uses a different donor protocol.")

    tokens = bank.get("tokens")
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 2
        or tokens.dtype != torch.float32
        or tokens.numel() == 0
    ):
        raise FCVScoringError(f"Token bank {path} has invalid float32 token storage.")
    if not torch.isfinite(tokens).all():
        raise FCVScoringError(f"Token bank {path} contains non-finite values.")
    token_count, embedding_dim = (int(tokens.shape[0]), int(tokens.shape[1]))
    if int(bank.get("token_count", -1)) != token_count or int(
        bank.get("embedding_dim", -1)
    ) != embedding_dim:
        raise FCVScoringError(f"Token-bank dimensions are inconsistent in {path}.")

    source_images, source_sample_id_to_index = _validate_source_images(
        bank, expected_label
    )
    if int(bank.get("source_image_count", -1)) != len(source_images):
        raise FCVScoringError(f"Token-bank source count is inconsistent in {path}.")
    for key in (*PROVENANCE_TENSOR_KEYS, "token_patch_score"):
        value = bank.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) != token_count:
            raise FCVScoringError(f"Token-bank provenance {key!r} is invalid in {path}.")
    source_indices = bank["token_source_image_index"].to(torch.long)
    if int(source_indices.min()) < 0 or int(source_indices.max()) >= len(source_images):
        raise FCVScoringError(f"Token-bank source indices are out of range in {path}.")
    source_classes = bank["token_source_class"].to(torch.long)
    if not bool(torch.all(source_classes == expected_label)):
        raise FCVScoringError(f"Token-bank class provenance is invalid in {path}.")
    patch_indices = bank["token_source_patch_idx"].to(torch.long)
    patch_count = int(config["model"]["patch_grid_size"]) ** 2
    if int(patch_indices.min()) < 0 or int(patch_indices.max()) >= patch_count:
        raise FCVScoringError(f"Token-bank patch indices are out of range in {path}.")
    patch_scores = bank["token_patch_score"].float()
    if not torch.isfinite(patch_scores).all() or float(patch_scores.max()) > float(
        config["fcv"]["background_patch_threshold"]
    ) + 1.0e-6:
        raise FCVScoringError(f"Token bank {path} contains unsafe background tokens.")

    bank["tokens"] = tokens.contiguous()
    bank["source_images"] = source_images
    bank["source_sample_id_to_index"] = source_sample_id_to_index
    bank["layout_sha256"] = token_bank_layout_sha256(bank)
    bank["artifact_path"] = str(path)
    bank["artifact_sha256"] = _sha256_file(path)
    return bank


def _plan_content_sha256(plan: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    metadata = {
        key: plan.get(key)
        for key in (
            "schema_version",
            "artifact_type",
            "training_fingerprint",
            "split",
            "validation_manifest_sha256",
            "manifest_bundle_sha256",
            "patch_mask_sha256",
            "patch_mask_summary_sha256",
            "patch_mask_preprocessing_sha256",
            "teacher_maps_sha256",
            "donor_samples_per_image",
            "donor_sampling_seed",
            "sampling_contract",
            "context_bank_layout_sha256",
            "sample_count",
            "eligible_sample_count",
        )
    }
    digest.update(_json_bytes(metadata))
    records = plan.get("records")
    if not isinstance(records, Sequence):
        raise FCVScoringError("Donor plan records are missing.")
    for record in records:
        if not isinstance(record, Mapping):
            raise FCVScoringError("Donor plan record is not a mapping.")
        digest.update(
            _json_bytes(
                {
                    key: record.get(key)
                    for key in (
                        "sample_id",
                        "metadata_index",
                        "label",
                        "fcv_eligible",
                        "donor_context_label",
                        "donor_context_name",
                    )
                }
            )
        )
        for key in ("background_idx", "donor_token_indices"):
            tensor = record.get(key)
            if not isinstance(tensor, torch.Tensor):
                raise FCVScoringError(f"Donor plan record has no tensor {key!r}.")
            tensor = tensor.detach().cpu().contiguous()
            digest.update(key.encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(_json_bytes(list(tensor.shape)))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _eligible_token_indices(bank: Mapping[str, Any], target_sample_id: str) -> torch.Tensor:
    token_count = int(bank["token_count"])
    available = torch.arange(token_count, dtype=torch.long)
    target_source_index = bank["source_sample_id_to_index"].get(target_sample_id)
    if target_source_index is not None:
        source_indices = bank["token_source_image_index"].to(torch.long)
        available = available[source_indices != int(target_source_index)]
    if available.numel() == 0:
        raise FCVScoringError(
            f"Self-donor exclusion leaves no tokens for target {target_sample_id}."
        )
    return available


def validate_opposite_donor_plan(
    config: Mapping[str, Any],
    source: TokenBankSource,
    banks: Mapping[int, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
    """Validate the complete cached draw plan against source masks and bank layout."""

    expected_values = {
        "artifact_type": "fcv_vit_opposite_donor_plan",
        "schema_version": 1,
        "training_fingerprint": candidate_training_fingerprint(config),
        "split": "biased_validation",
        "validation_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "patch_mask_sha256": source.patch_mask_sha256,
        "patch_mask_summary_sha256": source.patch_mask_summary_sha256,
        "patch_mask_preprocessing_sha256": source.patch_mask_preprocessing_sha256,
        "teacher_maps_sha256": source.teacher_maps_sha256,
        "donor_samples_per_image": int(config["fcv"]["donor_samples_per_image"]),
        "donor_sampling_seed": int(config["reproducibility"]["donor_sampling_seed"]),
        "sampling_contract": dict(config["fcv"]["donor_bank"]),
        "sample_count": source.sample_count,
    }
    for key, expected in expected_values.items():
        if plan.get(key) != expected:
            raise FCVScoringError(
                f"Donor-plan field {key!r} is stale: {plan.get(key)!r} "
                f"versus {expected!r}."
            )
    expected_layouts = {
        str(label): str(banks[label]["layout_sha256"]) for label in CONTEXT_NAMES
    }
    if plan.get("context_bank_layout_sha256") != expected_layouts:
        raise FCVScoringError("Donor-plan bank layout does not match this candidate.")
    saved_content_hash = str(plan.get("plan_content_sha256", ""))
    if len(saved_content_hash) != 64 or _plan_content_sha256(plan) != saved_content_hash:
        raise FCVScoringError("Donor-plan content hash is missing or invalid.")

    frame = _source_frame(source)
    records = plan.get("records")
    if not isinstance(records, Sequence) or len(records) != len(frame):
        raise FCVScoringError("Donor plan does not cover the full validation manifest.")
    by_sample_id: Dict[str, Mapping[str, Any]] = {}
    eligible_count = 0
    donor_samples = int(config["fcv"]["donor_samples_per_image"])
    for row, plan_record in zip(frame.itertuples(index=False), records):
        if not isinstance(plan_record, Mapping):
            raise FCVScoringError("Donor-plan records must be mappings.")
        sample_id = str(row.sample_id)
        record = source.records_by_sample_id[sample_id]
        label = int(row.label)
        expected_donor_label = OPPOSITE_CONTEXT[label]
        if (
            str(plan_record.get("sample_id")) != sample_id
            or int(plan_record.get("metadata_index", -1)) != int(row.metadata_index)
            or int(plan_record.get("label", -1)) != label
            or bool(plan_record.get("fcv_eligible")) != bool(record["fcv_eligible"])
            or int(plan_record.get("donor_context_label", -1)) != expected_donor_label
            or plan_record.get("donor_context_name") != CONTEXT_NAMES[expected_donor_label]
        ):
            raise FCVScoringError(f"Stale donor-plan metadata for {sample_id}.")
        background_idx = plan_record.get("background_idx")
        donor_indices = plan_record.get("donor_token_indices")
        if not isinstance(background_idx, torch.Tensor) or not isinstance(
            donor_indices, torch.Tensor
        ):
            raise FCVScoringError(f"Donor-plan tensors are missing for {sample_id}.")
        background_idx = background_idx.to(dtype=torch.long, device="cpu").flatten()
        expected_background = record["background_idx"].to(
            dtype=torch.long, device="cpu"
        ).flatten()
        if not torch.equal(background_idx, expected_background):
            raise FCVScoringError(f"Donor-plan background positions changed for {sample_id}.")
        if bool(record["fcv_eligible"]):
            eligible_count += 1
            expected_shape = (donor_samples, int(background_idx.numel()))
            if donor_indices.dtype != torch.long or tuple(donor_indices.shape) != expected_shape:
                raise FCVScoringError(
                    f"Donor indices for {sample_id} have shape/dtype "
                    f"{tuple(donor_indices.shape)}/{donor_indices.dtype}, expected "
                    f"{expected_shape}/torch.int64."
                )
            bank = banks[expected_donor_label]
            if donor_indices.numel() and (
                int(donor_indices.min()) < 0
                or int(donor_indices.max()) >= int(bank["token_count"])
            ):
                raise FCVScoringError(f"Out-of-range donor token index for {sample_id}.")
            target_source_index = bank["source_sample_id_to_index"].get(sample_id)
            if target_source_index is not None:
                sampled_source = bank["token_source_image_index"].to(torch.long).index_select(
                    0, donor_indices.flatten()
                )
                if bool(torch.any(sampled_source == int(target_source_index))):
                    raise FCVScoringError(f"Self-donor token sampled for {sample_id}.")
        elif tuple(donor_indices.shape) != (0, 0):
            raise FCVScoringError(
                f"Ineligible target {sample_id} unexpectedly has donor draws."
            )
        if sample_id in by_sample_id:
            raise FCVScoringError(f"Duplicate donor-plan sample ID: {sample_id}")
        by_sample_id[sample_id] = plan_record
    if int(plan.get("eligible_sample_count", -1)) != eligible_count:
        raise FCVScoringError("Donor-plan eligible count is inconsistent.")
    return by_sample_id


def prepare_opposite_donor_plan(
    config: Mapping[str, Any],
    source: TokenBankSource,
    banks: Mapping[int, Mapping[str, Any]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Create the candidate-independent, deterministic five-draw donor plan."""

    output_path = Path(output_path).expanduser().resolve()
    if set(banks) != set(CONTEXT_NAMES):
        raise FCVScoringError("Both land and water reference banks are required.")
    if output_path.is_file() and not overwrite:
        existing = _load_trusted_torch_artifact(output_path)
        validate_opposite_donor_plan(config, source, banks, existing)
        return existing

    frame = _source_frame(source)
    donor_samples = int(config["fcv"]["donor_samples_per_image"])
    donor_seed = int(config["reproducibility"]["donor_sampling_seed"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(donor_seed)
    records: List[Dict[str, Any]] = []
    eligible_count = 0
    draws_by_context = {0: 0, 1: 0}
    for row in frame.itertuples(index=False):
        sample_id = str(row.sample_id)
        label = int(row.label)
        record = source.records_by_sample_id[sample_id]
        background_idx = record["background_idx"].detach().to(
            dtype=torch.long, device="cpu"
        ).flatten()
        donor_label = OPPOSITE_CONTEXT[label]
        if bool(record["fcv_eligible"]):
            available = _eligible_token_indices(banks[donor_label], sample_id)
            offsets = torch.randint(
                0,
                int(available.numel()),
                (donor_samples, int(background_idx.numel())),
                generator=generator,
                dtype=torch.long,
            )
            donor_indices = available.index_select(0, offsets.flatten()).reshape_as(offsets)
            eligible_count += 1
            draws_by_context[donor_label] += int(donor_indices.numel())
        else:
            donor_indices = torch.empty((0, 0), dtype=torch.long)
        records.append(
            {
                "sample_id": sample_id,
                "metadata_index": int(row.metadata_index),
                "label": label,
                "fcv_eligible": bool(record["fcv_eligible"]),
                "donor_context_label": donor_label,
                "donor_context_name": CONTEXT_NAMES[donor_label],
                "background_idx": background_idx,
                "donor_token_indices": donor_indices,
            }
        )

    plan: MutableMapping[str, Any] = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_opposite_donor_plan",
        "training_fingerprint": candidate_training_fingerprint(config),
        "split": "biased_validation",
        "validation_manifest_path": str(source.manifest_path),
        "validation_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_path": str(source.manifest_bundle_path),
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "patch_mask_path": str(source.patch_mask_path),
        "patch_mask_sha256": source.patch_mask_sha256,
        "patch_mask_summary_path": str(source.patch_mask_summary_path),
        "patch_mask_summary_sha256": source.patch_mask_summary_sha256,
        "patch_mask_preprocessing_sha256": source.patch_mask_preprocessing_sha256,
        "teacher_maps_sha256": source.teacher_maps_sha256,
        "donor_samples_per_image": donor_samples,
        "donor_sampling_seed": donor_seed,
        "sampling_contract": dict(config["fcv"]["donor_bank"]),
        "reference_candidate_id": str(banks[0]["candidate_id"]),
        "context_bank_layout_sha256": {
            str(label): str(banks[label]["layout_sha256"]) for label in CONTEXT_NAMES
        },
        "context_token_count": {
            str(label): int(banks[label]["token_count"]) for label in CONTEXT_NAMES
        },
        "sample_count": source.sample_count,
        "eligible_sample_count": eligible_count,
        "ineligible_sample_count": source.sample_count - eligible_count,
        "sampled_token_draws_by_donor_context": {
            str(label): int(value) for label, value in draws_by_context.items()
        },
        "records": records,
    }
    plan["plan_content_sha256"] = _plan_content_sha256(plan)
    validate_opposite_donor_plan(config, source, banks, plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(plan, output_path)
    return plan


def make_counterfactual_token_batch(
    target_tokens: torch.Tensor,
    background_idx: torch.Tensor,
    donor_tokens: torch.Tensor,
    donor_token_indices: torch.Tensor,
) -> torch.Tensor:
    """Replace only target background positions for all cached donor draws."""

    if target_tokens.ndim != 2 or donor_tokens.ndim != 2:
        raise FCVScoringError("Target and donor token tensors must have shape [N,D].")
    if target_tokens.shape[1] != donor_tokens.shape[1]:
        raise FCVScoringError("Target and donor embedding dimensions differ.")
    background_idx = background_idx.to(dtype=torch.long, device=target_tokens.device)
    donor_token_indices = donor_token_indices.to(
        dtype=torch.long, device=donor_tokens.device
    )
    if donor_token_indices.ndim != 2 or donor_token_indices.shape[1] != len(
        background_idx
    ):
        raise FCVScoringError("Donor draws must have shape [K, num_background].")
    if len(background_idx) == 0:
        raise FCVScoringError("FCV cannot swap an empty background set.")
    if int(background_idx.min()) < 0 or int(background_idx.max()) >= target_tokens.shape[0]:
        raise FCVScoringError("Target background position is out of range.")
    if int(donor_token_indices.min()) < 0 or int(
        donor_token_indices.max()
    ) >= donor_tokens.shape[0]:
        raise FCVScoringError("Donor token index is out of range.")
    donor_samples = int(donor_token_indices.shape[0])
    counterfactual = target_tokens.unsqueeze(0).repeat(donor_samples, 1, 1)
    sampled = donor_tokens.index_select(0, donor_token_indices.flatten()).reshape(
        donor_samples,
        len(background_idx),
        target_tokens.shape[1],
    )
    sampled = sampled.to(device=target_tokens.device, dtype=target_tokens.dtype)
    counterfactual[:, background_idx, :] = sampled
    return counterfactual


def _score_summary_is_reusable(
    summary: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    candidate_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source: TokenBankSource,
    donor_plan_path: Path,
    donor_plan_sha256: str,
    reconstruction_reports: Mapping[str, Any],
    counterfactual_forward_batch_size: int,
) -> bool:
    if (
        summary.get("artifact_type") != "fcv_vit_candidate_score_summary"
        or summary.get("schema_version") != 1
        or summary.get("status") != "complete"
        or summary.get("candidate_id") != candidate_id
        or summary.get("training_fingerprint") != candidate_training_fingerprint(config)
        or summary.get("checkpoint_path") != str(checkpoint_path)
        or summary.get("checkpoint_sha256") != checkpoint_sha256
        or summary.get("validation_manifest_sha256") != source.manifest_sha256
        or summary.get("manifest_bundle_sha256") != source.manifest_bundle_sha256
        or summary.get("patch_mask_sha256") != source.patch_mask_sha256
        or summary.get("patch_mask_summary_sha256")
        != source.patch_mask_summary_sha256
        or summary.get("patch_mask_preprocessing_sha256")
        != source.patch_mask_preprocessing_sha256
        or summary.get("teacher_maps_sha256") != source.teacher_maps_sha256
        or summary.get("software_versions") != software_versions()
        or summary.get("software_fingerprint") != software_fingerprint()
        or summary.get("donor_plan_path") != str(donor_plan_path)
        or summary.get("donor_plan_sha256") != donor_plan_sha256
        or summary.get("reconstruction_reports") != dict(reconstruction_reports)
        or summary.get("identity_swap_uses_production_replacement_path") is not True
        or int(summary.get("identity_swap_path_sample_count", 0)) <= 0
        or float(summary.get("identity_swap_inside_token_max_abs_error", -1.0)) != 0.0
        or float(summary.get("identity_swap_outside_token_max_abs_error", -1.0)) != 0.0
        or not isinstance(summary.get("real_swap_integrity_diagnostics"), Mapping)
        or float(
            summary["real_swap_integrity_diagnostics"].get(
                "foreground_token_max_abs_error", -1.0
            )
        )
        != 0.0
        or float(
            summary["real_swap_integrity_diagnostics"].get(
                "donor_reconstruction_max_abs_error", -1.0
            )
        )
        != 0.0
        or int(
            summary["real_swap_integrity_diagnostics"].get(
                "replaced_token_changed_count", 0
            )
        )
        <= 0
        or summary.get("execution")
        != {
            "validation_batch_size": source.batch_size,
            "validation_num_workers": source.num_workers,
            "counterfactual_forward_batch_size": counterfactual_forward_batch_size,
        }
    ):
        return False
    score_path = Path(str(summary.get("score_csv_path", "")))
    expected_size = int(summary.get("score_csv_size_bytes", -1))
    expected_hash = str(summary.get("score_csv_sha256", ""))
    files_valid = (
        expected_size > 0
        and len(expected_hash) == 64
        and score_path.is_file()
        and score_path.stat().st_size == expected_size
        and _sha256_file(score_path) == expected_hash
    )
    if not files_valid:
        return False
    try:
        validate_fcv_summary_against_frame(summary, pd.read_csv(score_path), config)
    except (OSError, ValueError, TypeError, KeyError, FCVScoringError):
        return False
    return True


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise FCVScoringError("Cannot compute a metric over zero samples.")
    return float(sum(values) / len(values))


def _numeric_distribution(values: pd.Series) -> Dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if numeric.empty or not bool(numeric.map(math.isfinite).all()):
        raise FCVScoringError("Token diagnostic distribution is empty or non-finite.")
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "q25": float(numeric.quantile(0.25)),
        "median": float(numeric.quantile(0.50)),
        "q75": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
    }


def _assert_close(name: str, observed: Any, expected: float) -> None:
    if observed is None or not np.isclose(
        float(observed), float(expected), rtol=0.0, atol=1.0e-7
    ):
        raise FCVScoringError(
            f"FCV summary metric {name} does not reproduce from its hashed CSV: "
            f"summary={observed!r}, recomputed={expected!r}."
        )


def recompute_fcv_metrics_from_frame(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Recompute every selection-facing FCV metric from per-image draw records."""

    required = {
        "sample_id",
        "label",
        "fcv_eligible",
        "p_y_original",
        "original_cross_entropy",
        "pred_original",
        "correct_original",
        "p_y_counterfactual_mean",
        "p_y_counterfactual_std",
        "pred_counterfactual_majority",
        "correct_counterfactual_majority",
        "counterfactual_draw_accuracy",
        "counterfactual_correct_draws",
        "counterfactual_confidence_drop",
        "donor_draw_count",
        "p_y_counterfactual_draws",
        "pred_counterfactual_draws",
        "real_swap_foreground_max_abs_error",
        "real_swap_donor_reconstruction_max_abs_error",
        "real_swap_replaced_token_draw_count",
        "real_swap_replaced_token_changed_count",
        "real_swap_replaced_token_changed_fraction",
        "real_swap_replacement_delta_mean",
        "real_swap_replacement_delta_max",
        "target_donor_cosine_similarity_mean",
        "target_nearest_donor_cosine_mean",
        "donor_unique_source_images",
        "donor_max_source_fraction",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FCVScoringError(f"FCV score CSV is missing columns: {missing}")
    if frame.empty or frame["sample_id"].astype(str).duplicated().any():
        raise FCVScoringError("FCV score CSV must contain unique non-empty samples.")
    eligible_mask = frame["fcv_eligible"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    normalized = frame.copy()
    num_classes = int(config["model"]["num_classes"])
    original_probabilities = normalized["p_y_original"].astype(float).to_numpy()
    original_losses = normalized["original_cross_entropy"].astype(float).to_numpy()
    if not np.isfinite(original_probabilities).all() or (
        (original_probabilities < 0.0) | (original_probabilities > 1.0)
    ).any():
        raise FCVScoringError("FCV original probabilities are invalid.")
    if not np.isfinite(original_losses).all() or (original_losses < 0.0).any():
        raise FCVScoringError("FCV original cross-entropies are invalid.")
    for index in normalized.index:
        label = int(normalized.at[index, "label"])
        original_prediction = int(normalized.at[index, "pred_original"])
        if not 0 <= label < num_classes or not 0 <= original_prediction < num_classes:
            raise FCVScoringError("FCV labels or original predictions are invalid.")
        expected_original_correct = int(original_prediction == label)
        if int(normalized.at[index, "correct_original"]) != expected_original_correct:
            raise FCVScoringError("FCV original correctness does not reproduce.")
        if not bool(eligible_mask.loc[index]):
            continue
        foreground_error = float(
            normalized.at[index, "real_swap_foreground_max_abs_error"]
        )
        donor_error = float(
            normalized.at[index, "real_swap_donor_reconstruction_max_abs_error"]
        )
        replaced_count = int(
            normalized.at[index, "real_swap_replaced_token_draw_count"]
        )
        changed_count = int(
            normalized.at[index, "real_swap_replaced_token_changed_count"]
        )
        changed_fraction = float(
            normalized.at[index, "real_swap_replaced_token_changed_fraction"]
        )
        delta_mean = float(normalized.at[index, "real_swap_replacement_delta_mean"])
        delta_max = float(normalized.at[index, "real_swap_replacement_delta_max"])
        if foreground_error != 0.0 or donor_error != 0.0:
            raise FCVScoringError(
                "Real FCV swap changed foreground tokens or failed donor reconstruction."
            )
        if (
            replaced_count <= 0
            or changed_count < 0
            or changed_count > replaced_count
            or not np.isclose(
                changed_fraction,
                changed_count / replaced_count,
                rtol=0.0,
                atol=1.0e-12,
            )
            or not 0.0 <= changed_fraction <= 1.0
            or not np.isfinite([delta_mean, delta_max]).all()
            or delta_mean < 0.0
            or delta_max < delta_mean
        ):
            raise FCVScoringError("Real FCV replacement diagnostics are invalid.")
        draw_count = int(normalized.at[index, "donor_draw_count"])
        if draw_count <= 0:
            raise FCVScoringError("FCV donor_draw_count must be positive.")
        probabilities = json.loads(normalized.at[index, "p_y_counterfactual_draws"])
        predictions = json.loads(normalized.at[index, "pred_counterfactual_draws"])
        if len(probabilities) != draw_count or len(predictions) != draw_count:
            raise FCVScoringError("FCV draw arrays do not match donor_draw_count.")
        probabilities = np.asarray(probabilities, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.int64)
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0.0) | (probabilities > 1.0)
        ).any():
            raise FCVScoringError("FCV draw probabilities are invalid.")
        if ((predictions < 0) | (predictions >= num_classes)).any():
            raise FCVScoringError("FCV draw predictions are invalid.")
        correct_draws = int((predictions == label).sum())
        draw_accuracy = float(correct_draws / draw_count)
        majority = int(np.bincount(predictions, minlength=num_classes).argmax())
        mean_probability = float(probabilities.mean())
        std_probability = float(probabilities.std(ddof=0))
        confidence_drop = float(
            float(normalized.at[index, "p_y_original"]) - mean_probability
        )
        checks = {
            "counterfactual_correct_draws": correct_draws,
            "pred_counterfactual_majority": majority,
            "correct_counterfactual_majority": int(majority == label),
        }
        for column, expected in checks.items():
            if int(normalized.at[index, column]) != expected:
                raise FCVScoringError(f"FCV row field {column} does not reproduce.")
        for column, expected in (
            ("counterfactual_draw_accuracy", draw_accuracy),
            ("p_y_counterfactual_mean", mean_probability),
            ("p_y_counterfactual_std", std_probability),
            ("counterfactual_confidence_drop", confidence_drop),
        ):
            if not np.isclose(
                float(normalized.at[index, column]), expected, rtol=0.0, atol=1.0e-7
            ):
                raise FCVScoringError(f"FCV row field {column} does not reproduce.")
    eligible = normalized.loc[eligible_mask].copy()
    if eligible.empty:
        raise FCVScoringError("FCV score CSV contains no eligible samples.")
    original_accuracy = float(normalized["correct_original"].astype(float).mean())
    counterfactual_accuracy = float(
        eligible["counterfactual_draw_accuracy"].astype(float).mean()
    )
    weights = config["fcv"]["primary_selector"]
    primary = float(
        float(weights["original_accuracy_weight"]) * original_accuracy
        + float(weights["counterfactual_accuracy_weight"])
        * counterfactual_accuracy
    )
    originally_correct = eligible[eligible["correct_original"].astype(int) == 1]
    per_class = {
        str(int(label)): {
            "eligible_count": int(len(class_frame)),
            "original_accuracy": float(
                class_frame["correct_original"].astype(float).mean()
            ),
            "counterfactual_draw_accuracy": float(
                class_frame["counterfactual_draw_accuracy"].astype(float).mean()
            ),
            "counterfactual_true_class_probability": float(
                class_frame["p_y_counterfactual_mean"].astype(float).mean()
            ),
        }
        for label, class_frame in eligible.groupby("label", sort=True)
    }
    eligible_diagnostics = normalized.loc[eligible_mask]
    replaced_count = int(
        eligible_diagnostics["real_swap_replaced_token_draw_count"].astype(int).sum()
    )
    changed_count = int(
        eligible_diagnostics["real_swap_replaced_token_changed_count"].astype(int).sum()
    )
    if replaced_count <= 0 or changed_count <= 0:
        raise FCVScoringError("The real opposite-context intervention cohort was a no-op.")
    swap_diagnostics = {
        "foreground_token_max_abs_error": float(
            eligible_diagnostics["real_swap_foreground_max_abs_error"].astype(float).max()
        ),
        "donor_reconstruction_max_abs_error": float(
            eligible_diagnostics[
                "real_swap_donor_reconstruction_max_abs_error"
            ].astype(float).max()
        ),
        "replaced_token_draw_count": replaced_count,
        "replaced_token_changed_count": changed_count,
        "replaced_token_changed_fraction": float(changed_count / replaced_count),
        "replacement_delta_mean": float(
            np.average(
                eligible_diagnostics["real_swap_replacement_delta_mean"].astype(float),
                weights=eligible_diagnostics[
                    "real_swap_replaced_token_draw_count"
                ].astype(int),
            )
        ),
        "replacement_delta_max": float(
            eligible_diagnostics["real_swap_replacement_delta_max"].astype(float).max()
        ),
        "eligible_intervention_count": int(len(eligible_diagnostics)),
    }
    token_distribution_global_means = {
        column: float(eligible_diagnostics[column].astype(float).mean())
        for column in (
            "target_donor_cosine_similarity_mean",
            "target_nearest_donor_cosine_mean",
            "donor_unique_source_images",
            "donor_max_source_fraction",
        )
    }
    return {
        "sample_count": int(len(normalized)),
        "eligible_sample_count": int(len(eligible)),
        "original_accuracy": original_accuracy,
        "original_loss": float(original_losses.mean()),
        "counterfactual_accuracy": counterfactual_accuracy,
        "counterfactual_majority_accuracy": float(
            eligible["correct_counterfactual_majority"].astype(float).mean()
        ),
        "counterfactual_true_class_probability": float(
            eligible["p_y_counterfactual_mean"].astype(float).mean()
        ),
        "original_true_class_probability": float(
            normalized["p_y_original"].astype(float).mean()
        ),
        "mean_confidence_drop": float(
            eligible["counterfactual_confidence_drop"].astype(float).mean()
        ),
        "conditional_accuracy_originally_correct": (
            float(
                originally_correct["counterfactual_draw_accuracy"].astype(float).mean()
            )
            if not originally_correct.empty
            else None
        ),
        "conditional_sample_count": int(len(originally_correct)),
        "primary_selector_score": primary,
        "per_class": per_class,
        "swap_diagnostics": swap_diagnostics,
        "token_distribution_global_means": token_distribution_global_means,
    }


def validate_fcv_summary_against_frame(
    summary: Mapping[str, Any],
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    recomputed = recompute_fcv_metrics_from_frame(frame, config)
    fields = {
        "biased_validation_accuracy_recomputed": "original_accuracy",
        "biased_validation_loss_recomputed": "original_loss",
        "opposite_context_counterfactual_accuracy": "counterfactual_accuracy",
        "opposite_context_counterfactual_majority_accuracy": (
            "counterfactual_majority_accuracy"
        ),
        "opposite_context_true_class_probability": (
            "counterfactual_true_class_probability"
        ),
        "original_true_class_probability": "original_true_class_probability",
        "mean_counterfactual_confidence_drop": "mean_confidence_drop",
        "primary_selector_score": "primary_selector_score",
    }
    for summary_name, recomputed_name in fields.items():
        _assert_close(summary_name, summary.get(summary_name), recomputed[recomputed_name])
    if int(summary.get("validation_sample_count", -1)) != recomputed["sample_count"]:
        raise FCVScoringError("FCV validation sample count does not reproduce.")
    if int(summary.get("fcv_eligible_sample_count", -1)) != recomputed[
        "eligible_sample_count"
    ]:
        raise FCVScoringError("FCV eligible sample count does not reproduce.")
    if int(summary.get("conditional_counterfactual_sample_count", -1)) != recomputed[
        "conditional_sample_count"
    ]:
        raise FCVScoringError("FCV conditional sample count does not reproduce.")
    conditional = recomputed["conditional_accuracy_originally_correct"]
    if conditional is None:
        if summary.get("conditional_counterfactual_accuracy_originally_correct") is not None:
            raise FCVScoringError("FCV conditional accuracy should be null.")
    else:
        _assert_close(
            "conditional_counterfactual_accuracy_originally_correct",
            summary.get("conditional_counterfactual_accuracy_originally_correct"),
            conditional,
        )
    summary_per_class = summary.get("per_class_fcv_metrics")
    if not isinstance(summary_per_class, Mapping) or set(summary_per_class) != set(
        recomputed["per_class"]
    ):
        raise FCVScoringError("FCV per-class metric keys do not reproduce.")
    for label, expected_metrics in recomputed["per_class"].items():
        observed_metrics = summary_per_class[label]
        if int(observed_metrics.get("eligible_count", -1)) != int(
            expected_metrics["eligible_count"]
        ):
            raise FCVScoringError("FCV per-class eligible counts do not reproduce.")
        for metric_name in (
            "original_accuracy",
            "counterfactual_draw_accuracy",
            "counterfactual_true_class_probability",
        ):
            _assert_close(
                f"per_class.{label}.{metric_name}",
                observed_metrics.get(metric_name),
                expected_metrics[metric_name],
            )
    observed_swap = summary.get("real_swap_integrity_diagnostics")
    if not isinstance(observed_swap, Mapping):
        raise FCVScoringError("FCV summary lacks real-swap integrity diagnostics.")
    for key in (
        "foreground_token_max_abs_error",
        "donor_reconstruction_max_abs_error",
        "replaced_token_changed_fraction",
        "replacement_delta_mean",
        "replacement_delta_max",
    ):
        _assert_close(f"real_swap.{key}", observed_swap.get(key), recomputed["swap_diagnostics"][key])
    for key in (
        "replaced_token_draw_count",
        "replaced_token_changed_count",
        "eligible_intervention_count",
    ):
        if int(observed_swap.get(key, -1)) != int(recomputed["swap_diagnostics"][key]):
            raise FCVScoringError(f"FCV real-swap diagnostic {key} does not reproduce.")
    observed_token_diagnostics = summary.get("token_distribution_diagnostics")
    if not isinstance(observed_token_diagnostics, Mapping) or not isinstance(
        observed_token_diagnostics.get("global_means"), Mapping
    ):
        raise FCVScoringError("FCV summary lacks token-distribution diagnostics.")
    for key, expected in recomputed["token_distribution_global_means"].items():
        _assert_close(
            f"token_distribution.global_means.{key}",
            observed_token_diagnostics["global_means"].get(key),
            expected,
        )
    return recomputed


def score_candidate_fcv(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    source: TokenBankSource,
    token_bank_dir: str | Path,
    donor_plan_path: str | Path,
    output_dir: str | Path,
    *,
    reconstruction_reports: Mapping[str, Any],
    device: str | torch.device = "cuda",
    counterfactual_forward_batch_size: int = 256,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Score one candidate with cached opposite-context token substitutions."""

    if counterfactual_forward_batch_size <= 0:
        raise FCVScoringError("counterfactual_forward_batch_size must be positive.")
    if counterfactual_forward_batch_size != int(
        config["execution"]["fcv_counterfactual_forward_batch_size"]
    ):
        raise FCVScoringError(
            "FCV counterfactual forward batch size differs from the locked value."
        )
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    token_bank_dir = Path(token_bank_dir).expanduser().resolve()
    donor_plan_path = Path(donor_plan_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not donor_plan_path.is_file():
        raise FileNotFoundError(f"Missing cached donor plan: {donor_plan_path}")
    model, checkpoint = load_candidate_model(config, checkpoint_path, device=device)
    candidate_id = str(checkpoint.get("candidate_id", ""))
    if not candidate_id or Path(candidate_id).name != candidate_id:
        raise FCVScoringError(f"Unsafe candidate ID: {candidate_id!r}")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    banks = {
        label: load_background_bank(
            config,
            token_bank_dir / f"{candidate_id}_{context_name}.pt",
            source,
            expected_label=label,
            expected_candidate_id=candidate_id,
            expected_checkpoint_sha256=checkpoint_sha256,
        )
        for label, context_name in CONTEXT_NAMES.items()
    }
    donor_plan = _load_trusted_torch_artifact(donor_plan_path)
    plan_by_sample_id = validate_opposite_donor_plan(
        config, source, banks, donor_plan
    )
    donor_plan_sha256 = _sha256_file(donor_plan_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / f"{candidate_id}.csv"
    summary_path = output_dir / f"{candidate_id}_summary.json"
    if summary_path.is_file() and not overwrite:
        with summary_path.open("r", encoding="utf-8") as handle:
            existing_summary = json.load(handle)
        if _score_summary_is_reusable(
            existing_summary,
            config=config,
            candidate_id=candidate_id,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            source=source,
            donor_plan_path=donor_plan_path,
            donor_plan_sha256=donor_plan_sha256,
            reconstruction_reports=reconstruction_reports,
            counterfactual_forward_batch_size=counterfactual_forward_batch_size,
        ):
            existing_summary = dict(existing_summary)
            existing_summary["status"] = "reused"
            return existing_summary
        raise FCVScoringError(
            f"Existing FCV score summary is stale: {summary_path}. "
            "Use --overwrite to replace it."
        )

    model.eval()
    model_device = torch.device(device)
    bank_tokens = {
        label: bank["tokens"].to(model_device, non_blocking=True)
        for label, bank in banks.items()
    }
    rows: List[Dict[str, Any]] = []
    donor_samples = int(config["fcv"]["donor_samples_per_image"])
    num_classes = int(config["model"]["num_classes"])
    reconstruction_max_abs_error = 0.0
    identity_swap_max_abs_error = 0.0
    identity_swap_inside_token_max_abs_error = 0.0
    identity_swap_outside_token_max_abs_error = 0.0
    identity_swap_path_sample_count = 0
    real_swap_foreground_max_abs_error = 0.0
    real_swap_donor_reconstruction_max_abs_error = 0.0
    real_swap_replaced_token_draw_count = 0
    real_swap_replaced_token_changed_count = 0
    real_swap_replacement_delta_sum = 0.0
    real_swap_replacement_delta_max = 0.0
    original_loss_values: List[float] = []
    with torch.inference_mode():
        for images, labels, sample_ids in source.loader:
            images = images.to(model_device, non_blocking=True)
            labels_device = labels.to(model_device, dtype=torch.long, non_blocking=True)
            raw_tokens = extract_raw_patch_tokens(model, images).float()
            normal_logits = model(images).float()
            reconstructed_logits = forward_from_patch_tokens(model, raw_tokens).float()
            reconstruction_max_abs_error = max(
                reconstruction_max_abs_error,
                float((normal_logits - reconstructed_logits).abs().max().item()),
            )
            original_logits = normal_logits
            # Production-path null intervention: call the exact FCV replacement
            # function while using this target's own raw background values as
            # the donor bank. This catches replacement-index/path regressions.
            identity_items: List[torch.Tensor] = []
            for batch_index, sample_id_value in enumerate(sample_ids):
                identity_indices = source.records_by_sample_id[
                    str(sample_id_value)
                ]["background_idx"].to(dtype=torch.long, device=model_device)
                target = raw_tokens[batch_index]
                if identity_indices.numel() == 0:
                    identity_items.append(target.clone())
                    continue
                replaced = make_counterfactual_token_batch(
                    target,
                    identity_indices,
                    target,
                    identity_indices.unsqueeze(0),
                )[0]
                inside_error = float(
                    (
                        replaced.index_select(0, identity_indices)
                        - target.index_select(0, identity_indices)
                    )
                    .abs()
                    .max()
                    .item()
                )
                outside_mask = torch.ones(
                    target.shape[0], dtype=torch.bool, device=model_device
                )
                outside_mask[identity_indices] = False
                outside_error = (
                    float((replaced[outside_mask] - target[outside_mask]).abs().max().item())
                    if bool(outside_mask.any())
                    else 0.0
                )
                identity_swap_inside_token_max_abs_error = max(
                    identity_swap_inside_token_max_abs_error, inside_error
                )
                identity_swap_outside_token_max_abs_error = max(
                    identity_swap_outside_token_max_abs_error, outside_error
                )
                identity_swap_path_sample_count += 1
                identity_items.append(replaced)
            identity_tokens = torch.stack(identity_items, dim=0)
            identity_logits = forward_from_patch_tokens(model, identity_tokens).float()
            identity_swap_max_abs_error = max(
                identity_swap_max_abs_error,
                float((normal_logits - identity_logits).abs().max().item()),
            )
            batch_original_losses = (
                F.cross_entropy(original_logits, labels_device, reduction="none")
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            )
            original_loss_values.extend(float(value) for value in batch_original_losses)
            original_probabilities = original_logits.float().softmax(dim=1)
            original_predictions = original_logits.argmax(dim=1)

            counterfactual_batches: List[torch.Tensor] = []
            eligible_batch_items: List[tuple[int, Mapping[str, Any]]] = []
            batch_rows: List[Dict[str, Any]] = []
            for batch_index, sample_id_value in enumerate(sample_ids):
                sample_id = str(sample_id_value)
                label = int(labels[batch_index].item())
                patch_record = source.records_by_sample_id[sample_id]
                plan_record = plan_by_sample_id[sample_id]
                coverage = patch_record.get("coverage", {})
                original_probability = float(
                    original_probabilities[batch_index, label].item()
                )
                original_prediction = int(original_predictions[batch_index].item())
                row: Dict[str, Any] = {
                    "candidate_id": candidate_id,
                    "sample_id": sample_id,
                    "metadata_index": int(patch_record["metadata_index"]),
                    "label": label,
                    "fcv_eligible": bool(patch_record["fcv_eligible"]),
                    "eligibility_reason": str(
                        patch_record.get("eligibility_reason", "unknown")
                    ),
                    "p_y_original": original_probability,
                    "original_cross_entropy": float(
                        batch_original_losses[batch_index]
                    ),
                    "pred_original": original_prediction,
                    "correct_original": int(original_prediction == label),
                    "p_y_counterfactual_mean": None,
                    "p_y_counterfactual_std": None,
                    "pred_counterfactual_majority": None,
                    "correct_counterfactual_majority": None,
                    "counterfactual_draw_accuracy": None,
                    "counterfactual_correct_draws": None,
                    "counterfactual_confidence_drop": None,
                    "num_background_patches_swapped": int(
                        plan_record["background_idx"].numel()
                    ),
                    "background_coverage": float(coverage.get("background_frac", 0.0)),
                    "evidence_coverage": float(coverage.get("evidence_frac", 0.0)),
                    "ambiguous_coverage": float(coverage.get("ambiguous_frac", 0.0)),
                    "donor_context_label": int(plan_record["donor_context_label"]),
                    "donor_context_name": str(plan_record["donor_context_name"]),
                    "donor_draw_count": donor_samples,
                    "p_y_counterfactual_draws": None,
                    "pred_counterfactual_draws": None,
                    "target_background_token_norm_mean": None,
                    "donor_token_norm_mean": None,
                    "target_donor_cosine_similarity_mean": None,
                    "target_nearest_donor_cosine_mean": None,
                    "donor_unique_source_images": None,
                    "donor_max_source_fraction": None,
                    "real_swap_foreground_max_abs_error": None,
                    "real_swap_donor_reconstruction_max_abs_error": None,
                    "real_swap_replaced_token_draw_count": None,
                    "real_swap_replaced_token_changed_count": None,
                    "real_swap_replaced_token_changed_fraction": None,
                    "real_swap_replacement_delta_mean": None,
                    "real_swap_replacement_delta_max": None,
                }
                batch_rows.append(row)
                if bool(patch_record["fcv_eligible"]):
                    donor_label = int(plan_record["donor_context_label"])
                    donor_indices = plan_record["donor_token_indices"].to(
                        dtype=torch.long, device=model_device
                    )
                    target_background = raw_tokens[batch_index].index_select(
                        0,
                        plan_record["background_idx"].to(
                            dtype=torch.long, device=model_device
                        ),
                    )
                    sampled_donors = bank_tokens[donor_label].index_select(
                        0, donor_indices.flatten()
                    )
                    repeated_targets = target_background.repeat(donor_samples, 1)
                    paired_cosine = F.cosine_similarity(
                        repeated_targets, sampled_donors, dim=1
                    )
                    target_normalized = F.normalize(target_background, dim=1)
                    donor_normalized = F.normalize(sampled_donors, dim=1)
                    nearest_cosine = (
                        target_normalized @ donor_normalized.transpose(0, 1)
                    ).max(dim=1).values
                    donor_source_indices = banks[donor_label][
                        "token_source_image_index"
                    ].to(torch.long).index_select(0, donor_indices.cpu().flatten())
                    source_counts = torch.bincount(donor_source_indices)
                    nonzero_source_counts = source_counts[source_counts > 0]
                    row.update(
                        {
                            "target_background_token_norm_mean": float(
                                target_background.norm(dim=1).mean().item()
                            ),
                            "donor_token_norm_mean": float(
                                sampled_donors.norm(dim=1).mean().item()
                            ),
                            "target_donor_cosine_similarity_mean": float(
                                paired_cosine.mean().item()
                            ),
                            "target_nearest_donor_cosine_mean": float(
                                nearest_cosine.mean().item()
                            ),
                            "donor_unique_source_images": int(
                                nonzero_source_counts.numel()
                            ),
                            "donor_max_source_fraction": float(
                                nonzero_source_counts.max().item()
                                / donor_source_indices.numel()
                            ),
                        }
                    )
                    counterfactual_tokens = make_counterfactual_token_batch(
                            raw_tokens[batch_index],
                            plan_record["background_idx"],
                            bank_tokens[donor_label],
                            plan_record["donor_token_indices"],
                    )
                    background_idx = plan_record["background_idx"].to(
                        dtype=torch.long, device=model_device
                    )
                    expected_donors = sampled_donors.reshape(
                        donor_samples, len(background_idx), raw_tokens.shape[-1]
                    ).to(
                        device=model_device,
                        dtype=raw_tokens.dtype,
                    )
                    donor_error = float(
                        (
                            counterfactual_tokens.index_select(1, background_idx)
                            - expected_donors
                        ).abs().max().item()
                    )
                    foreground_mask = torch.ones(
                        raw_tokens.shape[1], dtype=torch.bool, device=model_device
                    )
                    foreground_mask[background_idx] = False
                    foreground_error = (
                        float(
                            (
                                counterfactual_tokens[:, foreground_mask, :]
                                - raw_tokens[batch_index, foreground_mask, :].unsqueeze(0)
                            ).abs().max().item()
                        )
                        if bool(foreground_mask.any())
                        else 0.0
                    )
                    replacement_delta = (
                        counterfactual_tokens.index_select(1, background_idx)
                        - target_background.unsqueeze(0)
                    ).norm(dim=2)
                    replaced_count = int(replacement_delta.numel())
                    changed_count = int((replacement_delta != 0.0).sum().item())
                    delta_sum = float(replacement_delta.sum().item())
                    delta_max = float(replacement_delta.max().item())
                    if foreground_error != 0.0 or donor_error != 0.0:
                        raise FCVScoringError(
                            "Real opposite-context replacement violated token integrity: "
                            f"sample={sample_id} foreground={foreground_error:.3e} "
                            f"donor={donor_error:.3e}."
                        )
                    row.update(
                        {
                            "real_swap_foreground_max_abs_error": foreground_error,
                            "real_swap_donor_reconstruction_max_abs_error": donor_error,
                            "real_swap_replaced_token_draw_count": replaced_count,
                            "real_swap_replaced_token_changed_count": changed_count,
                            "real_swap_replaced_token_changed_fraction": changed_count
                            / replaced_count,
                            "real_swap_replacement_delta_mean": delta_sum
                            / replaced_count,
                            "real_swap_replacement_delta_max": delta_max,
                        }
                    )
                    real_swap_foreground_max_abs_error = max(
                        real_swap_foreground_max_abs_error, foreground_error
                    )
                    real_swap_donor_reconstruction_max_abs_error = max(
                        real_swap_donor_reconstruction_max_abs_error, donor_error
                    )
                    real_swap_replaced_token_draw_count += replaced_count
                    real_swap_replaced_token_changed_count += changed_count
                    real_swap_replacement_delta_sum += delta_sum
                    real_swap_replacement_delta_max = max(
                        real_swap_replacement_delta_max, delta_max
                    )
                    counterfactual_batches.append(counterfactual_tokens)
                    eligible_batch_items.append((len(batch_rows) - 1, plan_record))

            if counterfactual_batches:
                all_counterfactual_tokens = torch.cat(counterfactual_batches, dim=0)
                logits_chunks = []
                for start in range(
                    0,
                    len(all_counterfactual_tokens),
                    counterfactual_forward_batch_size,
                ):
                    logits_chunks.append(
                        forward_from_patch_tokens(
                            model,
                            all_counterfactual_tokens[
                                start : start + counterfactual_forward_batch_size
                            ],
                        ).float()
                    )
                counterfactual_logits = torch.cat(logits_chunks, dim=0)
                expected = len(eligible_batch_items) * donor_samples
                if len(counterfactual_logits) != expected:
                    raise FCVScoringError("Counterfactual output count is inconsistent.")
                offset = 0
                for row_index, _ in eligible_batch_items:
                    row = batch_rows[row_index]
                    label = int(row["label"])
                    draw_logits = counterfactual_logits[offset : offset + donor_samples]
                    offset += donor_samples
                    draw_probabilities = draw_logits.softmax(dim=1)
                    true_class_probabilities = draw_probabilities[:, label]
                    draw_predictions = draw_logits.argmax(dim=1)
                    correct_draws = int((draw_predictions == label).sum().item())
                    prediction_counts = torch.bincount(
                        draw_predictions, minlength=num_classes
                    )
                    majority_prediction = int(prediction_counts.argmax().item())
                    (
                        probability_values,
                        mean_probability,
                        std_probability,
                    ) = _canonical_probability_draw_statistics(
                        true_class_probabilities
                    )
                    row.update(
                        {
                            "p_y_counterfactual_mean": mean_probability,
                            "p_y_counterfactual_std": std_probability,
                            "pred_counterfactual_majority": majority_prediction,
                            "correct_counterfactual_majority": int(
                                majority_prediction == label
                            ),
                            "counterfactual_draw_accuracy": correct_draws
                            / donor_samples,
                            "counterfactual_correct_draws": correct_draws,
                            "counterfactual_confidence_drop": float(
                                row["p_y_original"] - mean_probability
                            ),
                            "p_y_counterfactual_draws": json.dumps(
                                probability_values,
                                separators=(",", ":"),
                            ),
                            "pred_counterfactual_draws": json.dumps(
                                [int(value) for value in draw_predictions.cpu().tolist()],
                                separators=(",", ":"),
                            ),
                        }
                    )
            rows.extend(batch_rows)

    frame = pd.DataFrame(rows)
    if len(frame) != source.sample_count or frame["sample_id"].duplicated().any():
        raise FCVScoringError("Candidate score rows do not cover validation exactly once.")
    expected_order = _source_frame(source)["sample_id"].astype(str).tolist()
    if frame["sample_id"].astype(str).tolist() != expected_order:
        raise FCVScoringError("Candidate score row order differs from the public manifest.")
    eligible = frame[frame["fcv_eligible"]].copy()
    if eligible.empty:
        raise FCVScoringError("No validation samples are eligible for FCV scoring.")
    originally_correct_eligible = eligible[eligible["correct_original"] == 1]
    original_accuracy = float(frame["correct_original"].mean())
    original_loss = _mean(original_loss_values)
    checkpoint_metrics = checkpoint.get("metrics", {})
    checkpoint_accuracy = float(checkpoint_metrics.get("biased_val_accuracy", float("nan")))
    checkpoint_loss = float(checkpoint_metrics.get("biased_val_loss", float("nan")))
    if original_accuracy != checkpoint_accuracy:
        raise FCVScoringError(
            "Float32 ordinary validation accuracy differs between Step 4 and Step 7: "
            f"{checkpoint_accuracy} versus {original_accuracy}."
        )
    if abs(original_loss - checkpoint_loss) > 1.0e-6:
        raise FCVScoringError(
            "Float32 ordinary validation loss differs between Step 4 and Step 7: "
            f"{checkpoint_loss} versus {original_loss}."
        )
    if reconstruction_max_abs_error >= 1.0e-5:
        raise FCVScoringError(
            "Normal and resumed ViT forwards differ by "
            f"{reconstruction_max_abs_error:.3e}, exceeding 1e-5."
        )
    if identity_swap_max_abs_error >= 1.0e-5:
        raise FCVScoringError(
            "Target-token identity swap changed logits by "
            f"{identity_swap_max_abs_error:.3e}, exceeding 1e-5."
        )
    if identity_swap_path_sample_count == 0:
        raise FCVScoringError("Identity swap never exercised a non-empty replacement mask.")
    if (
        identity_swap_inside_token_max_abs_error != 0.0
        or identity_swap_outside_token_max_abs_error != 0.0
    ):
        raise FCVScoringError(
            "Production-path identity replacement changed raw token values: "
            f"inside={identity_swap_inside_token_max_abs_error:.3e}, "
            f"outside={identity_swap_outside_token_max_abs_error:.3e}."
        )
    if real_swap_foreground_max_abs_error != 0.0:
        raise FCVScoringError("Real FCV swaps changed foreground token values.")
    if real_swap_donor_reconstruction_max_abs_error != 0.0:
        raise FCVScoringError("Real FCV swaps did not reconstruct selected donor tokens.")
    if (
        real_swap_replaced_token_draw_count <= 0
        or real_swap_replaced_token_changed_count <= 0
    ):
        raise FCVScoringError("The real opposite-context intervention cohort was a no-op.")
    fcv_draw_accuracy = float(eligible["counterfactual_draw_accuracy"].mean())
    fcv_majority_accuracy = float(eligible["correct_counterfactual_majority"].mean())
    weights = config["fcv"]["primary_selector"]
    primary_score = (
        float(weights["original_accuracy_weight"]) * original_accuracy
        + float(weights["counterfactual_accuracy_weight"]) * fcv_draw_accuracy
    )

    _atomic_csv(frame, score_path)
    score_sha256 = _sha256_file(score_path)
    per_class_metrics = {}
    for label, class_frame in eligible.groupby("label", sort=True):
        per_class_metrics[str(int(label))] = {
            "eligible_count": int(len(class_frame)),
            "original_accuracy": float(class_frame["correct_original"].mean()),
            "counterfactual_draw_accuracy": float(
                class_frame["counterfactual_draw_accuracy"].mean()
            ),
            "counterfactual_true_class_probability": float(
                class_frame["p_y_counterfactual_mean"].mean()
            ),
        }
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_candidate_score_summary",
        "status": "complete",
        "candidate_id": candidate_id,
        "run": dict(checkpoint["run"]),
        "epoch": int(checkpoint["epoch"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "training_fingerprint": checkpoint["training_fingerprint"],
        "campaign_provenance_path": checkpoint.get(
            "campaign_provenance_path"
        ),
        "campaign_provenance_sha256": checkpoint.get(
            "campaign_provenance_sha256"
        ),
        "campaign_bindings_sha256": checkpoint.get(
            "campaign_bindings_sha256"
        ),
        "pretrained_provenance_path": checkpoint.get(
            "pretrained_provenance_path"
        ),
        "pretrained_provenance_sha256": checkpoint.get(
            "pretrained_provenance_sha256"
        ),
        "pretrained_backbone_sha256": checkpoint.get(
            "pretrained_backbone_sha256"
        ),
        "initial_model_state_sha256": checkpoint.get(
            "initial_model_state_sha256"
        ),
        "software_versions": software_versions(),
        "software_fingerprint": software_fingerprint(),
        "validation_manifest_path": str(source.manifest_path),
        "validation_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_path": str(source.manifest_bundle_path),
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "patch_mask_path": str(source.patch_mask_path),
        "patch_mask_sha256": source.patch_mask_sha256,
        "patch_mask_summary_path": str(source.patch_mask_summary_path),
        "patch_mask_summary_sha256": source.patch_mask_summary_sha256,
        "patch_mask_preprocessing_sha256": source.patch_mask_preprocessing_sha256,
        "teacher_maps_sha256": source.teacher_maps_sha256,
        "reconstruction_reports": dict(reconstruction_reports),
        "execution": {
            "validation_batch_size": source.batch_size,
            "validation_num_workers": source.num_workers,
            "counterfactual_forward_batch_size": counterfactual_forward_batch_size,
        },
        "donor_plan_path": str(donor_plan_path),
        "donor_plan_sha256": donor_plan_sha256,
        "donor_plan_content_sha256": donor_plan["plan_content_sha256"],
        "donor_samples_per_image": donor_samples,
        "validation_sample_count": source.sample_count,
        "fcv_eligible_sample_count": int(len(eligible)),
        "fcv_ineligible_sample_count": int(source.sample_count - len(eligible)),
        "fcv_eligible_fraction": float(len(eligible) / source.sample_count),
        "biased_validation_accuracy_recomputed": original_accuracy,
        "biased_validation_accuracy_checkpoint": checkpoint_accuracy,
        "biased_validation_loss_recomputed": original_loss,
        "biased_validation_loss_checkpoint": checkpoint_loss,
        "normal_vs_resumed_max_abs_error": reconstruction_max_abs_error,
        "identity_swap_max_abs_error": identity_swap_max_abs_error,
        "identity_swap_inside_token_max_abs_error": (
            identity_swap_inside_token_max_abs_error
        ),
        "identity_swap_outside_token_max_abs_error": (
            identity_swap_outside_token_max_abs_error
        ),
        "identity_swap_path_sample_count": identity_swap_path_sample_count,
        "identity_swap_uses_production_replacement_path": True,
        "real_swap_integrity_diagnostics": {
            "foreground_token_max_abs_error": real_swap_foreground_max_abs_error,
            "donor_reconstruction_max_abs_error": (
                real_swap_donor_reconstruction_max_abs_error
            ),
            "replaced_token_draw_count": real_swap_replaced_token_draw_count,
            "replaced_token_changed_count": real_swap_replaced_token_changed_count,
            "replaced_token_changed_fraction": (
                real_swap_replaced_token_changed_count
                / real_swap_replaced_token_draw_count
            ),
            "replacement_delta_mean": (
                real_swap_replacement_delta_sum
                / real_swap_replaced_token_draw_count
            ),
            "replacement_delta_max": real_swap_replacement_delta_max,
            "eligible_intervention_count": int(len(eligible)),
        },
        "opposite_context_counterfactual_accuracy": fcv_draw_accuracy,
        "opposite_context_counterfactual_majority_accuracy": fcv_majority_accuracy,
        "opposite_context_true_class_probability": float(
            eligible["p_y_counterfactual_mean"].mean()
        ),
        "original_true_class_probability": float(frame["p_y_original"].mean()),
        "mean_counterfactual_confidence_drop": float(
            eligible["counterfactual_confidence_drop"].mean()
        ),
        "per_class_fcv_metrics": per_class_metrics,
        "token_distribution_diagnostics": {
            "global_means": {
                column: float(eligible[column].mean())
                for column in (
                    "target_background_token_norm_mean",
                    "donor_token_norm_mean",
                    "target_donor_cosine_similarity_mean",
                    "target_nearest_donor_cosine_mean",
                    "donor_unique_source_images",
                    "donor_max_source_fraction",
                )
            },
            "global_distributions": {
                column: _numeric_distribution(eligible[column])
                for column in (
                    "target_background_token_norm_mean",
                    "donor_token_norm_mean",
                    "target_donor_cosine_similarity_mean",
                    "target_nearest_donor_cosine_mean",
                    "donor_unique_source_images",
                    "donor_max_source_fraction",
                )
            },
            "per_class": {
                str(int(label)): {
                    column: _numeric_distribution(class_frame[column])
                    for column in (
                        "target_background_token_norm_mean",
                        "donor_token_norm_mean",
                        "target_donor_cosine_similarity_mean",
                        "target_nearest_donor_cosine_mean",
                        "donor_unique_source_images",
                        "donor_max_source_fraction",
                    )
                }
                for label, class_frame in eligible.groupby("label", sort=True)
            },
        },
        "conditional_counterfactual_accuracy_originally_correct": (
            float(originally_correct_eligible["counterfactual_draw_accuracy"].mean())
            if not originally_correct_eligible.empty
            else None
        ),
        "conditional_counterfactual_sample_count": int(
            len(originally_correct_eligible)
        ),
        "primary_selector_name": str(weights["name"]),
        "primary_selector_score": float(primary_score),
        "score_csv_path": str(score_path),
        "score_csv_size_bytes": score_path.stat().st_size,
        "score_csv_sha256": score_sha256,
        "token_banks": {
            CONTEXT_NAMES[label]: {
                "path": str(banks[label]["artifact_path"]),
                "sha256": str(banks[label]["artifact_sha256"]),
                "layout_sha256": str(banks[label]["layout_sha256"]),
                "token_count": int(banks[label]["token_count"]),
            }
            for label in CONTEXT_NAMES
        },
    }
    _atomic_json(summary, summary_path)
    del model, checkpoint, banks, bank_tokens, donor_plan, plan_by_sample_id
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def aggregate_fcv_score_summaries(
    config: Mapping[str, Any],
    score_dir: str | Path,
    output_csv: str | Path,
    output_summary: str | Path,
    *,
    source: TokenBankSource,
    donor_plan_path: str | Path,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Validate and index all Step 7 candidate summaries without test access."""

    score_dir = Path(score_dir).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    output_summary = Path(output_summary).expanduser().resolve()
    donor_plan_path = Path(donor_plan_path).expanduser().resolve()
    donor_plan_sha256 = _sha256_file(donor_plan_path)
    expected_fingerprint = candidate_training_fingerprint(config)
    selected_epochs = candidate_epochs(config)
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    invalid: List[Dict[str, str]] = []
    for run in enumerate_sweep_runs(config):
        for epoch in selected_epochs:
            candidate_id = run.candidate_id(epoch)
            summary_path = score_dir / f"{candidate_id}_summary.json"
            if not summary_path.is_file():
                missing.append(candidate_id)
                continue
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                valid = (
                    summary.get("artifact_type") == "fcv_vit_candidate_score_summary"
                    and summary.get("schema_version") == 1
                    and summary.get("status") == "complete"
                    and summary.get("candidate_id") == candidate_id
                    and summary.get("training_fingerprint") == expected_fingerprint
                    and summary.get("validation_manifest_sha256") == source.manifest_sha256
                    and summary.get("manifest_bundle_sha256")
                    == source.manifest_bundle_sha256
                    and summary.get("patch_mask_sha256") == source.patch_mask_sha256
                    and summary.get("patch_mask_summary_sha256")
                    == source.patch_mask_summary_sha256
                    and summary.get("patch_mask_preprocessing_sha256")
                    == source.patch_mask_preprocessing_sha256
                    and summary.get("teacher_maps_sha256")
                    == source.teacher_maps_sha256
                    and summary.get("software_versions") == software_versions()
                    and summary.get("software_fingerprint") == software_fingerprint()
                    and summary.get("donor_plan_path") == str(donor_plan_path)
                    and summary.get("donor_plan_sha256") == donor_plan_sha256
                    and int(summary.get("validation_sample_count", -1))
                    == source.sample_count
                    and summary.get("execution")
                    == {
                        "validation_batch_size": source.batch_size,
                        "validation_num_workers": source.num_workers,
                        "counterfactual_forward_batch_size": int(
                            config["execution"][
                                "fcv_counterfactual_forward_batch_size"
                            ]
                        ),
                    }
                )
                score_path = Path(str(summary.get("score_csv_path", "")))
                checkpoint_path = Path(str(summary.get("checkpoint_path", "")))
                checkpoint_sha256 = str(summary.get("checkpoint_sha256", ""))
                token_banks = summary.get("token_banks", {})
                current_banks_valid = isinstance(token_banks, Mapping)
                if current_banks_valid:
                    for bank_details in token_banks.values():
                        if not isinstance(bank_details, Mapping):
                            current_banks_valid = False
                            break
                        bank_path = Path(str(bank_details.get("path", "")))
                        current_banks_valid = bool(
                            bank_path.is_file()
                            and _sha256_file(bank_path) == bank_details.get("sha256")
                        )
                        if not current_banks_valid:
                            break
                if not current_banks_valid and isinstance(token_banks, Mapping):
                    receipt_path = (
                        Path(config["paths"]["output_root"])
                        / config["outputs"]["token_banks"]
                        / "cleanup_receipts"
                        / f"{candidate_id}.json"
                    )
                    current_banks_valid = validate_cleanup_receipt(
                        receipt_path,
                        candidate_id=candidate_id,
                        checkpoint_path=checkpoint_path,
                        checkpoint_sha256=checkpoint_sha256,
                        training_fingerprint=expected_fingerprint,
                        token_banks=token_banks,
                    ) is not None
                if not valid or (
                    not score_path.is_file()
                    or score_path.stat().st_size
                    != int(summary.get("score_csv_size_bytes", -1))
                    or _sha256_file(score_path) != summary.get("score_csv_sha256")
                    or not checkpoint_path.is_file()
                    or _sha256_file(checkpoint_path) != checkpoint_sha256
                    or not current_banks_valid
                ):
                    raise FCVScoringError("stale summary or score CSV provenance")
                score_frame = pd.read_csv(score_path)
                recomputed = validate_fcv_summary_against_frame(
                    summary, score_frame, config
                )
                if recomputed["sample_count"] != source.sample_count:
                    raise FCVScoringError(
                        "FCV score CSV does not cover the frozen validation source."
                    )
                rows.append(
                    {
                        "run_index": run.run_index,
                        "candidate_id": candidate_id,
                        "epoch": epoch,
                        "seed": run.seed,
                        "learning_rate": run.learning_rate,
                        "weight_decay": run.weight_decay,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha256,
                        "biased_validation_accuracy": recomputed["original_accuracy"],
                        "fcv_counterfactual_accuracy": recomputed[
                            "counterfactual_accuracy"
                        ],
                        "fcv_counterfactual_majority_accuracy": recomputed[
                            "counterfactual_majority_accuracy"
                        ],
                        "fcv_true_class_probability": recomputed[
                            "counterfactual_true_class_probability"
                        ],
                        "fcv_confidence_drop": recomputed["mean_confidence_drop"],
                        "fcv_conditional_accuracy_originally_correct": recomputed[
                            "conditional_accuracy_originally_correct"
                        ],
                        "fcv_eligible_sample_count": recomputed[
                            "eligible_sample_count"
                        ],
                        "primary_selector_score": recomputed[
                            "primary_selector_score"
                        ],
                        "score_csv_path": str(score_path),
                        "score_csv_sha256": str(summary["score_csv_sha256"]),
                        "token_bank_land_sha256": str(
                            token_banks[CONTEXT_NAMES[0]]["sha256"]
                        ),
                        "token_bank_water_sha256": str(
                            token_banks[CONTEXT_NAMES[1]]["sha256"]
                        ),
                        "per_class_fcv_metrics_json": json.dumps(
                            recomputed["per_class"], sort_keys=True
                        ),
                        "identity_swap_max_abs_error": float(
                            summary["identity_swap_max_abs_error"]
                        ),
                        "identity_swap_inside_token_max_abs_error": float(
                            summary["identity_swap_inside_token_max_abs_error"]
                        ),
                        "identity_swap_outside_token_max_abs_error": float(
                            summary["identity_swap_outside_token_max_abs_error"]
                        ),
                        "identity_swap_path_sample_count": int(
                            summary["identity_swap_path_sample_count"]
                        ),
                        "real_swap_integrity_diagnostics_json": json.dumps(
                            summary["real_swap_integrity_diagnostics"], sort_keys=True
                        ),
                        "token_distribution_diagnostics_json": json.dumps(
                            summary["token_distribution_diagnostics"], sort_keys=True
                        ),
                        "summary_path": str(summary_path),
                    }
                )
            except (OSError, ValueError, KeyError, TypeError, FCVScoringError) as exc:
                invalid.append({"candidate_id": candidate_id, "error": str(exc)})

    if (missing or invalid) and not allow_incomplete:
        raise FCVScoringError(
            f"FCV score pool is incomplete: missing={len(missing)} "
            f"invalid={len(invalid)}."
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["run_index", "epoch"]).reset_index(drop=True)
        if frame["candidate_id"].duplicated().any():
            raise FCVScoringError("Duplicate candidate IDs in FCV score index.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(frame, output_csv)
    expected_count = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_score_pool_summary",
        "status": "complete" if not missing and not invalid else "incomplete",
        "candidate_count": len(frame),
        "expected_candidate_count": expected_count,
        "missing_candidate_count": len(missing),
        "missing_candidate_preview": missing[:10],
        "invalid_candidate_count": len(invalid),
        "invalid_candidate_preview": invalid[:10],
        "training_fingerprint": expected_fingerprint,
        "validation_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "patch_mask_sha256": source.patch_mask_sha256,
        "patch_mask_summary_sha256": source.patch_mask_summary_sha256,
        "patch_mask_preprocessing_sha256": source.patch_mask_preprocessing_sha256,
        "teacher_maps_sha256": source.teacher_maps_sha256,
        "software_versions": software_versions(),
        "software_fingerprint": software_fingerprint(),
        "donor_plan_path": str(donor_plan_path),
        "donor_plan_sha256": donor_plan_sha256,
        "output_csv": str(output_csv),
        "output_csv_sha256": _sha256_file(output_csv),
        "checkpoint_and_bank_hashes_bound": True,
        "selection_metrics_recomputed_from_hashed_csvs": True,
        "execution": {
            "validation_batch_size": source.batch_size,
            "validation_num_workers": source.num_workers,
            "counterfactual_forward_batch_size": int(
                config["execution"]["fcv_counterfactual_forward_batch_size"]
            ),
        },
        "identity_swap_candidate_failure_count": int(
            (
                (frame["identity_swap_max_abs_error"].astype(float) >= 1.0e-5)
                | (frame["identity_swap_inside_token_max_abs_error"].astype(float) != 0.0)
                | (frame["identity_swap_outside_token_max_abs_error"].astype(float) != 0.0)
                | (frame["identity_swap_path_sample_count"].astype(int) <= 0)
            ).sum()
        )
        if not frame.empty
        else 0,
        "identity_swap_max_abs_error_across_candidates": (
            float(frame["identity_swap_max_abs_error"].astype(float).max())
            if not frame.empty
            else None
        ),
        "token_diagnostic_candidate_count": int(
            frame["token_distribution_diagnostics_json"].notna().sum()
        )
        if not frame.empty
        else 0,
    }
    _atomic_json(summary, output_summary)
    return summary
