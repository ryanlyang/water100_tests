"""Essential semantic and corruption controls for ViT-FCV Step 8."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch

from .candidate_training import (
    candidate_training_fingerprint,
    enumerate_sweep_runs,
    software_fingerprint,
    software_versions,
)
from .config import candidate_epochs
from .fcv_scoring import (
    OPPOSITE_CONTEXT,
    load_background_bank,
    make_counterfactual_token_batch,
    validate_fcv_summary_against_frame,
    validate_opposite_donor_plan,
)
from .token_banks import CONTEXT_NAMES, TokenBankSource
from .vit_counterfactual_forward import (
    extract_raw_patch_tokens,
    forward_from_patch_tokens,
    load_candidate_model,
)


CONTROL_NAMES = (
    "same_context",
    "random_mask",
    "shuffled_mask",
    "evidence_swap",
)


class FCVControlError(ValueError):
    """Raised when a Step 8 control or its provenance is invalid."""


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
    """Return draw values and statistics using their persisted representation.

    CUDA reductions are allowed to accumulate float32 values in a different
    order across architectures.  In particular, GH200 can differ by more than
    the strict artifact-validation tolerance from NumPy's float64 mean over
    the values subsequently written to JSON.  The draw list is the canonical
    raw artifact, so derive every persisted statistic from that exact list.
    """

    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise FCVControlError(
            "Control probability draws must be a non-empty one-dimensional tensor."
        )
    values = [
        float(value)
        for value in probabilities.detach().to(device="cpu").tolist()
    ]
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or ((array < 0.0) | (array > 1.0)).any():
        raise FCVControlError("Control probability draws are invalid.")
    return values, float(array.mean()), float(array.std(ddof=0))


def recompute_control_metrics_from_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    """Recompute one control's aggregate metrics from its hashed draw CSV."""

    required = {
        "sample_id",
        "label",
        "control_eligible",
        "p_y_original",
        "pred_original",
        "correct_original",
        "donor_draw_count",
        "p_y_control_mean",
        "p_y_control_std",
        "pred_control_majority",
        "correct_control_majority",
        "control_draw_accuracy",
        "control_correct_draws",
        "control_confidence_drop",
        "num_positions_swapped",
        "p_y_control_draws",
        "pred_control_draws",
        "swap_preserved_token_max_abs_error",
        "swap_donor_reconstruction_max_abs_error",
        "swap_replaced_token_draw_count",
        "swap_replaced_token_changed_count",
        "swap_replaced_token_changed_fraction",
        "swap_replacement_delta_mean",
        "swap_replacement_delta_max",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FCVControlError(f"Control CSV is missing columns: {missing}")
    if frame.empty or frame["sample_id"].astype(str).duplicated().any():
        raise FCVControlError("Control CSV must contain unique non-empty samples.")
    eligible_mask = frame["control_eligible"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    original_probabilities = frame["p_y_original"].astype(float).to_numpy()
    if not np.isfinite(original_probabilities).all() or (
        (original_probabilities < 0.0) | (original_probabilities > 1.0)
    ).any():
        raise FCVControlError("Control original probabilities are invalid.")
    for index in frame.index:
        label = int(frame.at[index, "label"])
        prediction = int(frame.at[index, "pred_original"])
        if label not in {0, 1} or prediction not in {0, 1}:
            raise FCVControlError("Control labels or original predictions are invalid.")
        if int(frame.at[index, "correct_original"]) != int(prediction == label):
            raise FCVControlError("Control original correctness does not reproduce.")
        if not bool(eligible_mask.loc[index]):
            continue
        draw_count = int(frame.at[index, "donor_draw_count"])
        if draw_count <= 0:
            raise FCVControlError("Control donor_draw_count must be positive.")
        probabilities = np.asarray(
            json.loads(frame.at[index, "p_y_control_draws"]), dtype=np.float64
        )
        predictions = np.asarray(
            json.loads(frame.at[index, "pred_control_draws"]), dtype=np.int64
        )
        if len(probabilities) != draw_count or len(predictions) != draw_count:
            raise FCVControlError("Control draw arrays have stale lengths.")
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0.0) | (probabilities > 1.0)
        ).any():
            raise FCVControlError("Control draw probabilities are invalid.")
        if ((predictions < 0) | (predictions > 1)).any():
            raise FCVControlError("Control draw predictions are invalid.")
        correct_draws = int((predictions == label).sum())
        majority = int(np.bincount(predictions, minlength=2).argmax())
        probability_mean = float(probabilities.mean())
        probability_std = float(probabilities.std(ddof=0))
        expected_values = {
            "control_correct_draws": correct_draws,
            "pred_control_majority": majority,
            "correct_control_majority": int(majority == label),
        }
        for column, expected in expected_values.items():
            if int(frame.at[index, column]) != expected:
                raise FCVControlError(f"Control row field {column} does not reproduce.")
        floating = {
            "control_draw_accuracy": float(correct_draws / draw_count),
            "p_y_control_mean": probability_mean,
            "p_y_control_std": probability_std,
            "control_confidence_drop": float(
                float(frame.at[index, "p_y_original"]) - probability_mean
            ),
        }
        for column, expected in floating.items():
            if not np.isclose(
                float(frame.at[index, column]), expected, rtol=0.0, atol=1.0e-7
            ):
                raise FCVControlError(f"Control row field {column} does not reproduce.")
        preserved_error = float(
            frame.at[index, "swap_preserved_token_max_abs_error"]
        )
        donor_error = float(
            frame.at[index, "swap_donor_reconstruction_max_abs_error"]
        )
        replaced_count = int(frame.at[index, "swap_replaced_token_draw_count"])
        changed_count = int(frame.at[index, "swap_replaced_token_changed_count"])
        changed_fraction = float(
            frame.at[index, "swap_replaced_token_changed_fraction"]
        )
        delta_mean = float(frame.at[index, "swap_replacement_delta_mean"])
        delta_max = float(frame.at[index, "swap_replacement_delta_max"])
        if preserved_error != 0.0 or donor_error != 0.0:
            raise FCVControlError(
                "Control replacement violated preserved-token or donor integrity."
            )
        if replaced_count <= 0 or changed_count < 0 or changed_count > replaced_count:
            raise FCVControlError("Control replacement counts are invalid.")
        if not np.isclose(
            changed_fraction,
            changed_count / replaced_count,
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise FCVControlError("Control changed-token fraction does not reproduce.")
        if (
            not np.isfinite(delta_mean)
            or not np.isfinite(delta_max)
            or delta_mean < 0.0
            or delta_max < 0.0
            # Float32 parallel reduction can round a mean a few ulps above
            # the separately reduced maximum; allow only that numerical slack.
            or delta_mean > delta_max + 1.0e-6 * max(1.0, delta_max)
        ):
            raise FCVControlError(
                "Control replacement deltas are invalid: "
                f"sample={frame.at[index, 'sample_id']} "
                f"mean={delta_mean!r} max={delta_max!r}."
            )
    eligible = frame.loc[eligible_mask].copy()
    if eligible.empty:
        raise FCVControlError("Control CSV contains no eligible rows.")
    replaced_count = int(eligible["swap_replaced_token_draw_count"].astype(int).sum())
    changed_count = int(
        eligible["swap_replaced_token_changed_count"].astype(int).sum()
    )
    if replaced_count <= 0 or changed_count <= 0:
        raise FCVControlError("The control intervention cohort was a complete no-op.")
    weighted_delta_sum = float(
        (
            eligible["swap_replacement_delta_mean"].astype(float)
            * eligible["swap_replaced_token_draw_count"].astype(float)
        ).sum()
    )
    return {
        "sample_count": int(len(frame)),
        "original_accuracy": float(frame["correct_original"].astype(float).mean()),
        "eligible_sample_count": int(len(eligible)),
        "eligible_fraction": float(len(eligible) / len(frame)),
        "counterfactual_accuracy": float(
            eligible["control_draw_accuracy"].astype(float).mean()
        ),
        "counterfactual_majority_accuracy": float(
            eligible["correct_control_majority"].astype(float).mean()
        ),
        "true_class_probability": float(
            eligible["p_y_control_mean"].astype(float).mean()
        ),
        "mean_confidence_drop": float(
            eligible["control_confidence_drop"].astype(float).mean()
        ),
        "mean_positions_swapped": float(
            eligible["num_positions_swapped"].astype(float).mean()
        ),
        "swap_preserved_token_max_abs_error": float(
            eligible["swap_preserved_token_max_abs_error"].astype(float).max()
        ),
        "swap_donor_reconstruction_max_abs_error": float(
            eligible["swap_donor_reconstruction_max_abs_error"].astype(float).max()
        ),
        "swap_replaced_token_draw_count": replaced_count,
        "swap_replaced_token_changed_count": changed_count,
        "swap_replaced_token_changed_fraction": float(
            changed_count / replaced_count
        ),
        "swap_replacement_delta_mean": float(weighted_delta_sum / replaced_count),
        "swap_replacement_delta_max": float(
            eligible["swap_replacement_delta_max"].astype(float).max()
        ),
    }


def _validate_metric_mapping(
    observed: Mapping[str, Any], expected: Mapping[str, Any], context: str
) -> None:
    for key, value in expected.items():
        if key in {"sample_count", "original_accuracy"}:
            continue
        if key in {
            "eligible_sample_count",
            "swap_replaced_token_draw_count",
            "swap_replaced_token_changed_count",
        }:
            if int(observed.get(key, -1)) != int(value):
                raise FCVControlError(f"{context}.{key} does not reproduce.")
        elif not np.isclose(
            float(observed.get(key, float("nan"))),
            float(value),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise FCVControlError(f"{context}.{key} does not reproduce.")


def _recompute_control_diagnostics(
    config: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]],
    opposite: Mapping[str, Any],
) -> Dict[str, Any]:
    """Derive every cross-control diagnostic from raw-CSV recomputations."""

    opposite_accuracy = float(opposite["counterfactual_accuracy"])
    opposite_drop = float(opposite["mean_confidence_drop"])
    same = controls["same_context"]
    values = {
        "biased_validation_accuracy_recomputed": float(
            same["original_accuracy"]
        ),
        "opposite_context_counterfactual_accuracy": opposite_accuracy,
        "opposite_context_mean_confidence_drop": opposite_drop,
        "same_minus_opposite_accuracy": float(
            same["counterfactual_accuracy"] - opposite_accuracy
        ),
        "opposite_minus_same_confidence_drop": float(
            opposite_drop - same["mean_confidence_drop"]
        ),
        "random_mask_minus_opposite_accuracy": float(
            controls["random_mask"]["counterfactual_accuracy"]
            - opposite_accuracy
        ),
        "shuffled_mask_minus_opposite_accuracy": float(
            controls["shuffled_mask"]["counterfactual_accuracy"]
            - opposite_accuracy
        ),
        "evidence_vs_background_sensitivity_gap": float(
            controls["evidence_swap"]["mean_confidence_drop"]
            - opposite_drop
        ),
    }
    warning_policy = config["fcv"]["controls"]["diagnostic_warning_policy"]
    warnings: List[Dict[str, Any]] = []
    for name, value, threshold in (
        (
            "same_context_accuracy_not_better_than_opposite",
            values["same_minus_opposite_accuracy"],
            float(warning_policy["same_minus_opposite_accuracy_min"]),
        ),
        (
            "opposite_drop_not_larger_than_same_context",
            values["opposite_minus_same_confidence_drop"],
            float(warning_policy["opposite_minus_same_confidence_drop_min"]),
        ),
        (
            "evidence_swap_not_more_sensitive_than_background_swap",
            values["evidence_vs_background_sensitivity_gap"],
            float(warning_policy["evidence_vs_background_sensitivity_gap_min"]),
        ),
    ):
        if value < threshold:
            warnings.append(
                {"name": name, "value": value, "warning_threshold": threshold}
            )
    return {
        **values,
        "diagnostic_warning_policy": dict(warning_policy),
        "diagnostic_warning_count": len(warnings),
        "diagnostic_status": "warning" if warnings else "passed",
        "diagnostic_warnings": warnings,
    }


def _load_torch(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise FCVControlError(f"Torch artifact is not a mapping: {path}")
    return payload


def _source_frame(source: TokenBankSource) -> pd.DataFrame:
    frame = getattr(getattr(source.loader, "dataset", None), "frame", None)
    if not isinstance(frame, pd.DataFrame) or len(frame) != source.sample_count:
        raise FCVControlError("Control source has no valid public manifest frame.")
    return frame


def _tensor_digest(digest: Any, name: str, value: torch.Tensor) -> None:
    value = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_json_bytes(list(value.shape)))
    digest.update(value.numpy().tobytes())


def _evidence_layout_sha256(layout: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        _json_bytes(
            {
                "label": layout.get("label"),
                "token_count": layout.get("token_count"),
                "source_image_count": layout.get("source_image_count"),
                "source_images": layout.get("source_images"),
            }
        )
    )
    _tensor_digest(digest, "token_source_image_index", layout["token_source_image_index"])
    _tensor_digest(digest, "token_source_patch_idx", layout["token_source_patch_idx"])
    return digest.hexdigest()


def build_evidence_layouts(source: TokenBankSource) -> Dict[int, Dict[str, Any]]:
    """Build model-independent evidence-bank provenance from Step 3 masks."""

    frame = _source_frame(source)
    layouts: Dict[int, Dict[str, Any]] = {}
    for label in CONTEXT_NAMES:
        source_images: List[Dict[str, Any]] = []
        source_indices: List[torch.Tensor] = []
        patch_indices: List[torch.Tensor] = []
        for row in frame.itertuples(index=False):
            if int(row.label) != label:
                continue
            sample_id = str(row.sample_id)
            record = source.records_by_sample_id[sample_id]
            evidence_idx = record.get("evidence_idx")
            if not isinstance(evidence_idx, torch.Tensor):
                raise FCVControlError(f"Step 3 record has no evidence_idx: {sample_id}")
            evidence_idx = evidence_idx.detach().to(dtype=torch.long, device="cpu").flatten()
            evidence_eligible = bool(
                record.get("evidence_control_eligible", evidence_idx.numel() > 0)
            )
            if not evidence_eligible or evidence_idx.numel() == 0:
                continue
            source_index = len(source_images)
            source_images.append(
                {
                    "source_image_index": source_index,
                    "sample_id": sample_id,
                    "metadata_index": int(row.metadata_index),
                    "label": label,
                }
            )
            source_indices.append(
                torch.full((evidence_idx.numel(),), source_index, dtype=torch.int32)
            )
            patch_indices.append(evidence_idx.to(torch.int32))
        if not source_images or not patch_indices:
            raise FCVControlError(
                f"No evidence-token donors exist for {CONTEXT_NAMES[label]}."
            )
        token_source_image_index = torch.cat(source_indices).contiguous()
        token_source_patch_idx = torch.cat(patch_indices).contiguous()
        layout: Dict[str, Any] = {
            "label": label,
            "source_images": source_images,
            "source_sample_id_to_index": {
                item["sample_id"]: item["source_image_index"]
                for item in source_images
            },
            "token_source_image_index": token_source_image_index,
            "token_source_patch_idx": token_source_patch_idx,
            "token_count": int(token_source_patch_idx.numel()),
            "source_image_count": len(source_images),
        }
        layout["layout_sha256"] = _evidence_layout_sha256(layout)
        layouts[label] = layout
    return layouts


def _available_bank_tokens(bank: Mapping[str, Any], target_sample_id: str) -> torch.Tensor:
    available = torch.arange(int(bank["token_count"]), dtype=torch.long)
    source_index = bank["source_sample_id_to_index"].get(target_sample_id)
    if source_index is not None:
        provenance = bank["token_source_image_index"].to(torch.long)
        available = available[provenance != int(source_index)]
    if available.numel() == 0:
        raise FCVControlError(
            f"Self-donor exclusion leaves no tokens for {target_sample_id}."
        )
    return available


def _sample_indices(
    available: torch.Tensor,
    donor_samples: int,
    position_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if position_count <= 0:
        return torch.empty((0, 0), dtype=torch.long)
    offsets = torch.randint(
        0,
        int(available.numel()),
        (donor_samples, position_count),
        generator=generator,
        dtype=torch.long,
    )
    return available.index_select(0, offsets.flatten()).reshape_as(offsets)


def _control_plan_content_sha256(plan: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        _json_bytes(
            {
                key: plan.get(key)
                for key in (
                    "schema_version",
                    "artifact_type",
                    "training_fingerprint",
                    "validation_manifest_sha256",
                    "manifest_bundle_sha256",
                    "patch_mask_sha256",
                    "opposite_donor_plan_sha256",
                    "opposite_donor_plan_content_sha256",
                    "donor_samples_per_image",
                    "control_sampling_seed",
                    "control_substream_seeds",
                    "context_bank_layout_sha256",
                    "evidence_bank_layout_sha256",
                    "sample_count",
                )
            }
        )
    )
    evidence_layouts = plan.get("evidence_bank_layouts")
    if not isinstance(evidence_layouts, Mapping):
        raise FCVControlError("Control plan has no evidence layouts.")
    for label in CONTEXT_NAMES:
        layout = evidence_layouts.get(str(label))
        if not isinstance(layout, Mapping):
            raise FCVControlError(f"Missing evidence layout for label {label}.")
        digest.update(_json_bytes(layout["source_images"]))
        _tensor_digest(digest, f"evidence_source_{label}", layout["token_source_image_index"])
        _tensor_digest(digest, f"evidence_patch_{label}", layout["token_source_patch_idx"])
    records = plan.get("records")
    if not isinstance(records, Sequence):
        raise FCVControlError("Control plan has no records.")
    for record in records:
        if not isinstance(record, Mapping):
            raise FCVControlError("Control-plan record must be a mapping.")
        digest.update(
            _json_bytes(
                {
                    key: record.get(key)
                    for key in (
                        "sample_id",
                        "metadata_index",
                        "label",
                        "fcv_eligible",
                        "evidence_control_eligible",
                        "shuffled_mask_source_sample_id",
                    )
                }
            )
        )
        for key in (
            "background_idx",
            "evidence_idx",
            "same_context_donor_token_indices",
            "random_mask_idx",
            "shuffled_background_idx",
            "shuffled_opposite_donor_token_indices",
            "evidence_opposite_donor_token_indices",
        ):
            value = record.get(key)
            if not isinstance(value, torch.Tensor):
                raise FCVControlError(f"Control-plan record has no tensor {key!r}.")
            _tensor_digest(digest, key, value)
    return digest.hexdigest()


def validate_control_plan(
    config: Mapping[str, Any],
    source: TokenBankSource,
    banks: Mapping[int, Mapping[str, Any]],
    opposite_plan: Mapping[str, Any],
    opposite_plan_path: Path,
    plan: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
    """Validate every cached control mask and donor draw."""

    expected = {
        "artifact_type": "fcv_vit_control_plan",
        "schema_version": 1,
        "training_fingerprint": candidate_training_fingerprint(config),
        "split": "biased_validation",
        "validation_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "patch_mask_sha256": source.patch_mask_sha256,
        "patch_mask_summary_sha256": source.patch_mask_summary_sha256,
        "patch_mask_preprocessing_sha256": source.patch_mask_preprocessing_sha256,
        "teacher_maps_sha256": source.teacher_maps_sha256,
        "opposite_donor_plan_path": str(opposite_plan_path),
        "opposite_donor_plan_sha256": _sha256_file(opposite_plan_path),
        "opposite_donor_plan_content_sha256": opposite_plan["plan_content_sha256"],
        "donor_samples_per_image": int(config["fcv"]["donor_samples_per_image"]),
        "control_sampling_seed": int(config["reproducibility"]["control_sampling_seed"]),
        "sample_count": source.sample_count,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise FCVControlError(
                f"Control-plan field {key!r} is stale: {plan.get(key)!r} "
                f"versus {value!r}."
            )
    expected_background_layouts = {
        str(label): str(banks[label]["layout_sha256"]) for label in CONTEXT_NAMES
    }
    if plan.get("context_bank_layout_sha256") != expected_background_layouts:
        raise FCVControlError("Control plan does not match candidate bank layout.")
    expected_evidence_layouts = build_evidence_layouts(source)
    expected_evidence_hashes = {
        str(label): expected_evidence_layouts[label]["layout_sha256"]
        for label in CONTEXT_NAMES
    }
    if plan.get("evidence_bank_layout_sha256") != expected_evidence_hashes:
        raise FCVControlError("Control plan has stale evidence-bank provenance.")
    content_hash = str(plan.get("plan_content_sha256", ""))
    if len(content_hash) != 64 or _control_plan_content_sha256(plan) != content_hash:
        raise FCVControlError("Control-plan content hash is invalid.")

    main_by_id = validate_opposite_donor_plan(
        config, source, banks, opposite_plan
    )
    frame = _source_frame(source)
    records = plan.get("records")
    if not isinstance(records, Sequence) or len(records) != len(frame):
        raise FCVControlError("Control plan does not cover the public manifest.")
    by_id: Dict[str, Mapping[str, Any]] = {}
    patch_count = int(config["model"]["patch_grid_size"]) ** 2
    donor_samples = int(config["fcv"]["donor_samples_per_image"])
    for row, control_record in zip(frame.itertuples(index=False), records):
        if not isinstance(control_record, Mapping):
            raise FCVControlError("Control records must be mappings.")
        sample_id = str(row.sample_id)
        label = int(row.label)
        patch_record = source.records_by_sample_id[sample_id]
        main_record = main_by_id[sample_id]
        evidence_idx = patch_record["evidence_idx"].to(torch.long).flatten()
        evidence_eligible = (
            bool(patch_record["fcv_eligible"])
            and bool(
                patch_record.get(
                    "evidence_control_eligible", evidence_idx.numel() > 0
                )
            )
            and evidence_idx.numel() > 0
        )
        if (
            str(control_record.get("sample_id")) != sample_id
            or int(control_record.get("metadata_index", -1)) != int(row.metadata_index)
            or int(control_record.get("label", -1)) != label
            or bool(control_record.get("fcv_eligible"))
            != bool(patch_record["fcv_eligible"])
            or bool(control_record.get("evidence_control_eligible")) != evidence_eligible
        ):
            raise FCVControlError(f"Stale control metadata for {sample_id}.")
        background_idx = control_record["background_idx"].to(torch.long).flatten()
        if not torch.equal(background_idx, main_record["background_idx"].to(torch.long)):
            raise FCVControlError(f"Background positions changed for {sample_id}.")
        if not torch.equal(
            control_record["evidence_idx"].to(torch.long).flatten(), evidence_idx
        ):
            raise FCVControlError(f"Evidence positions changed for {sample_id}.")
        if bool(patch_record["fcv_eligible"]):
            count = int(background_idx.numel())
            same_indices = control_record["same_context_donor_token_indices"]
            random_idx = control_record["random_mask_idx"].to(torch.long)
            shuffled_idx = control_record["shuffled_background_idx"].to(torch.long)
            shuffled_donors = control_record["shuffled_opposite_donor_token_indices"]
            if tuple(same_indices.shape) != (donor_samples, count):
                raise FCVControlError(f"Invalid same-context draws for {sample_id}.")
            if same_indices.numel() and (
                int(same_indices.min()) < 0
                or int(same_indices.max()) >= int(banks[label]["token_count"])
            ):
                raise FCVControlError(f"Same-context donor index is invalid for {sample_id}.")
            target_source = banks[label]["source_sample_id_to_index"].get(sample_id)
            if target_source is not None:
                sampled_sources = banks[label]["token_source_image_index"].to(
                    torch.long
                ).index_select(0, same_indices.flatten())
                if bool(torch.any(sampled_sources == int(target_source))):
                    raise FCVControlError(f"Same-context self-donor used for {sample_id}.")
            if (
                random_idx.numel() != count
                or torch.unique(random_idx).numel() != count
                or int(random_idx.min()) < 0
                or int(random_idx.max()) >= patch_count
            ):
                raise FCVControlError(f"Invalid matched random mask for {sample_id}.")
            shuffled_source = str(control_record["shuffled_mask_source_sample_id"])
            if not shuffled_source or shuffled_source == sample_id:
                raise FCVControlError(f"Shuffled mask is not from another image: {sample_id}.")
            expected_shuffled = source.records_by_sample_id[shuffled_source][
                "background_idx"
            ].to(torch.long)
            if not torch.equal(shuffled_idx, expected_shuffled):
                raise FCVControlError(f"Shuffled mask values changed for {sample_id}.")
            if tuple(shuffled_donors.shape) != (
                donor_samples,
                int(shuffled_idx.numel()),
            ):
                raise FCVControlError(f"Invalid shuffled-mask draws for {sample_id}.")
            opposite_bank = banks[OPPOSITE_CONTEXT[label]]
            if shuffled_donors.numel() and (
                int(shuffled_donors.min()) < 0
                or int(shuffled_donors.max()) >= int(opposite_bank["token_count"])
            ):
                raise FCVControlError(f"Invalid shuffled donor index for {sample_id}.")
        else:
            for key in (
                "same_context_donor_token_indices",
                "random_mask_idx",
                "shuffled_background_idx",
                "shuffled_opposite_donor_token_indices",
            ):
                if control_record[key].numel() != 0:
                    raise FCVControlError(
                        f"Ineligible sample {sample_id} has cached {key}."
                    )
        evidence_donors = control_record["evidence_opposite_donor_token_indices"]
        if evidence_eligible:
            opposite_layout = expected_evidence_layouts[OPPOSITE_CONTEXT[label]]
            if tuple(evidence_donors.shape) != (
                donor_samples,
                int(evidence_idx.numel()),
            ):
                raise FCVControlError(f"Invalid evidence-swap draws for {sample_id}.")
            if evidence_donors.numel() and (
                int(evidence_donors.min()) < 0
                or int(evidence_donors.max()) >= int(opposite_layout["token_count"])
            ):
                raise FCVControlError(f"Evidence donor index is invalid for {sample_id}.")
        elif evidence_donors.numel() != 0:
            raise FCVControlError(f"Evidence-ineligible target has draws: {sample_id}.")
        by_id[sample_id] = control_record
    return by_id


def prepare_control_plan(
    config: Mapping[str, Any],
    source: TokenBankSource,
    banks: Mapping[int, Mapping[str, Any]],
    opposite_plan: Mapping[str, Any],
    opposite_plan_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Cache all Step 8 control positions and donor indices once."""

    opposite_plan_path = Path(opposite_plan_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.is_file() and not overwrite:
        existing = _load_torch(output_path)
        validate_control_plan(
            config, source, banks, opposite_plan, opposite_plan_path, existing
        )
        return existing
    main_by_id = validate_opposite_donor_plan(config, source, banks, opposite_plan)
    frame = _source_frame(source)
    evidence_layouts = build_evidence_layouts(source)
    donor_samples = int(config["fcv"]["donor_samples_per_image"])
    base_seed = int(config["reproducibility"]["control_sampling_seed"])
    substream_seeds = {
        "same_context": base_seed + 11,
        "random_mask": base_seed + 23,
        "shuffled_mask": base_seed + 37,
        "shuffled_donors": base_seed + 53,
        "evidence_donors": base_seed + 71,
    }
    generators = {
        name: torch.Generator(device="cpu").manual_seed(seed)
        for name, seed in substream_seeds.items()
    }
    eligible_ids = [
        str(row.sample_id)
        for row in frame.itertuples(index=False)
        if bool(source.records_by_sample_id[str(row.sample_id)]["fcv_eligible"])
    ]
    if len(eligible_ids) < 2:
        raise FCVControlError("Shuffled-mask control requires two eligible images.")
    shuffle_offset = int(
        torch.randint(
            1,
            len(eligible_ids),
            (1,),
            generator=generators["shuffled_mask"],
        ).item()
    )
    shuffled_source_by_id = {
        sample_id: eligible_ids[(index + shuffle_offset) % len(eligible_ids)]
        for index, sample_id in enumerate(eligible_ids)
    }
    patch_count = int(config["model"]["patch_grid_size"]) ** 2
    records: List[Dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        sample_id = str(row.sample_id)
        label = int(row.label)
        patch_record = source.records_by_sample_id[sample_id]
        background_idx = patch_record["background_idx"].detach().to(
            dtype=torch.long, device="cpu"
        ).flatten()
        evidence_idx = patch_record["evidence_idx"].detach().to(
            dtype=torch.long, device="cpu"
        ).flatten()
        fcv_eligible = bool(patch_record["fcv_eligible"])
        evidence_eligible = (
            fcv_eligible
            and bool(
                patch_record.get(
                    "evidence_control_eligible", evidence_idx.numel() > 0
                )
            )
            and evidence_idx.numel() > 0
        )
        if fcv_eligible:
            same_available = _available_bank_tokens(banks[label], sample_id)
            same_donors = _sample_indices(
                same_available,
                donor_samples,
                int(background_idx.numel()),
                generators["same_context"],
            )
            random_idx = torch.randperm(
                patch_count, generator=generators["random_mask"]
            )[: background_idx.numel()].sort().values
            shuffled_source = shuffled_source_by_id[sample_id]
            shuffled_idx = source.records_by_sample_id[shuffled_source][
                "background_idx"
            ].detach().to(dtype=torch.long, device="cpu").flatten()
            opposite_available = _available_bank_tokens(
                banks[OPPOSITE_CONTEXT[label]], sample_id
            )
            shuffled_donors = _sample_indices(
                opposite_available,
                donor_samples,
                int(shuffled_idx.numel()),
                generators["shuffled_donors"],
            )
        else:
            same_donors = torch.empty((0, 0), dtype=torch.long)
            random_idx = torch.empty((0,), dtype=torch.long)
            shuffled_source = ""
            shuffled_idx = torch.empty((0,), dtype=torch.long)
            shuffled_donors = torch.empty((0, 0), dtype=torch.long)
        if evidence_eligible:
            opposite_evidence = evidence_layouts[OPPOSITE_CONTEXT[label]]
            evidence_available = torch.arange(
                int(opposite_evidence["token_count"]), dtype=torch.long
            )
            evidence_donors = _sample_indices(
                evidence_available,
                donor_samples,
                int(evidence_idx.numel()),
                generators["evidence_donors"],
            )
        else:
            evidence_donors = torch.empty((0, 0), dtype=torch.long)
        records.append(
            {
                "sample_id": sample_id,
                "metadata_index": int(row.metadata_index),
                "label": label,
                "fcv_eligible": fcv_eligible,
                "evidence_control_eligible": evidence_eligible,
                "background_idx": background_idx,
                "evidence_idx": evidence_idx,
                "same_context_donor_token_indices": same_donors,
                "random_mask_idx": random_idx,
                "shuffled_mask_source_sample_id": shuffled_source,
                "shuffled_background_idx": shuffled_idx,
                "shuffled_opposite_donor_token_indices": shuffled_donors,
                "evidence_opposite_donor_token_indices": evidence_donors,
            }
        )
    serialized_evidence_layouts = {
        str(label): evidence_layouts[label] for label in CONTEXT_NAMES
    }
    plan: MutableMapping[str, Any] = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_control_plan",
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
        "opposite_donor_plan_path": str(opposite_plan_path),
        "opposite_donor_plan_sha256": _sha256_file(opposite_plan_path),
        "opposite_donor_plan_content_sha256": opposite_plan["plan_content_sha256"],
        "donor_samples_per_image": donor_samples,
        "control_sampling_seed": base_seed,
        "control_substream_seeds": substream_seeds,
        "shuffled_mask_derangement_offset": shuffle_offset,
        "context_bank_layout_sha256": {
            str(label): str(banks[label]["layout_sha256"])
            for label in CONTEXT_NAMES
        },
        "evidence_bank_layout_sha256": {
            str(label): str(evidence_layouts[label]["layout_sha256"])
            for label in CONTEXT_NAMES
        },
        "evidence_bank_layouts": serialized_evidence_layouts,
        "sample_count": source.sample_count,
        "fcv_eligible_sample_count": len(eligible_ids),
        "evidence_eligible_sample_count": sum(
            int(record["evidence_control_eligible"]) for record in records
        ),
        "records": records,
    }
    plan["plan_content_sha256"] = _control_plan_content_sha256(plan)
    validate_control_plan(
        config, source, banks, opposite_plan, opposite_plan_path, plan
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(plan, output_path)
    return plan


def _build_evidence_token_banks(
    raw_tokens: torch.Tensor,
    source: TokenBankSource,
    expected_layouts: Mapping[str, Mapping[str, Any]],
) -> Dict[int, torch.Tensor]:
    frame = _source_frame(source)
    sample_to_row = {
        str(sample_id): index
        for index, sample_id in enumerate(frame["sample_id"].astype(str).tolist())
    }
    banks: Dict[int, torch.Tensor] = {}
    for label in CONTEXT_NAMES:
        layout = expected_layouts[str(label)]
        chunks = []
        for source_image in layout["source_images"]:
            sample_id = str(source_image["sample_id"])
            row_index = sample_to_row[sample_id]
            evidence_idx = source.records_by_sample_id[sample_id]["evidence_idx"].to(
                torch.long
            )
            chunks.append(raw_tokens[row_index].index_select(0, evidence_idx))
        tokens = torch.cat(chunks, dim=0).float().contiguous()
        if int(tokens.shape[0]) != int(layout["token_count"]):
            raise FCVControlError(f"Evidence token count changed for label {label}.")
        if not torch.isfinite(tokens).all():
            raise FCVControlError(f"Evidence bank for label {label} is non-finite.")
        banks[label] = tokens
    return banks


def _control_summary_reusable(
    summary: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    candidate_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source: TokenBankSource,
    opposite_plan_path: Path,
    opposite_plan_sha256: str,
    control_plan_path: Path,
    control_plan_sha256: str,
    step7_summary_path: Path,
    step7_summary_sha256: str,
    reconstruction_reports: Mapping[str, Any],
    target_batch_size: int,
    counterfactual_forward_batch_size: int,
) -> bool:
    if (
        summary.get("artifact_type") != "fcv_vit_candidate_control_summary"
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
        or summary.get("opposite_donor_plan_path") != str(opposite_plan_path)
        or summary.get("opposite_donor_plan_sha256") != opposite_plan_sha256
        or summary.get("control_plan_path") != str(control_plan_path)
        or summary.get("control_plan_sha256") != control_plan_sha256
        or summary.get("step7_summary_path") != str(step7_summary_path)
        or summary.get("step7_summary_sha256") != step7_summary_sha256
        or summary.get("reconstruction_reports") != dict(reconstruction_reports)
        or summary.get("diagnostic_warning_policy")
        != dict(config["fcv"]["controls"]["diagnostic_warning_policy"])
        or summary.get("execution")
        != {
            "validation_batch_size": source.batch_size,
            "validation_num_workers": source.num_workers,
            "target_batch_size": target_batch_size,
            "counterfactual_forward_batch_size": counterfactual_forward_batch_size,
        }
    ):
        return False
    files = summary.get("score_csvs")
    if not isinstance(files, Mapping) or set(files) != set(CONTROL_NAMES):
        return False
    recomputed_controls: Dict[str, Dict[str, Any]] = {}
    for control_name in CONTROL_NAMES:
        details = files[control_name]
        if not isinstance(details, Mapping):
            return False
        path = Path(str(details.get("path", "")))
        if (
            not path.is_file()
            or path.stat().st_size != int(details.get("size_bytes", -1))
            or _sha256_file(path) != details.get("sha256")
        ):
            return False
        try:
            recomputed_controls[control_name] = recompute_control_metrics_from_frame(
                pd.read_csv(path)
            )
            _validate_metric_mapping(
                summary["controls"][control_name],
                recomputed_controls[control_name],
                control_name,
            )
        except (OSError, ValueError, TypeError, KeyError, FCVControlError):
            return False
    try:
        with step7_summary_path.open("r", encoding="utf-8") as handle:
            step7_summary = json.load(handle)
        step7_csv = Path(str(step7_summary.get("score_csv_path", "")))
        if (
            not step7_csv.is_file()
            or _sha256_file(step7_csv) != step7_summary.get("score_csv_sha256")
        ):
            return False
        opposite = validate_fcv_summary_against_frame(
            step7_summary, pd.read_csv(step7_csv), config
        )
        expected_differences = _recompute_control_diagnostics(
            config, recomputed_controls, opposite
        )
        for name, expected in expected_differences.items():
            if name in {
                "diagnostic_warning_policy",
                "diagnostic_warning_count",
                "diagnostic_status",
                "diagnostic_warnings",
            }:
                continue
            if not np.isclose(
                float(summary.get(name, float("nan"))),
                float(expected),
                rtol=0.0,
                atol=1.0e-7,
            ):
                return False
        if (
            summary.get("diagnostic_warning_policy")
            != expected_differences["diagnostic_warning_policy"]
            or int(summary.get("diagnostic_warning_count", -1))
            != expected_differences["diagnostic_warning_count"]
            or summary.get("diagnostic_status")
            != expected_differences["diagnostic_status"]
        ):
            return False
        observed_warnings = summary.get("diagnostic_warnings")
        expected_warnings = expected_differences["diagnostic_warnings"]
        if not isinstance(observed_warnings, list) or len(observed_warnings) != len(
            expected_warnings
        ):
            return False
        for observed, expected in zip(observed_warnings, expected_warnings):
            if (
                observed.get("name") != expected["name"]
                or not np.isclose(
                    float(observed.get("value", float("nan"))),
                    expected["value"],
                    rtol=0.0,
                    atol=1.0e-7,
                )
                or not np.isclose(
                    float(observed.get("warning_threshold", float("nan"))),
                    expected["warning_threshold"],
                    rtol=0.0,
                    atol=1.0e-7,
                )
            ):
                return False
    except (OSError, ValueError, TypeError, KeyError, FCVControlError):
        return False
    return True


def _score_one_control(
    *,
    control_name: str,
    config: Mapping[str, Any],
    model: torch.nn.Module,
    raw_tokens: torch.Tensor,
    original_probabilities: torch.Tensor,
    original_predictions: torch.Tensor,
    source: TokenBankSource,
    main_by_id: Mapping[str, Mapping[str, Any]],
    control_by_id: Mapping[str, Mapping[str, Any]],
    background_bank_tokens: Mapping[int, torch.Tensor],
    evidence_bank_tokens: Mapping[int, torch.Tensor],
    device: torch.device,
    target_batch_size: int,
    forward_batch_size: int,
) -> pd.DataFrame:
    if control_name not in CONTROL_NAMES:
        raise FCVControlError(f"Unknown control: {control_name}")
    frame = _source_frame(source)
    donor_samples = int(config["fcv"]["donor_samples_per_image"])
    num_classes = int(config["model"]["num_classes"])
    rows: List[Dict[str, Any]] = []
    for row_index, manifest_row in enumerate(frame.itertuples(index=False)):
        sample_id = str(manifest_row.sample_id)
        label = int(manifest_row.label)
        patch_record = source.records_by_sample_id[sample_id]
        control_record = control_by_id[sample_id]
        if control_name == "evidence_swap":
            eligible = bool(control_record["evidence_control_eligible"])
            positions = control_record["evidence_idx"]
            donor_label = OPPOSITE_CONTEXT[label]
            donor_indices = control_record["evidence_opposite_donor_token_indices"]
            position_source = "target_teacher_evidence"
            mask_source_sample_id = sample_id
        elif control_name == "same_context":
            eligible = bool(control_record["fcv_eligible"])
            positions = control_record["background_idx"]
            donor_label = label
            donor_indices = control_record["same_context_donor_token_indices"]
            position_source = "target_teacher_background"
            mask_source_sample_id = sample_id
        elif control_name == "random_mask":
            eligible = bool(control_record["fcv_eligible"])
            positions = control_record["random_mask_idx"]
            donor_label = OPPOSITE_CONTEXT[label]
            donor_indices = main_by_id[sample_id]["donor_token_indices"]
            position_source = "matched_count_uniform_random"
            mask_source_sample_id = "RANDOM"
        else:
            eligible = bool(control_record["fcv_eligible"])
            positions = control_record["shuffled_background_idx"]
            donor_label = OPPOSITE_CONTEXT[label]
            donor_indices = control_record["shuffled_opposite_donor_token_indices"]
            position_source = "other_image_teacher_background"
            mask_source_sample_id = str(
                control_record["shuffled_mask_source_sample_id"]
            )
        rows.append(
            {
                "control_name": control_name,
                "sample_id": sample_id,
                "metadata_index": int(manifest_row.metadata_index),
                "label": label,
                "control_eligible": eligible,
                "eligibility_reason": (
                    "eligible"
                    if eligible
                    else str(patch_record.get("eligibility_reason", "ineligible"))
                ),
                "p_y_original": float(original_probabilities[row_index, label]),
                "pred_original": int(original_predictions[row_index]),
                "correct_original": int(int(original_predictions[row_index]) == label),
                "donor_context_label": donor_label,
                "donor_context_name": CONTEXT_NAMES[donor_label],
                "position_source": position_source,
                "mask_source_sample_id": mask_source_sample_id,
                "num_positions_swapped": int(positions.numel()),
                "donor_draw_count": donor_samples,
                "p_y_control_mean": None,
                "p_y_control_std": None,
                "pred_control_majority": None,
                "correct_control_majority": None,
                "control_draw_accuracy": None,
                "control_correct_draws": None,
                "control_confidence_drop": None,
                "p_y_control_draws": None,
                "pred_control_draws": None,
                "swap_preserved_token_max_abs_error": None,
                "swap_donor_reconstruction_max_abs_error": None,
                "swap_replaced_token_draw_count": None,
                "swap_replaced_token_changed_count": None,
                "swap_replaced_token_changed_fraction": None,
                "swap_replacement_delta_mean": None,
                "swap_replacement_delta_max": None,
            }
        )

    with torch.inference_mode():
        for start in range(0, len(rows), target_batch_size):
            stop = min(start + target_batch_size, len(rows))
            counterfactual_batches: List[torch.Tensor] = []
            eligible_row_indices: List[int] = []
            for row_index in range(start, stop):
                row = rows[row_index]
                if not row["control_eligible"]:
                    continue
                sample_id = str(row["sample_id"])
                control_record = control_by_id[sample_id]
                main_record = main_by_id[sample_id]
                label = int(row["label"])
                if control_name == "same_context":
                    positions = control_record["background_idx"]
                    donor_indices = control_record["same_context_donor_token_indices"]
                    donor_tokens = background_bank_tokens[label]
                elif control_name == "random_mask":
                    positions = control_record["random_mask_idx"]
                    donor_indices = main_record["donor_token_indices"]
                    donor_tokens = background_bank_tokens[OPPOSITE_CONTEXT[label]]
                elif control_name == "shuffled_mask":
                    positions = control_record["shuffled_background_idx"]
                    donor_indices = control_record[
                        "shuffled_opposite_donor_token_indices"
                    ]
                    donor_tokens = background_bank_tokens[OPPOSITE_CONTEXT[label]]
                else:
                    positions = control_record["evidence_idx"]
                    donor_indices = control_record[
                        "evidence_opposite_donor_token_indices"
                    ]
                    donor_tokens = evidence_bank_tokens[OPPOSITE_CONTEXT[label]]
                target_tokens = raw_tokens[row_index].to(
                    device, non_blocking=True
                )
                counterfactual_tokens = make_counterfactual_token_batch(
                    target_tokens,
                    positions,
                    donor_tokens,
                    donor_indices,
                )
                positions_device = positions.to(dtype=torch.long, device=device)
                donor_indices_device = donor_indices.to(
                    dtype=torch.long, device=donor_tokens.device
                )
                expected_donors = donor_tokens.index_select(
                    0, donor_indices_device.flatten()
                ).reshape(
                    donor_samples,
                    len(positions_device),
                    target_tokens.shape[-1],
                ).to(device=device, dtype=target_tokens.dtype)
                donor_error = float(
                    (
                        counterfactual_tokens.index_select(1, positions_device)
                        - expected_donors
                    ).abs().max().item()
                )
                preserved_mask = torch.ones(
                    target_tokens.shape[0], dtype=torch.bool, device=device
                )
                preserved_mask[positions_device] = False
                preserved_error = (
                    float(
                        (
                            counterfactual_tokens[:, preserved_mask, :]
                            - target_tokens[preserved_mask, :].unsqueeze(0)
                        ).abs().max().item()
                    )
                    if bool(preserved_mask.any())
                    else 0.0
                )
                target_replaced = target_tokens.index_select(
                    0, positions_device
                ).unsqueeze(0)
                replacement_delta = (
                    counterfactual_tokens.index_select(1, positions_device)
                    - target_replaced
                ).norm(dim=2)
                replaced_count = int(replacement_delta.numel())
                changed_count = int((replacement_delta != 0.0).sum().item())
                if preserved_error != 0.0 or donor_error != 0.0:
                    raise FCVControlError(
                        "Control replacement violated token integrity: "
                        f"control={control_name} sample={sample_id} "
                        f"preserved={preserved_error:.3e} donor={donor_error:.3e}."
                    )
                row.update(
                    {
                        "swap_preserved_token_max_abs_error": preserved_error,
                        "swap_donor_reconstruction_max_abs_error": donor_error,
                        "swap_replaced_token_draw_count": replaced_count,
                        "swap_replaced_token_changed_count": changed_count,
                        "swap_replaced_token_changed_fraction": (
                            changed_count / replaced_count
                        ),
                        "swap_replacement_delta_mean": float(
                            replacement_delta.mean().item()
                        ),
                        "swap_replacement_delta_max": float(
                            replacement_delta.max().item()
                        ),
                    }
                )
                counterfactual_batches.append(counterfactual_tokens)
                eligible_row_indices.append(row_index)
            if not counterfactual_batches:
                continue
            all_tokens = torch.cat(counterfactual_batches, dim=0)
            logits_chunks = []
            for cf_start in range(0, len(all_tokens), forward_batch_size):
                logits_chunks.append(
                    forward_from_patch_tokens(
                        model, all_tokens[cf_start : cf_start + forward_batch_size]
                    ).float()
                )
            logits = torch.cat(logits_chunks, dim=0)
            if len(logits) != len(eligible_row_indices) * donor_samples:
                raise FCVControlError("Control output count is inconsistent.")
            offset = 0
            for row_index in eligible_row_indices:
                row = rows[row_index]
                label = int(row["label"])
                draw_logits = logits[offset : offset + donor_samples]
                offset += donor_samples
                probabilities = draw_logits.softmax(dim=1)[:, label]
                predictions = draw_logits.argmax(dim=1)
                correct_draws = int((predictions == label).sum().item())
                majority = int(
                    torch.bincount(predictions, minlength=num_classes).argmax().item()
                )
                (
                    probability_values,
                    mean_probability,
                    std_probability,
                ) = _canonical_probability_draw_statistics(probabilities)
                row.update(
                    {
                        "p_y_control_mean": mean_probability,
                        "p_y_control_std": std_probability,
                        "pred_control_majority": majority,
                        "correct_control_majority": int(majority == label),
                        "control_draw_accuracy": correct_draws / donor_samples,
                        "control_correct_draws": correct_draws,
                        "control_confidence_drop": float(
                            row["p_y_original"] - mean_probability
                        ),
                        "p_y_control_draws": json.dumps(
                            probability_values,
                            separators=(",", ":"),
                        ),
                        "pred_control_draws": json.dumps(
                            [int(value) for value in predictions.cpu().tolist()],
                            separators=(",", ":"),
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    # Validate all persisted invariants now so a no-op control cannot reach its
    # summary even before the CSV is written and re-read by the caller.
    recompute_control_metrics_from_frame(frame)
    return frame


def score_candidate_controls(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    source: TokenBankSource,
    token_bank_dir: str | Path,
    opposite_plan_path: str | Path,
    control_plan_path: str | Path,
    step7_score_dir: str | Path,
    output_dir: str | Path,
    *,
    reconstruction_reports: Mapping[str, Any],
    device: str | torch.device = "cuda",
    target_batch_size: int = 16,
    counterfactual_forward_batch_size: int = 256,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Score all four controls for one candidate with one model load."""

    if target_batch_size <= 0 or counterfactual_forward_batch_size <= 0:
        raise FCVControlError("Control batch sizes must be positive.")
    if target_batch_size != int(config["execution"]["control_target_batch_size"]):
        raise FCVControlError("Control target batch size differs from the locked value.")
    if counterfactual_forward_batch_size != int(
        config["execution"]["control_counterfactual_forward_batch_size"]
    ):
        raise FCVControlError(
            "Control counterfactual forward batch size differs from the locked value."
        )
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    token_bank_dir = Path(token_bank_dir).expanduser().resolve()
    opposite_plan_path = Path(opposite_plan_path).expanduser().resolve()
    control_plan_path = Path(control_plan_path).expanduser().resolve()
    step7_score_dir = Path(step7_score_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    model, checkpoint = load_candidate_model(config, checkpoint_path, device=device)
    candidate_id = str(checkpoint.get("candidate_id", ""))
    if not candidate_id or Path(candidate_id).name != candidate_id:
        raise FCVControlError(f"Unsafe candidate ID: {candidate_id!r}")
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
    opposite_plan = _load_torch(opposite_plan_path)
    main_by_id = validate_opposite_donor_plan(config, source, banks, opposite_plan)
    control_plan = _load_torch(control_plan_path)
    control_by_id = validate_control_plan(
        config,
        source,
        banks,
        opposite_plan,
        opposite_plan_path,
        control_plan,
    )
    opposite_plan_sha256 = _sha256_file(opposite_plan_path)
    control_plan_sha256 = _sha256_file(control_plan_path)
    step7_summary_path = step7_score_dir / f"{candidate_id}_summary.json"
    if not step7_summary_path.is_file():
        raise FileNotFoundError(f"Missing Step 7 score summary: {step7_summary_path}")
    with step7_summary_path.open("r", encoding="utf-8") as handle:
        step7_summary = json.load(handle)
    if (
        step7_summary.get("artifact_type") != "fcv_vit_candidate_score_summary"
        or step7_summary.get("status") != "complete"
        or step7_summary.get("candidate_id") != candidate_id
        or step7_summary.get("checkpoint_sha256") != checkpoint_sha256
        or step7_summary.get("donor_plan_sha256") != opposite_plan_sha256
        or step7_summary.get("manifest_bundle_sha256")
        != source.manifest_bundle_sha256
        or step7_summary.get("patch_mask_summary_sha256")
        != source.patch_mask_summary_sha256
        or step7_summary.get("teacher_maps_sha256") != source.teacher_maps_sha256
        or step7_summary.get("execution")
        != {
            "validation_batch_size": source.batch_size,
            "validation_num_workers": source.num_workers,
            "counterfactual_forward_batch_size": int(
                config["execution"]["fcv_counterfactual_forward_batch_size"]
            ),
        }
    ):
        raise FCVControlError("Step 7 candidate summary is stale or incompatible.")
    step7_score_path = Path(str(step7_summary.get("score_csv_path", "")))
    if (
        not step7_score_path.is_file()
        or _sha256_file(step7_score_path) != step7_summary.get("score_csv_sha256")
    ):
        raise FCVControlError("Step 7 per-image score CSV is missing or stale.")
    try:
        step7_recomputed = validate_fcv_summary_against_frame(
            step7_summary, pd.read_csv(step7_score_path), config
        )
    except Exception as exc:
        raise FCVControlError(
            f"Step 7 metrics do not reproduce from the hashed score CSV: {exc}"
        ) from exc
    step7_summary_sha256 = _sha256_file(step7_summary_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{candidate_id}_controls_summary.json"
    if summary_path.is_file() and not overwrite:
        with summary_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if _control_summary_reusable(
            existing,
            config=config,
            candidate_id=candidate_id,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            source=source,
            opposite_plan_path=opposite_plan_path,
            opposite_plan_sha256=opposite_plan_sha256,
            control_plan_path=control_plan_path,
            control_plan_sha256=control_plan_sha256,
            step7_summary_path=step7_summary_path,
            step7_summary_sha256=step7_summary_sha256,
            reconstruction_reports=reconstruction_reports,
            target_batch_size=target_batch_size,
            counterfactual_forward_batch_size=counterfactual_forward_batch_size,
        ):
            existing = dict(existing)
            existing["status"] = "reused"
            return existing
        raise FCVControlError(
            f"Existing control summary is stale: {summary_path}. Use --overwrite."
        )

    model.eval()
    model_device = torch.device(device)
    raw_chunks: List[torch.Tensor] = []
    probability_chunks: List[torch.Tensor] = []
    prediction_chunks: List[torch.Tensor] = []
    observed_sample_ids: List[str] = []
    with torch.inference_mode():
        for images, _, sample_ids in source.loader:
            images = images.to(model_device, non_blocking=True)
            raw = extract_raw_patch_tokens(model, images).float()
            logits = forward_from_patch_tokens(model, raw).float()
            raw_chunks.append(raw.cpu())
            probability_chunks.append(logits.softmax(dim=1).cpu())
            prediction_chunks.append(logits.argmax(dim=1).cpu())
            observed_sample_ids.extend(str(value) for value in sample_ids)
    expected_ids = _source_frame(source)["sample_id"].astype(str).tolist()
    if observed_sample_ids != expected_ids:
        raise FCVControlError("Raw-token cache order differs from the public manifest.")
    raw_tokens = torch.cat(raw_chunks, dim=0).contiguous()
    original_probabilities = torch.cat(probability_chunks, dim=0)
    original_predictions = torch.cat(prediction_chunks, dim=0)
    evidence_bank_cpu = _build_evidence_token_banks(
        raw_tokens,
        source,
        control_plan["evidence_bank_layouts"],
    )
    background_bank_tokens = {
        label: banks[label]["tokens"].to(model_device, non_blocking=True)
        for label in CONTEXT_NAMES
    }
    evidence_bank_tokens = {
        label: evidence_bank_cpu[label].to(model_device, non_blocking=True)
        for label in CONTEXT_NAMES
    }

    score_csvs: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, Dict[str, Any]] = {}
    recomputed_original_accuracy: float | None = None
    for control_name in CONTROL_NAMES:
        frame = _score_one_control(
            control_name=control_name,
            config=config,
            model=model,
            raw_tokens=raw_tokens,
            original_probabilities=original_probabilities,
            original_predictions=original_predictions,
            source=source,
            main_by_id=main_by_id,
            control_by_id=control_by_id,
            background_bank_tokens=background_bank_tokens,
            evidence_bank_tokens=evidence_bank_tokens,
            device=model_device,
            target_batch_size=target_batch_size,
            forward_batch_size=counterfactual_forward_batch_size,
        )
        eligible = frame[frame["control_eligible"]].copy()
        if eligible.empty:
            raise FCVControlError(f"No eligible rows for control {control_name}.")
        original_accuracy = float(frame["correct_original"].mean())
        if recomputed_original_accuracy is None:
            recomputed_original_accuracy = original_accuracy
        elif abs(recomputed_original_accuracy - original_accuracy) > 1.0e-12:
            raise FCVControlError("Original accuracy changed between controls.")
        csv_path = output_dir / f"{candidate_id}_{control_name}.csv"
        _atomic_csv(frame, csv_path)
        score_csvs[control_name] = {
            "path": str(csv_path),
            "size_bytes": csv_path.stat().st_size,
            "sha256": _sha256_file(csv_path),
        }
        # The summary is derived from the serialized record, including direct
        # replacement integrity diagnostics, so resume/aggregation validates
        # exactly the bytes retained for audit.
        metrics[control_name] = recompute_control_metrics_from_frame(
            pd.read_csv(csv_path)
        )
    if recomputed_original_accuracy is None:
        raise FCVControlError("No original accuracy was computed.")
    step7_original = float(step7_recomputed["original_accuracy"])
    if abs(recomputed_original_accuracy - step7_original) > 1.0e-8:
        raise FCVControlError(
            "Step 8 original accuracy does not reproduce the Step 7 score: "
            f"{recomputed_original_accuracy} versus {step7_original}."
        )
    diagnostics = _recompute_control_diagnostics(config, metrics, step7_recomputed)
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_candidate_control_summary",
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
        "validation_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_path": str(source.manifest_bundle_path),
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "patch_mask_sha256": source.patch_mask_sha256,
        "patch_mask_summary_sha256": source.patch_mask_summary_sha256,
        "patch_mask_preprocessing_sha256": source.patch_mask_preprocessing_sha256,
        "teacher_maps_sha256": source.teacher_maps_sha256,
        "opposite_donor_plan_path": str(opposite_plan_path),
        "opposite_donor_plan_sha256": opposite_plan_sha256,
        "control_plan_path": str(control_plan_path),
        "control_plan_sha256": control_plan_sha256,
        "step7_summary_path": str(step7_summary_path),
        "step7_summary_sha256": step7_summary_sha256,
        "reconstruction_reports": dict(reconstruction_reports),
        "execution": {
            "validation_batch_size": source.batch_size,
            "validation_num_workers": source.num_workers,
            "target_batch_size": target_batch_size,
            "counterfactual_forward_batch_size": counterfactual_forward_batch_size,
        },
        "validation_sample_count": source.sample_count,
        "biased_validation_accuracy_recomputed": diagnostics[
            "biased_validation_accuracy_recomputed"
        ],
        "opposite_context_counterfactual_accuracy": diagnostics[
            "opposite_context_counterfactual_accuracy"
        ],
        "opposite_context_mean_confidence_drop": diagnostics[
            "opposite_context_mean_confidence_drop"
        ],
        "controls": metrics,
        "same_minus_opposite_accuracy": diagnostics[
            "same_minus_opposite_accuracy"
        ],
        "opposite_minus_same_confidence_drop": diagnostics[
            "opposite_minus_same_confidence_drop"
        ],
        "random_mask_minus_opposite_accuracy": diagnostics[
            "random_mask_minus_opposite_accuracy"
        ],
        "shuffled_mask_minus_opposite_accuracy": diagnostics[
            "shuffled_mask_minus_opposite_accuracy"
        ],
        "evidence_vs_background_sensitivity_gap": diagnostics[
            "evidence_vs_background_sensitivity_gap"
        ],
        "diagnostic_warning_policy": diagnostics["diagnostic_warning_policy"],
        "diagnostic_warning_count": diagnostics["diagnostic_warning_count"],
        "diagnostic_status": diagnostics["diagnostic_status"],
        "diagnostic_warnings": diagnostics["diagnostic_warnings"],
        "score_csvs": score_csvs,
    }
    _atomic_json(summary, summary_path)
    del (
        model,
        checkpoint,
        banks,
        background_bank_tokens,
        evidence_bank_tokens,
        evidence_bank_cpu,
        raw_tokens,
        opposite_plan,
        control_plan,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def aggregate_control_summaries(
    config: Mapping[str, Any],
    control_score_dir: str | Path,
    output_csv: str | Path,
    output_summary: str | Path,
    *,
    source: TokenBankSource,
    opposite_plan_path: str | Path,
    control_plan_path: str | Path,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Index all four-control summaries without loading Oracle or test data."""

    control_score_dir = Path(control_score_dir).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    output_summary = Path(output_summary).expanduser().resolve()
    opposite_plan_path = Path(opposite_plan_path).expanduser().resolve()
    control_plan_path = Path(control_plan_path).expanduser().resolve()
    opposite_sha = _sha256_file(opposite_plan_path)
    control_sha = _sha256_file(control_plan_path)
    expected_fingerprint = candidate_training_fingerprint(config)
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    invalid: List[Dict[str, str]] = []
    selected_epochs = candidate_epochs(config)
    for run in enumerate_sweep_runs(config):
        for epoch in selected_epochs:
            candidate_id = run.candidate_id(epoch)
            summary_path = control_score_dir / f"{candidate_id}_controls_summary.json"
            if not summary_path.is_file():
                missing.append(candidate_id)
                continue
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                valid = (
                    summary.get("artifact_type")
                    == "fcv_vit_candidate_control_summary"
                    and summary.get("schema_version") == 1
                    and summary.get("status") == "complete"
                    and summary.get("candidate_id") == candidate_id
                    and summary.get("training_fingerprint") == expected_fingerprint
                    and summary.get("validation_manifest_sha256")
                    == source.manifest_sha256
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
                    and summary.get("opposite_donor_plan_path")
                    == str(opposite_plan_path)
                    and summary.get("opposite_donor_plan_sha256") == opposite_sha
                    and summary.get("control_plan_path") == str(control_plan_path)
                    and summary.get("control_plan_sha256") == control_sha
                    and summary.get("diagnostic_warning_policy")
                    == dict(config["fcv"]["controls"]["diagnostic_warning_policy"])
                    and summary.get("execution")
                    == {
                        "validation_batch_size": source.batch_size,
                        "validation_num_workers": source.num_workers,
                        "target_batch_size": int(
                            config["execution"]["control_target_batch_size"]
                        ),
                        "counterfactual_forward_batch_size": int(
                            config["execution"][
                                "control_counterfactual_forward_batch_size"
                            ]
                        ),
                    }
                )
                files = summary.get("score_csvs")
                metrics = summary.get("controls")
                step7_summary_path = Path(str(summary.get("step7_summary_path", "")))
                checkpoint_path = Path(str(summary.get("checkpoint_path", "")))
                checkpoint_sha256 = str(summary.get("checkpoint_sha256", ""))
                if (
                    not valid
                    or not isinstance(files, Mapping)
                    or not isinstance(metrics, Mapping)
                    or set(files) != set(CONTROL_NAMES)
                    or set(metrics) != set(CONTROL_NAMES)
                    or not step7_summary_path.is_file()
                    or _sha256_file(step7_summary_path)
                    != summary.get("step7_summary_sha256")
                    or not checkpoint_path.is_file()
                    or _sha256_file(checkpoint_path) != checkpoint_sha256
                ):
                    raise FCVControlError("stale summary metadata")
                for control_name in CONTROL_NAMES:
                    details = files[control_name]
                    path = Path(str(details["path"]))
                    if (
                        not path.is_file()
                        or path.stat().st_size != int(details["size_bytes"])
                        or _sha256_file(path) != details["sha256"]
                    ):
                        raise FCVControlError(f"invalid {control_name} score CSV")
                if not _control_summary_reusable(
                    summary,
                    config=config,
                    candidate_id=candidate_id,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha256,
                    source=source,
                    opposite_plan_path=opposite_plan_path,
                    opposite_plan_sha256=opposite_sha,
                    control_plan_path=control_plan_path,
                    control_plan_sha256=control_sha,
                    step7_summary_path=step7_summary_path,
                    step7_summary_sha256=str(summary["step7_summary_sha256"]),
                    reconstruction_reports=summary["reconstruction_reports"],
                    target_batch_size=int(
                        config["execution"]["control_target_batch_size"]
                    ),
                    counterfactual_forward_batch_size=int(
                        config["execution"][
                            "control_counterfactual_forward_batch_size"
                        ]
                    ),
                ):
                    raise FCVControlError(
                        "control metrics do not reproduce from hashed raw CSVs"
                    )
                recomputed_controls = {
                    name: recompute_control_metrics_from_frame(
                        pd.read_csv(Path(str(files[name]["path"])))
                    )
                    for name in CONTROL_NAMES
                }
                with step7_summary_path.open("r", encoding="utf-8") as handle:
                    step7_summary = json.load(handle)
                step7_csv = Path(str(step7_summary["score_csv_path"]))
                opposite = validate_fcv_summary_against_frame(
                    step7_summary, pd.read_csv(step7_csv), config
                )
                diagnostics = _recompute_control_diagnostics(
                    config, recomputed_controls, opposite
                )
                row: Dict[str, Any] = {
                    "run_index": run.run_index,
                    "candidate_id": candidate_id,
                    "epoch": epoch,
                    "seed": run.seed,
                    "learning_rate": run.learning_rate,
                    "weight_decay": run.weight_decay,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "biased_validation_accuracy": diagnostics[
                        "biased_validation_accuracy_recomputed"
                    ],
                    "opposite_context_counterfactual_accuracy": diagnostics[
                        "opposite_context_counterfactual_accuracy"
                    ],
                    "same_minus_opposite_accuracy": diagnostics[
                        "same_minus_opposite_accuracy"
                    ],
                    "random_mask_minus_opposite_accuracy": diagnostics[
                        "random_mask_minus_opposite_accuracy"
                    ],
                    "shuffled_mask_minus_opposite_accuracy": diagnostics[
                        "shuffled_mask_minus_opposite_accuracy"
                    ],
                    "evidence_vs_background_sensitivity_gap": diagnostics[
                        "evidence_vs_background_sensitivity_gap"
                    ],
                    "diagnostic_status": diagnostics["diagnostic_status"],
                    "diagnostic_warning_count": diagnostics[
                        "diagnostic_warning_count"
                    ],
                    "diagnostic_warnings": json.dumps(
                        diagnostics["diagnostic_warnings"], separators=(",", ":")
                    ),
                    "summary_path": str(summary_path),
                }
                for control_name in CONTROL_NAMES:
                    for metric_name, value in recomputed_controls[control_name].items():
                        if metric_name in {"sample_count", "original_accuracy"}:
                            continue
                        row[f"{control_name}_{metric_name}"] = value
                    row[f"{control_name}_score_csv"] = files[control_name]["path"]
                rows.append(row)
            except (OSError, ValueError, KeyError, TypeError, FCVControlError) as exc:
                invalid.append({"candidate_id": candidate_id, "error": str(exc)})
    if (missing or invalid) and not allow_incomplete:
        raise FCVControlError(
            f"FCV control pool is incomplete: missing={len(missing)} "
            f"invalid={len(invalid)}."
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["run_index", "epoch"]).reset_index(drop=True)
        if frame["candidate_id"].duplicated().any():
            raise FCVControlError("Duplicate candidates in the control index.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(frame, output_csv)
    expected_count = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_control_pool_summary",
        "status": "complete" if not missing and not invalid else "incomplete",
        "candidate_count": len(frame),
        "expected_candidate_count": expected_count,
        "control_count_per_candidate": len(CONTROL_NAMES),
        "candidate_warning_count": int(
            (frame["diagnostic_warning_count"] > 0).sum()
        ) if not frame.empty else 0,
        "total_diagnostic_warning_count": int(
            frame["diagnostic_warning_count"].sum()
        ) if not frame.empty else 0,
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
        "opposite_donor_plan_path": str(opposite_plan_path),
        "opposite_donor_plan_sha256": opposite_sha,
        "control_plan_path": str(control_plan_path),
        "control_plan_sha256": control_sha,
        "output_csv": str(output_csv),
        "output_csv_sha256": _sha256_file(output_csv),
        "selection_metrics_recomputed_from_hashed_csvs": True,
        "execution": {
            "validation_batch_size": source.batch_size,
            "validation_num_workers": source.num_workers,
            "target_batch_size": int(
                config["execution"]["control_target_batch_size"]
            ),
            "counterfactual_forward_batch_size": int(
                config["execution"]["control_counterfactual_forward_batch_size"]
            ),
        },
    }
    _atomic_json(summary, output_summary)
    return summary
