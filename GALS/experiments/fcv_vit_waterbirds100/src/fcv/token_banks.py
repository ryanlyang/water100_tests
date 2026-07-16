"""Build model-specific raw-patch background banks for FCV Step 6."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .candidate_training import (
    CandidateTrainingError,
    PublicManifestDataset,
    build_transforms,
    candidate_training_fingerprint,
    enumerate_sweep_runs,
    get_sweep_run,
    software_fingerprint,
    software_versions,
)
from .config import candidate_epochs
from .manifest_provenance import (
    ManifestProvenanceError,
    validate_manifest_bundle,
)
from .vit_counterfactual_forward import extract_raw_patch_tokens, load_candidate_model


CONTEXT_NAMES = {0: "land_context", 1: "water_context"}
REQUIRED_PATCH_RECORD_KEYS = {
    "sample_id",
    "metadata_index",
    "label",
    "patch_scores",
    "background_idx",
    "fcv_eligible",
    "teacher_map_path",
    "teacher_map_sha256",
}


class TokenBankError(ValueError):
    """Raised when Step 6 provenance or token-bank invariants fail."""


@dataclass(frozen=True)
class TokenBankSource:
    """Validated, reusable validation loader and Step 3 masks."""

    manifest_path: Path
    patch_mask_path: Path
    manifest_sha256: str
    manifest_bundle_path: Path
    manifest_bundle_sha256: str
    patch_mask_sha256: str
    patch_mask_summary_path: Path
    patch_mask_summary_sha256: str
    patch_mask_preprocessing_sha256: str
    teacher_maps_sha256: str
    records_by_sample_id: Mapping[str, Mapping[str, Any]]
    loader: DataLoader
    sample_count: int
    eligible_counts_by_label: Mapping[int, int]
    batch_size: int
    num_workers: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


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


def _load_trusted_torch_artifact(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TokenBankError(f"Torch artifact is not a mapping: {path}")
    return payload


def _validate_patch_mask_artifact(
    config: Mapping[str, Any],
    artifact: Mapping[str, Any],
    manifest_sha256: str,
    manifest_bundle_sha256: str,
) -> Dict[str, Mapping[str, Any]]:
    if artifact.get("artifact_type") != "fcv_vit_patch_masks":
        raise TokenBankError("Step 6 requires the Step 3 fcv_vit_patch_masks artifact.")
    if int(artifact.get("schema_version", -1)) != 2:
        raise TokenBankError(
            f"Unsupported patch-mask schema: {artifact.get('schema_version')}"
        )
    if artifact.get("split") != "biased_validation":
        raise TokenBankError("Token banks may only use the biased_validation split.")
    if artifact.get("manifest_sha256") != manifest_sha256:
        raise TokenBankError(
            "Patch-mask artifact and public validation manifest hashes disagree."
        )
    if artifact.get("manifest_bundle_sha256") != manifest_bundle_sha256:
        raise TokenBankError(
            "Patch-mask artifact and Step-2 manifest-bundle hashes disagree."
        )
    model_cfg = config["model"]
    fcv_cfg = config["fcv"]
    expected_values = {
        "image_size": int(model_cfg["image_size"]),
        "patch_size": int(model_cfg["patch_size"]),
        "patch_grid_size": int(model_cfg["patch_grid_size"]),
        "patch_count": int(model_cfg["patch_grid_size"]) ** 2,
        "evidence_threshold": float(fcv_cfg["evidence_patch_threshold"]),
        "background_threshold": float(fcv_cfg["background_patch_threshold"]),
        "minimum_background_patches": int(fcv_cfg["minimum_background_patches"]),
    }
    for key, expected in expected_values.items():
        if artifact.get(key) != expected:
            raise TokenBankError(
                f"Patch-mask field {key!r} is stale: {artifact.get(key)!r} "
                f"versus expected {expected!r}."
            )
    teacher_cfg = config["data"]["teacher_maps"]
    expected_preprocessing = {
        "teacher_map_source": str(teacher_cfg["source"]),
        "teacher_map_format": str(teacher_cfg["format"]),
        "foreground_class_ids": [
            int(value) for value in teacher_cfg["foreground_class_ids"]
        ],
        "normalize_to_unit_interval": bool(
            teacher_cfg["normalize_to_unit_interval"]
        ),
        "interpolation": str(teacher_cfg["interpolation"]),
        "spatial_transform": str(teacher_cfg["spatial_transform"]),
        "eval_resize_size": int(
            config["training"]["augmentation"]["eval_resize_size"]
        ),
        "image_size": int(model_cfg["image_size"]),
        "patch_size": int(model_cfg["patch_size"]),
        "patch_grid_size": int(model_cfg["patch_grid_size"]),
        "evidence_threshold": float(fcv_cfg["evidence_patch_threshold"]),
        "background_threshold": float(fcv_cfg["background_patch_threshold"]),
        "minimum_background_patches": int(fcv_cfg["minimum_background_patches"]),
        "minimum_eligible_fraction": float(fcv_cfg["minimum_eligible_fraction"]),
        "minimum_eligible_count_per_class": int(
            fcv_cfg["minimum_eligible_count_per_class"]
        ),
        "ambiguous_patch_policy": str(fcv_cfg["ambiguous_patch_policy"]),
    }
    if artifact.get("preprocessing_config") != expected_preprocessing:
        raise TokenBankError("Step 3 preprocessing configuration is stale.")
    if artifact.get("preprocessing_config_sha256") != _sha256_json(
        expected_preprocessing
    ):
        raise TokenBankError("Step 3 preprocessing fingerprint is stale.")
    records = artifact.get("records")
    if not isinstance(records, Sequence) or not records:
        raise TokenBankError("Patch-mask artifact has no records.")
    by_sample_id: Dict[str, Mapping[str, Any]] = {}
    expected_patch_count = expected_values["patch_count"]
    for record in records:
        if not isinstance(record, Mapping):
            raise TokenBankError("Patch-mask records must be mappings.")
        missing = sorted(REQUIRED_PATCH_RECORD_KEYS.difference(record))
        if missing:
            raise TokenBankError(f"Patch-mask record is missing keys: {missing}")
        sample_id = str(record["sample_id"])
        if sample_id in by_sample_id:
            raise TokenBankError(f"Duplicate patch-mask sample ID: {sample_id}")
        label = int(record["label"])
        if label not in CONTEXT_NAMES:
            raise TokenBankError(f"Patch-mask label must be binary, found {label}.")
        scores = record["patch_scores"]
        background_idx = record["background_idx"]
        if not isinstance(scores, torch.Tensor) or scores.numel() != expected_patch_count:
            raise TokenBankError(f"Invalid patch_scores for {sample_id}.")
        scores = scores.detach().to(dtype=torch.float32, device="cpu").flatten()
        if not torch.isfinite(scores).all():
            raise TokenBankError(f"Non-finite patch_scores for {sample_id}.")
        if not isinstance(background_idx, torch.Tensor):
            raise TokenBankError(f"Invalid background_idx for {sample_id}.")
        background_idx = background_idx.detach().to(dtype=torch.long, device="cpu").flatten()
        if background_idx.numel() and (
            int(background_idx.min()) < 0
            or int(background_idx.max()) >= expected_patch_count
        ):
            raise TokenBankError(f"Out-of-range background patch index for {sample_id}.")
        if torch.unique(background_idx).numel() != background_idx.numel():
            raise TokenBankError(f"Duplicate background patch indices for {sample_id}.")
        if background_idx.numel() and float(
            scores.index_select(0, background_idx).max()
        ) > float(fcv_cfg["background_patch_threshold"]) + 1.0e-6:
            raise TokenBankError(
                f"Patch-mask artifact marks a non-background patch safe for {sample_id}."
            )
        if bool(record["fcv_eligible"]) and background_idx.numel() < int(
            fcv_cfg["minimum_background_patches"]
        ):
            raise TokenBankError(
                f"Eligible record {sample_id} has too few safe background patches."
            )
        teacher_map_path = Path(str(record["teacher_map_path"])).expanduser().resolve()
        teacher_map_sha256 = str(record["teacher_map_sha256"])
        if (
            not teacher_map_path.is_file()
            or len(teacher_map_sha256) != 64
            or _sha256_file(teacher_map_path) != teacher_map_sha256
        ):
            raise TokenBankError(
                f"Teacher-map bytes changed after Step 3 for {sample_id}."
            )
        by_sample_id[sample_id] = record
    current_teacher_maps_sha256 = _sha256_json(
        {
            "teacher_maps": [
                {
                    "sample_id": str(record["sample_id"]),
                    "sha256": str(record["teacher_map_sha256"]),
                }
                for record in records
            ]
        }
    )
    if artifact.get("teacher_maps_sha256") != current_teacher_maps_sha256:
        raise TokenBankError("Step 3 teacher-map aggregate hash is stale.")
    return by_sample_id


def prepare_token_bank_source(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    patch_mask_path: str | Path,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    check_images: bool = True,
) -> TokenBankSource:
    """Validate and cache the only public data source allowed for Step 6."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    patch_mask_path = Path(patch_mask_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing public validation manifest: {manifest_path}")
    if not patch_mask_path.is_file():
        raise FileNotFoundError(f"Missing Step 3 patch-mask artifact: {patch_mask_path}")
    try:
        manifest_binding = validate_manifest_bundle(
            config, manifest_path, "biased_validation"
        )
    except ManifestProvenanceError as exc:
        raise TokenBankError(str(exc)) from exc
    manifest_sha256 = manifest_binding.manifest_sha256
    patch_mask_sha256 = _sha256_file(patch_mask_path)
    summary_path = patch_mask_path.with_name("patch_masks_val_summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing successful Step 3 summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        patch_summary = json.load(handle)
    if not isinstance(patch_summary, Mapping):
        raise TokenBankError("Step 3 summary must be a JSON mapping.")
    patch_artifact = _load_trusted_torch_artifact(patch_mask_path)
    records_by_sample_id = _validate_patch_mask_artifact(
        config,
        patch_artifact,
        manifest_sha256,
        manifest_binding.bundle_sha256,
    )
    preprocessing_sha256 = str(patch_artifact.get("preprocessing_config_sha256", ""))
    teacher_maps_sha256 = str(patch_artifact.get("teacher_maps_sha256", ""))
    summary_expected = {
        "schema_version": 2,
        "artifact_type": "fcv_vit_patch_masks",
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "manifest_bundle_path": str(manifest_binding.bundle_path),
        "manifest_bundle_sha256": manifest_binding.bundle_sha256,
        "patch_mask_path": str(patch_mask_path),
        "patch_mask_sha256": patch_mask_sha256,
        "preprocessing_config_sha256": preprocessing_sha256,
        "teacher_maps_sha256": teacher_maps_sha256,
    }
    for key, expected in summary_expected.items():
        if patch_summary.get(key) != expected:
            raise TokenBankError(
                f"Step 3 summary field {key!r} is stale: "
                f"{patch_summary.get(key)!r} versus {expected!r}."
            )
    audit_path = Path(str(patch_summary.get("audit_path", "")))
    overlay_index_path = Path(str(patch_summary.get("preflight_overlay_index", "")))
    if (
        not audit_path.is_file()
        or _sha256_file(audit_path) != patch_summary.get("audit_sha256")
        or not overlay_index_path.is_file()
        or _sha256_file(overlay_index_path)
        != patch_summary.get("preflight_overlay_index_sha256")
    ):
        raise TokenBankError("Step 3 diagnostic artifacts are missing or stale.")

    transform = build_transforms(config)["eval"]
    dataset = PublicManifestDataset(
        manifest_path,
        "biased_validation",
        transform,
        check_images=check_images,
    )
    manifest_ids = dataset.frame["sample_id"].astype(str).tolist()
    if set(manifest_ids) != set(records_by_sample_id) or len(manifest_ids) != len(
        records_by_sample_id
    ):
        raise TokenBankError(
            "Public validation manifest and patch-mask sample IDs do not match exactly."
        )
    # Donor eligibility is intentionally broader than target eligibility: every
    # image with at least one threshold-safe background token contributes.
    eligible_counts = {0: 0, 1: 0}
    for row in dataset.frame.itertuples(index=False):
        sample_id = str(row.sample_id)
        record = records_by_sample_id[sample_id]
        if int(record["metadata_index"]) != int(row.metadata_index):
            raise TokenBankError(f"Metadata index mismatch for {sample_id}.")
        if int(record["label"]) != int(row.label):
            raise TokenBankError(f"Label mismatch for {sample_id}.")
        if int(record["background_idx"].numel()) > 0:
            eligible_counts[int(row.label)] += 1
    for label, count in eligible_counts.items():
        if count < 2:
            raise TokenBankError(
                f"{CONTEXT_NAMES[label]} needs at least two safe-background donor images "
                f"to support self-donor exclusion; found {count}."
            )

    resolved_batch_size = int(
        config["execution"]["token_bank_batch_size"]
        if batch_size is None
        else batch_size
    )
    resolved_workers = int(
        config["execution"]["token_bank_num_workers"]
        if num_workers is None
        else num_workers
    )
    if resolved_batch_size <= 0 or resolved_workers < 0:
        raise TokenBankError("batch_size must be positive and num_workers non-negative.")
    if resolved_batch_size != int(config["execution"]["token_bank_batch_size"]):
        raise TokenBankError("Token extraction batch size differs from the locked value.")
    if resolved_workers != int(config["execution"]["token_bank_num_workers"]):
        raise TokenBankError("Token extraction worker count differs from the locked value.")
    loader = DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=resolved_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=resolved_workers > 0,
        drop_last=False,
    )
    if len(preprocessing_sha256) != 64:
        raise TokenBankError("Patch-mask artifact has no valid preprocessing hash.")
    if len(teacher_maps_sha256) != 64:
        raise TokenBankError("Patch-mask artifact has no valid teacher-map aggregate hash.")
    return TokenBankSource(
        manifest_path=manifest_path,
        patch_mask_path=patch_mask_path,
        manifest_sha256=manifest_sha256,
        manifest_bundle_path=manifest_binding.bundle_path,
        manifest_bundle_sha256=manifest_binding.bundle_sha256,
        patch_mask_sha256=patch_mask_sha256,
        patch_mask_summary_path=summary_path,
        patch_mask_summary_sha256=_sha256_file(summary_path),
        patch_mask_preprocessing_sha256=preprocessing_sha256,
        teacher_maps_sha256=teacher_maps_sha256,
        records_by_sample_id=records_by_sample_id,
        loader=loader,
        sample_count=len(dataset),
        eligible_counts_by_label=eligible_counts,
        batch_size=resolved_batch_size,
        num_workers=resolved_workers,
    )


class _BankAccumulator:
    def __init__(self, label: int, patch_grid_size: int) -> None:
        self.label = label
        self.patch_grid_size = patch_grid_size
        self.source_images: List[Dict[str, Any]] = []
        self.token_chunks: List[torch.Tensor] = []
        self.source_image_indices: List[torch.Tensor] = []
        self.source_classes: List[torch.Tensor] = []
        self.patch_indices: List[torch.Tensor] = []
        self.patch_rows: List[torch.Tensor] = []
        self.patch_columns: List[torch.Tensor] = []
        self.patch_scores: List[torch.Tensor] = []

    def add(
        self,
        *,
        sample_id: str,
        metadata_index: int,
        tokens: torch.Tensor,
        background_idx: torch.Tensor,
        patch_scores: torch.Tensor,
    ) -> None:
        if tokens.ndim != 2 or tokens.shape[0] != background_idx.numel():
            raise TokenBankError(f"Token/index shape mismatch for {sample_id}.")
        source_index = len(self.source_images)
        self.source_images.append(
            {
                "source_image_index": source_index,
                "sample_id": sample_id,
                "metadata_index": int(metadata_index),
                "label": self.label,
            }
        )
        count = int(background_idx.numel())
        indices = background_idx.detach().to(dtype=torch.int32, device="cpu")
        self.token_chunks.append(tokens.detach().to(dtype=torch.float32, device="cpu"))
        self.source_image_indices.append(
            torch.full((count,), source_index, dtype=torch.int32)
        )
        self.source_classes.append(torch.full((count,), self.label, dtype=torch.int8))
        self.patch_indices.append(indices)
        self.patch_rows.append((indices // self.patch_grid_size).to(torch.int16))
        self.patch_columns.append((indices % self.patch_grid_size).to(torch.int16))
        self.patch_scores.append(
            patch_scores.index_select(0, background_idx.to(torch.long)).to(torch.float32)
        )

    def finalize(self) -> Dict[str, Any]:
        if len(self.source_images) < 2 or not self.token_chunks:
            raise TokenBankError(
                f"{CONTEXT_NAMES[self.label]} has insufficient eligible donor images."
            )
        tokens = torch.cat(self.token_chunks, dim=0).contiguous()
        if not torch.isfinite(tokens).all():
            raise TokenBankError(f"{CONTEXT_NAMES[self.label]} contains non-finite tokens.")
        token_source_image_index = torch.cat(self.source_image_indices).contiguous()
        token_source_patch_idx = torch.cat(self.patch_indices).contiguous()
        token_count = int(tokens.shape[0])
        return {
            "tokens": tokens,
            "token_source_image_index": token_source_image_index,
            "token_source_class": torch.cat(self.source_classes).contiguous(),
            "token_source_patch_idx": token_source_patch_idx,
            "token_source_patch_row": torch.cat(self.patch_rows).contiguous(),
            "token_source_patch_col": torch.cat(self.patch_columns).contiguous(),
            "token_patch_score": torch.cat(self.patch_scores).contiguous(),
            "source_images": self.source_images,
            "source_sample_id_to_index": {
                item["sample_id"]: item["source_image_index"]
                for item in self.source_images
            },
            "token_count": token_count,
            "source_image_count": len(self.source_images),
        }


def _bank_paths(output_dir: Path, candidate_id: str) -> Dict[int, Path]:
    return {
        label: output_dir / f"{candidate_id}_{context_name}.pt"
        for label, context_name in CONTEXT_NAMES.items()
    }


def _completed_summary_is_valid(
    summary: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    candidate_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source: TokenBankSource,
    reconstruction_reports: Mapping[str, Any],
) -> bool:
    if summary.get("artifact_type") != "fcv_vit_token_bank_summary":
        return False
    if summary.get("status") != "complete" or summary.get("candidate_id") != candidate_id:
        return False
    if summary.get("training_fingerprint") != candidate_training_fingerprint(config):
        return False
    if summary.get("checkpoint_path") != str(checkpoint_path):
        return False
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        return False
    if summary.get("validation_manifest_sha256") != source.manifest_sha256:
        return False
    if summary.get("manifest_bundle_sha256") != source.manifest_bundle_sha256:
        return False
    if summary.get("patch_mask_sha256") != source.patch_mask_sha256:
        return False
    if summary.get("patch_mask_summary_sha256") != source.patch_mask_summary_sha256:
        return False
    if (
        summary.get("patch_mask_preprocessing_sha256")
        != source.patch_mask_preprocessing_sha256
    ):
        return False
    if summary.get("teacher_maps_sha256") != source.teacher_maps_sha256:
        return False
    active_versions = software_versions()
    if summary.get("software_versions") != active_versions:
        return False
    if summary.get("software_fingerprint") != software_fingerprint(active_versions):
        return False
    if summary.get("reconstruction_reports") != dict(reconstruction_reports):
        return False
    if summary.get("execution") != {
        "batch_size": source.batch_size,
        "num_workers": source.num_workers,
    }:
        return False
    banks = summary.get("banks")
    if not isinstance(banks, Mapping):
        return False
    for context_name in CONTEXT_NAMES.values():
        details = banks.get(context_name)
        if not isinstance(details, Mapping):
            return False
        path = Path(str(details.get("path", "")))
        expected_size = int(details.get("file_size_bytes", -1))
        expected_sha256 = str(details.get("sha256", ""))
        if (
            expected_size <= 0
            or len(expected_sha256) != 64
            or not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256_file(path) != expected_sha256
        ):
            return False
    return True


def build_background_token_banks(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    source: TokenBankSource,
    output_dir: str | Path,
    *,
    reconstruction_reports: Mapping[str, Any],
    device: str | torch.device = "cuda",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Extract and persist land/water safe-background tokens for one candidate."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    model, checkpoint = load_candidate_model(config, checkpoint_path, device=device)
    candidate_id = str(checkpoint.get("candidate_id", ""))
    if not candidate_id:
        raise TokenBankError("Candidate checkpoint has no candidate_id.")
    if Path(candidate_id).name != candidate_id:
        raise TokenBankError(f"Unsafe candidate_id for artifact naming: {candidate_id!r}")
    saved_manifest_hashes = checkpoint.get("manifest_sha256")
    if not isinstance(saved_manifest_hashes, Mapping) or saved_manifest_hashes.get(
        "biased_validation"
    ) != source.manifest_sha256:
        raise TokenBankError(
            "Candidate checkpoint was trained/evaluated with a different biased "
            "validation manifest."
        )
    if saved_manifest_hashes.get("manifest_bundle") != source.manifest_bundle_sha256:
        raise TokenBankError(
            "Candidate checkpoint was trained with a different Step-2 manifest bundle."
        )
    active_versions = software_versions()
    if checkpoint.get("software_versions") != active_versions:
        raise TokenBankError(
            "Candidate checkpoint software differs from the active token-bank runtime."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    bank_paths = _bank_paths(output_dir, candidate_id)
    summary_path = output_dir / f"{candidate_id}_summary.json"
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if summary_path.is_file() and not overwrite:
        with summary_path.open("r", encoding="utf-8") as handle:
            existing_summary = json.load(handle)
        if _completed_summary_is_valid(
            existing_summary,
            config=config,
            candidate_id=candidate_id,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            source=source,
            reconstruction_reports=reconstruction_reports,
        ):
            existing_summary = dict(existing_summary)
            existing_summary["status"] = "reused"
            return existing_summary
        raise TokenBankError(
            f"Existing token-bank summary is stale or incomplete: {summary_path}. "
            "Use --overwrite to replace it."
        )

    patch_grid_size = int(config["model"]["patch_grid_size"])
    accumulators = {
        label: _BankAccumulator(label, patch_grid_size) for label in CONTEXT_NAMES
    }
    skipped_no_safe_background = {0: 0, 1: 0}
    processed_by_label = {0: 0, 1: 0}
    processed_images = 0
    model.eval()
    model_device = torch.device(device)
    with torch.inference_mode():
        for images, labels, sample_ids in source.loader:
            images = images.to(model_device, non_blocking=True)
            raw_tokens = extract_raw_patch_tokens(model, images).float().cpu()
            labels = labels.to(dtype=torch.long, device="cpu")
            for batch_index, sample_id_value in enumerate(sample_ids):
                sample_id = str(sample_id_value)
                label = int(labels[batch_index].item())
                record = source.records_by_sample_id[sample_id]
                if int(record["label"]) != label:
                    raise TokenBankError(f"Runtime label mismatch for {sample_id}.")
                background_idx = record["background_idx"].detach().to(
                    dtype=torch.long,
                    device="cpu",
                )
                if background_idx.numel() == 0:
                    skipped_no_safe_background[label] += 1
                    continue
                patch_scores = record["patch_scores"].detach().to(
                    dtype=torch.float32,
                    device="cpu",
                )
                selected = raw_tokens[batch_index].index_select(0, background_idx)
                accumulators[label].add(
                    sample_id=sample_id,
                    metadata_index=int(record["metadata_index"]),
                    tokens=selected,
                    background_idx=background_idx,
                    patch_scores=patch_scores,
                )
                processed_by_label[label] += 1
                processed_images += 1

    if processed_by_label != dict(source.eligible_counts_by_label):
        raise TokenBankError(
            "Token extraction did not account for every eligible donor image: "
            f"processed={processed_by_label}, expected={source.eligible_counts_by_label}."
        )

    bank_summaries: Dict[str, Dict[str, Any]] = {}
    for label, context_name in CONTEXT_NAMES.items():
        values = accumulators[label].finalize()
        payload = {
            "schema_version": 1,
            "artifact_type": "fcv_vit_background_token_bank",
            "candidate_id": candidate_id,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "training_fingerprint": checkpoint["training_fingerprint"],
            "software_versions": active_versions,
            "software_fingerprint": software_fingerprint(active_versions),
            "model": dict(checkpoint["model"]),
            "split": "biased_validation",
            "context_name": context_name,
            "context_label_proxy": label,
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
            "patch_grid_size": patch_grid_size,
            "embedding_dim": int(values["tokens"].shape[1]),
            "storage_dtype": "float32",
            "sampling_contract": dict(config["fcv"]["donor_bank"]),
            "execution": {
                "batch_size": source.batch_size,
                "num_workers": source.num_workers,
            },
            **values,
        }
        path = bank_paths[label]
        _atomic_torch_save(payload, path)
        bank_summaries[context_name] = {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "context_label_proxy": label,
            "token_count": int(values["token_count"]),
            "source_image_count": int(values["source_image_count"]),
            "embedding_dim": int(values["tokens"].shape[1]),
        }
        del payload, values

    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_token_bank_summary",
        "status": "complete",
        "candidate_id": candidate_id,
        "run": dict(checkpoint["run"]),
        "epoch": int(checkpoint["epoch"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "training_fingerprint": checkpoint["training_fingerprint"],
        "software_versions": active_versions,
        "software_fingerprint": software_fingerprint(active_versions),
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
            "batch_size": source.batch_size,
            "num_workers": source.num_workers,
        },
        "validation_sample_count": source.sample_count,
        "safe_background_donor_image_count": processed_images,
        "safe_background_donor_counts_by_label": {
            str(key): int(value) for key, value in processed_by_label.items()
        },
        "skipped_no_safe_background_by_label": {
            str(key): int(value) for key, value in skipped_no_safe_background.items()
        },
        "donor_cohort_policy": "all_images_with_at_least_one_safe_background_token",
        "banks": bank_summaries,
    }
    _atomic_json(summary, summary_path)
    del model, checkpoint, accumulators
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def candidate_checkpoints_for_run(
    config: Mapping[str, Any],
    candidate_root: str | Path,
    run_index: int,
) -> List[Path]:
    """Return the exact ordered reduced-pool checkpoints for one sweep run."""

    run = get_sweep_run(config, run_index)
    run_dir = Path(candidate_root) / run.run_id
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing Step 4 run metrics: {metrics_path}")
    metrics = pd.read_csv(metrics_path).sort_values("epoch")
    expected_epochs = candidate_epochs(config)
    if metrics["epoch"].astype(int).tolist() != expected_epochs:
        raise CandidateTrainingError(
            f"Run {run_index} does not contain the complete ordered epoch set."
        )
    expected_ids = [run.candidate_id(epoch) for epoch in expected_epochs]
    if metrics["candidate_id"].astype(str).tolist() != expected_ids:
        raise CandidateTrainingError(f"Run {run_index} candidate IDs are inconsistent.")
    checkpoints = [Path(str(value)).expanduser().resolve() for value in metrics["checkpoint_path"]]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Run {run_index} has missing candidate checkpoints: {missing[:5]}"
        )
    return checkpoints


def aggregate_token_bank_summaries(
    config: Mapping[str, Any],
    token_bank_dir: str | Path,
    output_csv: str | Path,
    output_summary: str | Path,
    *,
    manifest_sha256: str,
    patch_mask_sha256: str,
    source: TokenBankSource | None = None,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Validate and index all candidate/context bank artifacts without loading tokens."""

    token_bank_dir = Path(token_bank_dir)
    output_csv = Path(output_csv)
    output_summary = Path(output_summary)
    expected_fingerprint = candidate_training_fingerprint(config)
    rows = []
    missing_candidates = []
    invalid_candidates = []
    selected_epochs = candidate_epochs(config)
    for run in enumerate_sweep_runs(config):
        for epoch in selected_epochs:
            candidate_id = run.candidate_id(epoch)
            summary_path = token_bank_dir / f"{candidate_id}_summary.json"
            if not summary_path.is_file():
                missing_candidates.append(candidate_id)
                continue
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                candidate_rows = []
                valid = (
                    summary.get("artifact_type") == "fcv_vit_token_bank_summary"
                    and summary.get("status") == "complete"
                    and summary.get("candidate_id") == candidate_id
                    and summary.get("training_fingerprint") == expected_fingerprint
                    and summary.get("validation_manifest_sha256") == manifest_sha256
                    and summary.get("patch_mask_sha256") == patch_mask_sha256
                )
                if source is not None:
                    valid = bool(
                        valid
                        and summary.get("patch_mask_summary_sha256")
                        == source.patch_mask_summary_sha256
                        and summary.get("patch_mask_preprocessing_sha256")
                        == source.patch_mask_preprocessing_sha256
                        and summary.get("teacher_maps_sha256")
                        == source.teacher_maps_sha256
                        and summary.get("manifest_bundle_sha256")
                        == source.manifest_bundle_sha256
                        and summary.get("software_versions") == software_versions()
                        and summary.get("software_fingerprint")
                        == software_fingerprint()
                        and summary.get("execution")
                        == {
                            "batch_size": source.batch_size,
                            "num_workers": source.num_workers,
                        }
                    )
                checkpoint_path = Path(str(summary.get("checkpoint_path", "")))
                checkpoint_sha256 = str(summary.get("checkpoint_sha256", ""))
                if (
                    not checkpoint_path.is_file()
                    or len(checkpoint_sha256) != 64
                    or _sha256_file(checkpoint_path) != checkpoint_sha256
                ):
                    raise TokenBankError("candidate checkpoint bytes changed")
                banks = summary.get("banks", {})
                if not valid or not isinstance(banks, Mapping):
                    raise TokenBankError("stale summary metadata")
                for label, context_name in CONTEXT_NAMES.items():
                    details = banks.get(context_name)
                    if not isinstance(details, Mapping):
                        raise TokenBankError(f"missing {context_name}")
                    path = Path(str(details.get("path", "")))
                    expected_size = int(details.get("file_size_bytes", -1))
                    expected_sha256 = str(details.get("sha256", ""))
                    if (
                        expected_size <= 0
                        or len(expected_sha256) != 64
                        or not path.is_file()
                        or path.stat().st_size != expected_size
                        or _sha256_file(path) != expected_sha256
                    ):
                        raise TokenBankError(f"invalid {context_name} file")
                    candidate_rows.append(
                        {
                            "run_index": run.run_index,
                            "candidate_id": candidate_id,
                            "epoch": epoch,
                            "seed": run.seed,
                            "learning_rate": run.learning_rate,
                            "weight_decay": run.weight_decay,
                            "context_name": context_name,
                            "context_label_proxy": label,
                            "token_count": int(details["token_count"]),
                            "source_image_count": int(details["source_image_count"]),
                            "embedding_dim": int(details["embedding_dim"]),
                            "bank_path": str(path),
                            "bank_sha256": expected_sha256,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": checkpoint_sha256,
                            "file_size_bytes": int(details["file_size_bytes"]),
                            "summary_path": str(summary_path.resolve()),
                        }
                    )
                rows.extend(candidate_rows)
            except (OSError, ValueError, KeyError, TypeError, TokenBankError) as exc:
                invalid_candidates.append({"candidate_id": candidate_id, "error": str(exc)})

    if (missing_candidates or invalid_candidates) and not allow_incomplete:
        raise TokenBankError(
            f"Token-bank pool is incomplete: missing={len(missing_candidates)} "
            f"invalid={len(invalid_candidates)}."
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["run_index", "epoch", "context_label_proxy"]
        ).reset_index(drop=True)
        if frame[["candidate_id", "context_name"]].duplicated().any():
            raise TokenBankError("Duplicate candidate/context rows in token-bank index.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(frame, output_csv)
    expected_candidates = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    summary = {
        "artifact_type": "fcv_vit_token_bank_pool_summary",
        "status": "complete" if not missing_candidates and not invalid_candidates else "incomplete",
        "candidate_count": int(frame["candidate_id"].nunique()) if not frame.empty else 0,
        "expected_candidate_count": expected_candidates,
        "bank_count": int(len(frame)),
        "expected_bank_count": expected_candidates * len(CONTEXT_NAMES),
        "missing_candidate_count": len(missing_candidates),
        "missing_candidate_preview": missing_candidates[:10],
        "invalid_candidate_count": len(invalid_candidates),
        "invalid_candidate_preview": invalid_candidates[:10],
        "training_fingerprint": expected_fingerprint,
        "validation_manifest_sha256": manifest_sha256,
        "manifest_bundle_sha256": (
            source.manifest_bundle_sha256 if source is not None else None
        ),
        "patch_mask_sha256": patch_mask_sha256,
        "patch_mask_summary_sha256": (
            source.patch_mask_summary_sha256 if source is not None else None
        ),
        "patch_mask_preprocessing_sha256": (
            source.patch_mask_preprocessing_sha256 if source is not None else None
        ),
        "teacher_maps_sha256": (
            source.teacher_maps_sha256 if source is not None else None
        ),
        "software_versions": software_versions() if source is not None else None,
        "software_fingerprint": software_fingerprint() if source is not None else None,
        "execution": (
            {
                "batch_size": source.batch_size,
                "num_workers": source.num_workers,
            }
            if source is not None
            else None
        ),
        "output_csv": str(output_csv.resolve()),
        "output_csv_sha256": _sha256_file(output_csv),
    }
    _atomic_json(summary, output_summary)
    return summary
