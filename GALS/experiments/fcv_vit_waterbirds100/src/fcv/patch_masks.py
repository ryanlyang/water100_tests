"""Preprocess R4RR teacher maps into ViT patch partitions for FCV."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, UnidentifiedImageError
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .manifest_provenance import (
    ManifestProvenanceError,
    validate_manifest_bundle,
)


REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "metadata_index",
    "image_path",
    "image_sha256",
    "label",
    "source_split",
    "study_split",
    "teacher_map_path",
    "teacher_map_exists",
}
FORBIDDEN_PUBLIC_COLUMNS = {"context", "group", "group_name", "place"}


class PatchMaskError(ValueError):
    """Raised when a teacher map cannot produce a valid patch partition."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _teacher_maps_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    entries = [
        {"sample_id": record["sample_id"], "sha256": record["teacher_map_sha256"]}
        for record in records
    ]
    return _sha256_json({"teacher_maps": entries})


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


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _ensure_writable_targets(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing patch-mask artifacts. Re-run with "
            "--overwrite: " + ", ".join(existing)
        )


def _voc_colormap() -> np.ndarray:
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for class_id in range(256):
        value = class_id
        for bit in range(8):
            colormap[class_id, 0] |= ((value >> 0) & 1) << (7 - bit)
            colormap[class_id, 1] |= ((value >> 1) & 1) << (7 - bit)
            colormap[class_id, 2] |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
    return colormap


VOC_COLORMAP = _voc_colormap()
VOC_RGB_TO_CLASS = {
    tuple(int(channel) for channel in color): class_id
    for class_id, color in enumerate(VOC_COLORMAP)
}


def _decode_voc_foreground(
    rgb: np.ndarray,
    foreground_class_ids: Sequence[int],
    path: Path,
) -> tuple[np.ndarray, Dict[str, Any]]:
    flat = rgb.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    class_ids = np.empty(len(unique_colors), dtype=np.int16)
    unknown = []
    for index, color in enumerate(unique_colors):
        key = tuple(int(channel) for channel in color)
        class_id = VOC_RGB_TO_CLASS.get(key)
        if class_id is None:
            unknown.append(key)
            continue
        class_ids[index] = class_id
    if unknown:
        raise PatchMaskError(
            f"Teacher cmap {path} contains non-VOC colors: {unknown[:5]}"
        )
    decoded = class_ids[inverse].reshape(rgb.shape[:2])
    foreground = np.isin(decoded, np.asarray(foreground_class_ids, dtype=np.int16))
    return foreground.astype(np.float32), {
        "decoded_class_ids": sorted(int(value) for value in np.unique(decoded)),
        "foreground_class_ids": [int(value) for value in foreground_class_ids],
        "foreground_pixel_fraction_source": float(foreground.mean()),
    }


def _load_teacher_map(
    path: Path,
    *,
    map_format: str,
    foreground_class_ids: Sequence[int],
) -> tuple[torch.Tensor, Dict[str, Any]]:
    try:
        with Image.open(path) as image:
            image.load()
            source_mode = image.mode
            source_size = image.size
            if map_format == "voc_colormap_class_ids":
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                array, decoder_metadata = _decode_voc_foreground(
                    rgb, foreground_class_ids, path
                )
            elif map_format == "binary_grayscale":
                grayscale = np.asarray(image.convert("L"), dtype=np.float32).copy()
                array = (grayscale > 0).astype(np.float32)
                decoder_metadata = {
                    "decoded_class_ids": [int(value) for value in np.unique(array)],
                    "foreground_class_ids": [1],
                    "foreground_pixel_fraction_source": float(array.mean()),
                }
            else:
                raise PatchMaskError(f"Unsupported teacher-map format: {map_format!r}")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PatchMaskError(f"Could not read teacher map {path}: {exc}") from exc

    if array.ndim != 2 or array.size == 0:
        raise PatchMaskError(
            f"Teacher map {path} must be a non-empty 2D image, found {array.shape}."
        )
    if not np.isfinite(array).all():
        raise PatchMaskError(f"Teacher map {path} contains non-finite values.")

    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    metadata = {
        "source_mode": source_mode,
        "source_width": int(source_size[0]),
        "source_height": int(source_size[1]),
        "teacher_map_format": map_format,
        "source_min_decoded": float(array.min()),
        "source_max_decoded": float(array.max()),
        "source_mean_decoded": float(array.mean()),
        **decoder_metadata,
    }
    return tensor, metadata


def _apply_eval_geometry(
    tensor: torch.Tensor,
    *,
    eval_resize_size: int,
    image_size: int,
    interpolation: str,
) -> torch.Tensor:
    interpolation_modes = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }
    if interpolation not in interpolation_modes:
        raise PatchMaskError(f"Unsupported interpolation: {interpolation!r}")
    resized = TF.resize(
        tensor,
        eval_resize_size,
        interpolation=interpolation_modes[interpolation],
        antialias=False,
    )
    return TF.center_crop(resized, [image_size, image_size])


def teacher_map_to_patch_scores(
    teacher_map_path: str | Path,
    *,
    image_size: int,
    patch_size: int,
    normalize_to_unit_interval: bool,
    interpolation: str,
    eval_resize_size: int,
    map_format: str,
    foreground_class_ids: Sequence[int],
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Decode one map, apply eval geometry, and pool over the ViT patch grid."""

    path = Path(teacher_map_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing teacher map: {path}")
    if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
        raise PatchMaskError(
            "image_size and patch_size must be positive, with image_size divisible "
            "by patch_size."
        )

    teacher_map, source_metadata = _load_teacher_map(
        path,
        map_format=map_format,
        foreground_class_ids=foreground_class_ids,
    )
    resized = _apply_eval_geometry(
        teacher_map,
        eval_resize_size=eval_resize_size,
        image_size=image_size,
        interpolation=interpolation,
    )
    if normalize_to_unit_interval:
        resized = resized.clamp(0.0, 1.0)

    patch_grid = F.avg_pool2d(resized, kernel_size=patch_size, stride=patch_size)
    patch_scores = patch_grid.flatten().to(dtype=torch.float32, device="cpu")
    expected = (image_size // patch_size) ** 2
    if patch_scores.numel() != expected:
        raise PatchMaskError(
            f"Expected {expected} patch scores for {path}, found {patch_scores.numel()}."
        )
    if not torch.isfinite(patch_scores).all():
        raise PatchMaskError(f"Patch scores for {path} contain non-finite values.")
    if normalize_to_unit_interval and (
        float(patch_scores.min()) < -1e-6 or float(patch_scores.max()) > 1.0 + 1e-6
    ):
        raise PatchMaskError(f"Normalized patch scores for {path} are outside [0, 1].")

    source_metadata.update(
        {
            "resized_min": float(resized.min()),
            "resized_max": float(resized.max()),
            "resized_mean": float(resized.mean()),
            "patch_min": float(patch_scores.min()),
            "patch_max": float(patch_scores.max()),
            "patch_mean": float(patch_scores.mean()),
        }
    )
    return patch_scores, source_metadata


def _aligned_image(path: Path, *, eval_resize_size: int, image_size: int) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image for patch-mask preflight: {path}")
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image = TF.resize(
                image,
                eval_resize_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            return TF.center_crop(image, [image_size, image_size]).copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PatchMaskError(f"Could not read preflight image {path}: {exc}") from exc


def _native_image_dimensions(path: Path) -> Dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source image for teacher-map alignment: {path}")
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PatchMaskError(f"Could not read source image {path}: {exc}") from exc
    return {"image_native_width": int(width), "image_native_height": int(height)}


def _write_preflight_overlay(
    image_path: Path,
    patch_scores: torch.Tensor,
    output_path: Path,
    *,
    eval_resize_size: int,
    image_size: int,
    patch_size: int,
    evidence_threshold: float,
    background_threshold: float,
) -> None:
    image = _aligned_image(
        image_path, eval_resize_size=eval_resize_size, image_size=image_size
    )
    scores = patch_scores.reshape(image_size // patch_size, image_size // patch_size)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for row in range(scores.shape[0]):
        for column in range(scores.shape[1]):
            value = float(scores[row, column])
            if value >= evidence_threshold:
                fill = (220, 40, 40, 95)
            elif value <= background_threshold:
                fill = (35, 105, 220, 55)
            else:
                fill = (245, 190, 40, 70)
            x0, y0 = column * patch_size, row * patch_size
            x1, y1 = x0 + patch_size - 1, y0 + patch_size - 1
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(255, 255, 255, 110))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def partition_patch_scores(
    patch_scores: torch.Tensor,
    *,
    evidence_threshold: float,
    background_threshold: float,
) -> Dict[str, torch.Tensor]:
    """Partition flattened patch scores into evidence/background/ambiguous indices."""

    scores = patch_scores.detach().to(dtype=torch.float32, device="cpu").flatten()
    if scores.numel() == 0 or not torch.isfinite(scores).all():
        raise PatchMaskError("Patch scores must be non-empty and finite.")
    if not 0.0 <= background_threshold < evidence_threshold <= 1.0:
        raise PatchMaskError(
            "Thresholds must satisfy 0 <= background < evidence <= 1."
        )

    evidence_idx = torch.nonzero(scores >= evidence_threshold, as_tuple=False).flatten()
    background_idx = torch.nonzero(scores <= background_threshold, as_tuple=False).flatten()
    ambiguous_idx = torch.nonzero(
        (scores > background_threshold) & (scores < evidence_threshold),
        as_tuple=False,
    ).flatten()

    combined = torch.cat([evidence_idx, background_idx, ambiguous_idx])
    if combined.numel() != scores.numel() or torch.unique(combined).numel() != scores.numel():
        raise PatchMaskError("Patch categories do not form a complete disjoint partition.")
    return {
        "evidence_idx": evidence_idx.to(torch.long),
        "background_idx": background_idx.to(torch.long),
        "ambiguous_idx": ambiguous_idx.to(torch.long),
    }


def _coverage(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _eligibility_reason(evidence_count: int, background_count: int, minimum_bg: int) -> str:
    reasons = []
    if evidence_count == 0:
        reasons.append("no_evidence_patches")
    if background_count < minimum_bg:
        reasons.append("insufficient_background_patches")
    return "eligible" if not reasons else ";".join(reasons)


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def prepare_patch_masks(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    require_all_eligible: bool = False,
) -> Dict[str, Any]:
    """Run Step 3 fail-closed so a failed rerun cannot expose stale masks."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "patch_masks": destination / "patch_masks_val.pt",
        "audit": destination / "patch_masks_val_audit.csv",
        "summary": destination / "patch_masks_val_summary.json",
        "preflight_index": destination / "preflight_overlays.csv",
    }
    _ensure_writable_targets(list(artifact_paths.values()), overwrite)
    if overwrite:
        for path in artifact_paths.values():
            if path.is_file():
                path.unlink()
        overlay_dir = destination / "preflight_overlays"
        if overlay_dir.is_dir():
            shutil.rmtree(overlay_dir)
    _atomic_json(
        {
            "schema_version": 2,
            "artifact_type": "fcv_vit_patch_masks",
            "status": "preflight_in_progress",
            "manifest_path": str(manifest_path),
            "patch_mask_path": str(artifact_paths["patch_masks"]),
            "stale_patch_mask_invalidated": True,
        },
        artifact_paths["summary"],
    )
    failure_context: Dict[str, Any] = {}
    try:
        return _prepare_patch_masks_impl(
            config,
            manifest_path,
            destination,
            overwrite=True,
            require_all_eligible=require_all_eligible,
            failure_context=failure_context,
        )
    except Exception as exc:
        existing: Dict[str, Any] = {}
        try:
            with artifact_paths["summary"].open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, ValueError, TypeError):
            existing = {}
        if existing.get("status") != "failed_acceptance":
            failure = {
                "schema_version": 2,
                "artifact_type": "fcv_vit_patch_masks",
                "status": "failed_preprocessing",
                "manifest_path": str(manifest_path),
                "manifest_sha256": (
                    _sha256_file(manifest_path) if manifest_path.is_file() else None
                ),
                "patch_mask_path": str(artifact_paths["patch_masks"]),
                "patch_mask_exists": artifact_paths["patch_masks"].is_file(),
                "stale_patch_mask_invalidated": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
                **failure_context,
            }
            if artifact_paths["audit"].is_file():
                partial_audit = pd.read_csv(artifact_paths["audit"])
                failure_row = {
                    "processing_status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **failure_context,
                }
                partial_audit = pd.concat(
                    [partial_audit, pd.DataFrame([failure_row])], ignore_index=True
                )
                _atomic_csv(partial_audit, artifact_paths["audit"])
            else:
                _atomic_csv(
                    pd.DataFrame(
                        [
                            {
                                "processing_status": "failed",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                **failure_context,
                            }
                        ]
                    ),
                    artifact_paths["audit"],
                )
            if not artifact_paths["preflight_index"].is_file():
                _atomic_csv(
                    pd.DataFrame(
                        columns=[
                            "sample_id",
                            "image_path",
                            "teacher_map_path",
                            "overlay_path",
                            "fcv_eligible",
                        ]
                    ),
                    artifact_paths["preflight_index"],
                )
            persisted_audit = pd.read_csv(artifact_paths["audit"])
            processing_status = persisted_audit.get(
                "processing_status",
                pd.Series([None] * len(persisted_audit)),
            )
            completed_before_failure = int(
                (~processing_status.astype(str).eq("failed")).sum()
            )
            failure.update(
                {
                    "audit_path": str(artifact_paths["audit"].resolve()),
                    "audit_sha256": _sha256_file(artifact_paths["audit"]),
                    "audit_row_count": int(len(persisted_audit)),
                    "partial_audit_row_count": int(len(persisted_audit)),
                    "completed_sample_count_before_failure": (
                        completed_before_failure
                    ),
                    "preflight_overlay_index": str(
                        artifact_paths["preflight_index"].resolve()
                    ),
                    "preflight_overlay_index_sha256": _sha256_file(
                        artifact_paths["preflight_index"]
                    ),
                    "preflight_overlay_count": int(
                        len(pd.read_csv(artifact_paths["preflight_index"]))
                    ),
                }
            )
            _atomic_json(failure, artifact_paths["summary"])
        if artifact_paths["patch_masks"].is_file():
            artifact_paths["patch_masks"].unlink()
        raise


def _prepare_patch_masks_impl(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    require_all_eligible: bool = False,
    failure_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create the Step 3 patch-mask artifact for the biased validation holdout."""

    manifest_path = Path(manifest_path)
    destination = Path(output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing FCV validation manifest: {manifest_path}. Run prepare_metadata.py first."
        )

    manifest = pd.read_csv(manifest_path)
    missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns))
    if missing_columns:
        raise PatchMaskError(f"Validation manifest is missing columns: {missing_columns}")
    leaked_columns = sorted(FORBIDDEN_PUBLIC_COLUMNS.intersection(manifest.columns))
    if leaked_columns:
        raise PatchMaskError(
            "FCV patch preprocessing must use the leakage-safe public manifest; found "
            f"analysis-only columns: {leaked_columns}"
        )
    if manifest.empty:
        raise PatchMaskError("FCV validation manifest is empty.")
    if manifest["sample_id"].duplicated().any():
        duplicates = manifest.loc[manifest["sample_id"].duplicated(), "sample_id"].tolist()
        raise PatchMaskError(f"Validation manifest has duplicate sample IDs: {duplicates[:5]}")
    if set(manifest["study_split"].astype(str)) != {"biased_validation"}:
        raise PatchMaskError(
            "Step 3 only accepts the train-derived biased_validation manifest."
        )
    if set(manifest["source_split"].astype(str)) != {"train"}:
        raise PatchMaskError("Step 3 requires source_split='train'.")
    try:
        manifest_binding = validate_manifest_bundle(
            config, manifest_path, "biased_validation"
        )
    except ManifestProvenanceError as exc:
        raise PatchMaskError(str(exc)) from exc
    if failure_context is not None:
        failure_context.update(
            {
                "manifest_bundle_path": str(manifest_binding.bundle_path),
                "manifest_bundle_sha256": manifest_binding.bundle_sha256,
            }
        )

    teacher_exists = manifest["teacher_map_exists"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    if not teacher_exists.all():
        missing = manifest.loc[~teacher_exists, "teacher_map_path"].head(5).tolist()
        raise FileNotFoundError(
            "Validation manifest reports missing teacher maps. Rebuild complete maps before "
            f"Step 3. First paths: {missing}"
        )

    model_cfg = config["model"]
    teacher_cfg = config["data"]["teacher_maps"]
    fcv_cfg = config["fcv"]
    image_size = int(model_cfg["image_size"])
    patch_size = int(model_cfg["patch_size"])
    expected_patch_count = int(model_cfg["patch_grid_size"]) ** 2
    eval_resize_size = int(config["training"]["augmentation"]["eval_resize_size"])
    evidence_threshold = float(fcv_cfg["evidence_patch_threshold"])
    background_threshold = float(fcv_cfg["background_patch_threshold"])
    minimum_background_patches = int(fcv_cfg["minimum_background_patches"])
    minimum_eligible_fraction = float(fcv_cfg["minimum_eligible_fraction"])
    minimum_eligible_count_per_class = int(
        fcv_cfg["minimum_eligible_count_per_class"]
    )
    preflight_overlay_count = int(fcv_cfg["preflight_overlay_count"])
    map_format = str(teacher_cfg["format"])
    foreground_class_ids = [
        int(value) for value in teacher_cfg["foreground_class_ids"]
    ]
    preprocessing_config = {
        "teacher_map_source": str(teacher_cfg["source"]),
        "teacher_map_format": map_format,
        "foreground_class_ids": foreground_class_ids,
        "normalize_to_unit_interval": bool(teacher_cfg["normalize_to_unit_interval"]),
        "interpolation": str(teacher_cfg["interpolation"]),
        "spatial_transform": str(teacher_cfg["spatial_transform"]),
        "eval_resize_size": eval_resize_size,
        "image_size": image_size,
        "patch_size": patch_size,
        "patch_grid_size": int(model_cfg["patch_grid_size"]),
        "evidence_threshold": evidence_threshold,
        "background_threshold": background_threshold,
        "minimum_background_patches": minimum_background_patches,
        "minimum_eligible_fraction": minimum_eligible_fraction,
        "minimum_eligible_count_per_class": minimum_eligible_count_per_class,
        "ambiguous_patch_policy": str(fcv_cfg["ambiguous_patch_policy"]),
    }
    preprocessing_config_sha256 = _sha256_json(preprocessing_config)

    destination.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "patch_masks": destination / "patch_masks_val.pt",
        "audit": destination / "patch_masks_val_audit.csv",
        "summary": destination / "patch_masks_val_summary.json",
        "preflight_index": destination / "preflight_overlays.csv",
    }
    _ensure_writable_targets(list(artifact_paths.values()), overwrite)

    records = []
    audit_rows = []
    overlay_rows = []
    overlay_indices = set(
        np.linspace(
            0,
            len(manifest) - 1,
            num=min(preflight_overlay_count, len(manifest)),
            dtype=int,
        ).tolist()
    )
    for row_index, row in enumerate(manifest.itertuples(index=False)):
        map_path = Path(str(row.teacher_map_path))
        image_path = Path(str(row.image_path)).expanduser().resolve()
        if failure_context is not None:
            failure_context.clear()
            failure_context.update(
                {
                    "sample_id": str(row.sample_id),
                    "metadata_index": int(row.metadata_index),
                    "label": int(row.label),
                    "image_path": str(image_path),
                    "teacher_map_path": str(map_path.expanduser().resolve()),
                    "processing_stage": "validate_image_hash",
                }
            )
        observed_image_sha256 = _sha256_file(image_path)
        if observed_image_sha256 != str(row.image_sha256):
            raise PatchMaskError(
                f"Image bytes changed after manifest creation for {image_path}: "
                f"expected {row.image_sha256}, observed {observed_image_sha256}."
            )
        if failure_context is not None:
            failure_context["processing_stage"] = "read_image_dimensions"
        image_metadata = _native_image_dimensions(image_path)
        if failure_context is not None:
            failure_context["processing_stage"] = "hash_teacher_map"
        teacher_map_sha256 = _sha256_file(map_path)
        if failure_context is not None:
            failure_context["processing_stage"] = "decode_and_partition_teacher_map"
        patch_scores, map_metadata = teacher_map_to_patch_scores(
            map_path,
            image_size=image_size,
            patch_size=patch_size,
            normalize_to_unit_interval=bool(teacher_cfg["normalize_to_unit_interval"]),
            interpolation=str(teacher_cfg["interpolation"]),
            eval_resize_size=eval_resize_size,
            map_format=map_format,
            foreground_class_ids=foreground_class_ids,
        )
        dimensions_match = (
            int(map_metadata["source_width"]) == image_metadata["image_native_width"]
            and int(map_metadata["source_height"])
            == image_metadata["image_native_height"]
        )
        if not dimensions_match:
            raise PatchMaskError(
                f"Teacher map {map_path} has native size "
                f"{map_metadata['source_width']}x{map_metadata['source_height']}, but "
                f"image {image_path} is {image_metadata['image_native_width']}x"
                f"{image_metadata['image_native_height']}."
            )
        if patch_scores.numel() != expected_patch_count:
            raise PatchMaskError(
                f"{row.sample_id} produced {patch_scores.numel()} patches; expected "
                f"{expected_patch_count}."
            )
        partition = partition_patch_scores(
            patch_scores,
            evidence_threshold=evidence_threshold,
            background_threshold=background_threshold,
        )
        evidence_count = int(partition["evidence_idx"].numel())
        background_count = int(partition["background_idx"].numel())
        ambiguous_count = int(partition["ambiguous_idx"].numel())
        reason = _eligibility_reason(
            evidence_count, background_count, minimum_background_patches
        )
        eligible = reason == "eligible"
        coverage = {
            "evidence_frac": _coverage(evidence_count, expected_patch_count),
            "background_frac": _coverage(background_count, expected_patch_count),
            "ambiguous_frac": _coverage(ambiguous_count, expected_patch_count),
        }
        record = {
            "image_id": str(row.sample_id),
            "sample_id": str(row.sample_id),
            "metadata_index": int(row.metadata_index),
            "label": int(row.label),
            "teacher_map_path": str(map_path.resolve()),
            "image_path": str(image_path),
            "image_sha256": observed_image_sha256,
            "teacher_map_sha256": teacher_map_sha256,
            "patch_scores": patch_scores,
            **partition,
            "coverage": coverage,
            "fcv_eligible": eligible,
            "evidence_control_eligible": evidence_count > 0,
            "eligibility_reason": reason,
        }
        records.append(record)
        audit_rows.append(
            {
                "sample_id": record["sample_id"],
                "metadata_index": record["metadata_index"],
                "label": record["label"],
                "teacher_map_path": record["teacher_map_path"],
                "teacher_map_sha256": teacher_map_sha256,
                "image_sha256": observed_image_sha256,
                **image_metadata,
                "native_dimensions_match": dimensions_match,
                **map_metadata,
                "evidence_count": evidence_count,
                "background_count": background_count,
                "ambiguous_count": ambiguous_count,
                **coverage,
                "fcv_eligible": eligible,
                "eligibility_reason": reason,
            }
        )
        if row_index in overlay_indices:
            overlay_path = (
                destination / "preflight_overlays" / f"{record['sample_id']}.png"
            )
            if failure_context is not None:
                failure_context["processing_stage"] = "write_preflight_overlay"
            _write_preflight_overlay(
                Path(record["image_path"]),
                patch_scores,
                overlay_path,
                eval_resize_size=eval_resize_size,
                image_size=image_size,
                patch_size=patch_size,
                evidence_threshold=evidence_threshold,
                background_threshold=background_threshold,
            )
            overlay_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "image_path": record["image_path"],
                    "teacher_map_path": record["teacher_map_path"],
                    "overlay_path": str(overlay_path.resolve()),
                    "fcv_eligible": eligible,
                    "evidence_count": evidence_count,
                    "background_count": background_count,
                }
            )
        # Persist the diagnostic prefix after every completed sample.  If a
        # later image/map/overlay fails, the outer fail-closed wrapper can
        # preserve every completed audit row instead of replacing it with a
        # generic one-line report.
        if failure_context is not None:
            failure_context["processing_stage"] = "persist_diagnostic_prefix"
        _atomic_csv(pd.DataFrame(audit_rows), artifact_paths["audit"])
        _atomic_csv(pd.DataFrame(overlay_rows), artifact_paths["preflight_index"])
        if failure_context is not None:
            failure_context["processing_stage"] = "completed_sample"

    audit = pd.DataFrame(audit_rows)
    eligible_count = int(audit["fcv_eligible"].sum())
    ineligible_count = int(len(audit) - eligible_count)
    eligible_by_class = {
        int(key): int(value)
        for key, value in audit.loc[audit["fcv_eligible"], "label"]
        .value_counts()
        .sort_index()
        .items()
    }
    class_ids = sorted(int(value) for value in audit["label"].unique())
    acceptance_errors = []
    if eligible_count / len(audit) < minimum_eligible_fraction:
        acceptance_errors.append(
            f"eligible_fraction={eligible_count / len(audit):.4f} is below "
            f"{minimum_eligible_fraction:.4f}"
        )
    for class_id in class_ids:
        count = eligible_by_class.get(class_id, 0)
        if count < minimum_eligible_count_per_class:
            acceptance_errors.append(
                f"class {class_id} has {count} eligible samples; requires "
                f"{minimum_eligible_count_per_class}"
            )
    if require_all_eligible and ineligible_count:
        acceptance_errors.append(
            f"require_all_eligible was set but {ineligible_count} samples are ineligible"
        )

    manifest_sha256 = _sha256_file(manifest_path)
    teacher_maps_sha256 = _teacher_maps_sha256(records)
    payload = {
        "schema_version": 2,
        "artifact_type": "fcv_vit_patch_masks",
        "split": "biased_validation",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "manifest_bundle_path": str(manifest_binding.bundle_path),
        "manifest_bundle_sha256": manifest_binding.bundle_sha256,
        "original_metadata_sha256": manifest_binding.original_metadata_sha256,
        "split_indices_sha256": manifest_binding.split_indices_sha256,
        "split_summary_sha256": manifest_binding.split_summary_sha256,
        "teacher_maps_sha256": teacher_maps_sha256,
        "preprocessing_config": preprocessing_config,
        "preprocessing_config_sha256": preprocessing_config_sha256,
        "image_size": image_size,
        "patch_size": patch_size,
        "patch_grid_size": int(model_cfg["patch_grid_size"]),
        "patch_count": expected_patch_count,
        "evidence_threshold": evidence_threshold,
        "background_threshold": background_threshold,
        "minimum_background_patches": minimum_background_patches,
        "normalization": "categorical_foreground_indicator_0_1",
        "teacher_map_format": map_format,
        "foreground_class_ids": foreground_class_ids,
        "spatial_transform": str(teacher_cfg["spatial_transform"]),
        "eval_resize_size": eval_resize_size,
        "interpolation": str(teacher_cfg["interpolation"]),
        "records": records,
        "sample_id_to_record_index": {
            record["sample_id"]: index for index, record in enumerate(records)
        },
    }
    summary = {
        "schema_version": 2,
        "artifact_type": payload["artifact_type"],
        "manifest_path": payload["manifest_path"],
        "manifest_sha256": manifest_sha256,
        "manifest_bundle_path": payload["manifest_bundle_path"],
        "manifest_bundle_sha256": payload["manifest_bundle_sha256"],
        "original_metadata_sha256": payload["original_metadata_sha256"],
        "split_indices_sha256": payload["split_indices_sha256"],
        "split_summary_sha256": payload["split_summary_sha256"],
        "teacher_maps_sha256": teacher_maps_sha256,
        "preprocessing_config": preprocessing_config,
        "preprocessing_config_sha256": preprocessing_config_sha256,
        "sample_count": int(len(records)),
        "eligible_count": eligible_count,
        "ineligible_count": ineligible_count,
        "eligible_fraction": float(eligible_count / len(records)),
        "eligible_counts_by_class": {
            str(class_id): eligible_by_class.get(class_id, 0)
            for class_id in class_ids
        },
        "acceptance_thresholds": {
            "minimum_eligible_fraction": minimum_eligible_fraction,
            "minimum_eligible_count_per_class": minimum_eligible_count_per_class,
        },
        "acceptance_errors": acceptance_errors,
        "status": "failed_acceptance" if acceptance_errors else "complete",
        "preflight_overlay_count": len(overlay_rows),
        "preflight_overlay_index": str(artifact_paths["preflight_index"].resolve()),
        "class_counts": {
            str(key): int(value)
            for key, value in audit["label"].value_counts().sort_index().items()
        },
        "ineligibility_reasons": {
            str(key): int(value)
            for key, value in audit.loc[
                ~audit["fcv_eligible"], "eligibility_reason"
            ].value_counts().items()
        },
        "thresholds": {
            "evidence": evidence_threshold,
            "background": background_threshold,
            "minimum_background_patches": minimum_background_patches,
        },
        "coverage_distributions": {
            "evidence_frac": _distribution(audit["evidence_frac"].tolist()),
            "background_frac": _distribution(audit["background_frac"].tolist()),
            "ambiguous_frac": _distribution(audit["ambiguous_frac"].tolist()),
        },
    }

    _atomic_csv(audit, artifact_paths["audit"])
    _atomic_csv(pd.DataFrame(overlay_rows), artifact_paths["preflight_index"])
    summary["audit_path"] = str(artifact_paths["audit"].resolve())
    summary["audit_sha256"] = _sha256_file(artifact_paths["audit"])
    summary["preflight_overlay_index_sha256"] = _sha256_file(
        artifact_paths["preflight_index"]
    )
    if acceptance_errors:
        _atomic_json(summary, artifact_paths["summary"])
        raise PatchMaskError(
            "Patch-mask preflight failed acceptance checks: "
            + "; ".join(acceptance_errors)
            + f". Diagnostics were written to {destination.resolve()}."
        )
    _atomic_torch_save(payload, artifact_paths["patch_masks"])
    summary["patch_mask_path"] = str(artifact_paths["patch_masks"].resolve())
    summary["patch_mask_sha256"] = _sha256_file(artifact_paths["patch_masks"])
    _atomic_json(summary, artifact_paths["summary"])
    return {
        "output_dir": str(destination.resolve()),
        "artifacts": {key: str(value.resolve()) for key, value in artifact_paths.items()},
        "summary": summary,
    }
