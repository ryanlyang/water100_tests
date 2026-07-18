"""Audit and project DecoyMNIST OpenCLIP+DINO maps into ViT patch space."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, UnidentifiedImageError

from decoy_data import locate_decoy_patch
from decoy_full_config import canonical_config_sha256, sha256_file
from decoy_manifest_provenance import (
    ManifestBinding,
    atomic_json,
    validate_manifest_bundle,
)


class TeacherMaskError(ValueError):
    """Raised when primary teacher maps fail the locked geometric preflight."""


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

CATEGORY_BACKGROUND = 0
CATEGORY_AMBIGUOUS = 1
CATEGORY_EVIDENCE = 2


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def teacher_map_candidates(
    teacher_root: str | Path,
    *,
    sample_id: str,
    image_rel_path: str,
    label: int,
) -> List[Path]:
    """Return the finite set of supported established DecoyMNIST map names."""

    root = Path(teacher_root).expanduser().resolve()
    stem = Path(str(image_rel_path)).stem
    names = [
        f"{sample_id}.png",
        f"{label}_{stem}.png",
        f"{stem}.png",
        f"train_{label}_{stem}.png",
    ]
    unique_names = list(dict.fromkeys(names))
    candidates = []
    for name in unique_names:
        candidates.extend((root / name, root / str(label) / name))
    return list(dict.fromkeys(path.resolve() for path in candidates))


def resolve_teacher_map(
    teacher_root: str | Path,
    *,
    sample_id: str,
    image_rel_path: str,
    label: int,
) -> Tuple[Path | None, List[Path]]:
    candidates = teacher_map_candidates(
        teacher_root,
        sample_id=sample_id,
        image_rel_path=image_rel_path,
        label=int(label),
    )
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None, candidates
    if len(existing) > 1:
        hashes = {sha256_file(path) for path in existing}
        if len(hashes) != 1:
            raise TeacherMaskError(
                f"Conflicting teacher-map aliases for {sample_id}: "
                + ", ".join(str(path) for path in existing)
            )
    return existing[0], candidates


def decode_voc_foreground(
    path: str | Path, foreground_class_ids: Sequence[int]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    map_path = Path(path).expanduser().resolve()
    try:
        with Image.open(map_path) as image:
            image.load()
            source_mode = image.mode
            source_size = image.size
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise TeacherMaskError(f"Could not read teacher map {map_path}: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise TeacherMaskError(f"Teacher map must decode to RGB, found {rgb.shape}.")
    if int(source_size[0]) != int(source_size[1]):
        raise TeacherMaskError(
            f"Teacher map must be square for direct-resize alignment: {map_path} "
            f"is {source_size[0]}x{source_size[1]}."
        )
    flat = rgb.reshape(-1, 3)
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    classes = np.empty(len(colors), dtype=np.int16)
    unknown = []
    for index, color in enumerate(colors):
        key = tuple(int(channel) for channel in color)
        class_id = VOC_RGB_TO_CLASS.get(key)
        if class_id is None:
            unknown.append(key)
        else:
            classes[index] = class_id
    if unknown:
        raise TeacherMaskError(
            f"Teacher map {map_path} contains non-VOC colors: {unknown[:5]}"
        )
    decoded = classes[inverse].reshape(rgb.shape[:2])
    foreground = np.isin(
        decoded, np.asarray([int(value) for value in foreground_class_ids])
    ).astype(np.uint8)
    return foreground, {
        "source_mode": source_mode,
        "source_width": int(source_size[0]),
        "source_height": int(source_size[1]),
        "decoded_class_ids": sorted(int(value) for value in np.unique(decoded)),
        "foreground_pixel_fraction_source": float(foreground.mean()),
    }


def resize_binary_mask(
    mask: np.ndarray, *, image_size: int, interpolation: str
) -> np.ndarray:
    if mask.ndim != 2:
        raise TeacherMaskError(f"Expected a two-dimensional mask, found {mask.shape}.")
    interpolation_modes = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }
    if interpolation not in interpolation_modes:
        raise TeacherMaskError(f"Unsupported map interpolation: {interpolation!r}")
    image = Image.fromarray((mask.astype(np.float32) * 255.0).astype(np.uint8), mode="L")
    resized = image.resize(
        (int(image_size), int(image_size)), resample=interpolation_modes[interpolation]
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def pool_patch_occupancy(
    resized_mask: np.ndarray, *, patch_size: int
) -> np.ndarray:
    if resized_mask.ndim != 2 or resized_mask.shape[0] != resized_mask.shape[1]:
        raise TeacherMaskError("Patch pooling requires a square 2D mask.")
    size = int(resized_mask.shape[0])
    if size % int(patch_size) != 0:
        raise TeacherMaskError("Mask size must be divisible by patch size.")
    grid = size // int(patch_size)
    scores = resized_mask.reshape(
        grid, int(patch_size), grid, int(patch_size)
    ).mean(axis=(1, 3))
    return scores.reshape(-1).astype(np.float32)


def partition_patch_scores(
    scores: np.ndarray, *, background_threshold: float, evidence_threshold: float
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise TeacherMaskError("Patch scores must be finite and nonempty.")
    if not 0.0 <= background_threshold < evidence_threshold <= 1.0:
        raise TeacherMaskError("Thresholds must satisfy 0 <= background < evidence <= 1.")
    categories = np.full(values.shape, CATEGORY_AMBIGUOUS, dtype=np.uint8)
    categories[values <= background_threshold] = CATEGORY_BACKGROUND
    categories[values >= evidence_threshold] = CATEGORY_EVIDENCE
    return categories


def exact_source_masks(
    image_path: str | Path, label: int, *, image_size: int
) -> Dict[str, np.ndarray]:
    path = Path(image_path).expanduser().resolve()
    with Image.open(path) as image:
        if image.mode != "L":
            raise TeacherMaskError(
                f"Expected mode-L DecoyMNIST source, found {image.mode!r}: {path}"
            )
        grayscale = np.asarray(image, dtype=np.uint8).copy()
    rows, columns = locate_decoy_patch(grayscale, int(label), "train")
    digit = (grayscale > 0).astype(np.uint8)
    digit[rows, columns] = 0
    decoy = np.zeros_like(digit)
    decoy[rows, columns] = 1
    return {
        "digit": resize_binary_mask(digit, image_size=image_size, interpolation="nearest"),
        "decoy": resize_binary_mask(decoy, image_size=image_size, interpolation="nearest"),
    }


def _mask_iou(left: np.ndarray, right: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(left) > 0.5
    b = np.asarray(right) > 0.5
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    total = int(a.sum() + b.sum())
    iou = float(intersection / union) if union else 1.0
    dice = float(2 * intersection / total) if total else 1.0
    return iou, dice


def _teacher_set_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = [
        {"sample_id": row["sample_id"], "sha256": row["teacher_map_sha256"]}
        for row in rows
    ]
    return _json_sha256({"teacher_maps": payload})


def _write_overlay(
    image_path: Path,
    scores: np.ndarray,
    output_path: Path,
    *,
    image_size: int,
    patch_size: int,
    background_threshold: float,
    evidence_threshold: float,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(
            (image_size, image_size), resample=Image.Resampling.BICUBIC
        )
    categories = partition_patch_scores(
        scores,
        background_threshold=background_threshold,
        evidence_threshold=evidence_threshold,
    ).reshape(image_size // patch_size, image_size // patch_size)
    draw = ImageDraw.Draw(image, "RGBA")
    colors = {
        CATEGORY_BACKGROUND: (35, 105, 220, 50),
        CATEGORY_AMBIGUOUS: (245, 190, 40, 75),
        CATEGORY_EVIDENCE: (220, 40, 40, 100),
    }
    for row in range(categories.shape[0]):
        for column in range(categories.shape[1]):
            x0, y0 = column * patch_size, row * patch_size
            x1, y1 = x0 + patch_size - 1, y0 + patch_size - 1
            draw.rectangle(
                (x0, y0, x1, y1),
                fill=colors[int(categories[row, column])],
                outline=(255, 255, 255, 100),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _class_counts(frame: pd.DataFrame, mask: pd.Series) -> Dict[str, int]:
    selected = frame.loc[mask, "label"].astype(int).value_counts().sort_index()
    return {str(label): int(selected.get(label, 0)) for label in range(10)}


def _artifact_paths(destination: Path) -> Dict[str, Path]:
    return {
        "coverage": destination / "teacher_map_coverage.csv",
        "missing": destination / "missing_teacher_maps.csv",
        "regeneration": destination / "missing_map_regeneration_request.json",
        "audit": destination / "projected_teacher_masks_audit.csv",
        "masks": destination / "projected_teacher_masks.npz",
        "provenance": destination / "projected_teacher_masks_provenance.json",
        "overlay_index": destination / "preflight_overlays.csv",
        "summary": destination / "teacher_mask_preflight_summary.json",
    }


def _ensure_writable(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to replace teacher-mask preflight artifacts without --overwrite: "
            + ", ".join(existing)
        )


def _write_failure(
    summary_path: Path,
    *,
    config: Mapping[str, Any],
    manifest_path: Path,
    status: str,
    error: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    atomic_json(
        {
            "artifact_type": "fcv_vit_decoymnist_teacher_mask_preflight",
            "artifact_version": 1,
            "status": status,
            "error": error,
            "config_sha256": canonical_config_sha256(config),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            **dict(details or {}),
        },
        summary_path,
    )


def prepare_teacher_masks(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Resolve all primary maps, project to 14x14, and fail closed on audit errors."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = _artifact_paths(destination)
    _ensure_writable(list(artifacts.values()), overwrite)
    if overwrite:
        for key in ("masks", "provenance"):
            if artifacts[key].is_file():
                artifacts[key].unlink()

    binding = validate_manifest_bundle(config, manifest_path, "biased_validation")
    manifest = pd.read_csv(manifest_path)
    teacher_root = Path(config["paths"]["teacher_map_root"]).expanduser().resolve()
    if not teacher_root.is_dir():
        raise FileNotFoundError(f"Missing primary teacher-map root: {teacher_root}")
    data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
    teacher_cfg = config["data"]["teacher_maps"]
    fcv_cfg = config["fcv"]
    model_cfg = config["model"]
    image_size = int(model_cfg["image_size"])
    patch_size = int(model_cfg["patch_size"])
    patch_count = int(model_cfg["patch_grid_size"]) ** 2
    background_threshold = float(fcv_cfg["background_patch_threshold"])
    evidence_threshold = float(fcv_cfg["evidence_patch_threshold"])

    coverage_rows = []
    resolved_paths: Dict[str, Path] = {}
    for row in manifest.itertuples(index=False):
        resolved, candidates = resolve_teacher_map(
            teacher_root,
            sample_id=str(row.sample_id),
            image_rel_path=str(row.image_rel_path),
            label=int(row.label),
        )
        expected = candidates[2] if len(candidates) > 2 else candidates[0]
        coverage_rows.append(
            {
                "sample_id": str(row.sample_id),
                "image_rel_path": str(row.image_rel_path),
                "label": int(row.label),
                "teacher_map_found": resolved is not None,
                "teacher_map_path": str(resolved) if resolved is not None else "",
                "expected_primary_path": str(expected),
            }
        )
        if resolved is not None:
            resolved_paths[str(row.sample_id)] = resolved
    coverage = pd.DataFrame(coverage_rows)
    missing = coverage.loc[~coverage["teacher_map_found"]].copy()
    _atomic_csv(coverage, artifacts["coverage"])
    _atomic_csv(missing, artifacts["missing"])
    regeneration_request = {
        "artifact_type": "fcv_vit_decoymnist_missing_teacher_map_request",
        "status": "not_needed" if missing.empty else "required",
        "missing_count": int(len(missing)),
        "missing_manifest": str(artifacts["missing"]),
        "teacher_source_must_remain": teacher_cfg["source"],
        "do_not_substitute_another_teacher": True,
        "established_generator": str(
            Path(config["paths"]["repository_root"])
            / "RightForTheRightRegions"
            / "pipelines"
            / "generate_r4rr_maps"
            / "generate_pseudo_masks_DecoyMNIST.py"
        ),
        "instruction": (
            "Regenerate only IDs in missing_teacher_maps.csv with the frozen "
            "OpenCLIP+DINO pipeline, then rerun this preflight."
        ),
    }
    atomic_json(regeneration_request, artifacts["regeneration"])
    if not missing.empty:
        error = f"Primary teacher-map coverage is incomplete: {len(missing)} missing."
        _write_failure(
            artifacts["summary"],
            config=config,
            manifest_path=manifest_path,
            status="failed_missing_teacher_maps",
            error=error,
            details={
                "missing_count": int(len(missing)),
                "missing_manifest": str(artifacts["missing"]),
                "regeneration_request": str(artifacts["regeneration"]),
            },
        )
        raise FileNotFoundError(error)

    overlay_indices = set(
        np.linspace(
            0,
            len(manifest) - 1,
            num=min(int(teacher_cfg["preflight_overlay_count"]), len(manifest)),
            dtype=int,
        ).tolist()
    )
    scores_rows: List[np.ndarray] = []
    category_rows: List[np.ndarray] = []
    eligible_rows: List[bool] = []
    audit_rows: List[Dict[str, Any]] = []
    overlay_rows = []
    try:
        for row_index, row in enumerate(manifest.itertuples(index=False)):
            sample = str(row.sample_id)
            image_path = (data_root / str(row.image_rel_path)).resolve()
            observed_image_hash = sha256_file(image_path)
            if observed_image_hash != str(row.image_sha256):
                raise TeacherMaskError(
                    f"Source image changed after manifest creation: {sample}."
                )
            map_path = resolved_paths[sample]
            foreground, map_metadata = decode_voc_foreground(
                map_path, teacher_cfg["foreground_class_ids"]
            )
            resized_teacher = resize_binary_mask(
                foreground,
                image_size=image_size,
                interpolation=str(teacher_cfg["resize_interpolation"]),
            )
            scores = pool_patch_occupancy(resized_teacher, patch_size=patch_size)
            if scores.shape != (patch_count,):
                raise TeacherMaskError(
                    f"{sample} produced {scores.size} patch scores; expected {patch_count}."
                )
            categories = partition_patch_scores(
                scores,
                background_threshold=background_threshold,
                evidence_threshold=evidence_threshold,
            )
            exact = exact_source_masks(image_path, int(row.label), image_size=image_size)
            digit_iou, digit_dice = _mask_iou(resized_teacher, exact["digit"])
            decoy_patch_scores = pool_patch_occupancy(
                exact["decoy"], patch_size=patch_size
            )
            decoy_cells = np.flatnonzero(decoy_patch_scores > 0.0)
            if decoy_cells.size == 0:
                raise TeacherMaskError(f"Exact decoy occupies no ViT cells for {sample}.")
            decoy_categories = categories[decoy_cells]
            decoy_evidence_count = int(
                (decoy_categories == CATEGORY_EVIDENCE).sum()
            )
            decoy_background_count = int(
                (decoy_categories == CATEGORY_BACKGROUND).sum()
            )
            decoy_safe = bool(
                decoy_background_count == int(decoy_cells.size)
            )
            evidence_count = int((categories == CATEGORY_EVIDENCE).sum())
            background_count = int((categories == CATEGORY_BACKGROUND).sum())
            ambiguous_count = int((categories == CATEGORY_AMBIGUOUS).sum())
            reasons = []
            if evidence_count < 1:
                reasons.append("no_evidence_patches")
            if background_count < int(fcv_cfg["minimum_background_patches"]):
                reasons.append("insufficient_background_patches")
            if bool(fcv_cfg["require_decoy_region_safe_background"]) and not decoy_safe:
                reasons.append("decoy_region_not_safe_background")
            eligible = not reasons
            teacher_hash = sha256_file(map_path)
            audit_rows.append(
                {
                    "sample_id": sample,
                    "image_rel_path": str(row.image_rel_path),
                    "label": int(row.label),
                    "image_sha256": observed_image_hash,
                    "teacher_map_path": str(map_path),
                    "teacher_map_sha256": teacher_hash,
                    **map_metadata,
                    "patch_count": patch_count,
                    "evidence_count": evidence_count,
                    "background_count": background_count,
                    "ambiguous_count": ambiguous_count,
                    "decoy_patch_cell_count": int(decoy_cells.size),
                    "decoy_background_count": decoy_background_count,
                    "decoy_evidence_count": decoy_evidence_count,
                    "decoy_region_safe_background": decoy_safe,
                    "teacher_exact_digit_iou_analysis_only": digit_iou,
                    "teacher_exact_digit_dice_analysis_only": digit_dice,
                    "fcv_eligible": eligible,
                    "eligibility_reason": "eligible" if eligible else ";".join(reasons),
                }
            )
            scores_rows.append(scores)
            category_rows.append(categories)
            eligible_rows.append(eligible)
            if row_index in overlay_indices:
                overlay_path = destination / "preflight_overlays" / f"{sample}.png"
                _write_overlay(
                    image_path,
                    scores,
                    overlay_path,
                    image_size=image_size,
                    patch_size=patch_size,
                    background_threshold=background_threshold,
                    evidence_threshold=evidence_threshold,
                )
                overlay_rows.append(
                    {
                        "sample_id": sample,
                        "image_rel_path": str(row.image_rel_path),
                        "teacher_map_path": str(map_path),
                        "overlay_path": str(overlay_path),
                        "fcv_eligible": eligible,
                    }
                )
    except Exception as exc:
        if audit_rows:
            _atomic_csv(pd.DataFrame(audit_rows), artifacts["audit"])
        _atomic_csv(pd.DataFrame(overlay_rows), artifacts["overlay_index"])
        _write_failure(
            artifacts["summary"],
            config=config,
            manifest_path=manifest_path,
            status="failed_projection",
            error=str(exc),
            details={"completed_before_failure": len(audit_rows)},
        )
        raise

    audit = pd.DataFrame(audit_rows)
    overlays = pd.DataFrame(overlay_rows)
    _atomic_csv(audit, artifacts["audit"])
    _atomic_csv(overlays, artifacts["overlay_index"])
    eligible_mask = audit["fcv_eligible"].astype(bool)
    eligible_fraction = float(eligible_mask.mean())
    eligible_by_class = _class_counts(audit, eligible_mask)
    decoy_evidence_targets = int((audit["decoy_evidence_count"] > 0).sum())
    acceptance_errors = []
    if eligible_fraction < float(fcv_cfg["minimum_eligible_fraction"]):
        acceptance_errors.append(
            f"eligible_fraction={eligible_fraction:.6f} below "
            f"{fcv_cfg['minimum_eligible_fraction']}"
        )
    for label, count in eligible_by_class.items():
        if count < int(fcv_cfg["minimum_eligible_per_class"]):
            acceptance_errors.append(
                f"class {label} has {count} eligible; requires "
                f"{fcv_cfg['minimum_eligible_per_class']}"
            )
    if bool(fcv_cfg["reject_any_teacher_evidence_on_decoy"]) and decoy_evidence_targets:
        acceptance_errors.append(
            f"teacher evidence overlaps the decoy in {decoy_evidence_targets} targets"
        )

    teacher_set_hash = _teacher_set_sha256(audit_rows)
    summary = {
        "artifact_type": "fcv_vit_decoymnist_teacher_mask_preflight",
        "artifact_version": 1,
        "status": "failed_acceptance" if acceptance_errors else "accepted",
        "config_sha256": canonical_config_sha256(config),
        "manifest_path": str(manifest_path),
        "manifest_sha256": binding.manifest_sha256,
        "manifest_bundle_path": str(binding.bundle_path),
        "manifest_bundle_sha256": binding.bundle_sha256,
        "teacher_map_root": str(teacher_root),
        "teacher_map_count": int(len(audit)),
        "teacher_maps_sha256": teacher_set_hash,
        "patch_grid_size": int(model_cfg["patch_grid_size"]),
        "patch_count": patch_count,
        "evidence_threshold": evidence_threshold,
        "background_threshold": background_threshold,
        "eligible_count": int(eligible_mask.sum()),
        "eligible_fraction": eligible_fraction,
        "eligible_by_class": eligible_by_class,
        "decoy_evidence_target_count": decoy_evidence_targets,
        "mean_teacher_exact_digit_iou_analysis_only": float(
            audit["teacher_exact_digit_iou_analysis_only"].mean()
        ),
        "overlay_count": int(len(overlays)),
        "audit_path": str(artifacts["audit"]),
        "overlay_index": str(artifacts["overlay_index"]),
        "acceptance_errors": acceptance_errors,
    }
    if acceptance_errors:
        atomic_json(summary, artifacts["summary"])
        raise TeacherMaskError(
            "Teacher-mask acceptance failed; diagnostics were preserved: "
            + "; ".join(acceptance_errors)
        )

    _atomic_npz(
        artifacts["masks"],
        sample_ids=np.asarray(audit["sample_id"].astype(str).tolist(), dtype=str),
        labels=np.asarray(audit["label"], dtype=np.int16),
        patch_scores=np.stack(scores_rows).astype(np.float32),
        patch_categories=np.stack(category_rows).astype(np.uint8),
        fcv_eligible=np.asarray(eligible_rows, dtype=np.bool_),
    )
    provenance = {
        "artifact_type": "fcv_vit_decoymnist_projected_teacher_masks",
        "artifact_version": 1,
        "config_sha256": canonical_config_sha256(config),
        "manifest_sha256": binding.manifest_sha256,
        "manifest_bundle_sha256": binding.bundle_sha256,
        "teacher_maps_sha256": teacher_set_hash,
        "mask_artifact_path": str(artifacts["masks"]),
        "mask_artifact_sha256": sha256_file(artifacts["masks"]),
        "sample_count": int(len(audit)),
        "patch_count": patch_count,
        "category_encoding": {
            "background": CATEGORY_BACKGROUND,
            "ambiguous": CATEGORY_AMBIGUOUS,
            "evidence": CATEGORY_EVIDENCE,
        },
    }
    atomic_json(provenance, artifacts["provenance"])
    summary.update(
        {
            "mask_artifact_path": str(artifacts["masks"]),
            "mask_artifact_sha256": provenance["mask_artifact_sha256"],
            "provenance_path": str(artifacts["provenance"]),
            "provenance_sha256": sha256_file(artifacts["provenance"]),
        }
    )
    atomic_json(summary, artifacts["summary"])
    return {
        "output_dir": str(destination),
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "summary": summary,
    }


def load_projected_teacher_masks(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    mask_artifact_path: str | Path,
) -> Tuple[Dict[str, np.ndarray], ManifestBinding]:
    """Load only an accepted, manifest-bound, non-pickle NumPy mask artifact."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    mask_path = Path(mask_artifact_path).expanduser().resolve()
    binding = validate_manifest_bundle(config, manifest_path, "biased_validation")
    provenance_path = mask_path.parent / "projected_teacher_masks_provenance.json"
    summary_path = mask_path.parent / "teacher_mask_preflight_summary.json"
    if not mask_path.is_file() or not provenance_path.is_file() or not summary_path.is_file():
        raise TeacherMaskError("Projected-mask artifact is incomplete.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "accepted":
        raise TeacherMaskError("Projected masks did not pass preflight acceptance.")
    if (
        provenance.get("config_sha256") != canonical_config_sha256(config)
        or provenance.get("manifest_sha256") != binding.manifest_sha256
        or provenance.get("manifest_bundle_sha256") != binding.bundle_sha256
        or provenance.get("mask_artifact_sha256") != sha256_file(mask_path)
    ):
        raise TeacherMaskError("Projected-mask provenance is stale or mismatched.")
    with np.load(mask_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    manifest = pd.read_csv(manifest_path)
    if arrays["sample_ids"].astype(str).tolist() != manifest["sample_id"].astype(str).tolist():
        raise TeacherMaskError("Projected-mask rows do not align with the manifest.")
    expected_shape = (len(manifest), int(config["model"]["patch_grid_size"]) ** 2)
    if arrays["patch_scores"].shape != expected_shape:
        raise TeacherMaskError("Projected patch-score shape is incorrect.")
    if arrays["patch_categories"].shape != expected_shape:
        raise TeacherMaskError("Projected patch-category shape is incorrect.")
    return arrays, binding

