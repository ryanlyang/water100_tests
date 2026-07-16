"""Prepare leakage-safe Waterbirds100 manifests for the first FCV study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .manifest_provenance import write_manifest_bundle


CLASS_NAMES = {0: "Landbird", 1: "Waterbird"}
GROUP_NAMES = {
    0: "Land_on_Land",
    1: "Land_on_Water",
    2: "Water_on_Land",
    3: "Water_on_Water",
}

PUBLIC_COLUMNS = [
    "sample_id",
    "metadata_index",
    "image_path",
    "image_rel_path",
    "image_sha256",
    "label",
    "class_name",
    "source_split",
    "study_split",
    "teacher_map_path",
    "teacher_map_exists",
]

ANALYSIS_COLUMNS = PUBLIC_COLUMNS + ["context", "group", "group_name"]


class MetadataError(ValueError):
    """Raised when Waterbirds metadata violates the study protocol."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_values(values: Iterable[int]) -> str:
    payload = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _resolve_image_path(data_root: Path, image_name: str) -> Path:
    raw = Path(str(image_name).strip())
    candidates = [raw] if raw.is_absolute() else [data_root / raw, data_root / "images" / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _teacher_map_candidates(teacher_root: Path, image_name: str) -> Sequence[Path]:
    relative = Path(str(image_name).strip().lstrip("/"))
    parent = relative.parent.name
    parent_underscored = parent.replace(".", "_")
    base = relative.stem + ".png"
    flat_name = f"{parent_underscored}_{relative.stem}.png"
    return (
        teacher_root / flat_name,
        teacher_root / parent_underscored / base,
        teacher_root / parent / base,
        teacher_root / relative.with_suffix(".png"),
    )


def _resolve_teacher_map(teacher_root: Path, image_name: str) -> Tuple[Path, bool]:
    candidates = _teacher_map_candidates(teacher_root, image_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), True
    return candidates[0].resolve(), False


def _validate_binary_column(frame: pd.DataFrame, column: str) -> None:
    values = set(frame[column].dropna().astype(int).unique().tolist())
    if not values.issubset({0, 1}):
        raise MetadataError(f"Column {column!r} must be binary, found {sorted(values)}.")


def _stratified_holdout_indices(
    frame: pd.DataFrame,
    label_column: str,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split dataframe positions deterministically while preserving class ratios."""

    rng = np.random.default_rng(seed)
    train_positions = []
    validation_positions = []

    for label in sorted(frame[label_column].astype(int).unique().tolist()):
        positions = np.flatnonzero(frame[label_column].to_numpy(dtype=int) == label)
        shuffled = rng.permutation(positions)
        validation_count = int(round(len(positions) * validation_fraction))
        if validation_count <= 0 or validation_count >= len(positions):
            raise MetadataError(
                f"Class {label} cannot support a {validation_fraction:.3f} holdout "
                f"with only {len(positions)} samples."
            )
        validation_positions.extend(shuffled[:validation_count].tolist())
        train_positions.extend(shuffled[validation_count:].tolist())

    return (
        np.asarray(sorted(train_positions), dtype=np.int64),
        np.asarray(sorted(validation_positions), dtype=np.int64),
    )


def _materialize_manifest(
    frame: pd.DataFrame,
    data_root: Path,
    teacher_root: Path,
    image_column: str,
    label_column: str,
    context_column: str,
    source_split: str,
    study_split: str,
    include_analysis_columns: bool,
) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        metadata_index = int(getattr(row, "metadata_index"))
        image_rel_path = str(getattr(row, image_column))
        label = int(getattr(row, label_column))
        context = int(getattr(row, context_column))
        group = label * 2 + context
        image_path = _resolve_image_path(data_root, image_rel_path)
        teacher_path, teacher_exists = _resolve_teacher_map(teacher_root, image_rel_path)
        record = {
            "sample_id": f"wb100_{metadata_index:05d}",
            "metadata_index": metadata_index,
            "image_path": str(image_path),
            "image_rel_path": image_rel_path,
            "image_sha256": _sha256_file(image_path) if image_path.is_file() else "MISSING",
            "label": label,
            "class_name": CLASS_NAMES[label],
            "source_split": source_split,
            "study_split": study_split,
            "teacher_map_path": str(teacher_path),
            "teacher_map_exists": bool(teacher_exists),
        }
        if include_analysis_columns:
            record.update(
                {
                    "context": context,
                    "group": group,
                    "group_name": GROUP_NAMES[group],
                }
            )
        rows.append(record)

    columns = ANALYSIS_COLUMNS if include_analysis_columns else PUBLIC_COLUMNS
    return pd.DataFrame(rows, columns=columns)


def _count_values(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    counts = frame[column].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _split_summary(
    source_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    label_column: str,
    context_column: str,
) -> Dict[str, Any]:
    labels = source_frame[label_column].astype(int)
    contexts = source_frame[context_column].astype(int)
    groups = labels * 2 + contexts
    return {
        "count": int(len(source_frame)),
        "class_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "context_counts": {
            str(k): int(v) for k, v in contexts.value_counts().sort_index().items()
        },
        "group_counts": {str(k): int(v) for k, v in groups.value_counts().sort_index().items()},
        "aligned_fraction": float((labels == contexts).mean()) if len(source_frame) else 0.0,
        "teacher_maps_found": int(manifest["teacher_map_exists"].sum()),
        "teacher_maps_missing": int((~manifest["teacher_map_exists"]).sum()),
        "image_set_sha256": hashlib.sha256(
            json.dumps(
                manifest[["sample_id", "image_sha256"]].to_dict("records"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _ensure_writable_targets(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing split artifacts. Re-run with --overwrite: "
            + ", ".join(existing)
        )


def prepare_waterbirds100_manifests(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    check_images: bool = True,
    require_holdout_teacher_maps: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create deterministic public and analysis-only manifests for Step 2."""

    paths = config["paths"]
    data_cfg = config["data"]
    columns = data_cfg["metadata_columns"]
    split_values = data_cfg["split_values"]
    holdout_cfg = data_cfg["biased_train_holdout"]

    data_root = Path(paths["data_root"])
    metadata_path = Path(paths["metadata"])
    teacher_root = Path(paths["teacher_map_root"])
    destination = Path(output_dir)

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Waterbirds metadata: {metadata_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing Waterbirds data root: {data_root}")
    if not teacher_root.is_dir():
        raise FileNotFoundError(f"Missing teacher-map root: {teacher_root}")

    metadata = pd.read_csv(metadata_path)
    required_columns = {
        columns["image_path"],
        columns["label"],
        columns["context"],
        columns["split"],
    }
    missing_columns = sorted(required_columns.difference(metadata.columns))
    if missing_columns:
        raise MetadataError(f"Metadata is missing columns: {missing_columns}")

    metadata = metadata.copy()
    metadata.insert(0, "metadata_index", np.arange(len(metadata), dtype=np.int64))
    label_column = columns["label"]
    context_column = columns["context"]
    image_column = columns["image_path"]
    split_column = columns["split"]
    _validate_binary_column(metadata, label_column)
    _validate_binary_column(metadata, context_column)

    train_source = metadata[
        metadata[split_column].astype(int) == int(split_values["train"])
    ].copy()
    oracle_source = metadata[
        metadata[split_column].astype(int) == int(split_values["original_validation"])
    ].copy()
    test_source = metadata[
        metadata[split_column].astype(int) == int(split_values["test"])
    ].copy()
    if train_source.empty or oracle_source.empty or test_source.empty:
        raise MetadataError("Train, original-validation, and test splits must all be non-empty.")

    mismatched_train = train_source[
        train_source[label_column].astype(int) != train_source[context_column].astype(int)
    ]
    if holdout_cfg.get("require_complete_shortcut_correlation", False) and not mismatched_train.empty:
        raise MetadataError(
            "Waterbirds100 source training split is not completely correlated: "
            f"{len(mismatched_train)} of {len(train_source)} rows have y != place."
        )

    train_positions, holdout_positions = _stratified_holdout_indices(
        train_source,
        label_column,
        float(holdout_cfg["validation_fraction"]),
        int(holdout_cfg["split_seed"]),
    )
    candidate_train_source = train_source.iloc[train_positions].copy()
    biased_holdout_source = train_source.iloc[holdout_positions].copy()

    candidate_train = _materialize_manifest(
        candidate_train_source,
        data_root,
        teacher_root,
        image_column,
        label_column,
        context_column,
        "train",
        "candidate_train",
        include_analysis_columns=False,
    )
    biased_holdout = _materialize_manifest(
        biased_holdout_source,
        data_root,
        teacher_root,
        image_column,
        label_column,
        context_column,
        "train",
        "biased_validation",
        include_analysis_columns=False,
    )
    oracle_validation = _materialize_manifest(
        oracle_source,
        data_root,
        teacher_root,
        image_column,
        label_column,
        context_column,
        "original_validation",
        "oracle_validation_analysis_only",
        include_analysis_columns=True,
    )
    test = _materialize_manifest(
        test_source,
        data_root,
        teacher_root,
        image_column,
        label_column,
        context_column,
        "test",
        "test_analysis_only",
        include_analysis_columns=True,
    )

    if check_images:
        manifests = {
            "candidate_train": candidate_train,
            "biased_validation": biased_holdout,
            "oracle_validation": oracle_validation,
            "test": test,
        }
        for name, manifest in manifests.items():
            missing_images = [path for path in manifest["image_path"] if not Path(path).is_file()]
            if missing_images:
                preview = ", ".join(missing_images[:5])
                raise FileNotFoundError(
                    f"{name} has {len(missing_images)} missing images. First paths: {preview}"
                )

    missing_holdout_masks = biased_holdout[~biased_holdout["teacher_map_exists"]]
    if require_holdout_teacher_maps and not missing_holdout_masks.empty:
        preview = ", ".join(missing_holdout_masks["teacher_map_path"].head(5).tolist())
        raise FileNotFoundError(
            "The FCV holdout requires complete teacher-map coverage, but "
            f"{len(missing_holdout_masks)} maps are missing. First expected paths: {preview}"
        )

    train_ids = set(candidate_train["metadata_index"].astype(int).tolist())
    holdout_ids = set(biased_holdout["metadata_index"].astype(int).tolist())
    source_ids = set(train_source["metadata_index"].astype(int).tolist())
    if train_ids.intersection(holdout_ids):
        raise MetadataError("Candidate-train and biased-validation manifests overlap.")
    if train_ids.union(holdout_ids) != source_ids:
        raise MetadataError("Train-derived manifests do not partition the source training split.")

    destination.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "candidate_train": destination / "metadata_train.csv",
        "biased_validation": destination / "metadata_val.csv",
        "oracle_validation": destination / "metadata_oracle_val_analysis_only.csv",
        "test": destination / "metadata_test_analysis_only.csv",
        "indices": destination / "split_indices.json",
        "summary": destination / "split_summary.json",
        "bundle": destination / "manifest_bundle.json",
    }
    _ensure_writable_targets(list(artifact_paths.values()), overwrite)

    split_indices = {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": _sha256_file(metadata_path),
        "split_seed": int(holdout_cfg["split_seed"]),
        "stratify_by": str(holdout_cfg["stratify_by"]),
        "train_fraction": float(holdout_cfg["train_fraction"]),
        "validation_fraction": float(holdout_cfg["validation_fraction"]),
        "candidate_train_metadata_indices": sorted(train_ids),
        "biased_validation_metadata_indices": sorted(holdout_ids),
        "candidate_train_indices_sha256": _sha256_values(sorted(train_ids)),
        "biased_validation_indices_sha256": _sha256_values(sorted(holdout_ids)),
    }

    summary = {
        "protocol": {
            "source": "fixed class-stratified holdout from fully biased training split",
            "vanilla_visibility": "biased validation holdout only",
            "fcv_visibility": "biased validation holdout only",
            "oracle_visibility": "original mixed validation; analysis only",
            "test_visibility": "evaluation only",
        },
        "metadata_sha256": split_indices["metadata_sha256"],
        "splits": {
            "candidate_train": _split_summary(
                candidate_train_source, candidate_train, label_column, context_column
            ),
            "biased_validation": _split_summary(
                biased_holdout_source, biased_holdout, label_column, context_column
            ),
            "oracle_validation_analysis_only": _split_summary(
                oracle_source, oracle_validation, label_column, context_column
            ),
            "test_analysis_only": _split_summary(test_source, test, label_column, context_column),
        },
        "public_manifest_columns": PUBLIC_COLUMNS,
        "analysis_only_manifest_columns": ANALYSIS_COLUMNS,
    }

    _atomic_csv(candidate_train, artifact_paths["candidate_train"])
    _atomic_csv(biased_holdout, artifact_paths["biased_validation"])
    _atomic_csv(oracle_validation, artifact_paths["oracle_validation"])
    _atomic_csv(test, artifact_paths["test"])
    _atomic_json(split_indices, artifact_paths["indices"])
    _atomic_json(summary, artifact_paths["summary"])
    write_manifest_bundle(config, destination, artifact_paths)

    return {
        "output_dir": str(destination.resolve()),
        "artifacts": {key: str(value.resolve()) for key, value in artifact_paths.items()},
        "summary": summary,
    }
