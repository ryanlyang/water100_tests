"""Step 9 Oracle validation and leakage-safe selector construction."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from .candidate_training import (
    build_transforms,
    candidate_training_fingerprint,
    enumerate_sweep_runs,
)
from .config import candidate_epochs
from .manifest_provenance import (
    ManifestProvenanceError,
    validate_manifest_bundle,
)
from .vit_counterfactual_forward import load_candidate_model
from .waterbirds_metadata import GROUP_NAMES


ORACLE_REQUIRED_COLUMNS = {
    "sample_id",
    "metadata_index",
    "image_path",
    "image_sha256",
    "label",
    "context",
    "group",
    "group_name",
    "source_split",
    "study_split",
}
ORACLE_STUDY_SPLIT = "oracle_validation_analysis_only"
ORACLE_PER_IMAGE_COLUMNS = [
    "candidate_id",
    "sample_id",
    "label",
    "group",
    "prediction",
    "correct",
    "true_class_probability",
    "loss",
    "logits",
    "probabilities",
]
ORACLE_METRIC_COLUMNS = [
    "run_index",
    "candidate_id",
    "epoch",
    "seed",
    "learning_rate",
    "weight_decay",
    "checkpoint_path",
    "checkpoint_sha256",
    "oracle_validation_loss",
    "oracle_validation_accuracy",
    "oracle_validation_balanced_group_accuracy",
    "oracle_validation_worst_group_accuracy",
    "oracle_group_0_accuracy",
    "oracle_group_0_count",
    "oracle_group_1_accuracy",
    "oracle_group_1_count",
    "oracle_group_2_accuracy",
    "oracle_group_2_count",
    "oracle_group_3_accuracy",
    "oracle_group_3_count",
    "per_image_csv_path",
    "per_image_csv_sha256",
    "summary_path",
]


class SelectorError(ValueError):
    """Raised when Step 9 provenance, Oracle data, or selector inputs are invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def selector_analysis_fingerprint(config: Mapping[str, Any]) -> str:
    """Fingerprint only the settings that can affect Step 9 selection."""

    payload = {
        "study": config["study"],
        "model": config["model"],
        "training_fingerprint": candidate_training_fingerprint(config),
        "candidate_pool": config["candidate_pool"],
        "primary_selector": config["fcv"]["primary_selector"],
        "selector_analysis": config["fcv"]["selector_analysis"],
        "secondary_selectors": config["fcv"]["secondary_selectors"],
        "oracle_visibility": config["data"]["selector_visibility"]["oracle"],
        "test_metrics_must_not_affect_selection": config["evaluation"][
            "test_metrics_must_not_affect_selection"
        ],
    }
    return _sha256_json(payload)


class OracleManifestDataset(Dataset):
    """Analysis-only original validation data with explicit group labels."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Any,
        *,
        check_images: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing Oracle manifest: {self.manifest_path}")
        frame = pd.read_csv(self.manifest_path)
        missing = sorted(ORACLE_REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise SelectorError(
                f"Oracle manifest {self.manifest_path} is missing columns: {missing}"
            )
        splits = set(frame["study_split"].astype(str).unique().tolist())
        if splits != {ORACLE_STUDY_SPLIT}:
            raise SelectorError(
                f"Oracle evaluator requires study_split={ORACLE_STUDY_SPLIT!r}; "
                f"found {sorted(splits)}."
            )
        source_splits = set(frame["source_split"].astype(str).unique().tolist())
        if source_splits != {"original_validation"}:
            raise SelectorError(
                "Oracle validation requires source_split='original_validation'; "
                f"found {sorted(source_splits)}."
            )
        if frame["sample_id"].astype(str).duplicated().any():
            raise SelectorError("Oracle validation sample IDs must be unique.")
        labels = frame["label"].astype(int)
        contexts = frame["context"].astype(int)
        groups = frame["group"].astype(int)
        if set(labels.unique()) != {0, 1} or set(contexts.unique()) != {0, 1}:
            raise SelectorError("Oracle validation labels and contexts must both be binary.")
        if set(groups.unique()) != set(GROUP_NAMES):
            raise SelectorError("Oracle validation must contain all four Waterbirds groups.")
        expected_groups = labels * 2 + contexts
        if not np.array_equal(groups.to_numpy(), expected_groups.to_numpy()):
            raise SelectorError("Oracle group IDs are inconsistent with label/context values.")
        expected_names = groups.map(GROUP_NAMES).astype(str)
        if not np.array_equal(
            frame["group_name"].astype(str).to_numpy(), expected_names.to_numpy()
        ):
            raise SelectorError("Oracle group names are inconsistent with group IDs.")

        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.image_paths = [Path(str(value)) for value in self.frame["image_path"]]
        if check_images:
            missing_images = [str(path) for path in self.image_paths if not path.is_file()]
            if missing_images:
                raise FileNotFoundError(
                    f"Oracle validation has {len(missing_images)} missing images. "
                    f"First paths: {missing_images[:5]}"
                )
            changed_images = []
            for path, expected in zip(
                self.image_paths, self.frame["image_sha256"].astype(str)
            ):
                observed = _sha256_file(path)
                if observed != expected:
                    changed_images.append(str(path))
            if changed_images:
                raise SelectorError(
                    "Oracle image bytes differ from the frozen manifest. First paths: "
                    f"{changed_images[:5]}"
                )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, str]:
        row = self.frame.iloc[index]
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                image.load()
                image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read Oracle image {path}: {exc}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return (
            image,
            int(row["label"]),
            int(row["group"]),
            str(row["sample_id"]),
        )


@dataclass(frozen=True)
class OracleValidationSource:
    manifest_path: Path
    manifest_sha256: str
    manifest_bundle_path: Path
    manifest_bundle_sha256: str
    dataset: OracleManifestDataset
    loader: DataLoader
    batch_size: int
    num_workers: int

    @property
    def sample_count(self) -> int:
        return len(self.dataset)


def prepare_oracle_validation_source(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    check_images: bool = True,
) -> OracleValidationSource:
    """Create the explicitly privileged original-validation loader."""

    selector_cfg = config["fcv"]["selector_analysis"]
    if selector_cfg["oracle_precision"] != "float32":
        raise SelectorError("Oracle evaluation must use the locked float32 precision.")
    try:
        manifest_binding = validate_manifest_bundle(
            config, manifest_path, "oracle_validation"
        )
    except ManifestProvenanceError as exc:
        raise SelectorError(str(exc)) from exc
    transform = build_transforms(config)["eval"]
    dataset = OracleManifestDataset(
        manifest_path,
        transform,
        check_images=check_images,
    )
    resolved_batch_size = int(
        selector_cfg["oracle_batch_size"] if batch_size is None else batch_size
    )
    resolved_workers = int(
        config["training"]["num_workers"] if num_workers is None else num_workers
    )
    if resolved_batch_size != int(selector_cfg["oracle_batch_size"]):
        raise SelectorError("Oracle validation batch size differs from the locked value.")
    if resolved_workers != int(config["training"]["num_workers"]):
        raise SelectorError("Oracle validation worker count differs from the locked value.")
    loader = DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=resolved_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return OracleValidationSource(
        manifest_path=dataset.manifest_path,
        manifest_sha256=manifest_binding.manifest_sha256,
        manifest_bundle_path=manifest_binding.bundle_path,
        manifest_bundle_sha256=manifest_binding.bundle_sha256,
        dataset=dataset,
        loader=loader,
        batch_size=resolved_batch_size,
        num_workers=resolved_workers,
    )


def compute_waterbirds_group_metrics(
    *,
    loss_sum: float,
    correct: int,
    total: int,
    group_correct: Sequence[int],
    group_total: Sequence[int],
) -> Dict[str, Any]:
    if total <= 0:
        raise SelectorError("Oracle validation produced no samples.")
    if len(group_correct) != 4 or len(group_total) != 4:
        raise SelectorError("Oracle validation requires exactly four groups.")
    if any(int(count) <= 0 for count in group_total):
        raise SelectorError(f"Oracle validation has an empty group: {list(group_total)}")
    group_accuracies = [
        float(group_correct[group] / group_total[group]) for group in range(4)
    ]
    result: Dict[str, Any] = {
        "loss": float(loss_sum / total),
        "accuracy": float(correct / total),
        "balanced_group_accuracy": float(np.mean(group_accuracies)),
        "worst_group_accuracy": float(np.min(group_accuracies)),
        "sample_count": int(total),
    }
    for group in range(4):
        result[f"group_{group}_name"] = GROUP_NAMES[group]
        result[f"group_{group}_accuracy"] = group_accuracies[group]
        result[f"group_{group}_correct"] = int(group_correct[group])
        result[f"group_{group}_count"] = int(group_total[group])
    return result


def recompute_oracle_metrics_from_frame(
    frame: pd.DataFrame,
    source: OracleValidationSource,
    *,
    candidate_id: str,
) -> Dict[str, Any]:
    """Recompute every Oracle selection metric from per-example records."""

    missing = sorted(set(ORACLE_PER_IMAGE_COLUMNS).difference(frame.columns))
    if missing:
        raise SelectorError(f"Oracle per-example CSV is missing columns: {missing}")
    expected = source.dataset.frame.reset_index(drop=True)
    observed = frame.reset_index(drop=True)
    if len(observed) != source.sample_count:
        raise SelectorError("Oracle per-example row count differs from the manifest.")
    if set(observed["candidate_id"].astype(str).unique()) != {candidate_id}:
        raise SelectorError("Oracle per-example candidate identity is inconsistent.")
    if observed["sample_id"].astype(str).tolist() != expected[
        "sample_id"
    ].astype(str).tolist():
        raise SelectorError("Oracle per-example sample order differs from the manifest.")
    for column in ("label", "group"):
        if not np.array_equal(
            observed[column].astype(int).to_numpy(),
            expected[column].astype(int).to_numpy(),
        ):
            raise SelectorError(f"Oracle per-example {column} differs from the manifest.")

    labels = observed["label"].astype(int).to_numpy()
    groups = observed["group"].astype(int).to_numpy()
    predictions = observed["prediction"].astype(int).to_numpy()
    correctness = observed["correct"].astype(int).to_numpy()
    losses = pd.to_numeric(observed["loss"], errors="raise").to_numpy(float)
    true_probabilities = pd.to_numeric(
        observed["true_class_probability"], errors="raise"
    ).to_numpy(float)
    if (
        not np.isfinite(losses).all()
        or not np.isfinite(true_probabilities).all()
        or (losses < 0.0).any()
        or ((true_probabilities < 0.0) | (true_probabilities > 1.0)).any()
    ):
        raise SelectorError("Oracle per-example losses/probabilities are invalid.")
    if not np.array_equal(correctness, (predictions == labels).astype(int)):
        raise SelectorError("Oracle per-example correctness does not reproduce.")

    for index, row in observed.iterrows():
        try:
            logits = np.asarray(json.loads(str(row["logits"])), dtype=np.float64)
            probabilities = np.asarray(
                json.loads(str(row["probabilities"])), dtype=np.float64
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SelectorError("Oracle logits/probabilities are not valid JSON.") from exc
        if logits.shape != (2,) or probabilities.shape != (2,):
            raise SelectorError("Oracle logits/probabilities must contain two classes.")
        if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise SelectorError("Oracle logits/probabilities contain non-finite values.")
        shifted = logits - logits.max()
        expected_probabilities = np.exp(shifted) / np.exp(shifted).sum()
        if not np.allclose(
            probabilities, expected_probabilities, rtol=0.0, atol=1.0e-7
        ):
            raise SelectorError("Oracle probabilities do not reproduce from logits.")
        label = int(labels[index])
        if int(predictions[index]) != int(np.argmax(logits)):
            raise SelectorError("Oracle prediction does not reproduce from logits.")
        if not np.isclose(
            true_probabilities[index], probabilities[label], rtol=0.0, atol=1.0e-7
        ):
            raise SelectorError("Oracle true-class probability does not reproduce.")
        expected_loss = float(-np.log(max(probabilities[label], 1.0e-300)))
        if not np.isclose(losses[index], expected_loss, rtol=0.0, atol=1.0e-6):
            raise SelectorError("Oracle per-example loss does not reproduce.")

    group_total = [int((groups == group).sum()) for group in range(4)]
    group_correct = [
        int(correctness[groups == group].sum()) for group in range(4)
    ]
    return compute_waterbirds_group_metrics(
        loss_sum=float(losses.sum()),
        correct=int(correctness.sum()),
        total=len(observed),
        group_correct=group_correct,
        group_total=group_total,
    )


def validate_oracle_summary_against_frame(
    summary: Mapping[str, Any],
    frame: pd.DataFrame,
    source: OracleValidationSource,
) -> Dict[str, Any]:
    candidate_id = str(summary.get("candidate_id", ""))
    recomputed = recompute_oracle_metrics_from_frame(
        frame, source, candidate_id=candidate_id
    )
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SelectorError("Oracle summary has no metric mapping.")
    for key in ("loss", "accuracy", "balanced_group_accuracy", "worst_group_accuracy"):
        if not np.isclose(
            float(metrics.get(key, float("nan"))),
            float(recomputed[key]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise SelectorError(f"Oracle summary metric {key} does not reproduce.")
    if int(metrics.get("sample_count", -1)) != int(recomputed["sample_count"]):
        raise SelectorError("Oracle summary sample count does not reproduce.")
    for group in range(4):
        for suffix in ("correct", "count"):
            key = f"group_{group}_{suffix}"
            if int(metrics.get(key, -1)) != int(recomputed[key]):
                raise SelectorError(f"Oracle summary {key} does not reproduce.")
        key = f"group_{group}_accuracy"
        if not np.isclose(
            float(metrics.get(key, float("nan"))),
            float(recomputed[key]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise SelectorError(f"Oracle summary {key} does not reproduce.")
    return recomputed


def _existing_oracle_summary(
    path: Path,
    *,
    candidate_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source: OracleValidationSource,
    config: Mapping[str, Any],
) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    valid = (
        summary.get("schema_version") == 2
        and summary.get("artifact_type") == "fcv_vit_oracle_validation_summary"
        and summary.get("status") == "complete"
        and summary.get("candidate_id") == candidate_id
        and summary.get("checkpoint_path") == str(checkpoint_path)
        and summary.get("checkpoint_sha256") == checkpoint_sha256
        and summary.get("oracle_manifest_path") == str(source.manifest_path)
        and summary.get("oracle_manifest_sha256") == source.manifest_sha256
        and summary.get("manifest_bundle_sha256") == source.manifest_bundle_sha256
        and int(summary.get("oracle_sample_count", -1)) == source.sample_count
        and summary.get("training_fingerprint") == candidate_training_fingerprint(config)
        and summary.get("selector_analysis_fingerprint")
        == selector_analysis_fingerprint(config)
        and summary.get("precision") == "float32"
        and summary.get("execution")
        == {
            "batch_size": source.batch_size,
            "num_workers": source.num_workers,
        }
        and summary.get("test_data_accessed") is False
    )
    if not valid:
        return None
    score_path = Path(str(summary.get("per_image_csv_path", ""))).expanduser().resolve()
    if (
        not score_path.is_file()
        or summary.get("per_image_csv_sha256") != _sha256_file(score_path)
    ):
        return None
    try:
        validate_oracle_summary_against_frame(summary, pd.read_csv(score_path), source)
    except (OSError, ValueError, KeyError, TypeError, SelectorError):
        return None
    return summary


def evaluate_candidate_oracle(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    source: OracleValidationSource,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Evaluate one candidate on privileged Oracle validation, never test data."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing candidate checkpoint: {checkpoint_path}")
    try:
        epoch = int(checkpoint_path.stem.removeprefix("epoch_"))
    except ValueError as exc:
        raise SelectorError(f"Unrecognized candidate checkpoint name: {checkpoint_path}") from exc
    run_id = checkpoint_path.parent.parent.name
    candidate_id = f"{run_id}_epoch_{epoch:03d}"
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    summary_path = Path(output_dir).expanduser().resolve() / (
        f"{candidate_id}_oracle_summary.json"
    )
    per_image_path = Path(output_dir).expanduser().resolve() / (
        f"{candidate_id}_oracle_per_image.csv"
    )
    existing = _existing_oracle_summary(
        summary_path,
        candidate_id=candidate_id,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        source=source,
        config=config,
    )
    if existing is not None and not overwrite:
        result = dict(existing)
        result["invocation_status"] = "already_complete"
        return result
    if summary_path.exists() and not overwrite:
        raise SelectorError(
            f"Stale Oracle summary exists for {candidate_id}; use --overwrite to replace it."
        )

    Path(output_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Oracle validation but is unavailable.")
    model, checkpoint = load_candidate_model(config, checkpoint_path, device=device)
    if checkpoint.get("candidate_id") != candidate_id or int(checkpoint.get("epoch", -1)) != epoch:
        raise SelectorError("Candidate checkpoint identity does not match its path.")
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    group_correct = [0, 0, 0, 0]
    group_total = [0, 0, 0, 0]
    per_image_rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for images, labels, groups, sample_ids in source.loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images).float()
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise SelectorError(f"Oracle logits have invalid shape: {tuple(logits.shape)}")
            batch_size = int(labels.numel())
            per_example_loss = F.cross_entropy(logits, labels, reduction="none")
            loss_sum += float(per_example_loss.sum().item())
            probabilities = logits.softmax(dim=1)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += batch_size
            groups = groups.to(torch.long)
            labels_cpu = labels.cpu()
            predictions_cpu = predictions.cpu()
            logits_cpu = logits.cpu()
            probabilities_cpu = probabilities.cpu()
            losses_cpu = per_example_loss.cpu()
            groups_cpu = groups.to(torch.long).cpu()
            for batch_index, sample_id in enumerate(sample_ids):
                label = int(labels_cpu[batch_index].item())
                prediction = int(predictions_cpu[batch_index].item())
                per_image_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "sample_id": str(sample_id),
                        "label": label,
                        "group": int(groups_cpu[batch_index].item()),
                        "prediction": prediction,
                        "correct": int(prediction == label),
                        "true_class_probability": float(
                            probabilities_cpu[batch_index, label].item()
                        ),
                        "loss": float(losses_cpu[batch_index].item()),
                        "logits": json.dumps(
                            [float(value) for value in logits_cpu[batch_index].tolist()],
                            separators=(",", ":"),
                        ),
                        "probabilities": json.dumps(
                            [
                                float(value)
                                for value in probabilities_cpu[batch_index].tolist()
                            ],
                            separators=(",", ":"),
                        ),
                    }
                )
            for group in range(4):
                mask = groups_cpu == group
                count = int(mask.sum().item())
                group_total[group] += count
                if count:
                    group_correct[group] += int(
                        (predictions_cpu[mask] == labels_cpu[mask]).sum().item()
                    )
    metrics = compute_waterbirds_group_metrics(
        loss_sum=loss_sum,
        correct=correct,
        total=total,
        group_correct=group_correct,
        group_total=group_total,
    )
    per_image_frame = pd.DataFrame(per_image_rows, columns=ORACLE_PER_IMAGE_COLUMNS)
    recomputed = recompute_oracle_metrics_from_frame(
        per_image_frame, source, candidate_id=candidate_id
    )
    for key in ("loss", "accuracy", "balanced_group_accuracy", "worst_group_accuracy"):
        tolerance = 1.0e-6 if key == "loss" else 1.0e-12
        if not np.isclose(
            float(metrics[key]), float(recomputed[key]), rtol=0.0, atol=tolerance
        ):
            raise SelectorError(f"Oracle forward metric {key} failed raw recomputation.")
    metrics = recomputed
    _atomic_csv(per_image_frame, per_image_path)
    summary = {
        "schema_version": 2,
        "artifact_type": "fcv_vit_oracle_validation_summary",
        "status": "complete",
        "candidate_id": candidate_id,
        "run": dict(checkpoint["run"]),
        "epoch": epoch,
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
        "pretrained_provenance_sha256": checkpoint.get(
            "pretrained_provenance_sha256"
        ),
        "pretrained_backbone_sha256": checkpoint.get(
            "pretrained_backbone_sha256"
        ),
        "initial_model_state_sha256": checkpoint.get(
            "initial_model_state_sha256"
        ),
        "selector_analysis_fingerprint": selector_analysis_fingerprint(config),
        "oracle_manifest_path": str(source.manifest_path),
        "oracle_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_path": str(source.manifest_bundle_path),
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "oracle_sample_count": source.sample_count,
        "precision": "float32",
        "execution": {
            "batch_size": source.batch_size,
            "num_workers": source.num_workers,
        },
        "per_image_csv_path": str(per_image_path),
        "per_image_csv_sha256": _sha256_file(per_image_path),
        "metrics": metrics,
        "test_data_accessed": False,
    }
    _atomic_json(summary, summary_path)
    del model, checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = dict(summary)
    result["invocation_status"] = "complete"
    return result


def aggregate_oracle_summaries(
    config: Mapping[str, Any],
    oracle_dir: str | Path,
    output_csv: str | Path,
    output_summary: str | Path,
    *,
    source: OracleValidationSource,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Strictly index all candidate Oracle validation summaries."""

    oracle_dir = Path(oracle_dir).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    output_summary = Path(output_summary).expanduser().resolve()
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    invalid: List[Dict[str, str]] = []
    selected_epochs = candidate_epochs(config)
    expected_training_fingerprint = candidate_training_fingerprint(config)
    expected_selector_fingerprint = selector_analysis_fingerprint(config)
    for run in enumerate_sweep_runs(config):
        for epoch in selected_epochs:
            candidate_id = run.candidate_id(epoch)
            summary_path = oracle_dir / f"{candidate_id}_oracle_summary.json"
            if not summary_path.is_file():
                missing.append(candidate_id)
                continue
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                metrics = summary.get("metrics")
                valid = (
                    summary.get("schema_version") == 2
                    and summary.get("artifact_type")
                    == "fcv_vit_oracle_validation_summary"
                    and summary.get("status") == "complete"
                    and summary.get("candidate_id") == candidate_id
                    and summary.get("run") == asdict(run)
                    and int(summary.get("epoch", -1)) == epoch
                    and summary.get("training_fingerprint")
                    == expected_training_fingerprint
                    and summary.get("selector_analysis_fingerprint")
                    == expected_selector_fingerprint
                    and summary.get("oracle_manifest_path") == str(source.manifest_path)
                    and summary.get("oracle_manifest_sha256") == source.manifest_sha256
                    and summary.get("manifest_bundle_sha256")
                    == source.manifest_bundle_sha256
                    and int(summary.get("oracle_sample_count", -1))
                    == source.sample_count
                    and summary.get("precision") == "float32"
                    and summary.get("execution")
                    == {
                        "batch_size": source.batch_size,
                        "num_workers": source.num_workers,
                    }
                    and summary.get("test_data_accessed") is False
                    and isinstance(metrics, Mapping)
                )
                checkpoint_path = Path(str(summary.get("checkpoint_path", "")))
                checkpoint_sha256 = str(summary.get("checkpoint_sha256", ""))
                if (
                    not valid
                    or not checkpoint_path.is_file()
                    or _sha256_file(checkpoint_path) != checkpoint_sha256
                ):
                    raise SelectorError("stale Oracle summary provenance")
                per_image_path = Path(
                    str(summary.get("per_image_csv_path", ""))
                ).expanduser().resolve()
                if (
                    not per_image_path.is_file()
                    or summary.get("per_image_csv_sha256")
                    != _sha256_file(per_image_path)
                ):
                    raise SelectorError("stale Oracle per-example artifact")
                recomputed = validate_oracle_summary_against_frame(
                    summary, pd.read_csv(per_image_path), source
                )
                metrics = recomputed
                group_counts = [int(metrics[f"group_{group}_count"]) for group in range(4)]
                row: Dict[str, Any] = {
                    "run_index": run.run_index,
                    "candidate_id": candidate_id,
                    "epoch": epoch,
                    "seed": run.seed,
                    "learning_rate": run.learning_rate,
                    "weight_decay": run.weight_decay,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "oracle_validation_loss": float(metrics["loss"]),
                    "oracle_validation_accuracy": float(metrics["accuracy"]),
                    "oracle_validation_balanced_group_accuracy": float(
                        metrics["balanced_group_accuracy"]
                    ),
                    "oracle_validation_worst_group_accuracy": float(
                        metrics["worst_group_accuracy"]
                    ),
                    "per_image_csv_path": str(per_image_path),
                    "per_image_csv_sha256": str(summary["per_image_csv_sha256"]),
                    "summary_path": str(summary_path),
                }
                for group in range(4):
                    row[f"oracle_group_{group}_accuracy"] = float(
                        metrics[f"group_{group}_accuracy"]
                    )
                    row[f"oracle_group_{group}_count"] = group_counts[group]
                if not all(
                    np.isfinite(float(row[key]))
                    for key in (
                        "oracle_validation_loss",
                        "oracle_validation_accuracy",
                        "oracle_validation_balanced_group_accuracy",
                        "oracle_validation_worst_group_accuracy",
                    )
                ):
                    raise SelectorError("non-finite Oracle metric")
                rows.append(row)
            except (OSError, ValueError, KeyError, TypeError, SelectorError) as exc:
                invalid.append({"candidate_id": candidate_id, "error": str(exc)})

    if (missing or invalid) and not allow_incomplete:
        raise SelectorError(
            f"Oracle score pool is incomplete: missing={len(missing)} "
            f"invalid={len(invalid)}."
        )
    frame = pd.DataFrame(rows, columns=ORACLE_METRIC_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["run_index", "epoch"]).reset_index(drop=True)
        if frame["candidate_id"].duplicated().any():
            raise SelectorError("Duplicate candidate IDs in Oracle score index.")
    _atomic_csv(frame, output_csv)
    expected_count = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    result = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_oracle_score_pool_summary",
        "status": "complete" if not missing and not invalid else "incomplete",
        "candidate_count": len(frame),
        "expected_candidate_count": expected_count,
        "missing_candidate_count": len(missing),
        "missing_candidate_preview": missing[:10],
        "invalid_candidate_count": len(invalid),
        "invalid_candidate_preview": invalid[:10],
        "training_fingerprint": expected_training_fingerprint,
        "selector_analysis_fingerprint": expected_selector_fingerprint,
        "oracle_manifest_path": str(source.manifest_path),
        "oracle_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "oracle_per_image_metrics_recomputed": True,
        "oracle_execution": {
            "batch_size": source.batch_size,
            "num_workers": source.num_workers,
        },
        "test_data_accessed": False,
        "output_csv": str(output_csv),
        "output_csv_sha256": _sha256_file(output_csv),
    }
    _atomic_json(result, output_summary)
    return result


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
    *,
    require_unique_candidate_ids: bool = True,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise SelectorError(f"{name} is missing required columns: {missing}")
    if (
        require_unique_candidate_ids
        and frame["candidate_id"].astype(str).duplicated().any()
    ):
        raise SelectorError(f"{name} contains duplicate candidate IDs.")


def _validate_aggregate_summary(
    summary_path: str | Path,
    csv_path: Path,
    *,
    expected_artifact_type: str | None,
    expected_candidate_count: int,
) -> Dict[str, Any]:
    path = Path(summary_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregate summary: {path}")
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise SelectorError(f"Aggregate summary is not a mapping: {path}")
    if (
        summary.get("status") != "complete"
        or int(summary.get("candidate_count", -1)) != expected_candidate_count
        or Path(str(summary.get("output_csv", ""))).expanduser().resolve()
        != csv_path
        or summary.get("output_csv_sha256") != _sha256_file(csv_path)
    ):
        raise SelectorError(f"Aggregate summary is stale or incomplete: {path}")
    if (
        expected_artifact_type is not None
        and summary.get("artifact_type") != expected_artifact_type
    ):
        raise SelectorError(f"Unexpected aggregate summary type: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "output_csv_sha256": str(summary["output_csv_sha256"]),
    }


def _validate_candidate_sets(
    frames: Mapping[str, pd.DataFrame], expected_count: int
) -> List[str]:
    sets = {
        name: set(frame["candidate_id"].astype(str).tolist())
        for name, frame in frames.items()
    }
    for name, values in sets.items():
        if len(values) != expected_count:
            raise SelectorError(
                f"{name} contains {len(values)} candidates; expected {expected_count}."
            )
    reference_name = next(iter(sets))
    reference = sets[reference_name]
    for name, values in sets.items():
        if values != reference:
            raise SelectorError(
                f"Candidate IDs differ between {reference_name} and {name}."
            )
    return sorted(reference)


def _probability_retention_ratio(
    score_csv_path: str | Path,
    score_csv_sha256: str,
    epsilon: float,
    expected_candidate_id: str,
) -> float:
    path = Path(score_csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 7 per-image score CSV: {path}")
    if len(score_csv_sha256) != 64 or _sha256_file(path) != score_csv_sha256:
        raise SelectorError(
            f"Step 7 per-image score CSV bytes changed before selection: {path}"
        )
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        ["candidate_id", "fcv_eligible", "p_y_original", "p_y_counterfactual_mean"],
        str(path),
        require_unique_candidate_ids=False,
    )
    candidate_ids = set(frame["candidate_id"].astype(str).unique().tolist())
    if candidate_ids != {expected_candidate_id}:
        raise SelectorError(
            f"Step 7 score CSV {path} belongs to {sorted(candidate_ids)}, not "
            f"{expected_candidate_id!r}."
        )
    eligible_raw = frame["fcv_eligible"]
    if eligible_raw.dtype == bool:
        eligible_mask = eligible_raw
    else:
        normalized = eligible_raw.astype(str).str.strip().str.lower()
        if not set(normalized.unique()).issubset({"true", "false", "1", "0"}):
            raise SelectorError(f"Invalid fcv_eligible values in {path}.")
        eligible_mask = normalized.isin({"true", "1"})
    eligible = frame.loc[eligible_mask]
    if eligible.empty:
        raise SelectorError(f"No FCV-eligible rows in {path}.")
    original = pd.to_numeric(eligible["p_y_original"], errors="raise").to_numpy(float)
    counterfactual = pd.to_numeric(
        eligible["p_y_counterfactual_mean"], errors="raise"
    ).to_numpy(float)
    if not np.isfinite(original).all() or not np.isfinite(counterfactual).all():
        raise SelectorError(f"Non-finite probability in {path}.")
    return float(np.mean(counterfactual / np.maximum(original, epsilon)))


def _select_one(
    matrix: pd.DataFrame,
    *,
    selector_name: str,
    selector_family: str,
    score_column: str,
    direction: str,
    formula: str,
    availability: str,
) -> Dict[str, Any]:
    scores = pd.to_numeric(matrix[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(scores).all():
        raise SelectorError(f"Selector {selector_name} contains non-finite scores.")
    if direction == "maximize":
        best_score = float(np.max(scores))
    elif direction == "minimize":
        best_score = float(np.min(scores))
    else:
        raise SelectorError(f"Unsupported selector direction: {direction}")
    tied = matrix.loc[scores == best_score].copy()
    tied = tied.sort_values("candidate_id", kind="stable")
    selected = tied.iloc[0]
    hparams = {
        "learning_rate": float(selected["learning_rate"]),
        "weight_decay": float(selected["weight_decay"]),
        "seed": int(selected["seed"]),
        "epoch": int(selected["epoch"]),
    }
    row: Dict[str, Any] = {
        "selector_name": selector_name,
        "selector_family": selector_family,
        "availability": availability,
        "direction": direction,
        "selector_formula": formula,
        "selector_score": best_score,
        "selected_checkpoint_id": str(selected["candidate_id"]),
        "selected_checkpoint_path": str(selected["checkpoint_path"]),
        "selected_checkpoint_sha256": str(selected["checkpoint_sha256"]),
        "selected_hparams": json.dumps(hparams, sort_keys=True, separators=(",", ":")),
        "run_index": int(selected["run_index"]),
        "epoch": int(selected["epoch"]),
        "seed": int(selected["seed"]),
        "learning_rate": float(selected["learning_rate"]),
        "weight_decay": float(selected["weight_decay"]),
        "exact_tie_count": int(len(tied)),
        "tie_break_rule": "candidate_id_ascending",
    }
    report_columns = [
        "biased_val_loss",
        "biased_val_accuracy",
        "biased_validation_accuracy_recomputed",
        "fcv_counterfactual_accuracy",
        "fcv_counterfactual_majority_accuracy",
        "fcv_true_class_probability",
        "fcv_probability_retention_ratio",
        "fcv_confidence_drop",
        "same_context_counterfactual_accuracy",
        "same_context_mean_confidence_drop",
        "shortcut_sensitivity",
        "control_normalized_fcv_score",
        "primary_selector_score",
    ]
    for column in report_columns:
        if column in selected.index:
            row[column] = float(selected[column])
    return row


def build_selection_table(
    config: Mapping[str, Any],
    candidate_metrics_csv: str | Path,
    fcv_scores_csv: str | Path,
    control_scores_csv: str | Path,
    oracle_scores_csv: str | Path,
    output_table_csv: str | Path,
    output_matrix_csv: str | Path,
    output_summary_json: str | Path,
    *,
    candidate_metrics_summary_json: str | Path | None = None,
    fcv_scores_summary_json: str | Path | None = None,
    control_scores_summary_json: str | Path | None = None,
    oracle_scores_summary_json: str | Path | None = None,
) -> Dict[str, Any]:
    """Join all validation-only metrics and select candidates deterministically."""

    input_paths = {
        "candidate_metrics": Path(candidate_metrics_csv).expanduser().resolve(),
        "fcv_scores": Path(fcv_scores_csv).expanduser().resolve(),
        "control_scores": Path(control_scores_csv).expanduser().resolve(),
        "oracle_scores": Path(oracle_scores_csv).expanduser().resolve(),
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Step 9 input {name}: {path}")
    expected_count = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    aggregate_summaries: Dict[str, Dict[str, Any]] = {}
    unprivileged_summary_specs = (
        (
            "candidate_metrics",
            candidate_metrics_summary_json,
            "fcv_vit_candidate_pool_summary",
        ),
        (
            "fcv_scores",
            fcv_scores_summary_json,
            "fcv_vit_score_pool_summary",
        ),
        (
            "control_scores",
            control_scores_summary_json,
            "fcv_vit_control_pool_summary",
        ),
    )
    for name, summary_path, artifact_type in unprivileged_summary_specs:
        if summary_path is not None:
            aggregate_summaries[name] = _validate_aggregate_summary(
                summary_path,
                input_paths[name],
                expected_artifact_type=artifact_type,
                expected_candidate_count=expected_count,
            )
    candidate = pd.read_csv(input_paths["candidate_metrics"])
    fcv = pd.read_csv(input_paths["fcv_scores"])
    controls = pd.read_csv(input_paths["control_scores"])
    _require_columns(
        candidate,
        [
            "candidate_id",
            "run_index",
            "epoch",
            "seed",
            "learning_rate",
            "weight_decay",
            "biased_val_loss",
            "biased_val_accuracy",
            "checkpoint_path",
            "checkpoint_sha256",
        ],
        "candidate metrics",
    )
    _require_columns(
        fcv,
        [
            "candidate_id",
            "run_index",
            "epoch",
            "seed",
            "learning_rate",
            "weight_decay",
            "biased_validation_accuracy",
            "fcv_counterfactual_accuracy",
            "fcv_counterfactual_majority_accuracy",
            "fcv_true_class_probability",
            "fcv_confidence_drop",
            "primary_selector_score",
            "score_csv_path",
            "score_csv_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
        ],
        "FCV scores",
    )
    _require_columns(
        controls,
        [
            "candidate_id",
            "run_index",
            "epoch",
            "seed",
            "learning_rate",
            "weight_decay",
            "same_context_counterfactual_accuracy",
            "same_context_mean_confidence_drop",
            "checkpoint_path",
            "checkpoint_sha256",
        ],
        "control scores",
    )
    candidate_ids = _validate_candidate_sets(
        {
            "candidate_metrics": candidate,
            "fcv_scores": fcv,
            "control_scores": controls,
        },
        expected_count,
    )
    key_columns = [
        "candidate_id",
        "run_index",
        "epoch",
        "seed",
        "learning_rate",
        "weight_decay",
    ]
    ordered = candidate.set_index("candidate_id").loc[candidate_ids].reset_index()
    for name, frame in (("fcv", fcv), ("controls", controls)):
        aligned = frame.set_index("candidate_id").loc[candidate_ids].reset_index()
        for column in key_columns[1:]:
            left = ordered[column].to_numpy()
            right = aligned[column].to_numpy()
            if np.issubdtype(np.asarray(left).dtype, np.number):
                matches = np.allclose(left, right, rtol=0.0, atol=0.0)
            else:
                matches = np.array_equal(left, right)
            if not matches:
                raise SelectorError(f"{name} {column} values differ from candidate metrics.")

    fcv_aligned = fcv.set_index("candidate_id").loc[candidate_ids]
    controls_aligned = controls.set_index("candidate_id").loc[candidate_ids]
    matrix = ordered.copy()
    for name, aligned in (("fcv", fcv_aligned), ("controls", controls_aligned)):
        if not np.array_equal(
            matrix["checkpoint_path"].astype(str).to_numpy(),
            aligned["checkpoint_path"].astype(str).to_numpy(),
        ) or not np.array_equal(
            matrix["checkpoint_sha256"].astype(str).to_numpy(),
            aligned["checkpoint_sha256"].astype(str).to_numpy(),
        ):
            raise SelectorError(f"{name} checkpoint path/hash differs from candidates.")
    for path, expected_hash in zip(
        matrix["checkpoint_path"], matrix["checkpoint_sha256"]
    ):
        checkpoint_path = Path(str(path))
        if not checkpoint_path.is_file() or _sha256_file(checkpoint_path) != str(
            expected_hash
        ):
            raise SelectorError(f"Candidate checkpoint bytes changed: {checkpoint_path}")
    matrix["biased_validation_accuracy_recomputed"] = fcv_aligned[
        "biased_validation_accuracy"
    ].to_numpy(float)
    for column in (
        "fcv_counterfactual_accuracy",
        "fcv_counterfactual_majority_accuracy",
        "fcv_true_class_probability",
        "fcv_confidence_drop",
        "primary_selector_score",
        "score_csv_path",
        "score_csv_sha256",
    ):
        matrix[column] = fcv_aligned[column].to_numpy()
    matrix["same_context_counterfactual_accuracy"] = controls_aligned[
        "same_context_counterfactual_accuracy"
    ].to_numpy(float)
    matrix["same_context_mean_confidence_drop"] = controls_aligned[
        "same_context_mean_confidence_drop"
    ].to_numpy(float)
    if not np.array_equal(
        matrix["biased_val_accuracy"].to_numpy(float),
        matrix["biased_validation_accuracy_recomputed"].to_numpy(float),
    ):
        raise SelectorError(
            "Vanilla and FCV ordinary validation accuracy must be byte-identical."
        )

    selector_cfg = config["fcv"]["selector_analysis"]
    epsilon = float(selector_cfg["probability_ratio_epsilon"])
    matrix["fcv_probability_retention_ratio"] = [
        _probability_retention_ratio(path, score_hash, epsilon, str(candidate_id))
        for path, score_hash, candidate_id in zip(
            matrix["score_csv_path"],
            matrix["score_csv_sha256"],
            matrix["candidate_id"],
        )
    ]
    matrix["shortcut_sensitivity"] = (
        matrix["fcv_confidence_drop"]
        - matrix["same_context_mean_confidence_drop"]
    )
    control_lambda = float(selector_cfg["control_normalized_lambda"])
    matrix["control_normalized_fcv_score"] = (
        matrix["biased_validation_accuracy_recomputed"]
        - control_lambda * matrix["shortcut_sensitivity"]
    )
    primary = config["fcv"]["primary_selector"]
    recomputed_primary = (
        float(primary["original_accuracy_weight"])
        * matrix["biased_validation_accuracy_recomputed"]
        + float(primary["counterfactual_accuracy_weight"])
        * matrix["fcv_counterfactual_accuracy"]
    )
    if not np.allclose(
        matrix["primary_selector_score"],
        recomputed_primary,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise SelectorError("Stored Step 7 primary selector scores do not reproduce.")
    for value in selector_cfg["fcv_accuracy_lambdas"]:
        slug = str(float(value)).replace(".", "p")
        matrix[f"fcv_accuracy_lambda_{slug}_score"] = (
            matrix["biased_validation_accuracy_recomputed"]
            + float(value) * matrix["fcv_counterfactual_accuracy"]
        )

    selector_specs: List[Dict[str, str]] = [
        {
            "selector_name": "biased_validation_accuracy",
            "selector_family": "biased_validation",
            "score_column": "biased_val_accuracy",
            "direction": "maximize",
            "formula": "biased_val_accuracy",
            "availability": "unprivileged_train_holdout",
        },
        {
            "selector_name": "biased_validation_loss",
            "selector_family": "biased_validation",
            "score_column": "biased_val_loss",
            "direction": "minimize",
            "formula": "biased_val_loss",
            "availability": "unprivileged_train_holdout",
        },
        {
            "selector_name": "opposite_context_counterfactual_accuracy",
            "selector_family": "fcv",
            "score_column": "fcv_counterfactual_accuracy",
            "direction": "maximize",
            "formula": "mean counterfactual draw accuracy on FCV-eligible samples",
            "availability": "unprivileged_train_holdout",
        },
        {
            "selector_name": "opposite_context_true_class_probability",
            "selector_family": "fcv_stability",
            "score_column": "fcv_true_class_probability",
            "direction": "maximize",
            "formula": "mean p_y(counterfactual) on FCV-eligible samples",
            "availability": "unprivileged_train_holdout",
        },
        {
            "selector_name": "opposite_context_probability_retention_ratio",
            "selector_family": "fcv_stability",
            "score_column": "fcv_probability_retention_ratio",
            "direction": "maximize",
            "formula": "mean p_y(counterfactual)/max(p_y(original),epsilon)",
            "availability": "unprivileged_train_holdout",
        },
        {
            "selector_name": str(primary["name"]),
            "selector_family": "fcv_primary",
            "score_column": "primary_selector_score",
            "direction": "maximize",
            "formula": "0.5*original_accuracy + 0.5*opposite_fcv_accuracy",
            "availability": "unprivileged_train_holdout",
        },
    ]
    for value in selector_cfg["fcv_accuracy_lambdas"]:
        slug = str(float(value)).replace(".", "p")
        selector_specs.append(
            {
                "selector_name": f"fcv_accuracy_lambda_{slug}",
                "selector_family": "fcv_lambda_ablation",
                "score_column": f"fcv_accuracy_lambda_{slug}_score",
                "direction": "maximize",
                "formula": f"original_accuracy + {float(value):g}*opposite_fcv_accuracy",
                "availability": "unprivileged_train_holdout",
            }
        )
    selector_specs.append(
        {
            "selector_name": "control_normalized_fcv",
            "selector_family": "fcv_control_normalized",
            "score_column": "control_normalized_fcv_score",
            "direction": "maximize",
            "formula": (
                "original_accuracy - (opposite_confidence_drop - "
                "same_context_confidence_drop)"
            ),
            "availability": "unprivileged_train_holdout",
        }
    )
    unprivileged_rows = [
        _select_one(
            matrix,
            selector_name=spec["selector_name"],
            selector_family=spec["selector_family"],
            score_column=spec["score_column"],
            direction=spec["direction"],
            formula=spec["formula"],
            availability=spec["availability"],
        )
        for spec in selector_specs
    ]
    output_table_csv = Path(output_table_csv).expanduser().resolve()
    output_matrix_csv = Path(output_matrix_csv).expanduser().resolve()
    output_summary_json = Path(output_summary_json).expanduser().resolve()
    unprivileged_table_path = output_table_csv.with_name(
        "unprivileged_selections_frozen.csv"
    )
    unprivileged_matrix_path = output_matrix_csv.with_name(
        "unprivileged_candidate_matrix.csv"
    )
    unprivileged_matrix = matrix.sort_values(["run_index", "epoch"]).reset_index(
        drop=True
    )
    _atomic_csv(unprivileged_matrix, unprivileged_matrix_path)
    _atomic_csv(pd.DataFrame(unprivileged_rows), unprivileged_table_path)

    # Only after unprivileged choices are persisted and hashed may privileged
    # Oracle metrics enter memory or the analysis matrix.
    if oracle_scores_summary_json is not None:
        aggregate_summaries["oracle_scores"] = _validate_aggregate_summary(
            oracle_scores_summary_json,
            input_paths["oracle_scores"],
            expected_artifact_type="fcv_vit_oracle_score_pool_summary",
            expected_candidate_count=expected_count,
        )
    oracle = pd.read_csv(input_paths["oracle_scores"])
    _require_columns(oracle, ORACLE_METRIC_COLUMNS, "Oracle scores")
    _validate_candidate_sets(
        {"candidate_metrics": candidate, "oracle_scores": oracle}, expected_count
    )
    oracle_aligned = oracle.set_index("candidate_id").loc[candidate_ids]
    for column in key_columns[1:]:
        left = matrix[column].to_numpy()
        right = oracle_aligned[column].to_numpy()
        matches = (
            np.allclose(left, right, rtol=0.0, atol=0.0)
            if np.issubdtype(np.asarray(left).dtype, np.number)
            else np.array_equal(left, right)
        )
        if not matches:
            raise SelectorError(f"oracle {column} differs from candidate metrics.")
    if not np.array_equal(
        matrix["checkpoint_path"].astype(str).to_numpy(),
        oracle_aligned["checkpoint_path"].astype(str).to_numpy(),
    ) or not np.array_equal(
        matrix["checkpoint_sha256"].astype(str).to_numpy(),
        oracle_aligned["checkpoint_sha256"].astype(str).to_numpy(),
    ):
        raise SelectorError("Oracle checkpoint path/hash differs from candidate metrics.")
    for column in ORACLE_METRIC_COLUMNS:
        if column not in key_columns and column not in {
            "checkpoint_path",
            "checkpoint_sha256",
            "summary_path",
        }:
            matrix[column] = oracle_aligned[column].to_numpy()
    oracle_specs = [
        {
            "selector_name": "oracle_validation_worst_group_accuracy",
            "selector_family": "oracle",
            "score_column": "oracle_validation_worst_group_accuracy",
            "direction": "maximize",
            "formula": "minimum original-validation group accuracy",
            "availability": "privileged_analysis_only",
        },
        {
            "selector_name": "oracle_validation_balanced_group_accuracy",
            "selector_family": "oracle",
            "score_column": "oracle_validation_balanced_group_accuracy",
            "direction": "maximize",
            "formula": "mean original-validation group accuracy",
            "availability": "privileged_analysis_only",
        },
    ]
    oracle_rows = [
        _select_one(
            matrix,
            selector_name=spec["selector_name"],
            selector_family=spec["selector_family"],
            score_column=spec["score_column"],
            direction=spec["direction"],
            formula=spec["formula"],
            availability=spec["availability"],
        )
        for spec in oracle_specs
    ]
    selection_rows = unprivileged_rows + oracle_rows
    selection_table = pd.DataFrame(selection_rows)
    matrix = matrix.sort_values(["run_index", "epoch"]).reset_index(drop=True)
    _atomic_csv(matrix, output_matrix_csv)
    _atomic_csv(selection_table, output_table_csv)
    input_hashes = {name: _sha256_file(path) for name, path in input_paths.items()}
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_selection_table_summary",
        "status": "complete",
        "candidate_count": len(matrix),
        "selector_count": len(selection_table),
        "selector_analysis_fingerprint": selector_analysis_fingerprint(config),
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "input_sha256": input_hashes,
        "aggregate_summaries": aggregate_summaries,
        "selection_table_path": str(output_table_csv),
        "selection_table_sha256": _sha256_file(output_table_csv),
        "candidate_selector_matrix_path": str(output_matrix_csv),
        "candidate_selector_matrix_sha256": _sha256_file(output_matrix_csv),
        "unprivileged_selection_table_path": str(unprivileged_table_path),
        "unprivileged_selection_table_sha256": _sha256_file(
            unprivileged_table_path
        ),
        "unprivileged_candidate_matrix_path": str(unprivileged_matrix_path),
        "unprivileged_candidate_matrix_sha256": _sha256_file(
            unprivileged_matrix_path
        ),
        "unprivileged_selection_frozen_before_oracle_join": True,
        "deterministic_tie_break": selector_cfg["deterministic_tie_break"],
        "selected_candidates": {
            row["selector_name"]: row["selected_checkpoint_id"]
            for row in selection_rows
        },
        "selected_checkpoint_sha256": {
            row["selector_name"]: row["selected_checkpoint_sha256"]
            for row in selection_rows
        },
        "test_data_accessed": False,
        "test_metrics_deferred_to_step": 10,
    }
    _atomic_json(summary, output_summary_json)
    return summary
