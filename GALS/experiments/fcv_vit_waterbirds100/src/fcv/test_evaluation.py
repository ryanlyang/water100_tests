"""Step 10 evaluation of frozen Step 9 selections on Waterbirds test data."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

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
from .selectors import (
    compute_waterbirds_group_metrics,
    selector_analysis_fingerprint,
)
from .vit_counterfactual_forward import load_candidate_model
from .waterbirds_metadata import GROUP_NAMES


TEST_STUDY_SPLIT = "test_analysis_only"
TEST_REQUIRED_COLUMNS = {
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
SELECTION_REQUIRED_COLUMNS = {
    "selector_name",
    "selector_family",
    "availability",
    "direction",
    "selector_formula",
    "selector_score",
    "selected_checkpoint_id",
    "selected_checkpoint_path",
    "selected_checkpoint_sha256",
    "selected_hparams",
}
TEST_PER_IMAGE_COLUMNS = [
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


class FinalTestError(ValueError):
    """Raised when frozen selection or final-test provenance is invalid."""


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


def final_test_evaluation_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {
        "study": config["study"],
        "model": config["model"],
        "training_fingerprint": candidate_training_fingerprint(config),
        "final_test": config["evaluation"]["final_test"],
        "primary_test_metric": config["evaluation"]["primary_test_metric"],
        "secondary_test_metrics": config["evaluation"]["secondary_test_metrics"],
        "group_names": GROUP_NAMES,
    }
    return _sha256_json(payload)


@dataclass(frozen=True)
class SelectedCheckpoint:
    candidate_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    selectors: tuple[str, ...]


@dataclass
class FrozenSelection:
    selection_table_path: Path
    selection_table_sha256: str
    selection_summary_path: Path
    selector_matrix_path: Path
    selector_matrix_sha256: str
    table: pd.DataFrame
    unique_checkpoints: tuple[SelectedCheckpoint, ...]
    pool_checkpoints: tuple[SelectedCheckpoint, ...]


def frozen_pool_checkpoint(
    frozen: FrozenSelection,
    candidate_id: str,
    checkpoint_path: str | Path | None = None,
    *,
    verify_bytes: bool = True,
) -> SelectedCheckpoint:
    """Return one Step-9-frozen pool identity and optionally verify its bytes."""

    matches = [
        item for item in frozen.pool_checkpoints if item.candidate_id == str(candidate_id)
    ]
    if len(matches) != 1:
        raise FinalTestError(
            f"Candidate is absent or duplicated in the frozen Step 9 pool: {candidate_id}"
        )
    expected = matches[0]
    if checkpoint_path is not None:
        observed_path = Path(checkpoint_path).expanduser().resolve()
        if observed_path != expected.checkpoint_path:
            raise FinalTestError(
                "Candidate checkpoint path differs from the frozen Step 9 matrix: "
                f"{candidate_id}"
            )
    if verify_bytes:
        if (
            not expected.checkpoint_path.is_file()
            or _sha256_file(expected.checkpoint_path) != expected.checkpoint_sha256
        ):
            raise FinalTestError(
                "Candidate checkpoint bytes differ from the frozen Step 9 matrix: "
                f"{candidate_id}"
            )
    return expected


def load_frozen_selection(
    config: Mapping[str, Any],
    selection_table_path: str | Path,
    selection_summary_path: str | Path,
) -> FrozenSelection:
    """Validate and freeze Step 9 outputs before test data becomes accessible."""

    selection_table_path = Path(selection_table_path).expanduser().resolve()
    selection_summary_path = Path(selection_summary_path).expanduser().resolve()
    if not selection_table_path.is_file():
        raise FileNotFoundError(f"Missing Step 9 selection table: {selection_table_path}")
    if not selection_summary_path.is_file():
        raise FileNotFoundError(
            f"Missing Step 9 selection summary: {selection_summary_path}"
        )
    with selection_summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    table_sha256 = _sha256_file(selection_table_path)
    valid_summary = (
        summary.get("schema_version") == 1
        and summary.get("artifact_type") == "fcv_vit_selection_table_summary"
        and summary.get("status") == "complete"
        and summary.get("selector_analysis_fingerprint")
        == selector_analysis_fingerprint(config)
        and summary.get("selection_table_path") == str(selection_table_path)
        and summary.get("selection_table_sha256") == table_sha256
        and summary.get("test_data_accessed") is False
        and int(summary.get("test_metrics_deferred_to_step", -1)) == 10
    )
    if not valid_summary:
        raise FinalTestError("Step 9 selection summary is stale or incomplete.")
    matrix_path = Path(
        str(summary.get("candidate_selector_matrix_path", ""))
    ).expanduser().resolve()
    if (
        not matrix_path.is_file()
        or _sha256_file(matrix_path)
        != summary.get("candidate_selector_matrix_sha256")
    ):
        raise FinalTestError("Step 9 candidate selector matrix is missing or stale.")

    table = pd.read_csv(selection_table_path)
    missing = sorted(SELECTION_REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise FinalTestError(f"Selection table is missing columns: {missing}")
    test_columns = [column for column in table.columns if column.startswith("test_")]
    if test_columns:
        raise FinalTestError(
            f"Step 9 selection must be frozen before test metrics exist: {test_columns}"
        )
    if table.empty or table["selector_name"].astype(str).duplicated().any():
        raise FinalTestError("Selection table must contain unique selector rows.")
    if len(table) != int(summary.get("selector_count", -1)):
        raise FinalTestError("Selection table row count differs from its summary.")
    expected_selected = {
        str(selector): str(candidate)
        for selector, candidate in summary.get("selected_candidates", {}).items()
    }
    actual_selected = dict(
        zip(
            table["selector_name"].astype(str),
            table["selected_checkpoint_id"].astype(str),
        )
    )
    if actual_selected != expected_selected:
        raise FinalTestError("Selector-to-candidate mapping differs from Step 9 summary.")

    matrix = pd.read_csv(matrix_path)
    required_matrix = {"candidate_id", "checkpoint_path", "checkpoint_sha256"}
    if not required_matrix.issubset(matrix.columns):
        raise FinalTestError("Step 9 selector matrix lacks checkpoint identity columns.")
    if matrix["candidate_id"].astype(str).duplicated().any():
        raise FinalTestError("Step 9 selector matrix has duplicate candidate IDs.")
    expected_candidate_ids = [
        run.candidate_id(epoch)
        for run in enumerate_sweep_runs(config)
        for epoch in candidate_epochs(config)
    ]
    observed_candidate_ids = matrix["candidate_id"].astype(str).tolist()
    if observed_candidate_ids != expected_candidate_ids:
        missing_ids = sorted(set(expected_candidate_ids).difference(observed_candidate_ids))
        unexpected_ids = sorted(
            set(observed_candidate_ids).difference(expected_candidate_ids)
        )
        raise FinalTestError(
            "Step 9 selector matrix is not the complete locked candidate pool: "
            f"expected={len(expected_candidate_ids)} observed={len(observed_candidate_ids)} "
            f"missing={missing_ids[:5]} unexpected={unexpected_ids[:5]}."
        )
    matrix_paths = dict(
        zip(
            matrix["candidate_id"].astype(str),
            matrix["checkpoint_path"].astype(str),
        )
    )
    matrix_hashes = dict(
        zip(
            matrix["candidate_id"].astype(str),
            matrix["checkpoint_sha256"].astype(str),
        )
    )
    pool_checkpoints_list: List[SelectedCheckpoint] = []
    seen_pool_paths: set[Path] = set()
    for row in matrix.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        checkpoint_path = Path(str(row.checkpoint_path)).expanduser().resolve()
        checkpoint_sha256 = str(row.checkpoint_sha256)
        if len(checkpoint_sha256) != 64:
            raise FinalTestError(
                f"Step 9 matrix has an invalid checkpoint hash for {candidate_id}."
            )
        if checkpoint_path in seen_pool_paths:
            raise FinalTestError(
                f"Step 9 matrix maps multiple candidates to {checkpoint_path}."
            )
        seen_pool_paths.add(checkpoint_path)
        pool_checkpoints_list.append(
            SelectedCheckpoint(
                candidate_id=candidate_id,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                selectors=(),
            )
        )

    selected_paths: Dict[str, Path] = {}
    selected_hashes: Dict[str, str] = {}
    selectors_by_candidate: Dict[str, List[str]] = {}
    candidate_order: List[str] = []
    for row in table.itertuples(index=False):
        selector_name = str(row.selector_name)
        candidate_id = str(row.selected_checkpoint_id)
        checkpoint_path = Path(str(row.selected_checkpoint_path)).expanduser().resolve()
        checkpoint_sha256 = str(row.selected_checkpoint_sha256)
        if candidate_id not in matrix_paths:
            raise FinalTestError(f"Selected candidate is absent from Step 9 matrix: {candidate_id}")
        if str(checkpoint_path) != str(Path(matrix_paths[candidate_id]).expanduser().resolve()):
            raise FinalTestError(f"Selected checkpoint path differs for {candidate_id}.")
        if (
            checkpoint_sha256 != matrix_hashes[candidate_id]
            or not checkpoint_path.is_file()
            or _sha256_file(checkpoint_path) != checkpoint_sha256
        ):
            raise FinalTestError(
                f"Selected checkpoint bytes differ from frozen selection: {candidate_id}"
            )
        if candidate_id in selected_paths and selected_paths[candidate_id] != checkpoint_path:
            raise FinalTestError(f"Candidate {candidate_id} maps to multiple checkpoints.")
        if candidate_id not in selected_paths:
            selected_paths[candidate_id] = checkpoint_path
            selected_hashes[candidate_id] = checkpoint_sha256
            selectors_by_candidate[candidate_id] = []
            candidate_order.append(candidate_id)
        selectors_by_candidate[candidate_id].append(selector_name)

    unique_checkpoints = tuple(
        SelectedCheckpoint(
            candidate_id=candidate_id,
            checkpoint_path=selected_paths[candidate_id],
            checkpoint_sha256=selected_hashes[candidate_id],
            selectors=tuple(selectors_by_candidate[candidate_id]),
        )
        for candidate_id in candidate_order
    )
    if not bool(config["evaluation"]["final_test"]["evaluate_unique_checkpoints_once"]):
        raise FinalTestError("Step 10 must evaluate unique selected checkpoints once.")
    return FrozenSelection(
        selection_table_path=selection_table_path,
        selection_table_sha256=table_sha256,
        selection_summary_path=selection_summary_path,
        selector_matrix_path=matrix_path,
        selector_matrix_sha256=str(summary["candidate_selector_matrix_sha256"]),
        table=table,
        unique_checkpoints=unique_checkpoints,
        pool_checkpoints=tuple(pool_checkpoints_list),
    )


class TestManifestDataset(Dataset):
    """Evaluation-only Waterbirds test split with group labels."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Any,
        *,
        check_images: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing test manifest: {self.manifest_path}")
        frame = pd.read_csv(self.manifest_path)
        missing = sorted(TEST_REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise FinalTestError(f"Test manifest is missing columns: {missing}")
        if set(frame["study_split"].astype(str).unique()) != {TEST_STUDY_SPLIT}:
            raise FinalTestError(
                f"Step 10 requires study_split={TEST_STUDY_SPLIT!r}."
            )
        if set(frame["source_split"].astype(str).unique()) != {"test"}:
            raise FinalTestError("Step 10 requires the source test split.")
        if frame["sample_id"].astype(str).duplicated().any():
            raise FinalTestError("Test sample IDs must be unique.")
        labels = frame["label"].astype(int)
        contexts = frame["context"].astype(int)
        groups = frame["group"].astype(int)
        if set(labels.unique()) != {0, 1} or set(contexts.unique()) != {0, 1}:
            raise FinalTestError("Test labels and contexts must both be binary.")
        if set(groups.unique()) != set(GROUP_NAMES):
            raise FinalTestError("Test evaluation requires all four Waterbirds groups.")
        if not np.array_equal(groups.to_numpy(), (labels * 2 + contexts).to_numpy()):
            raise FinalTestError("Test group IDs are inconsistent with label/context.")
        if not np.array_equal(
            frame["group_name"].astype(str).to_numpy(),
            groups.map(GROUP_NAMES).astype(str).to_numpy(),
        ):
            raise FinalTestError("Test group names are inconsistent with group IDs.")
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.image_paths = [Path(str(value)) for value in self.frame["image_path"]]
        if check_images:
            missing_images = [str(path) for path in self.image_paths if not path.is_file()]
            if missing_images:
                raise FileNotFoundError(
                    f"Test manifest has {len(missing_images)} missing images. "
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
                raise FinalTestError(
                    "Test image bytes differ from the frozen manifest. First paths: "
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
            raise RuntimeError(f"Could not read test image {path}: {exc}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), int(row["group"]), str(row["sample_id"])


@dataclass(frozen=True)
class FinalTestSource:
    manifest_path: Path
    manifest_sha256: str
    manifest_bundle_path: Path
    manifest_bundle_sha256: str
    dataset: TestManifestDataset
    loader: DataLoader
    batch_size: int
    num_workers: int

    @property
    def sample_count(self) -> int:
        return len(self.dataset)


def prepare_final_test_source(
    config: Mapping[str, Any],
    manifest_path: str | Path,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    check_images: bool = True,
) -> FinalTestSource:
    final_cfg = config["evaluation"]["final_test"]
    if final_cfg["precision"] != "float32":
        raise FinalTestError("Step 10 inference must use float32.")
    try:
        manifest_binding = validate_manifest_bundle(config, manifest_path, "test")
    except ManifestProvenanceError as exc:
        raise FinalTestError(str(exc)) from exc
    dataset = TestManifestDataset(
        manifest_path,
        build_transforms(config)["eval"],
        check_images=check_images,
    )
    resolved_batch_size = int(
        final_cfg["batch_size"] if batch_size is None else batch_size
    )
    resolved_workers = int(
        config["training"]["num_workers"] if num_workers is None else num_workers
    )
    if resolved_batch_size != int(final_cfg["batch_size"]):
        raise FinalTestError("Final-test batch size differs from the locked value.")
    if resolved_workers != int(config["training"]["num_workers"]):
        raise FinalTestError("Final-test worker count differs from the locked value.")
    loader = DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=resolved_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        generator=torch.Generator(device="cpu").manual_seed(
            int(final_cfg["dataloader_seed"])
        ),
    )
    return FinalTestSource(
        manifest_path=dataset.manifest_path,
        manifest_sha256=manifest_binding.manifest_sha256,
        manifest_bundle_path=manifest_binding.bundle_path,
        manifest_bundle_sha256=manifest_binding.bundle_sha256,
        dataset=dataset,
        loader=loader,
        batch_size=resolved_batch_size,
        num_workers=resolved_workers,
    )


def _validate_test_metrics(metrics: Mapping[str, Any], expected_sample_count: int) -> None:
    required = {
        "loss",
        "accuracy",
        "balanced_group_accuracy",
        "worst_group_accuracy",
        "sample_count",
    }
    for group in range(4):
        required.update(
            {
                f"group_{group}_accuracy",
                f"group_{group}_correct",
                f"group_{group}_count",
            }
        )
    missing = sorted(required.difference(metrics))
    if missing:
        raise FinalTestError(f"Final-test metrics are missing fields: {missing}")
    sample_count = int(metrics["sample_count"])
    if sample_count != expected_sample_count:
        raise FinalTestError("Final-test metric sample count is inconsistent.")
    group_counts = [int(metrics[f"group_{group}_count"]) for group in range(4)]
    group_correct = [int(metrics[f"group_{group}_correct"]) for group in range(4)]
    if sum(group_counts) != sample_count or any(count <= 0 for count in group_counts):
        raise FinalTestError("Final-test group counts are invalid.")
    if any(correct < 0 or correct > count for correct, count in zip(group_correct, group_counts)):
        raise FinalTestError("Final-test group correct counts are invalid.")
    group_accuracies = [correct / count for correct, count in zip(group_correct, group_counts)]
    saved_group_accuracies = [
        float(metrics[f"group_{group}_accuracy"]) for group in range(4)
    ]
    if not np.allclose(
        saved_group_accuracies, group_accuracies, rtol=0.0, atol=1.0e-12
    ):
        raise FinalTestError("Final-test group accuracies do not reproduce.")
    expected_accuracy = sum(group_correct) / sample_count
    expected_balanced = float(np.mean(group_accuracies))
    expected_worst = float(np.min(group_accuracies))
    for key, expected in (
        ("accuracy", expected_accuracy),
        ("balanced_group_accuracy", expected_balanced),
        ("worst_group_accuracy", expected_worst),
    ):
        value = float(metrics[key])
        if not np.isfinite(value) or not np.isclose(value, expected, rtol=0.0, atol=1.0e-12):
            raise FinalTestError(f"Final-test {key} does not reproduce.")
    loss = float(metrics["loss"])
    if not np.isfinite(loss) or loss < 0.0:
        raise FinalTestError("Final-test loss is invalid.")


def recompute_test_metrics_from_frame(
    frame: pd.DataFrame,
    source: FinalTestSource,
    *,
    candidate_id: str,
) -> Dict[str, Any]:
    """Recompute all test metrics from the persisted per-example predictions."""

    missing = sorted(set(TEST_PER_IMAGE_COLUMNS).difference(frame.columns))
    if missing:
        raise FinalTestError(f"Test per-example CSV is missing columns: {missing}")
    expected = source.dataset.frame.reset_index(drop=True)
    observed = frame.reset_index(drop=True)
    if len(observed) != source.sample_count:
        raise FinalTestError("Test per-example row count differs from the manifest.")
    if set(observed["candidate_id"].astype(str).unique()) != {candidate_id}:
        raise FinalTestError("Test per-example candidate identity is inconsistent.")
    if observed["sample_id"].astype(str).tolist() != expected[
        "sample_id"
    ].astype(str).tolist():
        raise FinalTestError("Test per-example sample order differs from the manifest.")
    for column in ("label", "group"):
        if not np.array_equal(
            observed[column].astype(int).to_numpy(),
            expected[column].astype(int).to_numpy(),
        ):
            raise FinalTestError(f"Test per-example {column} differs from the manifest.")

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
        raise FinalTestError("Test per-example losses/probabilities are invalid.")
    if not np.array_equal(correctness, (predictions == labels).astype(int)):
        raise FinalTestError("Test per-example correctness does not reproduce.")

    for index, row in observed.iterrows():
        try:
            logits = np.asarray(json.loads(str(row["logits"])), dtype=np.float64)
            probabilities = np.asarray(
                json.loads(str(row["probabilities"])), dtype=np.float64
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FinalTestError("Test logits/probabilities are not valid JSON.") from exc
        if logits.shape != (2,) or probabilities.shape != (2,):
            raise FinalTestError("Test logits/probabilities must contain two classes.")
        if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise FinalTestError("Test logits/probabilities contain non-finite values.")
        shifted = logits - logits.max()
        expected_probabilities = np.exp(shifted) / np.exp(shifted).sum()
        if not np.allclose(
            probabilities, expected_probabilities, rtol=0.0, atol=1.0e-7
        ):
            raise FinalTestError("Test probabilities do not reproduce from logits.")
        label = int(labels[index])
        if int(predictions[index]) != int(np.argmax(logits)):
            raise FinalTestError("Test prediction does not reproduce from logits.")
        if not np.isclose(
            true_probabilities[index], probabilities[label], rtol=0.0, atol=1.0e-7
        ):
            raise FinalTestError("Test true-class probability does not reproduce.")
        expected_loss = float(-np.log(max(probabilities[label], 1.0e-300)))
        if not np.isclose(losses[index], expected_loss, rtol=0.0, atol=1.0e-6):
            raise FinalTestError("Test per-example loss does not reproduce.")

    group_total = [int((groups == group).sum()) for group in range(4)]
    group_correct = [
        int(correctness[groups == group].sum()) for group in range(4)
    ]
    metrics = compute_waterbirds_group_metrics(
        loss_sum=float(losses.sum()),
        correct=int(correctness.sum()),
        total=len(observed),
        group_correct=group_correct,
        group_total=group_total,
    )
    _validate_test_metrics(metrics, source.sample_count)
    return metrics


def validate_test_summary_against_frame(
    summary: Mapping[str, Any],
    frame: pd.DataFrame,
    source: FinalTestSource,
) -> Dict[str, Any]:
    recomputed = recompute_test_metrics_from_frame(
        frame, source, candidate_id=str(summary.get("candidate_id", ""))
    )
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise FinalTestError("Test summary has no metric mapping.")
    for key in ("loss", "accuracy", "balanced_group_accuracy", "worst_group_accuracy"):
        if not np.isclose(
            float(metrics.get(key, float("nan"))),
            float(recomputed[key]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise FinalTestError(f"Test summary metric {key} does not reproduce.")
    if int(metrics.get("sample_count", -1)) != int(recomputed["sample_count"]):
        raise FinalTestError("Test summary sample count does not reproduce.")
    for group in range(4):
        for suffix in ("correct", "count"):
            key = f"group_{group}_{suffix}"
            if int(metrics.get(key, -1)) != int(recomputed[key]):
                raise FinalTestError(f"Test summary {key} does not reproduce.")
        key = f"group_{group}_accuracy"
        if not np.isclose(
            float(metrics.get(key, float("nan"))),
            float(recomputed[key]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise FinalTestError(f"Test summary {key} does not reproduce.")
    return recomputed


def _existing_test_summary(
    path: Path,
    *,
    selected: SelectedCheckpoint,
    checkpoint_sha256: str,
    source: FinalTestSource,
    config: Mapping[str, Any],
) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    valid = (
        summary.get("schema_version") == 2
        and summary.get("artifact_type") == "fcv_vit_selected_test_summary"
        and summary.get("status") == "complete"
        and summary.get("candidate_id") == selected.candidate_id
        and summary.get("checkpoint_path") == str(selected.checkpoint_path)
        and summary.get("checkpoint_sha256") == checkpoint_sha256
        and summary.get("training_fingerprint") == candidate_training_fingerprint(config)
        and summary.get("final_test_evaluation_fingerprint")
        == final_test_evaluation_fingerprint(config)
        and summary.get("test_manifest_path") == str(source.manifest_path)
        and summary.get("test_manifest_sha256") == source.manifest_sha256
        and summary.get("manifest_bundle_sha256") == source.manifest_bundle_sha256
        and int(summary.get("test_sample_count", -1)) == source.sample_count
        and summary.get("precision") == "float32"
        and summary.get("execution")
        == {
            "batch_size": source.batch_size,
            "num_workers": source.num_workers,
        }
        and summary.get("test_data_accessed") is True
        and isinstance(summary.get("metrics"), Mapping)
    )
    if not valid:
        return None
    try:
        per_image_path = Path(
            str(summary.get("per_image_csv_path", ""))
        ).expanduser().resolve()
        if (
            not per_image_path.is_file()
            or summary.get("per_image_csv_sha256") != _sha256_file(per_image_path)
        ):
            return None
        validate_test_summary_against_frame(
            summary, pd.read_csv(per_image_path), source
        )
    except (OSError, ValueError, KeyError, TypeError, FinalTestError):
        return None
    return summary


def evaluate_checkpoint_test_metrics(
    config: Mapping[str, Any],
    candidate_id: str,
    checkpoint_path: str | Path,
    source: FinalTestSource,
    *,
    device: str | torch.device = "cuda",
) -> tuple[Dict[str, Any], str, pd.DataFrame]:
    """Run the locked float32 test inference shared by Steps 10 and 11."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for final-test inference but is unavailable.")
    model, checkpoint = load_candidate_model(config, checkpoint_path, device=device)
    if checkpoint.get("candidate_id") != candidate_id:
        raise FinalTestError(
            f"Loaded checkpoint identity differs from expected candidate {candidate_id}."
        )
    model.eval()
    per_image_rows: List[Dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for images, labels, groups, sample_ids in source.loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images).float()
                if logits.ndim != 2 or logits.shape[1] != 2:
                    raise FinalTestError(
                        f"Test logits have invalid shape: {tuple(logits.shape)}"
                    )
                per_example_loss = F.cross_entropy(logits, labels, reduction="none")
                probabilities = logits.softmax(dim=1)
                predictions = logits.argmax(dim=1)
                labels_cpu = labels.cpu()
                predictions_cpu = predictions.cpu()
                groups_cpu = groups.to(torch.long).cpu()
                logits_cpu = logits.cpu()
                probabilities_cpu = probabilities.cpu()
                losses_cpu = per_example_loss.cpu()
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
        per_image_frame = pd.DataFrame(
            per_image_rows, columns=TEST_PER_IMAGE_COLUMNS
        )
        metrics = recompute_test_metrics_from_frame(
            per_image_frame, source, candidate_id=candidate_id
        )
        training_fingerprint = str(checkpoint["training_fingerprint"])
        return metrics, training_fingerprint, per_image_frame
    finally:
        del model, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_selected_checkpoint(
    config: Mapping[str, Any],
    selected: SelectedCheckpoint,
    source: FinalTestSource,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Evaluate one already-selected candidate exactly once on test data."""

    checkpoint_sha256 = _sha256_file(selected.checkpoint_path)
    if checkpoint_sha256 != selected.checkpoint_sha256:
        raise FinalTestError(
            f"Checkpoint changed after Step 9 freeze: {selected.candidate_id}"
        )
    summary_path = Path(output_dir).expanduser().resolve() / (
        f"{selected.candidate_id}_test_summary.json"
    )
    per_image_path = Path(output_dir).expanduser().resolve() / (
        f"{selected.candidate_id}_test_per_image.csv"
    )
    existing = _existing_test_summary(
        summary_path,
        selected=selected,
        checkpoint_sha256=checkpoint_sha256,
        source=source,
        config=config,
    )
    if existing is not None and not overwrite:
        result = dict(existing)
        result["invocation_status"] = "already_complete"
        result["summary_path"] = str(summary_path)
        return result
    if summary_path.exists() and not overwrite:
        raise FinalTestError(
            f"Stale test summary exists for {selected.candidate_id}; use --overwrite."
        )
    metrics, training_fingerprint, per_image_frame = evaluate_checkpoint_test_metrics(
        config,
        selected.candidate_id,
        selected.checkpoint_path,
        source,
        device=device,
    )
    _atomic_csv(per_image_frame, per_image_path)
    metrics = recompute_test_metrics_from_frame(
        pd.read_csv(per_image_path), source, candidate_id=selected.candidate_id
    )
    summary = {
        "schema_version": 2,
        "artifact_type": "fcv_vit_selected_test_summary",
        "status": "complete",
        "candidate_id": selected.candidate_id,
        "checkpoint_path": str(selected.checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "training_fingerprint": training_fingerprint,
        "final_test_evaluation_fingerprint": final_test_evaluation_fingerprint(config),
        "test_manifest_path": str(source.manifest_path),
        "test_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_path": str(source.manifest_bundle_path),
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "test_sample_count": source.sample_count,
        "per_image_csv_path": str(per_image_path),
        "per_image_csv_sha256": _sha256_file(per_image_path),
        "precision": "float32",
        "execution": {
            "batch_size": source.batch_size,
            "num_workers": source.num_workers,
        },
        "metrics": metrics,
        "test_data_accessed": True,
        "selection_was_frozen_before_evaluation": True,
    }
    _atomic_json(summary, summary_path)
    result = dict(summary)
    result["invocation_status"] = "complete"
    result["summary_path"] = str(summary_path)
    return result


def assemble_final_test_results(
    config: Mapping[str, Any],
    frozen: FrozenSelection,
    source: FinalTestSource,
    candidate_summaries: Mapping[str, Mapping[str, Any]],
    output_csv: str | Path,
    output_summary: str | Path,
) -> Dict[str, Any]:
    """Expand unique candidate metrics to selector rows without test-based ordering."""

    expected_ids = {item.candidate_id for item in frozen.unique_checkpoints}
    if set(candidate_summaries) != expected_ids:
        raise FinalTestError("Final-test summaries do not match unique frozen candidates.")
    selected_by_id = {
        item.candidate_id: item for item in frozen.unique_checkpoints
    }
    metrics_by_candidate: Dict[str, Mapping[str, Any]] = {}
    summary_paths: Dict[str, str] = {}
    summary_hashes: Dict[str, str] = {}
    per_image_paths: Dict[str, str] = {}
    per_image_hashes: Dict[str, str] = {}
    for candidate_id, summary in candidate_summaries.items():
        if (
            summary.get("schema_version") != 2
            or summary.get("artifact_type") != "fcv_vit_selected_test_summary"
            or summary.get("status") != "complete"
            or summary.get("candidate_id") != candidate_id
            or summary.get("test_data_accessed") is not True
            or summary.get("selection_was_frozen_before_evaluation") is not True
            or summary.get("checkpoint_path")
            != str(selected_by_id[candidate_id].checkpoint_path)
            or summary.get("checkpoint_sha256")
            != selected_by_id[candidate_id].checkpoint_sha256
            or _sha256_file(selected_by_id[candidate_id].checkpoint_path)
            != selected_by_id[candidate_id].checkpoint_sha256
            or summary.get("training_fingerprint")
            != candidate_training_fingerprint(config)
            or summary.get("final_test_evaluation_fingerprint")
            != final_test_evaluation_fingerprint(config)
            or summary.get("test_manifest_path") != str(source.manifest_path)
            or summary.get("test_manifest_sha256") != source.manifest_sha256
            or summary.get("manifest_bundle_sha256")
            != source.manifest_bundle_sha256
            or int(summary.get("test_sample_count", -1)) != source.sample_count
            or summary.get("precision") != "float32"
            or summary.get("execution")
            != {
                "batch_size": source.batch_size,
                "num_workers": source.num_workers,
            }
        ):
            raise FinalTestError(f"Invalid final-test summary for {candidate_id}.")
        metrics = summary.get("metrics")
        if not isinstance(metrics, Mapping):
            raise FinalTestError(f"Missing final-test metrics for {candidate_id}.")
        per_image_path = Path(
            str(summary.get("per_image_csv_path", ""))
        ).expanduser().resolve()
        if (
            not per_image_path.is_file()
            or summary.get("per_image_csv_sha256") != _sha256_file(per_image_path)
        ):
            raise FinalTestError(
                f"Missing or stale per-example test records for {candidate_id}."
            )
        metrics = validate_test_summary_against_frame(
            summary, pd.read_csv(per_image_path), source
        )
        metrics_by_candidate[candidate_id] = metrics
        per_image_paths[candidate_id] = str(per_image_path)
        per_image_hashes[candidate_id] = _sha256_file(per_image_path)
        summary_path = Path(str(summary.get("summary_path", ""))).expanduser().resolve()
        if not summary_path.is_file():
            raise FinalTestError(f"Missing persisted test summary for {candidate_id}.")
        summary_paths[candidate_id] = str(summary_path)
        summary_hashes[candidate_id] = _sha256_file(summary_path)

    final = frozen.table.copy()
    if not bool(config["evaluation"]["final_test"]["preserve_selector_order"]):
        raise FinalTestError("Step 10 must preserve frozen selector order.")
    original_selector_order = final["selector_name"].astype(str).tolist()
    metric_columns = {
        "test_loss": "loss",
        "test_average_accuracy": "accuracy",
        "test_balanced_group_accuracy": "balanced_group_accuracy",
        "test_worst_group_accuracy": "worst_group_accuracy",
        "test_landbird_land_accuracy": "group_0_accuracy",
        "test_landbird_water_accuracy": "group_1_accuracy",
        "test_waterbird_land_accuracy": "group_2_accuracy",
        "test_waterbird_water_accuracy": "group_3_accuracy",
        "test_sample_count": "sample_count",
    }
    for output_column, metric_key in metric_columns.items():
        final[output_column] = [
            metrics_by_candidate[str(candidate_id)][metric_key]
            for candidate_id in final["selected_checkpoint_id"]
        ]
    final["test_summary_path"] = [
        summary_paths[str(candidate_id)]
        for candidate_id in final["selected_checkpoint_id"]
    ]
    final["test_per_image_path"] = [
        per_image_paths[str(candidate_id)]
        for candidate_id in final["selected_checkpoint_id"]
    ]
    if final["selector_name"].astype(str).tolist() != original_selector_order:
        raise FinalTestError("Selector order changed while adding test metrics.")
    output_csv = Path(output_csv).expanduser().resolve()
    output_summary = Path(output_summary).expanduser().resolve()
    _atomic_csv(final, output_csv)
    summary = {
        "schema_version": 2,
        "artifact_type": "fcv_vit_final_test_results_summary",
        "status": "complete",
        "selector_count": len(final),
        "unique_selected_checkpoint_count": len(frozen.unique_checkpoints),
        "selection_table_path": str(frozen.selection_table_path),
        "selection_table_sha256": frozen.selection_table_sha256,
        "candidate_selector_matrix_path": str(frozen.selector_matrix_path),
        "candidate_selector_matrix_sha256": frozen.selector_matrix_sha256,
        "final_test_evaluation_fingerprint": final_test_evaluation_fingerprint(config),
        "test_manifest_path": str(source.manifest_path),
        "test_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_path": str(source.manifest_bundle_path),
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "test_sample_count": source.sample_count,
        "execution": {
            "batch_size": source.batch_size,
            "num_workers": source.num_workers,
        },
        "candidate_test_summary_paths": summary_paths,
        "candidate_test_summary_sha256": summary_hashes,
        "candidate_test_per_image_paths": per_image_paths,
        "candidate_test_per_image_sha256": per_image_hashes,
        "final_test_results_path": str(output_csv),
        "final_test_results_sha256": _sha256_file(output_csv),
        "primary_test_metric": config["evaluation"]["primary_test_metric"],
        "selection_frozen_before_test": True,
        "test_metrics_affected_selection": False,
        "selector_order_preserved": True,
    }
    _atomic_json(summary, output_summary)
    return summary
