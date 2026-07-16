"""Step 11 post-hoc pool evaluation and Oracle gap-closure analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .candidate_training import (
    candidate_training_fingerprint,
    enumerate_sweep_runs,
)
from .test_evaluation import (
    FinalTestSource,
    FrozenSelection,
    evaluate_checkpoint_test_metrics,
    final_test_evaluation_fingerprint,
    frozen_pool_checkpoint,
    recompute_test_metrics_from_frame,
    validate_test_summary_against_frame,
)
from .token_banks import candidate_checkpoints_for_run


POOL_TEST_COLUMNS = [
    "run_index",
    "candidate_id",
    "epoch",
    "seed",
    "learning_rate",
    "weight_decay",
    "checkpoint_path",
    "checkpoint_sha256",
    "test_loss",
    "test_accuracy",
    "test_balanced_group_accuracy",
    "test_worst_group_accuracy",
    "test_group_0_accuracy",
    "test_group_0_count",
    "test_group_1_accuracy",
    "test_group_1_count",
    "test_group_2_accuracy",
    "test_group_2_count",
    "test_group_3_accuracy",
    "test_group_3_count",
    "summary_path",
    "per_image_path",
    "per_image_sha256",
]


class GapAnalysisError(ValueError):
    """Raised when Step 11 provenance or gap inputs are invalid."""


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


def gap_analysis_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {
        "study": config["study"],
        "model": config["model"],
        "candidate_pool": config["candidate_pool"],
        "training_fingerprint": candidate_training_fingerprint(config),
        "final_test_evaluation_fingerprint": final_test_evaluation_fingerprint(config),
        "gap_closure": config["evaluation"]["gap_closure"],
    }
    return _sha256_json(payload)


def _validate_metrics(metrics: Mapping[str, Any], sample_count: int) -> None:
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
        raise GapAnalysisError(f"Test metrics are missing fields: {missing}")
    if int(metrics["sample_count"]) != sample_count:
        raise GapAnalysisError("Test metric sample count is stale.")
    group_counts = [int(metrics[f"group_{group}_count"]) for group in range(4)]
    group_correct = [int(metrics[f"group_{group}_correct"]) for group in range(4)]
    if sum(group_counts) != sample_count or any(count <= 0 for count in group_counts):
        raise GapAnalysisError("Test metrics have invalid group counts.")
    if any(
        correct < 0 or correct > count
        for correct, count in zip(group_correct, group_counts)
    ):
        raise GapAnalysisError("Test metrics have invalid group correct counts.")
    group_accuracies = [
        float(metrics[f"group_{group}_accuracy"]) for group in range(4)
    ]
    expected_group = [
        float(correct / count)
        for correct, count in zip(group_correct, group_counts)
    ]
    if not np.allclose(group_accuracies, expected_group, rtol=0.0, atol=1.0e-12):
        raise GapAnalysisError("Test group accuracies do not reproduce.")
    expected_accuracy = float(sum(group_correct) / sample_count)
    expected_balanced = float(np.mean(group_accuracies))
    expected_worst = float(np.min(group_accuracies))
    for key, expected in (
        ("accuracy", expected_accuracy),
        ("balanced_group_accuracy", expected_balanced),
        ("worst_group_accuracy", expected_worst),
    ):
        value = float(metrics[key])
        if not np.isfinite(value) or not np.isclose(
            value, expected, rtol=0.0, atol=1.0e-12
        ):
            raise GapAnalysisError(f"Test metric {key} does not reproduce.")
    loss = float(metrics["loss"])
    if not np.isfinite(loss) or loss < 0.0:
        raise GapAnalysisError("Test loss is invalid.")


def _candidate_identity(
    config: Mapping[str, Any], checkpoint_path: Path
) -> tuple[Any, int, str]:
    try:
        epoch = int(checkpoint_path.stem.removeprefix("epoch_"))
    except ValueError as exc:
        raise GapAnalysisError(
            f"Unrecognized candidate checkpoint name: {checkpoint_path}"
        ) from exc
    runs = {run.run_id: run for run in enumerate_sweep_runs(config)}
    run_id = checkpoint_path.parent.parent.name
    if run_id not in runs:
        raise GapAnalysisError(
            f"Checkpoint is not in the locked candidate pool: {checkpoint_path}"
        )
    run = runs[run_id]
    if epoch < 1 or epoch > int(config["training"]["epochs"]):
        raise GapAnalysisError(f"Candidate epoch is outside the locked pool: {epoch}")
    return run, epoch, run.candidate_id(epoch)


def _existing_pool_summary(
    path: Path,
    *,
    candidate_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source: FinalTestSource,
    config: Mapping[str, Any],
    frozen: FrozenSelection,
) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    valid = (
        summary.get("schema_version") == 2
        and summary.get("artifact_type") == "fcv_vit_posthoc_pool_test_summary"
        and summary.get("status") == "complete"
        and summary.get("candidate_id") == candidate_id
        and summary.get("checkpoint_path") == str(checkpoint_path)
        and summary.get("checkpoint_sha256") == checkpoint_sha256
        and summary.get("training_fingerprint")
        == candidate_training_fingerprint(config)
        and summary.get("final_test_evaluation_fingerprint")
        == final_test_evaluation_fingerprint(config)
        and summary.get("gap_analysis_fingerprint") == gap_analysis_fingerprint(config)
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
        and summary.get("posthoc_pool_analysis_only") is True
        and summary.get("eligible_for_model_selection") is False
        and summary.get("test_metrics_affected_selection") is False
        and summary.get("selection_frozen_before_test") is True
        and summary.get("selection_table_path") == str(frozen.selection_table_path)
        and summary.get("selection_table_sha256") == frozen.selection_table_sha256
        and summary.get("selection_summary_path")
        == str(frozen.selection_summary_path)
        and summary.get("selection_summary_sha256")
        == _sha256_file(frozen.selection_summary_path)
        and summary.get("candidate_selector_matrix_path")
        == str(frozen.selector_matrix_path)
        and summary.get("candidate_selector_matrix_sha256")
        == frozen.selector_matrix_sha256
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
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return summary


def evaluate_pool_candidate_test(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    source: FinalTestSource,
    frozen: FrozenSelection,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Score one candidate on test strictly for post-hoc pool analysis."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing candidate checkpoint: {checkpoint_path}")
    run, epoch, candidate_id = _candidate_identity(config, checkpoint_path)
    try:
        frozen_identity = frozen_pool_checkpoint(
            frozen, candidate_id, checkpoint_path, verify_bytes=True
        )
    except ValueError as exc:
        raise GapAnalysisError(str(exc)) from exc
    checkpoint_sha256 = frozen_identity.checkpoint_sha256
    summary_path = Path(output_dir).expanduser().resolve() / (
        f"{candidate_id}_pool_test_summary.json"
    )
    per_image_path = Path(output_dir).expanduser().resolve() / (
        f"{candidate_id}_pool_test_per_image.csv"
    )
    existing = _existing_pool_summary(
        summary_path,
        candidate_id=candidate_id,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        source=source,
        config=config,
        frozen=frozen,
    )
    if existing is not None and not overwrite:
        result = dict(existing)
        result["invocation_status"] = "already_complete"
        result["summary_path"] = str(summary_path)
        return result
    if summary_path.exists() and not overwrite:
        raise GapAnalysisError(
            f"Stale post-hoc test summary exists for {candidate_id}; use --overwrite."
        )
    metrics, training_fingerprint, per_image_frame = evaluate_checkpoint_test_metrics(
        config,
        candidate_id,
        checkpoint_path,
        source,
        device=device,
    )
    _atomic_csv(per_image_frame, per_image_path)
    metrics = recompute_test_metrics_from_frame(
        pd.read_csv(per_image_path), source, candidate_id=candidate_id
    )
    summary = {
        "schema_version": 2,
        "artifact_type": "fcv_vit_posthoc_pool_test_summary",
        "status": "complete",
        "candidate_id": candidate_id,
        "run": asdict(run),
        "epoch": epoch,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "training_fingerprint": training_fingerprint,
        "final_test_evaluation_fingerprint": final_test_evaluation_fingerprint(config),
        "gap_analysis_fingerprint": gap_analysis_fingerprint(config),
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
        "posthoc_pool_analysis_only": True,
        "eligible_for_model_selection": False,
        "test_metrics_affected_selection": False,
        "selection_frozen_before_test": True,
        "selection_table_path": str(frozen.selection_table_path),
        "selection_table_sha256": frozen.selection_table_sha256,
        "selection_summary_path": str(frozen.selection_summary_path),
        "selection_summary_sha256": _sha256_file(frozen.selection_summary_path),
        "candidate_selector_matrix_path": str(frozen.selector_matrix_path),
        "candidate_selector_matrix_sha256": frozen.selector_matrix_sha256,
    }
    _atomic_json(summary, summary_path)
    result = dict(summary)
    result["invocation_status"] = "complete"
    result["summary_path"] = str(summary_path)
    return result


def aggregate_pool_test_summaries(
    config: Mapping[str, Any],
    candidate_root: str | Path,
    pool_test_dir: str | Path,
    output_csv: str | Path,
    output_summary: str | Path,
    *,
    source: FinalTestSource,
    frozen: FrozenSelection,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Strictly index post-hoc test scores for the complete candidate pool."""

    candidate_root = Path(candidate_root).expanduser().resolve()
    pool_test_dir = Path(pool_test_dir).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    output_summary = Path(output_summary).expanduser().resolve()
    expected_training = candidate_training_fingerprint(config)
    expected_final_test = final_test_evaluation_fingerprint(config)
    expected_gap = gap_analysis_fingerprint(config)
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    invalid: List[Dict[str, str]] = []
    for run in enumerate_sweep_runs(config):
        checkpoints = candidate_checkpoints_for_run(
            config, candidate_root, run.run_index
        )
        for epoch, checkpoint_path in enumerate(checkpoints, start=1):
            candidate_id = run.candidate_id(epoch)
            try:
                frozen_identity = frozen_pool_checkpoint(
                    frozen, candidate_id, checkpoint_path, verify_bytes=True
                )
            except ValueError as exc:
                invalid.append({"candidate_id": candidate_id, "error": str(exc)})
                continue
            summary_path = pool_test_dir / f"{candidate_id}_pool_test_summary.json"
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
                    == "fcv_vit_posthoc_pool_test_summary"
                    and summary.get("status") == "complete"
                    and summary.get("candidate_id") == candidate_id
                    and summary.get("run") == asdict(run)
                    and int(summary.get("epoch", -1)) == epoch
                    and summary.get("checkpoint_path") == str(checkpoint_path)
                    and summary.get("checkpoint_sha256")
                    == frozen_identity.checkpoint_sha256
                    and summary.get("training_fingerprint") == expected_training
                    and summary.get("final_test_evaluation_fingerprint")
                    == expected_final_test
                    and summary.get("gap_analysis_fingerprint") == expected_gap
                    and summary.get("test_manifest_path") == str(source.manifest_path)
                    and summary.get("test_manifest_sha256") == source.manifest_sha256
                    and summary.get("manifest_bundle_sha256")
                    == source.manifest_bundle_sha256
                    and int(summary.get("test_sample_count", -1))
                    == source.sample_count
                    and summary.get("precision") == "float32"
                    and summary.get("execution")
                    == {
                        "batch_size": source.batch_size,
                        "num_workers": source.num_workers,
                    }
                    and summary.get("test_data_accessed") is True
                    and summary.get("posthoc_pool_analysis_only") is True
                    and summary.get("eligible_for_model_selection") is False
                    and summary.get("test_metrics_affected_selection") is False
                    and summary.get("selection_frozen_before_test") is True
                    and summary.get("selection_table_path")
                    == str(frozen.selection_table_path)
                    and summary.get("selection_table_sha256")
                    == frozen.selection_table_sha256
                    and summary.get("selection_summary_path")
                    == str(frozen.selection_summary_path)
                    and summary.get("selection_summary_sha256")
                    == _sha256_file(frozen.selection_summary_path)
                    and summary.get("candidate_selector_matrix_path")
                    == str(frozen.selector_matrix_path)
                    and summary.get("candidate_selector_matrix_sha256")
                    == frozen.selector_matrix_sha256
                    and isinstance(metrics, Mapping)
                )
                if not valid:
                    raise GapAnalysisError("stale post-hoc pool-test provenance")
                per_image_path = Path(
                    str(summary.get("per_image_csv_path", ""))
                ).expanduser().resolve()
                if (
                    not per_image_path.is_file()
                    or summary.get("per_image_csv_sha256")
                    != _sha256_file(per_image_path)
                ):
                    raise GapAnalysisError("stale per-example pool-test records")
                try:
                    metrics = validate_test_summary_against_frame(
                        summary, pd.read_csv(per_image_path), source
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise GapAnalysisError(str(exc)) from exc
                row: Dict[str, Any] = {
                    "run_index": run.run_index,
                    "candidate_id": candidate_id,
                    "epoch": epoch,
                    "seed": run.seed,
                    "learning_rate": run.learning_rate,
                    "weight_decay": run.weight_decay,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": str(summary["checkpoint_sha256"]),
                    "test_loss": float(metrics["loss"]),
                    "test_accuracy": float(metrics["accuracy"]),
                    "test_balanced_group_accuracy": float(
                        metrics["balanced_group_accuracy"]
                    ),
                    "test_worst_group_accuracy": float(
                        metrics["worst_group_accuracy"]
                    ),
                    "summary_path": str(summary_path),
                    "per_image_path": str(per_image_path),
                    "per_image_sha256": _sha256_file(per_image_path),
                }
                for group in range(4):
                    row[f"test_group_{group}_accuracy"] = float(
                        metrics[f"group_{group}_accuracy"]
                    )
                    row[f"test_group_{group}_count"] = int(
                        metrics[f"group_{group}_count"]
                    )
                rows.append(row)
            except (OSError, ValueError, KeyError, TypeError, GapAnalysisError) as exc:
                invalid.append({"candidate_id": candidate_id, "error": str(exc)})
    if (missing or invalid) and not allow_incomplete:
        raise GapAnalysisError(
            f"Post-hoc candidate test pool is incomplete: missing={len(missing)} "
            f"invalid={len(invalid)}."
        )
    frame = pd.DataFrame(rows, columns=POOL_TEST_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["run_index", "epoch"]).reset_index(drop=True)
        if frame["candidate_id"].astype(str).duplicated().any():
            raise GapAnalysisError("Duplicate candidate IDs in pool test index.")
    _atomic_csv(frame, output_csv)
    expected_count = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    result = {
        "schema_version": 2,
        "artifact_type": "fcv_vit_posthoc_pool_test_index_summary",
        "status": "complete" if not missing and not invalid else "incomplete",
        "candidate_count": len(frame),
        "expected_candidate_count": expected_count,
        "missing_candidate_count": len(missing),
        "missing_candidate_preview": missing[:10],
        "invalid_candidate_count": len(invalid),
        "invalid_candidate_preview": invalid[:10],
        "training_fingerprint": expected_training,
        "final_test_evaluation_fingerprint": expected_final_test,
        "gap_analysis_fingerprint": expected_gap,
        "test_manifest_path": str(source.manifest_path),
        "test_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_path": str(source.manifest_bundle_path),
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "test_sample_count": source.sample_count,
        "execution": {
            "batch_size": source.batch_size,
            "num_workers": source.num_workers,
        },
        "output_csv": str(output_csv),
        "output_csv_sha256": _sha256_file(output_csv),
        "test_data_accessed": True,
        "posthoc_pool_analysis_only": True,
        "eligible_for_model_selection": False,
        "test_metrics_affected_selection": False,
        "selection_frozen_before_test": True,
        "selection_table_path": str(frozen.selection_table_path),
        "selection_table_sha256": frozen.selection_table_sha256,
        "selection_summary_path": str(frozen.selection_summary_path),
        "selection_summary_sha256": _sha256_file(frozen.selection_summary_path),
        "candidate_selector_matrix_path": str(frozen.selector_matrix_path),
        "candidate_selector_matrix_sha256": frozen.selector_matrix_sha256,
    }
    _atomic_json(result, output_summary)
    return result


def validate_final_test_results(
    config: Mapping[str, Any],
    frozen: FrozenSelection,
    source: FinalTestSource,
    results_csv: str | Path,
    results_summary: str | Path,
) -> pd.DataFrame:
    """Validate that Step 10 metrics preserve the exact frozen selection."""

    results_csv = Path(results_csv).expanduser().resolve()
    results_summary = Path(results_summary).expanduser().resolve()
    if not results_csv.is_file() or not results_summary.is_file():
        raise FileNotFoundError("Step 10 final-test results are missing.")
    with results_summary.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    valid = (
        summary.get("schema_version") == 2
        and summary.get("artifact_type") == "fcv_vit_final_test_results_summary"
        and summary.get("status") == "complete"
        and summary.get("selection_table_path") == str(frozen.selection_table_path)
        and summary.get("selection_table_sha256") == frozen.selection_table_sha256
        and summary.get("candidate_selector_matrix_path")
        == str(frozen.selector_matrix_path)
        and summary.get("candidate_selector_matrix_sha256")
        == frozen.selector_matrix_sha256
        and summary.get("final_test_evaluation_fingerprint")
        == final_test_evaluation_fingerprint(config)
        and summary.get("test_manifest_path") == str(source.manifest_path)
        and summary.get("test_manifest_sha256") == source.manifest_sha256
        and summary.get("manifest_bundle_sha256") == source.manifest_bundle_sha256
        and int(summary.get("test_sample_count", -1)) == source.sample_count
        and summary.get("execution")
        == {
            "batch_size": source.batch_size,
            "num_workers": source.num_workers,
        }
        and summary.get("final_test_results_path") == str(results_csv)
        and summary.get("final_test_results_sha256") == _sha256_file(results_csv)
        and summary.get("selection_frozen_before_test") is True
        and summary.get("test_metrics_affected_selection") is False
        and summary.get("selector_order_preserved") is True
    )
    if not valid:
        raise GapAnalysisError("Step 10 final-test summary is stale or unsafe.")
    frame = pd.read_csv(results_csv)
    if frame["selector_name"].astype(str).tolist() != frozen.table[
        "selector_name"
    ].astype(str).tolist():
        raise GapAnalysisError("Step 10 selector order differs from frozen Step 9.")
    for column in (
        "selected_checkpoint_id",
        "selected_checkpoint_path",
        "selected_checkpoint_sha256",
    ):
        if frame[column].astype(str).tolist() != frozen.table[column].astype(str).tolist():
            raise GapAnalysisError(f"Step 10 changed frozen {column} values.")
    required = {
        "test_average_accuracy",
        "test_balanced_group_accuracy",
        "test_worst_group_accuracy",
        "test_sample_count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise GapAnalysisError(f"Step 10 results are missing columns: {missing}")
    for column in required.difference({"test_sample_count"}):
        values = frame[column].astype(float).to_numpy()
        if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
            raise GapAnalysisError(f"Step 10 column {column} is invalid.")
    if not (frame["test_sample_count"].astype(int) == source.sample_count).all():
        raise GapAnalysisError("Step 10 sample counts differ from the test manifest.")
    per_image_paths = summary.get("candidate_test_per_image_paths")
    per_image_hashes = summary.get("candidate_test_per_image_sha256")
    if not isinstance(per_image_paths, Mapping) or not isinstance(
        per_image_hashes, Mapping
    ):
        raise GapAnalysisError("Step 10 lacks per-example test provenance.")
    for row in frame.itertuples(index=False):
        candidate_id = str(row.selected_checkpoint_id)
        per_image_path = Path(str(per_image_paths.get(candidate_id, ""))).expanduser().resolve()
        if (
            not per_image_path.is_file()
            or per_image_hashes.get(candidate_id) != _sha256_file(per_image_path)
        ):
            raise GapAnalysisError(
                f"Step 10 per-example records are stale for {candidate_id}."
            )
        metrics = recompute_test_metrics_from_frame(
            pd.read_csv(per_image_path), source, candidate_id=candidate_id
        )
        for column, key in (
            ("test_average_accuracy", "accuracy"),
            ("test_balanced_group_accuracy", "balanced_group_accuracy"),
            ("test_worst_group_accuracy", "worst_group_accuracy"),
        ):
            if not np.isclose(float(getattr(row, column)), float(metrics[key]), rtol=0.0, atol=1.0e-12):
                raise GapAnalysisError(
                    f"Step 10 {column} does not reproduce for {candidate_id}."
                )
    return frame


def load_complete_pool_index(
    config: Mapping[str, Any],
    source: FinalTestSource,
    pool_csv: str | Path,
    pool_summary: str | Path,
    frozen: FrozenSelection,
) -> pd.DataFrame:
    pool_csv = Path(pool_csv).expanduser().resolve()
    pool_summary = Path(pool_summary).expanduser().resolve()
    if not pool_csv.is_file() or not pool_summary.is_file():
        raise FileNotFoundError("Step 11 post-hoc pool test index is missing.")
    with pool_summary.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    expected_count = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    valid = (
        summary.get("schema_version") == 2
        and summary.get("artifact_type")
        == "fcv_vit_posthoc_pool_test_index_summary"
        and summary.get("status") == "complete"
        and int(summary.get("candidate_count", -1)) == expected_count
        and int(summary.get("expected_candidate_count", -1)) == expected_count
        and summary.get("gap_analysis_fingerprint") == gap_analysis_fingerprint(config)
        and summary.get("test_manifest_path") == str(source.manifest_path)
        and summary.get("test_manifest_sha256") == source.manifest_sha256
        and summary.get("manifest_bundle_sha256") == source.manifest_bundle_sha256
        and int(summary.get("test_sample_count", -1)) == source.sample_count
        and summary.get("output_csv") == str(pool_csv)
        and summary.get("output_csv_sha256") == _sha256_file(pool_csv)
        and summary.get("test_data_accessed") is True
        and summary.get("posthoc_pool_analysis_only") is True
        and summary.get("eligible_for_model_selection") is False
        and summary.get("test_metrics_affected_selection") is False
        and summary.get("selection_frozen_before_test") is True
        and summary.get("selection_table_path") == str(frozen.selection_table_path)
        and summary.get("selection_table_sha256") == frozen.selection_table_sha256
        and summary.get("selection_summary_path")
        == str(frozen.selection_summary_path)
        and summary.get("selection_summary_sha256")
        == _sha256_file(frozen.selection_summary_path)
        and summary.get("candidate_selector_matrix_path")
        == str(frozen.selector_matrix_path)
        and summary.get("candidate_selector_matrix_sha256")
        == frozen.selector_matrix_sha256
    )
    if not valid:
        raise GapAnalysisError("Post-hoc candidate pool index is stale or incomplete.")
    frame = pd.read_csv(pool_csv)
    missing = sorted(set(POOL_TEST_COLUMNS).difference(frame.columns))
    if missing or len(frame) != expected_count:
        raise GapAnalysisError(
            f"Pool test CSV is incomplete: missing_columns={missing}, rows={len(frame)}."
        )
    if frame["candidate_id"].astype(str).duplicated().any():
        raise GapAnalysisError("Pool test CSV contains duplicate candidate IDs.")
    values = frame["test_worst_group_accuracy"].astype(float).to_numpy()
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise GapAnalysisError("Pool test worst-group values are invalid.")
    for checkpoint_path, checkpoint_sha256 in zip(
        frame["checkpoint_path"], frame["checkpoint_sha256"]
    ):
        path = Path(str(checkpoint_path)).expanduser().resolve()
        if not path.is_file() or _sha256_file(path) != str(checkpoint_sha256):
            raise GapAnalysisError(f"Pool checkpoint bytes changed: {path}")
    for row in frame.itertuples(index=False):
        try:
            frozen_identity = frozen_pool_checkpoint(
                frozen,
                str(row.candidate_id),
                str(row.checkpoint_path),
                verify_bytes=True,
            )
        except ValueError as exc:
            raise GapAnalysisError(str(exc)) from exc
        if str(row.checkpoint_sha256) != frozen_identity.checkpoint_sha256:
            raise GapAnalysisError(
                "Step 11 checkpoint hash differs from frozen Step 9 for "
                f"{row.candidate_id}."
            )
        per_image_path = Path(str(row.per_image_path)).expanduser().resolve()
        if (
            not per_image_path.is_file()
            or _sha256_file(per_image_path) != str(row.per_image_sha256)
        ):
            raise GapAnalysisError(
                f"Pool per-example records changed: {row.candidate_id}."
            )
        metrics = recompute_test_metrics_from_frame(
            pd.read_csv(per_image_path), source, candidate_id=str(row.candidate_id)
        )
        for column, key in (
            ("test_accuracy", "accuracy"),
            ("test_balanced_group_accuracy", "balanced_group_accuracy"),
            ("test_worst_group_accuracy", "worst_group_accuracy"),
        ):
            if not np.isclose(float(getattr(row, column)), float(metrics[key]), rtol=0.0, atol=1.0e-12):
                raise GapAnalysisError(
                    f"Pool {column} does not reproduce for {row.candidate_id}."
                )
    return frame


def compute_gap_closure_summary(
    config: Mapping[str, Any],
    frozen: FrozenSelection,
    source: FinalTestSource,
    final_results_csv: str | Path,
    final_results_summary: str | Path,
    pool_csv: str | Path,
    pool_summary: str | Path,
    output_csv: str | Path,
    output_summary: str | Path,
) -> Dict[str, Any]:
    """Compute raw FCV-to-Oracle gap closure from already frozen selections."""

    final = validate_final_test_results(
        config, frozen, source, final_results_csv, final_results_summary
    )
    pool = load_complete_pool_index(config, source, pool_csv, pool_summary, frozen)
    gap_cfg = config["evaluation"]["gap_closure"]
    metric = str(gap_cfg["metric"])
    selectors = {
        "biased": str(gap_cfg["biased_selector"]),
        "fcv": str(gap_cfg["fcv_selector"]),
        "oracle": str(gap_cfg["oracle_selector"]),
    }
    selected_rows: Dict[str, pd.Series] = {}
    for role, selector_name in selectors.items():
        matches = final[final["selector_name"].astype(str) == selector_name]
        if len(matches) != 1:
            raise GapAnalysisError(
                f"Expected exactly one {role} selector row named {selector_name!r}."
            )
        selected_rows[role] = matches.iloc[0]
    values = {
        role: float(row[metric]) for role, row in selected_rows.items()
    }
    biased = values["biased"]
    fcv = values["fcv"]
    oracle = values["oracle"]
    fcv_gain = fcv - biased
    oracle_gap = oracle - biased
    epsilon = float(gap_cfg["denominator_epsilon"])
    if abs(oracle_gap) <= epsilon:
        gap_fraction = None
        gap_percent = None
        status = "undefined_zero_oracle_gap"
    else:
        gap_fraction = fcv_gain / oracle_gap
        gap_percent = 100.0 * gap_fraction
        status = (
            "defined_positive_oracle_gap"
            if oracle_gap > 0.0
            else "defined_negative_oracle_gap"
        )

    best_value = float(pool["test_worst_group_accuracy"].max())
    tied = pool[
        np.isclose(
            pool["test_worst_group_accuracy"].astype(float),
            best_value,
            rtol=0.0,
            atol=1.0e-12,
        )
    ].sort_values("candidate_id")
    upper = tied.iloc[0]
    upper_hparams = json.dumps(
        {
            "learning_rate": float(upper["learning_rate"]),
            "weight_decay": float(upper["weight_decay"]),
            "seed": int(upper["seed"]),
            "epoch": int(upper["epoch"]),
        },
        sort_keys=True,
    )
    row: Dict[str, Any] = {
        "metric": metric,
        "biased_selector": selectors["biased"],
        "biased_candidate_id": str(selected_rows["biased"]["selected_checkpoint_id"]),
        "biased_robust_test_performance": biased,
        "fcv_selector": selectors["fcv"],
        "fcv_candidate_id": str(selected_rows["fcv"]["selected_checkpoint_id"]),
        "fcv_robust_test_performance": fcv,
        "oracle_selector": selectors["oracle"],
        "oracle_candidate_id": str(
            selected_rows["oracle"]["selected_checkpoint_id"]
        ),
        "oracle_robust_test_performance": oracle,
        "fcv_gain_over_biased": fcv_gain,
        "oracle_gap_over_biased": oracle_gap,
        "gap_closed_fraction": gap_fraction,
        "gap_closed_percent": gap_percent,
        "gap_closure_status": status,
        "gap_fraction_clipped": False,
        "pool_upper_bound_candidate_id": str(upper["candidate_id"]),
        "pool_upper_bound_test_worst_group_accuracy": best_value,
        "pool_upper_bound_test_balanced_group_accuracy": float(
            upper["test_balanced_group_accuracy"]
        ),
        "pool_upper_bound_test_accuracy": float(upper["test_accuracy"]),
        "pool_upper_bound_hparams": upper_hparams,
        "biased_pool_regret": best_value - biased,
        "fcv_pool_regret": best_value - fcv,
        "oracle_pool_regret": best_value - oracle,
        "pool_candidate_count": len(pool),
        "pool_upper_bound_is_posthoc_unfair": True,
        "pool_scores_eligible_for_selection": False,
        "test_metrics_affected_selection": False,
    }
    output_csv = Path(output_csv).expanduser().resolve()
    output_summary = Path(output_summary).expanduser().resolve()
    _atomic_csv(pd.DataFrame([row]), output_csv)
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_gap_closure_summary",
        "status": "complete",
        "gap_analysis_fingerprint": gap_analysis_fingerprint(config),
        "selection_table_path": str(frozen.selection_table_path),
        "selection_table_sha256": frozen.selection_table_sha256,
        "final_test_results_path": str(Path(final_results_csv).expanduser().resolve()),
        "final_test_results_sha256": _sha256_file(
            Path(final_results_csv).expanduser().resolve()
        ),
        "candidate_pool_test_scores_path": str(Path(pool_csv).expanduser().resolve()),
        "candidate_pool_test_scores_sha256": _sha256_file(
            Path(pool_csv).expanduser().resolve()
        ),
        "test_manifest_path": str(source.manifest_path),
        "test_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_path": str(source.manifest_bundle_path),
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "test_sample_count": source.sample_count,
        "gap_closure": row,
        "output_csv": str(output_csv),
        "output_csv_sha256": _sha256_file(output_csv),
        "selection_was_frozen_before_test": True,
        "posthoc_pool_upper_bound_is_not_a_selector": True,
        "test_metrics_affected_selection": False,
    }
    _atomic_json(summary, output_summary)
    return summary
