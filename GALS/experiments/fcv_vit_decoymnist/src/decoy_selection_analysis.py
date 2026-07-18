"""Step-11 leakage-separated selection freeze and post-hoc reporting."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from decoy_campaign_preflight import source_tree_provenance, validate_launch_gate
from decoy_full_config import (
    canonical_config_sha256,
    candidate_epochs,
    enumerate_runs,
    sha256_file,
)
from decoy_manifest_provenance import atomic_json
from decoy_online_schema import (
    COMMON_COLUMNS,
    load_selector_namespace,
    namespace_columns,
    namespace_output_path,
    validate_namespace_row,
)
from decoy_online_study import assert_no_forbidden_persistence


class SelectionAnalysisError(RuntimeError):
    """Raised when selection or post-hoc analysis crosses a visibility boundary."""


SELECTOR_SCORE_COLUMNS = {
    "vanilla": "biased_validation_accuracy",
    "fcv": "harmonic_fcv_score",
    "oracle": "oracle_validation_accuracy",
    "posthoc": "test_accuracy",
}

FREEZE_TABLE = "frozen_selector_scores.csv"
FREEZE_SELECTIONS = "global_selections_pretest.csv"
FREEZE_SUMMARY = "selection_freeze_summary.json"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _expected_candidate_ids(config: Mapping[str, Any]) -> list[str]:
    return [
        run.candidate_id(epoch)
        for run in enumerate_runs(config)
        for epoch in candidate_epochs(config)
    ]


def select_best_candidate(frame: pd.DataFrame, score_column: str) -> pd.Series:
    """Select descending score, then ascending candidate ID for exact ties."""

    if score_column not in frame or "candidate_id" not in frame:
        raise SelectionAnalysisError(f"Missing selector column: {score_column}")
    values = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise SelectionAnalysisError(f"Non-finite selector score: {score_column}")
    ordered = frame.assign(_score=values).sort_values(
        ["_score", "candidate_id"], ascending=[False, True], kind="mergesort"
    )
    return ordered.iloc[0].drop(labels=["_score"])


def compute_gap_closure(
    vanilla_accuracy: float,
    fcv_accuracy: float,
    oracle_accuracy: float,
    *,
    epsilon: float = 1.0e-12,
) -> Dict[str, Any]:
    denominator = float(oracle_accuracy) - float(vanilla_accuracy)
    numerator = float(fcv_accuracy) - float(vanilla_accuracy)
    defined = denominator > float(epsilon)
    return {
        "vanilla_selected_test_accuracy": float(vanilla_accuracy),
        "fcv_selected_test_accuracy": float(fcv_accuracy),
        "oracle_selected_test_accuracy": float(oracle_accuracy),
        "numerator": numerator,
        "denominator": denominator,
        "defined": defined,
        "unclipped_gap_closed": float(numerator / denominator) if defined else None,
        "undefined_reason": None if defined else "oracle_not_above_vanilla",
    }


def _authenticate_complete_runs_without_reading_test(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    expected_preflight_receipt_sha256: str | None = None,
    expected_pretrained_backbone_sha256: str | None = None,
) -> Dict[str, Any]:
    """Verify all run receipts and hashes without parsing test/control values."""

    epochs = int(config["training"]["epochs"])
    config_hash = canonical_config_sha256(config)
    records: Dict[str, Any] = {}
    for run in enumerate_runs(config):
        summary_path = output_root / "run_summaries" / f"{run.run_id}.json"
        if not summary_path.is_file():
            raise SelectionAnalysisError(f"Missing completed run: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        valid = (
            summary.get("artifact_type") == "fcv_vit_decoymnist_online_run_summary"
            and summary.get("artifact_version") == 1
            and summary.get("status") == "complete"
            and summary.get("execution_mode") == "production"
            and summary.get("config_sha256") == config_hash
            and summary.get("run") == asdict(run)
            and int(summary.get("completed_candidate_count", -1)) == epochs
            and summary.get("test_metrics_used_for_training_or_selection") is False
            and (
                expected_preflight_receipt_sha256 is None
                or summary.get("preflight_receipt_sha256")
                == expected_preflight_receipt_sha256
            )
            and (
                expected_pretrained_backbone_sha256 is None
                or summary.get("pretrained_backbone_sha256")
                == expected_pretrained_backbone_sha256
            )
        )
        if not valid:
            raise SelectionAnalysisError(f"Stale completed-run receipt: {summary_path}")
        namespace_hashes: Dict[str, str] = {}
        for namespace in (
            "biased_validation",
            "fcv",
            "controls",
            "oracle_analysis_only",
            "test_analysis_only",
        ):
            path = namespace_output_path(output_root, namespace, run.run_id)
            artifact = summary.get("namespace_artifacts", {}).get(namespace, {})
            if (
                not path.is_file()
                or artifact.get("path") != str(path)
                or artifact.get("sha256") != sha256_file(path)
                or int(artifact.get("row_count", -1)) != epochs
            ):
                raise SelectionAnalysisError(
                    f"Stale {namespace} artifact for {run.run_id}."
                )
            namespace_hashes[namespace] = str(artifact["sha256"])
        records[run.run_id] = {
            "run_summary_path": str(summary_path),
            "run_summary_sha256": sha256_file(summary_path),
            "namespace_sha256": namespace_hashes,
        }
    return records


def _assert_common_identity(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    if left["candidate_id"].astype(str).tolist() != right["candidate_id"].astype(str).tolist():
        raise SelectionAnalysisError(f"{name} candidate ordering differs.")
    for column in COMMON_COLUMNS:
        a = left[column].to_numpy()
        b = right[column].to_numpy()
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            matches = np.allclose(a.astype(float), b.astype(float), rtol=0.0, atol=1.0e-15)
        else:
            matches = np.array_equal(a.astype(str), b.astype(str))
        if not matches:
            raise SelectionAnalysisError(f"{name} changed common field {column}.")


def _load_selector_pool(
    config: Mapping[str, Any], output_root: Path
) -> pd.DataFrame:
    biased_frames = []
    fcv_frames = []
    oracle_frames = []
    for run in enumerate_runs(config):
        biased_frames.append(
            load_selector_namespace(
                output_root, "vanilla", "biased_validation", run.run_id
            )
        )
        fcv_frames.append(
            load_selector_namespace(output_root, "fcv", "fcv", run.run_id)
        )
        oracle_frames.append(
            load_selector_namespace(
                output_root, "oracle", "oracle_analysis_only", run.run_id
            )
        )
    biased = pd.concat(biased_frames, ignore_index=True).sort_values("candidate_id")
    fcv = pd.concat(fcv_frames, ignore_index=True).sort_values("candidate_id")
    oracle = pd.concat(oracle_frames, ignore_index=True).sort_values("candidate_id")
    for frame in (biased, fcv, oracle):
        frame.reset_index(drop=True, inplace=True)
    _assert_common_identity(biased, fcv, "FCV")
    _assert_common_identity(biased, oracle, "Oracle")
    expected = sorted(_expected_candidate_ids(config))
    observed = biased["candidate_id"].astype(str).tolist()
    if observed != expected or len(observed) != int(
        config["candidate_pool"]["expected_candidate_states"]
    ):
        raise SelectionAnalysisError("The selector pool is not the exact 1,080 candidates.")
    matrix = biased[COMMON_COLUMNS + ["biased_validation_accuracy"]].copy()
    matrix["fcv_counterfactual_accuracy"] = pd.to_numeric(
        fcv["fcv_counterfactual_accuracy"], errors="raise"
    )
    matrix["harmonic_fcv_score"] = pd.to_numeric(
        fcv["harmonic_fcv_score"], errors="raise"
    )
    matrix["oracle_validation_accuracy"] = pd.to_numeric(
        oracle["oracle_validation_accuracy"], errors="raise"
    )
    if any(column.startswith("test_") for column in matrix.columns):
        raise SelectionAnalysisError("Test values entered the selection freeze.")
    return matrix.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def _selection_rows(matrix: pd.DataFrame, selectors: Sequence[str]) -> pd.DataFrame:
    rows = []
    for selector in selectors:
        score_column = SELECTOR_SCORE_COLUMNS[selector]
        selected = select_best_candidate(matrix, score_column)
        rows.append(
            {
                "selector": selector,
                "score_column": score_column,
                "selected_candidate_id": str(selected["candidate_id"]),
                "selected_score": float(selected[score_column]),
                "run_index": int(selected["run_index"]),
                "epoch": int(selected["epoch"]),
                "seed": int(selected["seed"]),
                "learning_rate": float(selected["learning_rate"]),
                "weight_decay": float(selected["weight_decay"]),
                "crop_scale_min": float(selected["crop_scale_min"]),
            }
        )
    return pd.DataFrame(rows)


def freeze_selector_matrix(
    config: Mapping[str, Any], output_root: str | Path
) -> Dict[str, Any]:
    """Freeze deployable/Oracle selections without loading official-test values."""

    root = Path(output_root).expanduser().resolve()
    gate = validate_launch_gate(config, root / "preflight" / "launch_gate.json")
    assert_no_forbidden_persistence(config, root)
    result_root = root / "selection_results"
    summary_path = result_root / FREEZE_SUMMARY
    if summary_path.exists():
        return load_frozen_selector_matrix(config, root)[2]
    partial = [
        result_root / FREEZE_TABLE,
        result_root / FREEZE_SELECTIONS,
    ]
    if any(path.exists() for path in partial):
        raise SelectionAnalysisError(
            "Partial selection freeze exists without its authenticated summary."
        )

    run_receipts = _authenticate_complete_runs_without_reading_test(
        config,
        root,
        expected_preflight_receipt_sha256=gate["preflight_receipt_sha256"],
        expected_pretrained_backbone_sha256=gate["pretrained_backbone_sha256"],
    )
    matrix = _load_selector_pool(config, root)
    selections = _selection_rows(matrix, ("vanilla", "fcv", "oracle"))
    matrix_path = result_root / FREEZE_TABLE
    selections_path = result_root / FREEZE_SELECTIONS
    _atomic_csv(matrix, matrix_path)
    _atomic_csv(selections, selections_path)
    source = source_tree_provenance()
    summary = {
        "artifact_type": "fcv_vit_decoymnist_selection_freeze",
        "artifact_version": 1,
        "status": "complete",
        "config_sha256": canonical_config_sha256(config),
        "source_tree_sha256": source["source_tree_sha256"],
        "launch_gate_path": gate["artifact_path"],
        "launch_gate_sha256": gate["artifact_sha256"],
        "candidate_count": int(len(matrix)),
        "run_count": len(run_receipts),
        "selector_matrix_path": str(matrix_path),
        "selector_matrix_sha256": sha256_file(matrix_path),
        "selection_table_path": str(selections_path),
        "selection_table_sha256": sha256_file(selections_path),
        "run_receipts": run_receipts,
        "test_namespace_content_accessed": False,
        "test_metrics_used_for_selection": False,
        "tie_break": "candidate_id_ascending",
    }
    atomic_json(summary, summary_path)
    assert_no_forbidden_persistence(config, root)
    return summary


def load_frozen_selector_matrix(
    config: Mapping[str, Any], output_root: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    root = Path(output_root).expanduser().resolve()
    result_root = root / "selection_results"
    summary_path = result_root / FREEZE_SUMMARY
    if not summary_path.is_file():
        raise SelectionAnalysisError("The selector matrix has not been frozen.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    matrix_path = result_root / FREEZE_TABLE
    selections_path = result_root / FREEZE_SELECTIONS
    source = source_tree_provenance()
    valid = (
        summary.get("artifact_type") == "fcv_vit_decoymnist_selection_freeze"
        and summary.get("artifact_version") == 1
        and summary.get("status") == "complete"
        and summary.get("config_sha256") == canonical_config_sha256(config)
        and summary.get("source_tree_sha256") == source["source_tree_sha256"]
        and summary.get("selector_matrix_path") == str(matrix_path)
        and summary.get("selector_matrix_sha256") == sha256_file(matrix_path)
        and summary.get("selection_table_path") == str(selections_path)
        and summary.get("selection_table_sha256") == sha256_file(selections_path)
        and summary.get("test_namespace_content_accessed") is False
        and summary.get("test_metrics_used_for_selection") is False
    )
    if not valid:
        raise SelectionAnalysisError("The selection freeze is stale.")
    matrix = pd.read_csv(matrix_path)
    selections = pd.read_csv(selections_path)
    if len(matrix) != int(config["candidate_pool"]["expected_candidate_states"]):
        raise SelectionAnalysisError("Frozen selector cardinality changed.")
    if any(column.startswith("test_") for column in matrix.columns):
        raise SelectionAnalysisError("Frozen selector matrix contains test values.")
    recomputed = _selection_rows(matrix, ("vanilla", "fcv", "oracle"))
    if not recomputed.equals(selections):
        raise SelectionAnalysisError("Frozen selections do not reproduce exactly.")
    for run_id, record in summary.get("run_receipts", {}).items():
        path = Path(str(record.get("run_summary_path", "")))
        if not path.is_file() or sha256_file(path) != record.get("run_summary_sha256"):
            raise SelectionAnalysisError(f"Run receipt changed after freeze: {run_id}")
    return matrix, selections, summary


def _load_posthoc_namespace(
    config: Mapping[str, Any], output_root: Path, namespace: str
) -> pd.DataFrame:
    frames = []
    for run in enumerate_runs(config):
        if namespace == "test_analysis_only":
            frame = load_selector_namespace(
                output_root, "posthoc", namespace, run.run_id
            )
        else:
            path = namespace_output_path(output_root, namespace, run.run_id)
            frame = pd.read_csv(path)
            if frame.columns.tolist() != namespace_columns(namespace):
                raise SelectionAnalysisError(f"Stale {namespace} header: {path}")
            for row in frame.to_dict("records"):
                validate_namespace_row(namespace, row)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values("candidate_id")
    return result.reset_index(drop=True)


def _correlation(left: np.ndarray, right: np.ndarray, kind: str) -> float | None:
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    value = spearmanr(left, right).statistic if kind == "spearman" else kendalltau(
        left, right, variant="b"
    ).statistic
    return float(value) if np.isfinite(value) else None


def rank_quality_rows(
    matrix: pd.DataFrame, selectors: Sequence[str], top_k_values: Sequence[int]
) -> pd.DataFrame:
    truth = matrix.sort_values(
        ["test_accuracy", "candidate_id"], ascending=[False, True], kind="mergesort"
    )
    truth_values = pd.to_numeric(matrix["test_accuracy"], errors="raise").to_numpy(float)
    rows = []
    for selector in selectors:
        score_column = SELECTOR_SCORE_COLUMNS[selector]
        scores = pd.to_numeric(matrix[score_column], errors="raise").to_numpy(float)
        ranked = matrix.sort_values(
            [score_column, "candidate_id"], ascending=[False, True], kind="mergesort"
        )
        spearman = _correlation(scores, truth_values, "spearman")
        kendall = _correlation(scores, truth_values, "kendall")
        for value in top_k_values:
            k = min(int(value), len(matrix))
            overlap = len(
                set(ranked.head(k)["candidate_id"].astype(str)).intersection(
                    set(truth.head(k)["candidate_id"].astype(str))
                )
            )
            rows.append(
                {
                    "selector": selector,
                    "score_column": score_column,
                    "spearman_vs_test_accuracy": spearman,
                    "kendall_b_vs_test_accuracy": kendall,
                    "top_k": k,
                    "top_k_overlap_count": overlap,
                    "top_k_recall": float(overlap / k),
                    "posthoc_analysis_only": True,
                }
            )
    return pd.DataFrame(rows)


def _selected_test_outcomes(
    matrix: pd.DataFrame, frozen: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for selected in frozen.itertuples(index=False):
        candidate = matrix.loc[
            matrix["candidate_id"].astype(str) == str(selected.selected_candidate_id)
        ]
        if len(candidate) != 1:
            raise SelectionAnalysisError("Frozen selected candidate disappeared.")
        row = candidate.iloc[0]
        rows.append(
            {
                "selector": str(selected.selector),
                "selected_candidate_id": str(row["candidate_id"]),
                "selector_score": float(selected.selected_score),
                "run_index": int(row["run_index"]),
                "epoch": int(row["epoch"]),
                "seed": int(row["seed"]),
                "learning_rate": float(row["learning_rate"]),
                "weight_decay": float(row["weight_decay"]),
                "crop_scale_min": float(row["crop_scale_min"]),
                **{
                    column: float(row[column])
                    for column in matrix.columns
                    if column.startswith("test_")
                },
            }
        )
    ceiling = select_best_candidate(matrix, "test_accuracy")
    rows.append(
        {
            "selector": "posthoc",
            "selected_candidate_id": str(ceiling["candidate_id"]),
            "selector_score": float(ceiling["test_accuracy"]),
            "run_index": int(ceiling["run_index"]),
            "epoch": int(ceiling["epoch"]),
            "seed": int(ceiling["seed"]),
            "learning_rate": float(ceiling["learning_rate"]),
            "weight_decay": float(ceiling["weight_decay"]),
            "crop_scale_min": float(ceiling["crop_scale_min"]),
            **{
                column: float(ceiling[column])
                for column in matrix.columns
                if column.startswith("test_")
            },
        }
    )
    return pd.DataFrame(rows)


def _seed_reports(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed, group in matrix.groupby("seed", sort=True):
        for selector in ("vanilla", "fcv", "oracle", "posthoc"):
            selected = select_best_candidate(group, SELECTOR_SCORE_COLUMNS[selector])
            rows.append(
                {
                    "seed": int(seed),
                    "selector": selector,
                    "selected_candidate_id": str(selected["candidate_id"]),
                    "selected_score": float(selected[SELECTOR_SCORE_COLUMNS[selector]]),
                    "test_accuracy": float(selected["test_accuracy"]),
                    "test_balanced_class_accuracy": float(
                        selected["test_balanced_class_accuracy"]
                    ),
                    "test_worst_class_accuracy": float(
                        selected["test_worst_class_accuracy"]
                    ),
                }
            )
    selections = pd.DataFrame(rows)
    summaries = []
    for selector, group in selections.groupby("selector", sort=True):
        summaries.append(
            {
                "selector": selector,
                "seed_count": int(len(group)),
                "selected_test_accuracy_mean": float(group["test_accuracy"].mean()),
                "selected_test_accuracy_std_sample": float(group["test_accuracy"].std(ddof=1)),
                "selected_test_balanced_class_accuracy_mean": float(
                    group["test_balanced_class_accuracy"].mean()
                ),
                "selected_test_worst_class_accuracy_mean": float(
                    group["test_worst_class_accuracy"].mean()
                ),
            }
        )
    return selections, pd.DataFrame(summaries)


def _crop_report(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for crop, group in matrix.groupby("crop_scale_min", sort=False):
        best = select_best_candidate(group, "test_accuracy")
        values = pd.to_numeric(group["test_accuracy"], errors="raise")
        rows.append(
            {
                "crop_scale_min": float(crop),
                "candidate_count": int(len(group)),
                "test_accuracy_mean": float(values.mean()),
                "test_accuracy_std_sample": float(values.std(ddof=1)),
                "test_accuracy_median": float(values.median()),
                "test_accuracy_min": float(values.min()),
                "test_accuracy_max": float(values.max()),
                "best_candidate_id_posthoc": str(best["candidate_id"]),
            }
        )
    return pd.DataFrame(rows).sort_values("crop_scale_min", ascending=False)


def _control_report(matrix: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    _assert_common_identity(matrix, controls, "Controls")
    rows = []
    primary = pd.to_numeric(matrix["fcv_counterfactual_accuracy"], errors="raise")
    test = pd.to_numeric(matrix["test_accuracy"], errors="raise")
    for control in (
        "same_context",
        "random_mask",
        "shuffled_teacher_mask",
        "evidence_swap",
        "exact_synthetic_mask_analysis_only",
    ):
        column = f"control_{control}_counterfactual_accuracy"
        values = pd.to_numeric(controls[column], errors="coerce")
        finite = np.isfinite(values.to_numpy(float))
        rows.append(
            {
                "control": control,
                "valid_candidate_count": int(finite.sum()),
                "counterfactual_accuracy_mean": float(values[finite].mean()) if finite.any() else None,
                "primary_minus_control_mean": float((primary[finite] - values[finite]).mean()) if finite.any() else None,
                "spearman_vs_test_accuracy": _correlation(
                    values[finite].to_numpy(float), test[finite].to_numpy(float), "spearman"
                ) if finite.sum() >= 2 else None,
                "warning_only": True,
                "used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def analyze_posthoc_results(
    config: Mapping[str, Any], output_root: str | Path
) -> Dict[str, Any]:
    """Attach analysis-only test values after selection and generate all reports."""

    root = Path(output_root).expanduser().resolve()
    frozen_matrix, frozen_selections, freeze_summary = load_frozen_selector_matrix(
        config, root
    )
    current_receipts = _authenticate_complete_runs_without_reading_test(config, root)
    if current_receipts != freeze_summary.get("run_receipts"):
        raise SelectionAnalysisError(
            "Online run artifacts changed after the selection freeze."
        )
    result_root = root / "selection_results"
    final_summary_path = result_root / "final_report_summary.json"
    if final_summary_path.exists():
        existing = json.loads(final_summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("config_sha256") == canonical_config_sha256(config)
            and existing.get("selection_freeze_sha256")
            == sha256_file(result_root / FREEZE_SUMMARY)
            and existing.get("status") == "complete"
        ):
            return existing
        raise SelectionAnalysisError("A stale final report already exists.")

    test = _load_posthoc_namespace(config, root, "test_analysis_only")
    controls = _load_posthoc_namespace(config, root, "controls")
    _assert_common_identity(frozen_matrix, test, "Test")
    matrix = frozen_matrix.copy()
    for column in test.columns:
        if column.startswith("test_"):
            matrix[column] = test[column].to_numpy()
    outcomes = _selected_test_outcomes(matrix, frozen_selections)
    accuracy_by_selector = dict(zip(outcomes["selector"], outcomes["test_accuracy"]))
    gap = compute_gap_closure(
        accuracy_by_selector["vanilla"],
        accuracy_by_selector["fcv"],
        accuracy_by_selector["oracle"],
        epsilon=float(config["evaluation"]["gap_closure"]["denominator_epsilon"]),
    )
    test_values = pd.to_numeric(matrix["test_accuracy"], errors="raise")
    headroom = {
        "candidate_count": int(len(matrix)),
        "test_accuracy_min": float(test_values.min()),
        "test_accuracy_max": float(test_values.max()),
        "test_accuracy_range": float(test_values.max() - test_values.min()),
        "vanilla_selected_test_accuracy": float(accuracy_by_selector["vanilla"]),
        "oracle_selected_test_accuracy": float(accuracy_by_selector["oracle"]),
        "posthoc_maximum_test_accuracy": float(accuracy_by_selector["posthoc"]),
        "posthoc_gain_over_vanilla": float(
            accuracy_by_selector["posthoc"] - accuracy_by_selector["vanilla"]
        ),
        "contains_candidate_above_chance": bool(test_values.max() > 0.10),
        "contains_candidate_with_majority_accuracy": bool(test_values.max() > 0.50),
        "robustness_thresholds_are_descriptive_not_selectors": True,
    }
    ranks = rank_quality_rows(
        matrix,
        ("vanilla", "fcv", "oracle"),
        config["evaluation"]["rank_analysis"]["top_k_values"],
    )
    seed_selections, seed_summary = _seed_reports(matrix)
    crop = _crop_report(matrix)
    control = _control_report(matrix, controls)

    artifacts = {
        "candidate_matrix_posthoc": matrix,
        "selector_test_outcomes": outcomes,
        "crop_regime_summary": crop,
        "rank_analysis": ranks,
        "seed_stratified_selections": seed_selections,
        "seed_stratified_summary": seed_summary,
        "control_diagnostics": control,
    }
    artifact_records: Dict[str, Any] = {}
    for name, frame in artifacts.items():
        path = result_root / f"{name}.csv"
        _atomic_csv(frame, path)
        artifact_records[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": int(len(frame)),
        }
    gap_path = result_root / "gap_closure.json"
    headroom_path = result_root / "candidate_pool_headroom.json"
    atomic_json(gap, gap_path)
    atomic_json(headroom, headroom_path)
    artifact_records["gap_closure"] = {"path": str(gap_path), "sha256": sha256_file(gap_path)}
    artifact_records["candidate_pool_headroom"] = {
        "path": str(headroom_path),
        "sha256": sha256_file(headroom_path),
    }
    source = source_tree_provenance()
    summary = {
        "artifact_type": "fcv_vit_decoymnist_final_report",
        "artifact_version": 1,
        "status": "complete",
        "config_sha256": canonical_config_sha256(config),
        "source_tree_sha256": source["source_tree_sha256"],
        "selection_freeze_path": str(result_root / FREEZE_SUMMARY),
        "selection_freeze_sha256": sha256_file(result_root / FREEZE_SUMMARY),
        "frozen_selector_matrix_sha256": freeze_summary["selector_matrix_sha256"],
        "candidate_count": int(len(matrix)),
        "selector_test_accuracy": {
            key: float(value) for key, value in accuracy_by_selector.items()
        },
        "gap_closure": gap,
        "candidate_pool_headroom": headroom,
        "artifacts": artifact_records,
        "test_metrics_attached_after_selection_freeze": True,
        "test_metrics_affected_selection": False,
        "controls_used_for_selection": False,
    }
    atomic_json(summary, final_summary_path)
    assert_no_forbidden_persistence(config, root)
    return summary
