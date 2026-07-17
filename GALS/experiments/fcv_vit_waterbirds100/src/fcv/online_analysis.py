"""Leakage-separated aggregation for the 540-candidate online FCV study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from .campaign_provenance import (
    load_campaign_provenance_receipt,
    verify_non_test_campaign_inputs,
)
from .candidate_training import (
    SweepRun,
    _load_trusted_checkpoint,
    candidate_training_fingerprint,
    enumerate_sweep_runs,
    pretrained_provenance_path,
)
from .controls import CONTROL_NAMES, recompute_control_metrics_from_frame
from .fcv_scoring import validate_fcv_summary_against_frame
from .online_schema import (
    ONLINE_TEST_COLUMNS,
    ONLINE_VALIDATION_COLUMNS,
    RETAINED_SELECTOR_SPECS,
)
from .selectors import (
    prepare_oracle_validation_source,
    validate_oracle_summary_against_frame,
)
from .storage import allocated_bytes, online_storage_breakdown


class OnlineAnalysisError(RuntimeError):
    """Raised when an online campaign cannot be frozen or audited exactly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _numeric_summary(frame: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise OnlineAnalysisError(f"Non-finite campaign diagnostic: {column}")
        result[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise OnlineAnalysisError(f"Expected JSON mapping: {path}")
    return value


def _concurrent_storage_projection(
    *,
    observed_committed_growth_bytes: int,
    transient_checkpoint_bytes_per_writer: int,
    writers: int,
    reserve_bytes: int,
) -> Dict[str, Any]:
    """Project eight-writer peak growth without double-applying a safety factor."""

    if (
        observed_committed_growth_bytes < 0
        or transient_checkpoint_bytes_per_writer <= 0
        or writers <= 0
        or reserve_bytes <= 0
    ):
        raise OnlineAnalysisError("Invalid concurrent-storage projection inputs.")
    # The measured committed growth already includes one writer's resume,
    # retained winner, plans, indexes, and evidence. Scale that observation
    # once to all writers, then separately add one checkpoint per writer for
    # the intentional crash-safe winner-replacement staging window. The 2x
    # safety factor belongs only to the full durable-campaign projection.
    projected_committed = observed_committed_growth_bytes * writers
    projected_transient_staging = transient_checkpoint_bytes_per_writer * writers
    projected_peak = projected_committed + projected_transient_staging
    if projected_peak > reserve_bytes:
        raise OnlineAnalysisError(
            "Measured online-epoch storage growth exceeds the configured "
            "concurrency reserve: "
            f"projected={projected_peak} bytes, reserve={reserve_bytes} bytes."
        )
    return {
        "projection_formula": (
            "observed_committed_growth_x_writers_plus_"
            "one_transient_checkpoint_per_writer"
        ),
        "projected_committed_growth_bytes": projected_committed,
        "projected_transient_staging_bytes": projected_transient_staging,
        "projected_concurrent_growth_bytes": projected_peak,
        "configured_concurrent_growth_reserve_bytes": reserve_bytes,
        "within_configured_reserve": True,
    }


def _storage_high_water_projection(
    config: Mapping[str, Any],
    output_root: Path,
    run_dir: Path,
    baseline_path: Path,
    candidate_id: str,
    observed_candidate_count: int,
) -> Dict[str, Any]:
    """Project both concurrent growth and all durable 540-candidate outputs."""

    baseline = _load_json(baseline_path)
    if (
        baseline.get("output_root") != str(output_root)
        or int(baseline.get("allocated_bytes", -1)) < 0
    ):
        raise OnlineAnalysisError("Online smoke storage baseline is incompatible.")
    observed_now = allocated_bytes(output_root)
    observed_growth = max(0, observed_now - int(baseline["allocated_bytes"]))
    baseline_breakdown = baseline.get("storage_breakdown_bytes")
    if not isinstance(baseline_breakdown, Mapping):
        raise OnlineAnalysisError("Online smoke baseline lacks a storage breakdown.")
    current_breakdown = online_storage_breakdown(output_root)

    def matching_allocated(root: Path) -> int:
        total = 0
        if root.is_dir():
            for path in root.glob(f"{candidate_id}*"):
                if path.is_file():
                    total += int(path.stat().st_blocks) * 512
                elif path.is_dir():
                    total += allocated_bytes(path)
        return total

    evidence_by_family = {
        "fcv": matching_allocated(output_root / "online_scores" / "fcv"),
        "controls": matching_allocated(
            output_root / "online_scores" / "controls"
        ),
        "oracle": matching_allocated(output_root / "online_scores" / "oracle"),
        "test": matching_allocated(output_root / "online_test_analysis_only"),
    }
    candidate_evidence_bytes = sum(evidence_by_family.values())
    if candidate_evidence_bytes <= 0:
        raise OnlineAnalysisError("Smoke produced no measurable candidate evidence.")
    retained_sizes = [
        path.stat().st_size
        for path in (run_dir / "retained_checkpoints").glob("*.pt")
        if path.is_file()
    ]
    # Durable epoch growth already contains the first retained winner. One
    # additional checkpoint per writer covers crash-safe replacement staging.
    transient_checkpoint_allowance = max(retained_sizes, default=0)
    resume_path = run_dir / "resume_state.pt"
    resume_bytes = (
        int(resume_path.stat().st_blocks) * 512 if resume_path.is_file() else 0
    )
    plan_bytes = allocated_bytes(run_dir / "plans")
    if transient_checkpoint_allowance <= 0 or resume_bytes <= 0 or plan_bytes <= 0:
        raise OnlineAnalysisError(
            "Smoke did not materialize the bounded checkpoint, resume, and plan state."
        )
    writers = int(config["storage"]["max_concurrent_streaming_runs"])
    configured_reserve = int(
        float(config["storage"]["worst_case_concurrent_growth_gib"])
        * (1024 ** 3)
    )
    concurrent = _concurrent_storage_projection(
        observed_committed_growth_bytes=observed_growth,
        transient_checkpoint_bytes_per_writer=transient_checkpoint_allowance,
        writers=writers,
        reserve_bytes=configured_reserve,
    )

    expected_candidates = int(
        config["candidate_pool"]["expected_candidate_checkpoints"]
    )
    expected_runs = int(config["candidate_pool"]["expected_training_runs"])
    max_retained = int(
        config["candidate_pool"]["max_retained_checkpoints_per_run"]
    )
    if observed_candidate_count <= 0:
        raise OnlineAnalysisError("Observed candidate count must be positive.")
    per_candidate_online_run_other = int(
        np.ceil(
            float(current_breakdown["online_run_other"])
            / float(observed_candidate_count)
        )
    )
    fixed_bytes = max(
        0,
        observed_now - int(current_breakdown["categorized_total"]),
    )
    full_campaign_safety = float(
        config["storage"]["full_campaign_projection_safety_factor"]
    )
    projected_evidence = candidate_evidence_bytes * expected_candidates
    projected_run_indexes = per_candidate_online_run_other * expected_candidates
    projected_retained = (
        transient_checkpoint_allowance * max_retained * expected_runs
    )
    projected_transient = transient_checkpoint_allowance * writers
    projected_resumes = resume_bytes * writers
    projected_plans = plan_bytes * expected_runs
    projected_variable = sum(
        (
            projected_evidence,
            projected_run_indexes,
            projected_retained,
            projected_transient,
            projected_resumes,
            projected_plans,
        )
    )
    projected_full_campaign = int(
        fixed_bytes + full_campaign_safety * projected_variable
    )
    launch_guard_bytes = int(
        float(config["storage"]["launch_guard_gib"]) * (1024 ** 3)
    )
    if projected_full_campaign > launch_guard_bytes:
        raise OnlineAnalysisError(
            "Measured durable artifacts project beyond the full-campaign launch "
            "guard: "
            f"projected={projected_full_campaign} bytes, "
            f"guard={launch_guard_bytes} bytes."
        )
    return {
        "baseline_path": str(baseline_path),
        "baseline_sha256": _sha256_file(baseline_path),
        "baseline_allocated_bytes": int(baseline["allocated_bytes"]),
        "observed_allocated_bytes": observed_now,
        "observed_single_epoch_growth_bytes": observed_growth,
        "transient_checkpoint_allowance_bytes_per_writer": (
            transient_checkpoint_allowance
        ),
        "configured_concurrent_writers": writers,
        **concurrent,
        "candidate_id_measured": candidate_id,
        "candidate_evidence_bytes_by_family": evidence_by_family,
        "candidate_evidence_bytes": candidate_evidence_bytes,
        "per_candidate_online_run_other_bytes": per_candidate_online_run_other,
        "retained_checkpoint_bytes_per_file": transient_checkpoint_allowance,
        "resume_state_bytes_per_active_writer": resume_bytes,
        "intervention_plan_bytes_per_run": plan_bytes,
        "expected_candidates": expected_candidates,
        "expected_runs": expected_runs,
        "full_campaign_safety_factor": full_campaign_safety,
        "projected_evidence_bytes": projected_evidence,
        "projected_online_run_index_bytes": projected_run_indexes,
        "projected_retained_checkpoint_bytes": projected_retained,
        "projected_transient_checkpoint_bytes": projected_transient,
        "projected_resume_state_bytes": projected_resumes,
        "projected_intervention_plan_bytes": projected_plans,
        "fixed_noncampaign_bytes": fixed_bytes,
        "projected_full_campaign_bytes": projected_full_campaign,
        "launch_guard_bytes": launch_guard_bytes,
        "within_full_campaign_launch_guard": True,
    }


def _runtime_projection(
    config: Mapping[str, Any], test_prefix: pd.DataFrame
) -> Dict[str, Any]:
    """Project measured full online-epoch time against the seven-day run limit."""

    values = pd.to_numeric(
        test_prefix["epoch_online_total_seconds"], errors="raise"
    ).to_numpy(float)
    if len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise OnlineAnalysisError("Smoke has no valid full-epoch timing measurements.")
    epochs = int(config["training"]["epochs"])
    safety = float(config["cluster"]["runtime_projection_safety_factor"])
    limit_seconds = float(config["cluster"]["online_run_time_limit_hours"]) * 3600.0
    projected = float(values.max()) * epochs * safety
    if projected > limit_seconds:
        raise OnlineAnalysisError(
            "Measured online epoch time projects beyond the seven-day run limit: "
            f"projected={projected:.1f}s, limit={limit_seconds:.1f}s."
        )
    return {
        "observed_epoch_count": int(len(values)),
        "observed_epoch_seconds": [float(value) for value in values],
        "observed_total_seconds": float(values.sum()),
        "observed_max_epoch_seconds": float(values.max()),
        "training_epochs_per_run": epochs,
        "safety_factor": safety,
        "projected_run_seconds": projected,
        "projected_run_hours": projected / 3600.0,
        "run_time_limit_seconds": limit_seconds,
        "run_time_limit_hours": limit_seconds / 3600.0,
        "within_run_time_limit": True,
    }


def _selector_specs(config: Mapping[str, Any]) -> List[Dict[str, str]]:
    primary = str(config["fcv"]["primary_selector"]["name"])
    specs: List[Dict[str, str]] = [
        {
            "name": "biased_validation_accuracy",
            "family": "biased_validation",
            "column": "biased_val_accuracy",
            "direction": "maximize",
            "availability": "unprivileged_train_holdout",
        },
        {
            "name": "biased_validation_loss",
            "family": "biased_validation",
            "column": "biased_val_loss",
            "direction": "minimize",
            "availability": "unprivileged_train_holdout",
        },
        {
            "name": "opposite_context_counterfactual_accuracy",
            "family": "fcv",
            "column": "fcv_counterfactual_accuracy",
            "direction": "maximize",
            "availability": "unprivileged_train_holdout",
        },
        {
            "name": "opposite_context_true_class_probability",
            "family": "fcv_stability",
            "column": "fcv_true_class_probability",
            "direction": "maximize",
            "availability": "unprivileged_train_holdout",
        },
        {
            "name": "opposite_context_probability_retention_ratio",
            "family": "fcv_stability",
            "column": "fcv_probability_retention_ratio",
            "direction": "maximize",
            "availability": "unprivileged_train_holdout",
        },
        {
            "name": primary,
            "family": "fcv_primary",
            "column": "primary_selector_score",
            "direction": "maximize",
            "availability": "unprivileged_train_holdout",
        },
    ]
    for value in config["fcv"]["selector_analysis"]["fcv_accuracy_lambdas"]:
        slug = str(float(value)).replace(".", "p")
        specs.append(
            {
                "name": f"fcv_accuracy_lambda_{slug}",
                "family": "fcv_lambda_ablation",
                "column": f"fcv_accuracy_lambda_{slug}_score",
                "direction": "maximize",
                "availability": "unprivileged_train_holdout",
            }
        )
    specs.extend(
        [
            {
                "name": "control_normalized_fcv",
                "family": "fcv_control_normalized",
                "column": "control_normalized_fcv_score",
                "direction": "maximize",
                "availability": "unprivileged_train_holdout",
            },
            {
                "name": "same_context_counterfactual_accuracy",
                "family": "fcv_control",
                "column": "same_context_counterfactual_accuracy",
                "direction": "maximize",
                "availability": "unprivileged_train_holdout_control",
            },
            {
                "name": "random_mask_counterfactual_accuracy",
                "family": "fcv_control",
                "column": "random_mask_counterfactual_accuracy",
                "direction": "maximize",
                "availability": "unprivileged_train_holdout_control",
            },
            {
                "name": "shuffled_mask_counterfactual_accuracy",
                "family": "fcv_control",
                "column": "shuffled_mask_counterfactual_accuracy",
                "direction": "maximize",
                "availability": "unprivileged_train_holdout_control",
            },
            {
                "name": "evidence_swap_counterfactual_accuracy",
                "family": "fcv_control",
                "column": "evidence_swap_counterfactual_accuracy",
                "direction": "maximize",
                "availability": "unprivileged_train_holdout_control",
            },
            {
                "name": "oracle_validation_worst_group_accuracy",
                "family": "oracle",
                "column": "oracle_validation_worst_group_accuracy",
                "direction": "maximize",
                "availability": "privileged_analysis_only",
            },
            {
                "name": "oracle_validation_balanced_group_accuracy",
                "family": "oracle",
                "column": "oracle_validation_balanced_group_accuracy",
                "direction": "maximize",
                "availability": "privileged_analysis_only",
            },
        ]
    )
    return specs


def _select(matrix: pd.DataFrame, spec: Mapping[str, str]) -> Dict[str, Any]:
    values = pd.to_numeric(matrix[spec["column"]], errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise OnlineAnalysisError(f"Selector has non-finite values: {spec['name']}")
    best = values.max() if spec["direction"] == "maximize" else values.min()
    tied = matrix.loc[values == best].sort_values("candidate_id", kind="stable")
    selected = tied.iloc[0]
    return {
        "selector_name": spec["name"],
        "selector_family": spec["family"],
        "availability": spec["availability"],
        "direction": spec["direction"],
        "score_column": spec["column"],
        "selector_score": float(best),
        "selected_checkpoint_id": str(selected["candidate_id"]),
        "run_index": int(selected["run_index"]),
        "epoch": int(selected["epoch"]),
        "seed": int(selected["seed"]),
        "learning_rate": float(selected["learning_rate"]),
        "weight_decay": float(selected["weight_decay"]),
        "checkpoint_sha256": str(selected["checkpoint_sha256"]),
        "exact_tie_count": int(len(tied)),
        "tie_break_rule": "candidate_id_ascending",
    }


def _load_validation_runs(
    config: Mapping[str, Any],
    output_root: Path,
    usecols: Sequence[str],
    campaign: Mapping[str, Any],
) -> pd.DataFrame:
    """Read only explicitly requested columns; callers cannot accidentally see test."""

    frames = []
    expected_epochs = list(range(1, int(config["training"]["epochs"]) + 1))
    fingerprint = candidate_training_fingerprint(config)
    for run in enumerate_sweep_runs(config):
        run_dir = output_root / "online_runs" / run.run_id
        metrics_path = run_dir / "validation_metrics.csv"
        summary_path = run_dir / "run_summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            raise OnlineAnalysisError(f"Missing complete online run: {run.run_id}")
        summary = _load_json(summary_path)
        if (
            summary.get("artifact_type") != "fcv_vit_online_run_summary"
            or summary.get("status") != "complete"
            or summary.get("run") != {
                "run_index": run.run_index,
                "learning_rate": run.learning_rate,
                "weight_decay": run.weight_decay,
                "seed": run.seed,
            }
            or summary.get("training_fingerprint") != fingerprint
            or summary.get("software_versions")
            != campaign["bindings"]["software_versions"]
            or summary.get("software_fingerprint")
            != campaign["bindings"]["software_fingerprint"]
            or summary.get("source_tree_sha256")
            != campaign["bindings"]["source_tree"]["source_tree_sha256"]
            or summary.get("campaign_provenance_path")
            != campaign["artifact_path"]
            or summary.get("campaign_provenance_sha256")
            != campaign["artifact_sha256"]
            or summary.get("campaign_bindings_sha256")
            != campaign["bindings_sha256"]
            or summary.get("pretrained_provenance_path")
            != campaign["bindings"]["pretrained"]["path"]
            or summary.get("pretrained_provenance_sha256")
            != campaign["bindings"]["pretrained"]["sha256"]
            or summary.get("pretrained_backbone_sha256")
            != campaign["bindings"]["pretrained"]["backbone_sha256"]
            or summary.get("initial_model_state_sha256")
            != campaign["bindings"]["initialization"][
                "initial_model_state_sha256_by_seed"
            ].get(str(run.seed))
            or summary.get("manifest_sha256")
            != {
                "candidate_train": campaign["bindings"]["manifests"][
                    "candidate_train"
                ]["sha256"],
                "biased_validation": campaign["bindings"]["manifests"][
                    "biased_validation"
                ]["sha256"],
            }
            or summary.get("validation_metrics_sha256") != _sha256_file(metrics_path)
            or summary.get("test_metrics_affected_selection") is not False
        ):
            raise OnlineAnalysisError(f"Stale online run summary: {summary_path}")
        frame = pd.read_csv(metrics_path, usecols=list(usecols))
        if (
            len(frame) != len(expected_epochs)
            or frame["epoch"].astype(int).tolist() != expected_epochs
            or set(frame["run_index"].astype(int)) != {run.run_index}
            or frame["candidate_id"].astype(str).tolist()
            != [run.candidate_id(epoch) for epoch in expected_epochs]
        ):
            raise OnlineAnalysisError(f"Incomplete candidate sequence: {metrics_path}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    if len(result) != expected or result["candidate_id"].astype(str).duplicated().any():
        raise OnlineAnalysisError("Online validation pool is incomplete or duplicated.")
    return result.sort_values(["run_index", "epoch"]).reset_index(drop=True)


def _assert_close(observed: Any, expected: Any, context: str) -> None:
    if not np.isclose(
        float(observed), float(expected), rtol=0.0, atol=1.0e-9
    ):
        raise OnlineAnalysisError(
            f"Persisted metric does not reproduce for {context}: "
            f"{observed!r} versus {expected!r}."
        )


def _validate_unprivileged_evidence(
    config: Mapping[str, Any],
    matrix: pd.DataFrame,
    campaign: Mapping[str, Any],
) -> None:
    """Recompute every unprivileged selection-facing value from hashed rows."""

    epsilon = float(
        config["fcv"]["selector_analysis"]["probability_ratio_epsilon"]
    )
    control_lambda = float(
        config["fcv"]["selector_analysis"]["control_normalized_lambda"]
    )
    for row in matrix.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        checkpoint_sha = str(row.checkpoint_sha256)
        fcv_path = Path(str(row.fcv_summary_path)).expanduser().resolve()
        if (
            not fcv_path.is_file()
            or _sha256_file(fcv_path) != str(row.fcv_summary_sha256)
        ):
            raise OnlineAnalysisError(f"Stale FCV summary: {candidate_id}")
        fcv = _load_json(fcv_path)
        expected_initial = campaign["bindings"]["initialization"][
            "initial_model_state_sha256_by_seed"
        ].get(str(int(row.seed)))
        biased_binding = campaign["bindings"]["manifests"]["biased_validation"]
        patch_binding = campaign["bindings"]["patch_masks"]
        score_path = Path(str(fcv.get("score_csv_path", ""))).expanduser().resolve()
        if (
            fcv.get("artifact_type") != "fcv_vit_candidate_score_summary"
            or fcv.get("status") != "complete"
            or fcv.get("candidate_id") != candidate_id
            or fcv.get("checkpoint_sha256") != checkpoint_sha
            or fcv.get("training_fingerprint")
            != campaign["bindings"]["training_fingerprint"]
            or fcv.get("campaign_provenance_path") != campaign["artifact_path"]
            or fcv.get("campaign_provenance_sha256")
            != campaign["artifact_sha256"]
            or fcv.get("campaign_bindings_sha256")
            != campaign["bindings_sha256"]
            or fcv.get("pretrained_provenance_path")
            != campaign["bindings"]["pretrained"]["path"]
            or fcv.get("pretrained_provenance_sha256")
            != campaign["bindings"]["pretrained"]["sha256"]
            or fcv.get("pretrained_backbone_sha256")
            != campaign["bindings"]["pretrained"]["backbone_sha256"]
            or fcv.get("initial_model_state_sha256") != expected_initial
            or fcv.get("software_versions")
            != campaign["bindings"]["software_versions"]
            or fcv.get("software_fingerprint")
            != campaign["bindings"]["software_fingerprint"]
            or fcv.get("validation_manifest_sha256") != biased_binding["sha256"]
            or fcv.get("manifest_bundle_sha256") != biased_binding["bundle_sha256"]
            or fcv.get("patch_mask_sha256") != patch_binding["sha256"]
            or fcv.get("patch_mask_summary_sha256")
            != patch_binding["summary_sha256"]
            or fcv.get("patch_mask_preprocessing_sha256")
            != patch_binding["preprocessing_config_sha256"]
            or fcv.get("teacher_maps_sha256")
            != patch_binding["teacher_maps_sha256"]
            or fcv.get("execution")
            != {
                "validation_batch_size": int(
                    config["execution"]["token_bank_batch_size"]
                ),
                "validation_num_workers": int(
                    config["execution"]["token_bank_num_workers"]
                ),
                "counterfactual_forward_batch_size": int(
                    config["execution"]["fcv_counterfactual_forward_batch_size"]
                ),
            }
            or not score_path.is_file()
            or _sha256_file(score_path) != fcv.get("score_csv_sha256")
        ):
            raise OnlineAnalysisError(f"Invalid FCV provenance: {candidate_id}")
        score_frame = pd.read_csv(score_path)
        fcv_metrics = validate_fcv_summary_against_frame(fcv, score_frame, config)
        fcv_expected = {
            "fcv_counterfactual_accuracy": fcv_metrics["counterfactual_accuracy"],
            "fcv_counterfactual_majority_accuracy": fcv_metrics[
                "counterfactual_majority_accuracy"
            ],
            "fcv_true_class_probability": fcv_metrics[
                "counterfactual_true_class_probability"
            ],
            "fcv_confidence_drop": fcv_metrics["mean_confidence_drop"],
            "primary_selector_score": fcv_metrics["primary_selector_score"],
            "biased_val_accuracy": fcv_metrics["original_accuracy"],
            "biased_val_loss": fcv_metrics["original_loss"],
        }
        eligible = score_frame[
            score_frame["fcv_eligible"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
        ]
        retention = (
            pd.to_numeric(eligible["p_y_counterfactual_mean"], errors="raise")
            / np.maximum(
                pd.to_numeric(eligible["p_y_original"], errors="raise"), epsilon
            )
        ).mean()
        fcv_expected["fcv_probability_retention_ratio"] = float(retention)
        for column, expected in fcv_expected.items():
            _assert_close(getattr(row, column), expected, f"{candidate_id}.{column}")
        batch_loss = float(row.biased_val_loss_batch_reduced_diagnostic)
        checkpoint_batch_loss = float(
            fcv["biased_validation_loss_checkpoint"]
        )
        _assert_close(
            batch_loss,
            checkpoint_batch_loss,
            f"{candidate_id}.biased_val_loss_batch_reduced_diagnostic",
        )
        if abs(batch_loss - float(fcv_metrics["original_loss"])) > 1.0e-6:
            raise OnlineAnalysisError(
                "Batch-reduced and canonical per-example validation losses differ "
                f"by more than 1e-6 for {candidate_id}."
            )

        controls_path = Path(str(row.controls_summary_path)).expanduser().resolve()
        if (
            not controls_path.is_file()
            or _sha256_file(controls_path) != str(row.controls_summary_sha256)
        ):
            raise OnlineAnalysisError(f"Stale control summary: {candidate_id}")
        controls = _load_json(controls_path)
        if (
            controls.get("artifact_type") != "fcv_vit_candidate_control_summary"
            or controls.get("status") != "complete"
            or controls.get("candidate_id") != candidate_id
            or controls.get("checkpoint_sha256") != checkpoint_sha
            or controls.get("training_fingerprint")
            != campaign["bindings"]["training_fingerprint"]
            or controls.get("campaign_provenance_path") != campaign["artifact_path"]
            or controls.get("campaign_provenance_sha256")
            != campaign["artifact_sha256"]
            or controls.get("campaign_bindings_sha256")
            != campaign["bindings_sha256"]
            or controls.get("pretrained_provenance_path")
            != campaign["bindings"]["pretrained"]["path"]
            or controls.get("pretrained_provenance_sha256")
            != campaign["bindings"]["pretrained"]["sha256"]
            or controls.get("pretrained_backbone_sha256")
            != campaign["bindings"]["pretrained"]["backbone_sha256"]
            or controls.get("initial_model_state_sha256") != expected_initial
            or controls.get("software_versions")
            != campaign["bindings"]["software_versions"]
            or controls.get("software_fingerprint")
            != campaign["bindings"]["software_fingerprint"]
            or controls.get("validation_manifest_sha256")
            != biased_binding["sha256"]
            or controls.get("manifest_bundle_sha256")
            != biased_binding["bundle_sha256"]
            or controls.get("patch_mask_sha256") != patch_binding["sha256"]
            or controls.get("patch_mask_summary_sha256")
            != patch_binding["summary_sha256"]
            or controls.get("patch_mask_preprocessing_sha256")
            != patch_binding["preprocessing_config_sha256"]
            or controls.get("teacher_maps_sha256")
            != patch_binding["teacher_maps_sha256"]
            or controls.get("execution")
            != {
                "validation_batch_size": int(
                    config["execution"]["token_bank_batch_size"]
                ),
                "validation_num_workers": int(
                    config["execution"]["token_bank_num_workers"]
                ),
                "target_batch_size": int(
                    config["execution"]["control_target_batch_size"]
                ),
                "counterfactual_forward_batch_size": int(
                    config["execution"][
                        "control_counterfactual_forward_batch_size"
                    ]
                ),
            }
            or controls.get("step7_summary_sha256") != str(row.fcv_summary_sha256)
        ):
            raise OnlineAnalysisError(f"Invalid control provenance: {candidate_id}")
        files = controls.get("score_csvs")
        if not isinstance(files, Mapping) or set(files) != set(CONTROL_NAMES):
            raise OnlineAnalysisError(f"Incomplete control evidence: {candidate_id}")
        recomputed: Dict[str, Mapping[str, Any]] = {}
        for name in CONTROL_NAMES:
            details = files[name]
            path = Path(str(details.get("path", ""))).expanduser().resolve()
            if (
                not path.is_file()
                or path.stat().st_size != int(details.get("size_bytes", -1))
                or _sha256_file(path) != details.get("sha256")
            ):
                raise OnlineAnalysisError(
                    f"Stale {name} control evidence: {candidate_id}"
                )
            recomputed[name] = recompute_control_metrics_from_frame(
                pd.read_csv(path)
            )
            observed = controls.get("controls", {}).get(name)
            if not isinstance(observed, Mapping):
                raise OnlineAnalysisError(
                    f"Missing {name} control metrics: {candidate_id}"
                )
            for key, value in recomputed[name].items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    _assert_close(
                        observed.get(key), value, f"{candidate_id}.{name}.{key}"
                    )
        control_expected = {
            "same_context_counterfactual_accuracy": recomputed["same_context"][
                "counterfactual_accuracy"
            ],
            "same_context_mean_confidence_drop": recomputed["same_context"][
                "mean_confidence_drop"
            ],
            "random_mask_counterfactual_accuracy": recomputed["random_mask"][
                "counterfactual_accuracy"
            ],
            "shuffled_mask_counterfactual_accuracy": recomputed["shuffled_mask"][
                "counterfactual_accuracy"
            ],
            "evidence_swap_counterfactual_accuracy": recomputed["evidence_swap"][
                "counterfactual_accuracy"
            ],
            "control_diagnostic_warning_count": int(
                controls["diagnostic_warning_count"]
            ),
            **{
                output_name: fcv_metrics["token_distribution_global_means"][source_name]
                for output_name, source_name in {
                    "target_donor_cosine_similarity_mean": (
                        "target_donor_cosine_similarity_mean"
                    ),
                    "target_nearest_donor_cosine_mean": (
                        "target_nearest_donor_cosine_mean"
                    ),
                    "donor_unique_source_images_mean": "donor_unique_source_images",
                    "donor_max_source_fraction_mean": "donor_max_source_fraction",
                }.items()
            },
            "real_swap_replaced_token_changed_fraction": fcv_metrics[
                "swap_diagnostics"
            ]["replaced_token_changed_fraction"],
            "real_swap_replacement_delta_mean": fcv_metrics["swap_diagnostics"][
                "replacement_delta_mean"
            ],
            "real_swap_replacement_delta_max": fcv_metrics["swap_diagnostics"][
                "replacement_delta_max"
            ],
            "real_swap_foreground_token_max_abs_error": fcv_metrics[
                "swap_diagnostics"
            ]["foreground_token_max_abs_error"],
            "real_swap_donor_reconstruction_max_abs_error": fcv_metrics[
                "swap_diagnostics"
            ]["donor_reconstruction_max_abs_error"],
        }
        shortcut = float(fcv_metrics["mean_confidence_drop"]) - float(
            recomputed["same_context"]["mean_confidence_drop"]
        )
        control_expected["shortcut_sensitivity"] = shortcut
        control_expected["control_normalized_fcv_score"] = float(
            row.biased_val_accuracy
        ) - control_lambda * shortcut
        for column, expected in control_expected.items():
            _assert_close(getattr(row, column), expected, f"{candidate_id}.{column}")
        if str(row.control_diagnostic_status) != str(controls["diagnostic_status"]):
            raise OnlineAnalysisError(
                f"Persisted control status does not reproduce: {candidate_id}"
            )


def _validate_oracle_evidence(
    config: Mapping[str, Any],
    output_root: Path,
    matrix: pd.DataFrame,
    campaign: Mapping[str, Any],
) -> None:
    source = prepare_oracle_validation_source(
        config,
        output_root
        / "split_manifests"
        / "metadata_oracle_val_analysis_only.csv",
        check_images=False,
    )
    for row in matrix.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        path = Path(str(row.oracle_summary_path)).expanduser().resolve()
        if not path.is_file() or _sha256_file(path) != str(row.oracle_summary_sha256):
            raise OnlineAnalysisError(f"Stale Oracle summary: {candidate_id}")
        summary = _load_json(path)
        expected_initial = campaign["bindings"]["initialization"][
            "initial_model_state_sha256_by_seed"
        ].get(str(int(row.seed)))
        oracle_binding = campaign["bindings"]["manifests"]["oracle_validation"]
        per_image = Path(
            str(summary.get("per_image_csv_path", ""))
        ).expanduser().resolve()
        if (
            summary.get("artifact_type") != "fcv_vit_oracle_validation_summary"
            or summary.get("status") != "complete"
            or summary.get("candidate_id") != candidate_id
            or summary.get("checkpoint_sha256") != str(row.checkpoint_sha256)
            or summary.get("training_fingerprint")
            != campaign["bindings"]["training_fingerprint"]
            or summary.get("campaign_provenance_path") != campaign["artifact_path"]
            or summary.get("campaign_provenance_sha256")
            != campaign["artifact_sha256"]
            or summary.get("campaign_bindings_sha256")
            != campaign["bindings_sha256"]
            or summary.get("pretrained_provenance_sha256")
            != campaign["bindings"]["pretrained"]["sha256"]
            or summary.get("pretrained_backbone_sha256")
            != campaign["bindings"]["pretrained"]["backbone_sha256"]
            or summary.get("initial_model_state_sha256") != expected_initial
            or summary.get("oracle_manifest_sha256") != oracle_binding["sha256"]
            or summary.get("manifest_bundle_sha256")
            != oracle_binding["bundle_sha256"]
            or summary.get("execution")
            != {
                "batch_size": int(
                    config["fcv"]["selector_analysis"]["oracle_batch_size"]
                ),
                "num_workers": int(config["training"]["num_workers"]),
            }
            or summary.get("test_data_accessed") is not False
            or not per_image.is_file()
            or _sha256_file(per_image) != summary.get("per_image_csv_sha256")
        ):
            raise OnlineAnalysisError(f"Invalid Oracle provenance: {candidate_id}")
        metrics = validate_oracle_summary_against_frame(
            summary, pd.read_csv(per_image), source
        )
        expected = {
            "oracle_validation_loss": metrics["loss"],
            "oracle_validation_accuracy": metrics["accuracy"],
            "oracle_validation_balanced_group_accuracy": metrics[
                "balanced_group_accuracy"
            ],
            "oracle_validation_worst_group_accuracy": metrics[
                "worst_group_accuracy"
            ],
            **{
                f"oracle_group_{group}_accuracy": metrics[
                    f"group_{group}_accuracy"
                ]
                for group in range(4)
            },
        }
        for column, value in expected.items():
            _assert_close(getattr(row, column), value, f"{candidate_id}.{column}")


def _bind_retained_primary_winners(
    config: Mapping[str, Any], output_root: Path, selections: pd.DataFrame
) -> pd.DataFrame:
    selections = selections.copy()
    selections["retained_checkpoint_path"] = ""
    primary = set(RETAINED_SELECTOR_SPECS)
    for index, row in selections.iterrows():
        selector = str(row["selector_name"])
        if selector not in primary:
            continue
        run = enumerate_sweep_runs(config)[int(row["run_index"])]
        retention_path = output_root / "online_runs" / run.run_id / "retention_state.json"
        retention = _load_json(retention_path)
        candidate_id = str(row["selected_checkpoint_id"])
        if retention.get("selectors", {}).get(selector) != candidate_id:
            raise OnlineAnalysisError(
                f"Global {selector} winner was not retained locally: {candidate_id}"
            )
        details = retention.get("checkpoints", {}).get(candidate_id)
        if not isinstance(details, Mapping):
            raise OnlineAnalysisError(f"No retained checkpoint binding for {candidate_id}")
        path = Path(str(details["path"])).expanduser().resolve()
        expected = str(row["checkpoint_sha256"])
        if not path.is_file() or str(details["sha256"]) != expected or _sha256_file(path) != expected:
            raise OnlineAnalysisError(f"Retained checkpoint bytes changed: {candidate_id}")
        selections.at[index, "retained_checkpoint_path"] = str(path)
    return selections


def _primary_checkpoint_bindings(selections: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Return the unique globally selected checkpoint bindings."""

    primary = set(RETAINED_SELECTOR_SPECS)
    observed = selections[selections["selector_name"].astype(str).isin(primary)]
    if set(observed["selector_name"].astype(str)) != primary or len(observed) != len(
        primary
    ):
        raise OnlineAnalysisError("Global primary selector rows are incomplete.")
    bindings: Dict[str, Dict[str, Any]] = {}
    for row in observed.itertuples(index=False):
        selector = str(row.selector_name)
        candidate_id = str(row.selected_checkpoint_id)
        raw_path = str(row.retained_checkpoint_path)
        path = Path(raw_path).expanduser().resolve()
        sha256 = str(row.checkpoint_sha256)
        if not candidate_id or not raw_path or len(sha256) != 64:
            raise OnlineAnalysisError(
                f"Global checkpoint binding is incomplete for {selector}."
            )
        existing = bindings.get(candidate_id)
        if existing is None:
            bindings[candidate_id] = {
                "candidate_id": candidate_id,
                "path": str(path),
                "sha256": sha256,
                "selectors": [selector],
            }
        elif existing["path"] != str(path) or existing["sha256"] != sha256:
            raise OnlineAnalysisError(
                f"Global candidate {candidate_id} has inconsistent checkpoint bindings."
            )
        else:
            existing["selectors"].append(selector)
    for details in bindings.values():
        details["selectors"] = sorted(details["selectors"])
    if not 1 <= len(bindings) <= len(primary):
        raise OnlineAnalysisError("Global checkpoint retention count is invalid.")
    return bindings


def _safe_retained_checkpoint_path(
    path: str | Path, output_root: Path, candidate_id: str
) -> Path:
    resolved = Path(path).expanduser().resolve()
    retained_root = (output_root / "online_runs").resolve()
    try:
        resolved.relative_to(retained_root)
    except ValueError as exc:
        raise OnlineAnalysisError(
            f"Refusing checkpoint cleanup outside online_runs: {resolved}"
        ) from exc
    if (
        resolved.parent.name != "retained_checkpoints"
        or resolved.name != f"{candidate_id}.pt"
    ):
        raise OnlineAnalysisError(f"Unexpected retained-checkpoint path: {resolved}")
    return resolved


def _local_checkpoint_inventory(
    output_root: Path,
    runs: Sequence[SweepRun],
    *,
    require_all_files: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Validate and inventory every per-run winner before destructive cleanup."""

    inventory: Dict[str, Dict[str, Any]] = {}
    expected_selectors = set(RETAINED_SELECTOR_SPECS)
    for run in runs:
        run_dir = output_root / "online_runs" / run.run_id
        retention_path = run_dir / "retention_state.json"
        retention = _load_json(retention_path)
        selectors = retention.get("selectors")
        checkpoints = retention.get("checkpoints")
        if (
            retention.get("artifact_type") != "fcv_vit_online_local_retention"
            or retention.get("status") != "complete"
            or not isinstance(selectors, Mapping)
            or set(selectors) != expected_selectors
            or not isinstance(checkpoints, Mapping)
            or int(retention.get("unique_checkpoint_count", -1)) != len(checkpoints)
            or len(checkpoints) > len(expected_selectors)
            or set(selectors.values()) != set(checkpoints)
        ):
            raise OnlineAnalysisError(f"Invalid local retention state: {retention_path}")
        expected_paths = set()
        for candidate_id, raw_details in checkpoints.items():
            candidate_id = str(candidate_id)
            if candidate_id in inventory or not isinstance(raw_details, Mapping):
                raise OnlineAnalysisError(
                    f"Duplicate or invalid retained candidate: {candidate_id}"
                )
            path = _safe_retained_checkpoint_path(
                str(raw_details.get("path", "")), output_root, candidate_id
            )
            sha256 = str(raw_details.get("sha256", ""))
            expected_for_candidate = sorted(
                selector
                for selector, selected in selectors.items()
                if str(selected) == candidate_id
            )
            if (
                len(sha256) != 64
                or sorted(raw_details.get("selectors", []))
                != expected_for_candidate
            ):
                raise OnlineAnalysisError(
                    f"Retained checkpoint failed pre-cleanup validation: {path}"
                )
            if path.exists() and (
                not path.is_file()
                or path.is_symlink()
                or _sha256_file(path) != sha256
            ):
                raise OnlineAnalysisError(
                    f"Retained checkpoint failed pre-cleanup validation: {path}"
                )
            if require_all_files and not path.is_file():
                raise OnlineAnalysisError(f"Missing retained checkpoint: {path}")
            expected_paths.add(path)
            inventory[candidate_id] = {
                "candidate_id": candidate_id,
                "run_id": run.run_id,
                "path": str(path),
                "sha256": sha256,
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "selectors": expected_for_candidate,
            }
        actual_paths = {
            path.resolve() for path in (run_dir / "retained_checkpoints").glob("*.pt")
        }
        paths_match = (
            actual_paths == expected_paths
            if require_all_files
            else actual_paths.issubset(expected_paths)
        )
        if not paths_match:
            raise OnlineAnalysisError(
                f"Untracked retained checkpoint exists for run {run.run_id}."
            )
    return inventory


def _cleanup_non_global_retained_checkpoints(
    output_root: Path,
    result_dir: Path,
    selections: pd.DataFrame,
    *,
    runs: Sequence[SweepRun],
    training_fingerprint: str,
    selection_table_path: Path,
    candidate_matrix_path: Path,
) -> Dict[str, Any]:
    """Prune local winners after global selection with a resumable cleanup plan."""

    output_root = output_root.expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    table_sha256 = _sha256_file(selection_table_path)
    matrix_sha256 = _sha256_file(candidate_matrix_path)
    keep = _primary_checkpoint_bindings(selections)
    keep_ids = set(keep)
    plan_path = result_dir / "online_checkpoint_cleanup_plan.json"
    receipt_path = result_dir / "online_checkpoint_cleanup_receipt.json"

    if plan_path.is_file():
        plan = _load_json(plan_path)
        planned_keep = plan.get("keep")
        planned_delete = plan.get("delete")
        if (
            plan.get("schema_version") != 1
            or plan.get("artifact_type")
            != "fcv_vit_online_checkpoint_cleanup_plan"
            or plan.get("status") != "ready"
            or plan.get("training_fingerprint") != training_fingerprint
            or plan.get("selection_table_sha256") != table_sha256
            or plan.get("candidate_matrix_sha256") != matrix_sha256
            or planned_keep != [keep[key] for key in sorted(keep)]
            or not isinstance(planned_delete, list)
        ):
            raise OnlineAnalysisError("Existing checkpoint cleanup plan is stale.")
        metadata = _local_checkpoint_inventory(
            output_root, runs, require_all_files=False
        )
        planned = {
            str(details.get("candidate_id", "")): details
            for details in planned_keep + planned_delete
            if isinstance(details, Mapping)
        }
        if set(planned) != set(metadata):
            raise OnlineAnalysisError(
                "Checkpoint cleanup plan differs from local retention manifests."
            )
        for candidate_id, expected in metadata.items():
            observed = planned[candidate_id]
            for key in ("candidate_id", "path", "sha256"):
                if str(observed.get(key, "")) != str(expected[key]):
                    raise OnlineAnalysisError(
                        "Checkpoint cleanup plan differs from local retention manifests."
                    )
            if candidate_id not in keep_ids and (
                str(observed.get("run_id", "")) != str(expected["run_id"])
                or sorted(observed.get("selectors", []))
                != sorted(expected["selectors"])
            ):
                raise OnlineAnalysisError(
                    "Checkpoint cleanup plan differs from local retention manifests."
                )
    else:
        inventory = _local_checkpoint_inventory(output_root, runs)
        if not keep_ids.issubset(inventory):
            raise OnlineAnalysisError("A global winner is absent from local retention.")
        for candidate_id, binding in keep.items():
            local = inventory[candidate_id]
            if local["path"] != binding["path"] or local["sha256"] != binding["sha256"]:
                raise OnlineAnalysisError(
                    f"Global/local checkpoint binding differs: {candidate_id}"
                )
        delete = [
            inventory[key] for key in sorted(set(inventory).difference(keep_ids))
        ]
        plan = {
            "schema_version": 1,
            "artifact_type": "fcv_vit_online_checkpoint_cleanup_plan",
            "status": "ready",
            "training_fingerprint": training_fingerprint,
            "selection_table_path": str(selection_table_path.resolve()),
            "selection_table_sha256": table_sha256,
            "candidate_matrix_path": str(candidate_matrix_path.resolve()),
            "candidate_matrix_sha256": matrix_sha256,
            "retained_checkpoint_count_before_cleanup": len(inventory),
            "keep": [keep[key] for key in sorted(keep)],
            "delete": delete,
        }
        _atomic_json(plan, plan_path)

    planned_keep = plan.get("keep")
    planned_delete = plan.get("delete")
    if not isinstance(planned_keep, list) or not isinstance(planned_delete, list):
        raise OnlineAnalysisError("Checkpoint cleanup plan has invalid file lists.")
    if int(plan.get("retained_checkpoint_count_before_cleanup", -1)) != (
        len(planned_keep) + len(planned_delete)
    ):
        raise OnlineAnalysisError("Checkpoint cleanup plan has an invalid file count.")
    all_ids = set()
    for details in planned_keep + planned_delete:
        if not isinstance(details, Mapping):
            raise OnlineAnalysisError("Checkpoint cleanup binding is not a mapping.")
        candidate_id = str(details.get("candidate_id", ""))
        if not candidate_id or candidate_id in all_ids:
            raise OnlineAnalysisError("Checkpoint cleanup plan has duplicate candidates.")
        all_ids.add(candidate_id)
        path = _safe_retained_checkpoint_path(
            str(details.get("path", "")), output_root, candidate_id
        )
        if len(str(details.get("sha256", ""))) != 64:
            raise OnlineAnalysisError("Checkpoint cleanup plan has an invalid hash.")
        if candidate_id in keep_ids:
            if (
                not path.is_file()
                or path.is_symlink()
                or _sha256_file(path) != str(details["sha256"])
            ):
                raise OnlineAnalysisError(f"Global retained checkpoint changed: {path}")

    for details in planned_delete:
        candidate_id = str(details["candidate_id"])
        path = _safe_retained_checkpoint_path(details["path"], output_root, candidate_id)
        if path.exists():
            if (
                not path.is_file()
                or path.is_symlink()
                or _sha256_file(path) != str(details["sha256"])
                or path.stat().st_size != int(details["size_bytes"])
            ):
                raise OnlineAnalysisError(
                    f"Refusing to delete changed retained checkpoint: {path}"
                )
            path.unlink()
    if any(Path(str(details["path"])).exists() for details in planned_delete):
        raise OnlineAnalysisError("Non-global checkpoint cleanup is incomplete.")

    receipt = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_checkpoint_cleanup_receipt",
        "status": "complete",
        "training_fingerprint": training_fingerprint,
        "cleanup_plan_path": str(plan_path.resolve()),
        "cleanup_plan_sha256": _sha256_file(plan_path),
        "selection_table_sha256": table_sha256,
        "candidate_matrix_sha256": matrix_sha256,
        "retained_checkpoint_count_before_cleanup": int(
            plan["retained_checkpoint_count_before_cleanup"]
        ),
        "deleted_checkpoint_count": len(planned_delete),
        "retained_checkpoint_count_after_cleanup": len(planned_keep),
        "retained_global_checkpoints": planned_keep,
        "deleted_local_checkpoints": planned_delete,
    }
    _atomic_json(receipt, receipt_path)
    return {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": _sha256_file(receipt_path),
        "retained_checkpoint_count_before_cleanup": receipt[
            "retained_checkpoint_count_before_cleanup"
        ],
        "deleted_checkpoint_count": receipt["deleted_checkpoint_count"],
        "retained_checkpoint_count_after_cleanup": receipt[
            "retained_checkpoint_count_after_cleanup"
        ],
    }


def _validate_post_freeze_checkpoint_cleanup(
    config: Mapping[str, Any],
    output_root: Path,
    selection_summary: Mapping[str, Any],
    selections: pd.DataFrame,
) -> None:
    receipt_path = Path(
        str(selection_summary.get("checkpoint_cleanup_receipt_path", ""))
    ).expanduser().resolve()
    if (
        selection_summary.get("post_freeze_checkpoint_cleanup_complete") is not True
        or not receipt_path.is_file()
        or _sha256_file(receipt_path)
        != selection_summary.get("checkpoint_cleanup_receipt_sha256")
    ):
        raise OnlineAnalysisError("Post-freeze checkpoint cleanup is not complete.")
    receipt = _load_json(receipt_path)
    plan_path = Path(str(receipt.get("cleanup_plan_path", ""))).expanduser().resolve()
    keep = _primary_checkpoint_bindings(selections)
    retained = receipt.get("retained_global_checkpoints")
    deleted = receipt.get("deleted_local_checkpoints")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type")
        != "fcv_vit_online_checkpoint_cleanup_receipt"
        or receipt.get("status") != "complete"
        or receipt.get("training_fingerprint")
        != candidate_training_fingerprint(config)
        or not plan_path.is_file()
        or _sha256_file(plan_path) != receipt.get("cleanup_plan_sha256")
        or receipt.get("selection_table_sha256")
        != selection_summary.get("selection_table_sha256")
        or receipt.get("candidate_matrix_sha256")
        != selection_summary.get("candidate_matrix_sha256")
        or retained != [keep[key] for key in sorted(keep)]
        or not isinstance(deleted, list)
        or int(receipt.get("retained_checkpoint_count_after_cleanup", -1))
        != len(keep)
        or int(receipt.get("deleted_checkpoint_count", -1)) != len(deleted)
        or int(receipt.get("retained_checkpoint_count_before_cleanup", -1))
        != len(keep) + len(deleted)
        or len(keep) > int(config["candidate_pool"]["max_final_retained_checkpoints"])
    ):
        raise OnlineAnalysisError("Post-freeze checkpoint cleanup receipt is stale.")
    for details in retained:
        if not isinstance(details, Mapping):
            raise OnlineAnalysisError("Retained checkpoint receipt entry is invalid.")
        path = _safe_retained_checkpoint_path(
            details["path"], output_root, str(details["candidate_id"])
        )
        if not path.is_file() or _sha256_file(path) != str(details["sha256"]):
            raise OnlineAnalysisError(f"Global retained checkpoint changed: {path}")
    for details in deleted:
        if not isinstance(details, Mapping):
            raise OnlineAnalysisError("Deleted checkpoint receipt entry is invalid.")
        path = _safe_retained_checkpoint_path(
            details["path"], output_root, str(details["candidate_id"])
        )
        if path.exists():
            raise OnlineAnalysisError(f"Non-global retained checkpoint survived: {path}")


def freeze_online_validation_selection(
    config: Mapping[str, Any], output_root: str | Path
) -> Dict[str, Any]:
    """Freeze every selector without opening any test-analysis artifact."""

    output_root = Path(output_root).expanduser().resolve()
    campaign = load_campaign_provenance_receipt(
        config, pretrained_path=pretrained_provenance_path(config)
    )
    verify_non_test_campaign_inputs(config)
    result_dir = output_root / "selection_results"
    identity = [
        "run_index",
        "candidate_id",
        "epoch",
        "seed",
        "learning_rate",
        "weight_decay",
        "checkpoint_sha256",
    ]
    oracle_columns = [
        "oracle_validation_loss",
        "oracle_validation_accuracy",
        "oracle_validation_balanced_group_accuracy",
        "oracle_validation_worst_group_accuracy",
        "oracle_group_0_accuracy",
        "oracle_group_1_accuracy",
        "oracle_group_2_accuracy",
        "oracle_group_3_accuracy",
        "oracle_summary_path",
        "oracle_summary_sha256",
    ]
    unprivileged_columns = [
        column
        for column in ONLINE_VALIDATION_COLUMNS
        if column not in oracle_columns
    ]
    # First freeze the non-privileged selectors using pandas usecols. Oracle
    # columns never enter this phase's memory.
    unprivileged = _load_validation_runs(
        config, output_root, unprivileged_columns, campaign
    )
    _validate_unprivileged_evidence(config, unprivileged, campaign)
    for value in config["fcv"]["selector_analysis"]["fcv_accuracy_lambdas"]:
        slug = str(float(value)).replace(".", "p")
        unprivileged[f"fcv_accuracy_lambda_{slug}_score"] = (
            unprivileged["biased_val_accuracy"]
            + float(value) * unprivileged["fcv_counterfactual_accuracy"]
        )
    specs = _selector_specs(config)
    unprivileged_specs = [spec for spec in specs if spec["family"] != "oracle"]
    unprivileged_selections = pd.DataFrame(
        [_select(unprivileged, spec) for spec in unprivileged_specs]
    )
    unprivileged_matrix_path = result_dir / "online_unprivileged_candidate_matrix.csv"
    unprivileged_table_path = result_dir / "online_unprivileged_selections_frozen.csv"
    _atomic_csv(unprivileged, unprivileged_matrix_path)
    _atomic_csv(unprivileged_selections, unprivileged_table_path)
    unprivileged_matrix_sha256 = _sha256_file(unprivileged_matrix_path)
    unprivileged_table_sha256 = _sha256_file(unprivileged_table_path)
    diagnostic_columns = [
        "run_index",
        "candidate_id",
        "epoch",
        "control_diagnostic_warning_count",
        "control_diagnostic_status",
        "fcv_counterfactual_accuracy",
        "same_context_counterfactual_accuracy",
        "random_mask_counterfactual_accuracy",
        "shuffled_mask_counterfactual_accuracy",
        "evidence_swap_counterfactual_accuracy",
        "target_donor_cosine_similarity_mean",
        "target_nearest_donor_cosine_mean",
        "donor_unique_source_images_mean",
        "donor_max_source_fraction_mean",
        "real_swap_replaced_token_changed_fraction",
        "real_swap_replacement_delta_mean",
        "real_swap_replacement_delta_max",
        "real_swap_foreground_token_max_abs_error",
        "real_swap_donor_reconstruction_max_abs_error",
    ]
    control_diagnostics = unprivileged[diagnostic_columns].copy()
    control_diagnostics_path = result_dir / "online_control_diagnostics.csv"
    _atomic_csv(control_diagnostics, control_diagnostics_path)
    control_diagnostics_summary_path = (
        result_dir / "online_control_diagnostics_summary.json"
    )
    control_diagnostic_metrics = diagnostic_columns[5:]
    control_diagnostics_summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_control_diagnostics_summary",
        "status": "warning" if (
            control_diagnostics["control_diagnostic_warning_count"].astype(int) > 0
        ).any() else "passed",
        "candidate_count": len(control_diagnostics),
        "candidate_warning_count": int(
            (
                control_diagnostics["control_diagnostic_warning_count"].astype(int)
                > 0
            ).sum()
        ),
        "total_warning_count": int(
            control_diagnostics["control_diagnostic_warning_count"].astype(int).sum()
        ),
        "metric_distributions": _numeric_summary(
            control_diagnostics, control_diagnostic_metrics
        ),
        "candidate_diagnostics_path": str(control_diagnostics_path.resolve()),
        "candidate_diagnostics_sha256": _sha256_file(control_diagnostics_path),
    }
    _atomic_json(control_diagnostics_summary, control_diagnostics_summary_path)
    control_diagnostics_summary_sha256 = _sha256_file(
        control_diagnostics_summary_path
    )
    unprivileged_receipt_path = result_dir / "online_unprivileged_freeze_receipt.json"
    unprivileged_receipt = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_unprivileged_freeze_receipt",
        "status": "complete",
        "training_fingerprint": candidate_training_fingerprint(config),
        "campaign_provenance_path": campaign["artifact_path"],
        "campaign_provenance_sha256": campaign["artifact_sha256"],
        "campaign_bindings_sha256": campaign["bindings_sha256"],
        "candidate_count": len(unprivileged),
        "selector_count": len(unprivileged_selections),
        "candidate_matrix_path": str(unprivileged_matrix_path.resolve()),
        "candidate_matrix_sha256": unprivileged_matrix_sha256,
        "selection_table_path": str(unprivileged_table_path.resolve()),
        "selection_table_sha256": unprivileged_table_sha256,
        "control_diagnostics_summary_path": str(
            control_diagnostics_summary_path.resolve()
        ),
        "control_diagnostics_summary_sha256": control_diagnostics_summary_sha256,
        "oracle_artifacts_opened_before_receipt": False,
        "test_artifacts_opened_before_receipt": False,
    }
    _atomic_json(unprivileged_receipt, unprivileged_receipt_path)
    unprivileged_receipt_sha256 = _sha256_file(unprivileged_receipt_path)

    # Only after the unprivileged choices exist and are hashed do we read the
    # privileged Oracle columns and append Oracle selectors.
    oracle = _load_validation_runs(
        config, output_root, identity + oracle_columns, campaign
    )
    if unprivileged["candidate_id"].astype(str).tolist() != oracle[
        "candidate_id"
    ].astype(str).tolist():
        raise OnlineAnalysisError("Oracle and unprivileged candidate order differs.")
    matrix = unprivileged.copy()
    for column in oracle_columns:
        matrix[column] = oracle[column].to_numpy()
    _validate_oracle_evidence(config, output_root, matrix, campaign)
    oracle_specs = [spec for spec in specs if spec["family"] == "oracle"]
    selections = pd.concat(
        [
            unprivileged_selections,
            pd.DataFrame([_select(matrix, spec) for spec in oracle_specs]),
        ],
        ignore_index=True,
    )
    selections = _bind_retained_primary_winners(config, output_root, selections)
    if any(str(column).startswith("test_") for column in matrix.columns):
        raise OnlineAnalysisError("Frozen validation matrix contains test metrics.")
    matrix_path = result_dir / "online_validation_candidate_matrix.csv"
    table_path = result_dir / "online_selections_frozen.csv"
    summary_path = result_dir / "online_selection_summary.json"
    _atomic_csv(matrix, matrix_path)
    _atomic_csv(selections, table_path)
    # The selection table and matrix are durable before any checkpoint is
    # deleted.  The cleanup plan makes this phase safe to resume after a
    # partial deletion, and keeps only the unique global primary winners.
    cleanup = _cleanup_non_global_retained_checkpoints(
        output_root,
        result_dir,
        selections,
        runs=enumerate_sweep_runs(config),
        training_fingerprint=candidate_training_fingerprint(config),
        selection_table_path=table_path,
        candidate_matrix_path=matrix_path,
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_selection_summary",
        "status": "complete",
        "candidate_count": len(matrix),
        "selector_count": len(selections),
        "training_fingerprint": candidate_training_fingerprint(config),
        "campaign_provenance_path": campaign["artifact_path"],
        "campaign_provenance_sha256": campaign["artifact_sha256"],
        "campaign_bindings_sha256": campaign["bindings_sha256"],
        "candidate_matrix_path": str(matrix_path.resolve()),
        "candidate_matrix_sha256": _sha256_file(matrix_path),
        "selection_table_path": str(table_path.resolve()),
        "selection_table_sha256": _sha256_file(table_path),
        "unprivileged_matrix_path": str(unprivileged_matrix_path.resolve()),
        "unprivileged_matrix_sha256": unprivileged_matrix_sha256,
        "unprivileged_selection_path": str(unprivileged_table_path.resolve()),
        "unprivileged_selection_sha256": unprivileged_table_sha256,
        "unprivileged_freeze_receipt_path": str(
            unprivileged_receipt_path.resolve()
        ),
        "unprivileged_freeze_receipt_sha256": unprivileged_receipt_sha256,
        "control_diagnostics_summary_path": str(
            control_diagnostics_summary_path.resolve()
        ),
        "control_diagnostics_summary_sha256": control_diagnostics_summary_sha256,
        "control_candidate_warning_count": control_diagnostics_summary[
            "candidate_warning_count"
        ],
        "control_total_warning_count": control_diagnostics_summary[
            "total_warning_count"
        ],
        "unprivileged_selection_frozen_before_oracle_join": True,
        "test_data_accessed": False,
        "test_artifact_tree_opened": False,
        "test_metrics_affected_selection": False,
        "post_freeze_checkpoint_cleanup_complete": True,
        "checkpoint_cleanup_receipt_path": cleanup["receipt_path"],
        "checkpoint_cleanup_receipt_sha256": cleanup["receipt_sha256"],
        "retained_checkpoint_count_before_cleanup": cleanup[
            "retained_checkpoint_count_before_cleanup"
        ],
        "deleted_checkpoint_count": cleanup["deleted_checkpoint_count"],
        "retained_checkpoint_count_after_cleanup": cleanup[
            "retained_checkpoint_count_after_cleanup"
        ],
        "selected_candidates": dict(
            zip(selections["selector_name"], selections["selected_checkpoint_id"])
        ),
    }
    _atomic_json(summary, summary_path)
    return summary


def _validate_test_evidence_rows(
    config: Mapping[str, Any],
    pool: pd.DataFrame,
    source: Any,
    campaign: Mapping[str, Any],
) -> None:
    """Recompute compact test rows from hashed per-image analysis-only evidence."""

    from .test_evaluation import validate_test_summary_against_frame

    for row in pool.itertuples(index=False):
        per_image = Path(str(row.per_image_path)).expanduser().resolve()
        summary = Path(str(row.summary_path)).expanduser().resolve()
        if (
            not per_image.is_file()
            or _sha256_file(per_image) != str(row.per_image_sha256)
            or not summary.is_file()
            or _sha256_file(summary) != str(row.summary_sha256)
        ):
            raise OnlineAnalysisError(f"Stale test evidence: {row.candidate_id}")
        payload = _load_json(summary)
        expected_initial = campaign["bindings"]["initialization"][
            "initial_model_state_sha256_by_seed"
        ].get(str(int(row.seed)))
        test_binding = campaign["bindings"]["manifests"]["test"]
        if (
            payload.get("candidate_id") != str(row.candidate_id)
            or payload.get("checkpoint_sha256") != str(row.checkpoint_sha256)
            or payload.get("training_fingerprint")
            != campaign["bindings"]["training_fingerprint"]
            or payload.get("campaign_provenance_path") != campaign["artifact_path"]
            or payload.get("campaign_provenance_sha256")
            != campaign["artifact_sha256"]
            or payload.get("campaign_bindings_sha256")
            != campaign["bindings_sha256"]
            or payload.get("pretrained_provenance_path")
            != campaign["bindings"]["pretrained"]["path"]
            or payload.get("pretrained_provenance_sha256")
            != campaign["bindings"]["pretrained"]["sha256"]
            or payload.get("pretrained_backbone_sha256")
            != campaign["bindings"]["pretrained"]["backbone_sha256"]
            or payload.get("initial_model_state_sha256") != expected_initial
            or payload.get("posthoc_analysis_only") is not True
            or payload.get("eligible_for_model_selection") is not False
            or payload.get("test_metrics_available_to_training_or_retention")
            is not False
            or payload.get("test_metrics_affected_selection") is not False
            or payload.get("per_image_csv_sha256") != str(row.per_image_sha256)
            or payload.get("test_manifest_sha256") != source.manifest_sha256
            or payload.get("test_manifest_sha256") != test_binding["sha256"]
            or payload.get("manifest_bundle_sha256")
            != source.manifest_bundle_sha256
            or payload.get("manifest_bundle_sha256")
            != test_binding["bundle_sha256"]
            or payload.get("execution")
            != {
                "batch_size": int(config["evaluation"]["final_test"]["batch_size"]),
                "num_workers": int(config["training"]["num_workers"]),
                "precision": str(config["evaluation"]["final_test"]["precision"]),
            }
        ):
            raise OnlineAnalysisError(f"Invalid test provenance: {row.candidate_id}")
        metrics = validate_test_summary_against_frame(
            payload, pd.read_csv(per_image), source
        )
        expected = {
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "test_balanced_group_accuracy": metrics["balanced_group_accuracy"],
            "test_worst_group_accuracy": metrics["worst_group_accuracy"],
            **{
                f"test_group_{group}_accuracy": metrics[
                    f"group_{group}_accuracy"
                ]
                for group in range(4)
            },
            **{
                f"test_group_{group}_count": metrics[f"group_{group}_count"]
                for group in range(4)
            },
        }
        for column, value in expected.items():
            _assert_close(getattr(row, column), value, f"{row.candidate_id}.{column}")


def _load_test_pool(
    config: Mapping[str, Any],
    output_root: Path,
    validation_matrix: pd.DataFrame,
    campaign: Mapping[str, Any],
) -> pd.DataFrame:
    # Importing and opening the test source is intentionally delayed until this
    # post-freeze function. The validation freezer never touches test bytes.
    from .test_evaluation import prepare_final_test_source

    source = prepare_final_test_source(
        config,
        output_root / "split_manifests" / "metadata_test_analysis_only.csv",
        check_images=True,
    )
    frames = []
    expected_epochs = list(range(1, int(config["training"]["epochs"]) + 1))
    for run in enumerate_sweep_runs(config):
        path = output_root / "online_runs" / run.run_id / "test_metrics_analysis_only.csv"
        if not path.is_file():
            raise OnlineAnalysisError(f"Missing online test index: {path}")
        frame = pd.read_csv(path, usecols=ONLINE_TEST_COLUMNS)
        if len(frame) != len(expected_epochs) or frame["epoch"].astype(int).tolist() != expected_epochs:
            raise OnlineAnalysisError(f"Incomplete online test index: {path}")
        frames.append(frame)
    pool = pd.concat(frames, ignore_index=True).sort_values(
        ["run_index", "epoch"]
    ).reset_index(drop=True)
    expected = int(config["candidate_pool"]["expected_candidate_checkpoints"])
    if len(pool) != expected or pool["candidate_id"].astype(str).duplicated().any():
        raise OnlineAnalysisError("Online test pool is incomplete or duplicated.")
    for column in (
        "candidate_id",
        "run_index",
        "epoch",
        "seed",
        "learning_rate",
        "weight_decay",
        "checkpoint_sha256",
    ):
        left = validation_matrix[column].to_numpy()
        right = pool[column].to_numpy()
        matches = (
            np.allclose(left, right, rtol=0.0, atol=0.0)
            if np.issubdtype(np.asarray(left).dtype, np.number)
            else np.array_equal(left.astype(str), right.astype(str))
        )
        if not matches:
            raise OnlineAnalysisError(f"Validation/test identity differs: {column}")
    _validate_test_evidence_rows(config, pool, source, campaign)
    return pool


def validate_online_run_prefix(
    config: Mapping[str, Any],
    output_root: str | Path,
    *,
    run_index: int,
    expected_epoch: int,
    require_resumed_from_epoch: int | None = None,
    storage_baseline_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Validate a real paused online producer before releasing the full array."""

    output_root = Path(output_root).expanduser().resolve()
    campaign = load_campaign_provenance_receipt(
        config, pretrained_path=pretrained_provenance_path(config)
    )
    verify_non_test_campaign_inputs(config)
    runs = enumerate_sweep_runs(config)
    if run_index < 0 or run_index >= len(runs):
        raise OnlineAnalysisError(f"Invalid online smoke run index: {run_index}")
    if expected_epoch < 1 or expected_epoch >= int(config["training"]["epochs"]):
        raise OnlineAnalysisError(
            "Online smoke must validate a non-final interrupted epoch prefix."
        )
    run = runs[run_index]
    run_dir = output_root / "online_runs" / run.run_id
    validation_path = run_dir / "validation_metrics.csv"
    test_path = run_dir / "test_metrics_analysis_only.csv"
    resume_path = run_dir / "resume_state.pt"
    for path in (validation_path, test_path, resume_path):
        if not path.is_file():
            raise OnlineAnalysisError(f"Missing online smoke artifact: {path}")
    validation = pd.read_csv(validation_path, usecols=ONLINE_VALIDATION_COLUMNS)
    test = pd.read_csv(test_path, usecols=ONLINE_TEST_COLUMNS)
    expected_epochs = list(range(1, expected_epoch + 1))
    expected_ids = [run.candidate_id(epoch) for epoch in expected_epochs]
    for frame, name in ((validation, "validation"), (test, "test")):
        if (
            len(frame) != expected_epoch
            or frame["epoch"].astype(int).tolist() != expected_epochs
            or frame["candidate_id"].astype(str).tolist() != expected_ids
            or set(frame["run_index"].astype(int)) != {run_index}
        ):
            raise OnlineAnalysisError(
                f"Online smoke {name} prefix is incomplete or non-contiguous."
            )

    oracle_columns = {
        "oracle_validation_loss",
        "oracle_validation_accuracy",
        "oracle_validation_balanced_group_accuracy",
        "oracle_validation_worst_group_accuracy",
        "oracle_group_0_accuracy",
        "oracle_group_1_accuracy",
        "oracle_group_2_accuracy",
        "oracle_group_3_accuracy",
        "oracle_summary_path",
        "oracle_summary_sha256",
    }
    unprivileged = validation[
        [column for column in ONLINE_VALIDATION_COLUMNS if column not in oracle_columns]
    ]
    _validate_unprivileged_evidence(config, unprivileged, campaign)
    _validate_oracle_evidence(config, output_root, validation, campaign)

    # Test evidence is opened only after validation and Oracle evidence have
    # passed; this smoke validator never exposes test values to retention.
    from .test_evaluation import prepare_final_test_source

    source = prepare_final_test_source(
        config,
        output_root / "split_manifests" / "metadata_test_analysis_only.csv",
        check_images=True,
    )
    _validate_test_evidence_rows(config, test, source, campaign)

    resume = _load_trusted_checkpoint(resume_path)
    if (
        resume.get("artifact_type") != "fcv_vit_online_resume_state"
        or int(resume.get("completed_epoch", -1)) != expected_epoch
        or resume.get("campaign_provenance_sha256") != campaign["artifact_sha256"]
        or resume.get("campaign_bindings_sha256") != campaign["bindings_sha256"]
        or resume.get("test_metric_values_stored_in_resume_state") is not False
        or "test_rows" in resume
        or len(resume.get("validation_rows", [])) != expected_epoch
        or resume.get("validation_metrics_sha256") != _sha256_file(validation_path)
        or resume.get("analysis_only_test_metrics_sha256") != _sha256_file(test_path)
        or int(resume.get("analysis_only_test_metrics_size_bytes", -1))
        != test_path.stat().st_size
    ):
        raise OnlineAnalysisError("Online smoke resume commit is stale or leaky.")
    if require_resumed_from_epoch is not None and int(
        resume.get("invocation_resumed_from_epoch", -1)
    ) != int(require_resumed_from_epoch):
        raise OnlineAnalysisError(
            "Online smoke did not resume from the required committed epoch."
        )

    storage_observation: Dict[str, Any] | None = None
    if storage_baseline_path is not None:
        baseline_path = Path(storage_baseline_path).expanduser().resolve()
        storage_observation = _storage_high_water_projection(
            config,
            output_root,
            run_dir,
            baseline_path,
            run.candidate_id(expected_epoch),
            expected_epoch,
        )
    runtime_observation = _runtime_projection(config, test)

    receipt_path = (
        output_root
        / "preflight"
        / f"online_smoke_run_{run_index:03d}_epoch_{expected_epoch:03d}.json"
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_real_smoke_receipt",
        "status": "passed",
        "run_index": run_index,
        "run_id": run.run_id,
        "validated_epoch_prefix": expected_epoch,
        "required_resumed_from_epoch": require_resumed_from_epoch,
        "campaign_provenance_path": campaign["artifact_path"],
        "campaign_provenance_sha256": campaign["artifact_sha256"],
        "campaign_bindings_sha256": campaign["bindings_sha256"],
        "validation_metrics_sha256": _sha256_file(validation_path),
        "test_metrics_analysis_only_sha256": _sha256_file(test_path),
        "resume_state_sha256": _sha256_file(resume_path),
        "fcv_controls_oracle_test_recomputed": True,
        "storage_high_water_projection": storage_observation,
        "runtime_projection": runtime_observation,
        "test_metrics_available_to_training_or_retention": False,
        "checkpoint_retention_expanded": False,
    }
    _atomic_json(receipt, receipt_path)
    result = dict(receipt)
    result["receipt_path"] = str(receipt_path.resolve())
    result["receipt_sha256"] = _sha256_file(receipt_path)
    return result


def _validated_online_smoke_receipt(
    output_root: Path,
    campaign: Mapping[str, Any],
    *,
    run_index: int,
    expected_epoch: int,
) -> Dict[str, Any]:
    """Load one immutable smoke-stage receipt for the current campaign."""

    if expected_epoch not in (1, 2):
        raise OnlineAnalysisError("Only smoke epochs 1 and 2 are reusable gates.")
    path = (
        output_root
        / "preflight"
        / f"online_smoke_run_{run_index:03d}_epoch_{expected_epoch:03d}.json"
    )
    if not path.is_file():
        raise OnlineAnalysisError(f"Missing online smoke receipt: {path}")
    receipt = _load_json(path)
    storage = receipt.get("storage_high_water_projection")
    runtime = receipt.get("runtime_projection")
    if (
        receipt.get("artifact_type") != "fcv_vit_online_real_smoke_receipt"
        or receipt.get("status") != "passed"
        or int(receipt.get("run_index", -1)) != run_index
        or int(receipt.get("validated_epoch_prefix", -1)) != expected_epoch
        or receipt.get("required_resumed_from_epoch")
        != (None if expected_epoch == 1 else 1)
        or receipt.get("campaign_provenance_path") != campaign["artifact_path"]
        or receipt.get("campaign_provenance_sha256")
        != campaign["artifact_sha256"]
        or receipt.get("campaign_bindings_sha256")
        != campaign["bindings_sha256"]
        or receipt.get("checkpoint_retention_expanded") is not False
        or not isinstance(storage, Mapping)
        or storage.get("within_configured_reserve") is not True
        or storage.get("within_full_campaign_launch_guard") is not True
        or not isinstance(runtime, Mapping)
        or runtime.get("within_run_time_limit") is not True
    ):
        raise OnlineAnalysisError(f"Stale or failed online smoke receipt: {path}")
    result = dict(receipt)
    result["receipt_path"] = str(path.resolve())
    result["receipt_sha256"] = _sha256_file(path)
    result["receipt_status"] = "reused"
    return result


def validate_reusable_online_smoke_receipt(
    config: Mapping[str, Any],
    output_root: str | Path,
    *,
    run_index: int = 0,
    expected_epoch: int,
) -> Dict[str, Any]:
    """Validate one completed smoke stage without reopening mutable run state."""

    output_root = Path(output_root).expanduser().resolve()
    campaign = load_campaign_provenance_receipt(
        config, pretrained_path=pretrained_provenance_path(config)
    )
    return _validated_online_smoke_receipt(
        output_root,
        campaign,
        run_index=run_index,
        expected_epoch=expected_epoch,
    )


def ensure_online_smoke_gate(
    config: Mapping[str, Any], output_root: str | Path, *, run_index: int = 0
) -> Dict[str, Any]:
    """Validate/backfill the campaign-bound smoke gate for restart-safe launch."""

    output_root = Path(output_root).expanduser().resolve()
    campaign = load_campaign_provenance_receipt(
        config, pretrained_path=pretrained_provenance_path(config)
    )
    receipts = []
    for epoch in (1, 2):
        receipt = _validated_online_smoke_receipt(
            output_root,
            campaign,
            run_index=run_index,
            expected_epoch=epoch,
        )
        receipts.append(
            {
                "epoch": epoch,
                "path": receipt["receipt_path"],
                "sha256": receipt["receipt_sha256"],
            }
        )

    gate_path = output_root / "preflight" / "online_smoke_gate_receipt.json"
    expected = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_smoke_gate_receipt",
        "status": "passed",
        "run_index": run_index,
        "campaign_provenance_path": campaign["artifact_path"],
        "campaign_provenance_sha256": campaign["artifact_sha256"],
        "campaign_bindings_sha256": campaign["bindings_sha256"],
        "smoke_receipts": receipts,
        "interruption_resume_validated": True,
        "full_campaign_storage_projection_validated": True,
        "seven_day_runtime_projection_validated": True,
        "checkpoint_retention_expanded": False,
    }
    if gate_path.is_file():
        observed = _load_json(gate_path)
        if observed == expected:
            result = dict(observed)
            result["gate_path"] = str(gate_path.resolve())
            result["gate_sha256"] = _sha256_file(gate_path)
            result["gate_status"] = "reused"
            return result
    # Recover safely from a crash after both individual smoke receipts were
    # committed but before the aggregate gate was written.
    _atomic_json(expected, gate_path)
    result = dict(expected)
    result["gate_path"] = str(gate_path.resolve())
    result["gate_sha256"] = _sha256_file(gate_path)
    result["gate_status"] = "created"
    return result


def _oriented(values: np.ndarray, direction: str) -> np.ndarray:
    return values if direction == "maximize" else -values


def _cluster_bootstrap(
    score: np.ndarray,
    target: np.ndarray,
    clusters: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> Dict[str, float]:
    unique = np.unique(clusters)
    generator = np.random.default_rng(seed)
    spearman_values: List[float] = []
    kendall_values: List[float] = []
    for _ in range(replicates):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == value) for value in sampled])
        rho = float(spearmanr(score[indices], target[indices]).statistic)
        tau = float(kendalltau(score[indices], target[indices], variant="b").statistic)
        if np.isfinite(rho):
            spearman_values.append(rho)
        if np.isfinite(tau):
            kendall_values.append(tau)
    alpha = (1.0 - confidence) / 2.0
    if not spearman_values or not kendall_values:
        raise OnlineAnalysisError("Cluster bootstrap produced no finite correlations.")
    return {
        "spearman_bootstrap_low": float(np.quantile(spearman_values, alpha)),
        "spearman_bootstrap_high": float(np.quantile(spearman_values, 1.0 - alpha)),
        "kendall_bootstrap_low": float(np.quantile(kendall_values, alpha)),
        "kendall_bootstrap_high": float(np.quantile(kendall_values, 1.0 - alpha)),
    }


def analyze_online_test_results(
    config: Mapping[str, Any], output_root: str | Path
) -> Dict[str, Any]:
    """After selection freeze, join post-hoc test outcomes and compute Steps 10--12."""

    output_root = Path(output_root).expanduser().resolve()
    campaign = load_campaign_provenance_receipt(
        config, pretrained_path=pretrained_provenance_path(config)
    )
    result_dir = output_root / "selection_results"
    selection_summary_path = result_dir / "online_selection_summary.json"
    selection_summary = _load_json(selection_summary_path)
    if (
        selection_summary.get("status") != "complete"
        or selection_summary.get("training_fingerprint")
        != candidate_training_fingerprint(config)
        or selection_summary.get("campaign_provenance_path")
        != campaign["artifact_path"]
        or selection_summary.get("campaign_provenance_sha256")
        != campaign["artifact_sha256"]
        or selection_summary.get("campaign_bindings_sha256")
        != campaign["bindings_sha256"]
        or int(selection_summary.get("candidate_count", -1))
        != int(config["candidate_pool"]["expected_candidate_checkpoints"])
        or selection_summary.get("test_data_accessed") is not False
        or selection_summary.get("test_artifact_tree_opened") is not False
        or selection_summary.get("test_metrics_affected_selection") is not False
    ):
        raise OnlineAnalysisError("Selection was not safely frozen before test analysis.")
    matrix_path = Path(selection_summary["candidate_matrix_path"])
    table_path = Path(selection_summary["selection_table_path"])
    unprivileged_receipt_path = Path(
        selection_summary["unprivileged_freeze_receipt_path"]
    )
    control_diagnostics_summary_path = Path(
        selection_summary["control_diagnostics_summary_path"]
    )
    if (
        _sha256_file(matrix_path) != selection_summary["candidate_matrix_sha256"]
        or _sha256_file(table_path) != selection_summary["selection_table_sha256"]
        or _sha256_file(Path(selection_summary["unprivileged_matrix_path"]))
        != selection_summary["unprivileged_matrix_sha256"]
        or _sha256_file(Path(selection_summary["unprivileged_selection_path"]))
        != selection_summary["unprivileged_selection_sha256"]
        or not unprivileged_receipt_path.is_file()
        or _sha256_file(unprivileged_receipt_path)
        != selection_summary["unprivileged_freeze_receipt_sha256"]
        or not control_diagnostics_summary_path.is_file()
        or _sha256_file(control_diagnostics_summary_path)
        != selection_summary["control_diagnostics_summary_sha256"]
    ):
        raise OnlineAnalysisError("Frozen selection artifacts changed before test join.")
    unprivileged_receipt = _load_json(unprivileged_receipt_path)
    if (
        unprivileged_receipt.get("artifact_type")
        != "fcv_vit_online_unprivileged_freeze_receipt"
        or unprivileged_receipt.get("status") != "complete"
        or unprivileged_receipt.get("campaign_provenance_path")
        != campaign["artifact_path"]
        or unprivileged_receipt.get("campaign_provenance_sha256")
        != campaign["artifact_sha256"]
        or unprivileged_receipt.get("campaign_bindings_sha256")
        != campaign["bindings_sha256"]
        or unprivileged_receipt.get("candidate_matrix_sha256")
        != selection_summary["unprivileged_matrix_sha256"]
        or unprivileged_receipt.get("selection_table_sha256")
        != selection_summary["unprivileged_selection_sha256"]
        or unprivileged_receipt.get("control_diagnostics_summary_sha256")
        != selection_summary["control_diagnostics_summary_sha256"]
        or unprivileged_receipt.get("oracle_artifacts_opened_before_receipt")
        is not False
        or unprivileged_receipt.get("test_artifacts_opened_before_receipt")
        is not False
    ):
        raise OnlineAnalysisError("Unprivileged freeze receipt is invalid.")
    matrix = pd.read_csv(matrix_path)
    if any(str(column).startswith("test_") for column in matrix.columns):
        raise OnlineAnalysisError("Frozen validation matrix already contains test data.")
    selections = pd.read_csv(table_path)
    _validate_post_freeze_checkpoint_cleanup(
        config, output_root, selection_summary, selections
    )
    pool = _load_test_pool(config, output_root, matrix, campaign)
    pool_path = result_dir / "online_test_candidate_matrix.csv"
    _atomic_csv(pool, pool_path)

    test_by_id = pool.set_index("candidate_id")
    final_rows = []
    for row in selections.to_dict("records"):
        candidate_id = str(row["selected_checkpoint_id"])
        if candidate_id not in test_by_id.index:
            raise OnlineAnalysisError(f"Selected candidate has no test result: {candidate_id}")
        test = test_by_id.loc[candidate_id]
        result = dict(row)
        for column in ONLINE_TEST_COLUMNS:
            if column.startswith("test_"):
                result[column] = test[column]
        final_rows.append(result)
    final = pd.DataFrame(final_rows)
    final_path = result_dir / "online_selected_test_results.csv"
    _atomic_csv(final, final_path)

    gap_cfg = config["evaluation"]["gap_closure"]
    metric = "test_worst_group_accuracy"
    selected_values = dict(zip(final["selector_name"], final[metric].astype(float)))
    biased = float(selected_values[gap_cfg["biased_selector"]])
    fcv = float(selected_values[gap_cfg["fcv_selector"]])
    oracle = float(selected_values[gap_cfg["oracle_selector"]])
    denominator = oracle - biased
    epsilon = float(gap_cfg["denominator_epsilon"])
    gap_fraction = None if abs(denominator) <= epsilon else (fcv - biased) / denominator
    best_index = pool[metric].astype(float).idxmax()
    best_row = pool.loc[best_index]
    gap = {
        "biased_selector": gap_cfg["biased_selector"],
        "fcv_selector": gap_cfg["fcv_selector"],
        "oracle_selector": gap_cfg["oracle_selector"],
        "biased_test_worst_group_accuracy": biased,
        "fcv_test_worst_group_accuracy": fcv,
        "oracle_test_worst_group_accuracy": oracle,
        "oracle_gap": denominator,
        "fcv_gain_over_biased": fcv - biased,
        "oracle_gap_fraction_closed": gap_fraction,
        "oracle_gap_percentage_closed": (
            None if gap_fraction is None else 100.0 * gap_fraction
        ),
        "pool_best_candidate_id": str(best_row["candidate_id"]),
        "pool_best_test_worst_group_accuracy": float(best_row[metric]),
        "pool_best_is_posthoc_oracle_only": True,
    }
    gap_path = result_dir / "online_gap_closure_summary.json"
    _atomic_json(gap, gap_path)

    analysis = matrix.merge(
        pool[
            [
                "candidate_id",
                "test_accuracy",
                "test_balanced_group_accuracy",
                "test_worst_group_accuracy",
            ]
        ],
        on="candidate_id",
        validate="one_to_one",
    )
    rank_cfg = config["evaluation"]["rank_analysis"]
    target = analysis["test_worst_group_accuracy"].to_numpy(float)
    candidate_ids = analysis["candidate_id"].astype(str).to_numpy()
    robust_order = np.lexsort((candidate_ids, -target))
    robust_top_ids = candidate_ids[robust_order]
    pool_best_id = str(robust_top_ids[0])
    robust_ranks = dict(
        zip(candidate_ids, rankdata(-target, method="average"))
    )
    selection_by_name = selections.set_index("selector_name")
    rank_rows = []
    selected_ids: Dict[str, str] = {}
    for index, spec in enumerate(rank_cfg["selectors"]):
        name = str(spec["name"])
        column = str(spec["score_column"])
        direction = str(spec["direction"])
        raw = analysis[column].to_numpy(float)
        score = _oriented(raw, direction)
        order = np.lexsort((candidate_ids, -score))
        selected_id = str(selection_by_name.loc[name, "selected_checkpoint_id"])
        selected_ids[name] = selected_id
        analysis[f"oriented_score__{name}"] = score
        if str(candidate_ids[order[0]]) != selected_id:
            raise OnlineAnalysisError(f"Frozen selector does not reproduce: {name}")
        selected_test = float(
            analysis.loc[analysis["candidate_id"] == selected_id, metric].iloc[0]
        )
        row = {
            "selector_name": name,
            "display_name": str(spec["display_name"]),
            "score_column": column,
            "direction": direction,
            "candidate_count": len(analysis),
            "spearman": float(spearmanr(score, target).statistic),
            "kendall_tau_b": float(kendalltau(score, target, variant="b").statistic),
            "selected_candidate_id": selected_id,
            "selected_test_worst_group_accuracy": selected_test,
            "selected_test_robust_rank": float(robust_ranks[selected_id]),
            "selection_regret_to_pool_best": float(best_row[metric]) - selected_test,
            "pool_best_candidate_id": pool_best_id,
            **_cluster_bootstrap(
                score,
                target,
                analysis[str(rank_cfg["clustered_bootstrap"]["cluster_column"])].to_numpy(),
                replicates=int(rank_cfg["clustered_bootstrap"]["replicates"]),
                seed=int(rank_cfg["clustered_bootstrap"]["seed"]) + index,
                confidence=float(rank_cfg["clustered_bootstrap"]["confidence_level"]),
            ),
        }
        for k in rank_cfg["top_k_values"]:
            k = int(k)
            selector_top = set(candidate_ids[order[:k]])
            robust_top = set(robust_top_ids[:k])
            overlap = len(selector_top.intersection(robust_top))
            row[f"top_{k}_overlap_count"] = overlap
            row[f"top_{k}_recall"] = overlap / k
            row[f"top_{k}_hit"] = overlap > 0
        rank_rows.append(row)
    rank = pd.DataFrame(rank_rows)
    rank_path = result_dir / "online_rank_correlation_results.csv"
    candidate_rank_path = result_dir / "online_candidate_rank_analysis.csv"
    _atomic_csv(rank, rank_path)
    _atomic_csv(analysis, candidate_rank_path)
    plot_paths: List[Path] = []
    if bool(rank_cfg["create_scatter_plots"]):
        from .rank_analysis import _plot_selector_scatters

        analysis["fcv_main_score"] = analysis["primary_selector_score"]
        plot_paths = _plot_selector_scatters(
            analysis,
            rank_cfg["selectors"],
            selected_ids,
            pool_best_id,
            output_root / str(config["outputs"]["plots"]),
        )
    summary_path = result_dir / "online_posthoc_analysis_summary.json"
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_posthoc_analysis_summary",
        "status": "complete",
        "campaign_provenance_path": campaign["artifact_path"],
        "campaign_provenance_sha256": campaign["artifact_sha256"],
        "campaign_bindings_sha256": campaign["bindings_sha256"],
        "candidate_count": len(analysis),
        "selector_count": len(final),
        "selection_summary_path": str(selection_summary_path.resolve()),
        "selection_summary_sha256": _sha256_file(selection_summary_path),
        "frozen_matrix_sha256": _sha256_file(matrix_path),
        "test_pool_path": str(pool_path.resolve()),
        "test_pool_sha256": _sha256_file(pool_path),
        "selected_test_results_path": str(final_path.resolve()),
        "selected_test_results_sha256": _sha256_file(final_path),
        "gap_closure_path": str(gap_path.resolve()),
        "gap_closure_sha256": _sha256_file(gap_path),
        "rank_results_path": str(rank_path.resolve()),
        "rank_results_sha256": _sha256_file(rank_path),
        "candidate_rank_path": str(candidate_rank_path.resolve()),
        "candidate_rank_sha256": _sha256_file(candidate_rank_path),
        "control_diagnostics_summary_path": str(
            control_diagnostics_summary_path.resolve()
        ),
        "control_diagnostics_summary_sha256": _sha256_file(
            control_diagnostics_summary_path
        ),
        "control_candidate_warning_count": selection_summary[
            "control_candidate_warning_count"
        ],
        "control_total_warning_count": selection_summary[
            "control_total_warning_count"
        ],
        "scatter_plot_paths": [str(path.resolve()) for path in plot_paths],
        "scatter_plot_sha256": {
            str(path.resolve()): _sha256_file(path) for path in plot_paths
        },
        "selection_frozen_before_test_artifacts_opened": True,
        "test_metrics_affected_selection": False,
        "posthoc_analysis_only": True,
        "analysis_fingerprint": _sha256_json(
            {
                "selection": _sha256_file(selection_summary_path),
                "test_pool": _sha256_file(pool_path),
                "config_training": candidate_training_fingerprint(config),
            }
        ),
    }
    _atomic_json(summary, summary_path)
    return summary
