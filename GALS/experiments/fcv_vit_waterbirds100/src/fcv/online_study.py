"""Online all-epoch training and scoring for the ViT-FCV first study.

Every epoch is a candidate, but ordinary candidate checkpoints and token banks
are temporary.  The audited file-oriented FCV/control evaluators run against a
node-local checkpoint while that epoch is current.  Only running per-run
winners for the three primary selectors are copied to persistent storage.

Test metrics are computed for every candidate for post-hoc rank analysis, but
they are written to a separate analysis-only artifact tree and are never
returned to the training/retention decision code.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .candidate_training import (
    CandidateTrainingError,
    PublicManifestDataset,
    SweepRun,
    _atomic_csv,
    _atomic_json,
    _atomic_torch_save,
    _candidate_checkpoint_payload,
    _load_trusted_checkpoint,
    _optimizer_to_device,
    _recursive_cpu,
    _restore_rng_state,
    _rng_state,
    _run_epoch,
    build_dataloaders,
    build_model,
    build_scheduler,
    candidate_training_fingerprint,
    enumerate_sweep_runs,
    load_pretrained_cache_provenance,
    pretrained_backbone_sha256,
    seed_everything,
    software_fingerprint,
    software_versions,
    source_tree_provenance,
    state_dict_sha256,
)
from .config import candidate_epochs
from .campaign_provenance import load_campaign_provenance_receipt
from .controls import CONTROL_NAMES, prepare_control_plan, score_candidate_controls
from .fcv_scoring import (
    load_background_bank,
    prepare_opposite_donor_plan,
    score_candidate_fcv,
)
from .manifest_provenance import ManifestProvenanceError, validate_manifest_bundle
from .online_schema import (
    ONLINE_TEST_COLUMNS,
    ONLINE_VALIDATION_COLUMNS,
    RETAINED_SELECTOR_SPECS,
)
from .selectors import (
    OracleValidationSource,
    evaluate_candidate_oracle,
    prepare_oracle_validation_source,
)
from .storage import assert_storage_budget
from .test_evaluation import (
    FinalTestSource,
    evaluate_checkpoint_test_metrics,
    prepare_final_test_source,
    recompute_test_metrics_from_frame,
)
from .token_banks import (
    CONTEXT_NAMES,
    TokenBankSource,
    build_background_token_banks,
    prepare_token_bank_source,
)


class OnlineStudyError(RuntimeError):
    """Raised when online scoring, retention, or recovery is ambiguous."""


@contextmanager
def _isolated_training_rng(train_generator: torch.Generator):
    """Prevent analysis-only evaluation from changing committed training RNG."""

    state = _rng_state(train_generator)
    try:
        yield state
    finally:
        _restore_rng_state(state, train_generator)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise OnlineStudyError(f"Expected a JSON mapping: {path}")
    return payload


def _test_index_prefix(
    path: Path,
    run: SweepRun,
    committed_epochs: int,
    *,
    allow_one_uncommitted_row: bool,
) -> pd.DataFrame:
    """Validate and return the committed prefix of the analysis-only index."""

    if committed_epochs == 0 and not path.is_file():
        return pd.DataFrame(columns=ONLINE_TEST_COLUMNS)
    if not path.is_file():
        raise OnlineStudyError(f"Missing committed analysis-only test index: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != list(ONLINE_TEST_COLUMNS):
        raise OnlineStudyError(f"Analysis-only test index schema changed: {path}")
    upper = committed_epochs + int(allow_one_uncommitted_row)
    if len(frame) < committed_epochs or len(frame) > upper:
        raise OnlineStudyError(
            f"Analysis-only test index is not a recoverable prefix: {path}"
        )
    prefix = frame.iloc[:committed_epochs].copy().reset_index(drop=True)
    expected_epochs = list(range(1, committed_epochs + 1))
    if (
        prefix["epoch"].astype(int).tolist() != expected_epochs
        or prefix["candidate_id"].astype(str).tolist()
        != [run.candidate_id(epoch) for epoch in expected_epochs]
        or (not prefix.empty and set(prefix["run_index"].astype(int)) != {run.run_index})
    ):
        raise OnlineStudyError(f"Analysis-only test index identity changed: {path}")
    return prefix


def _restore_committed_test_index(
    path: Path,
    run: SweepRun,
    committed_epochs: int,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> pd.DataFrame:
    """Validate or byte-truncate one uncommitted row without CSV reserialization."""

    prefix = _test_index_prefix(
        path,
        run,
        committed_epochs,
        allow_one_uncommitted_row=True,
    )
    observed_rows = len(pd.read_csv(path))
    expected_size_bytes = int(expected_size_bytes)
    if len(expected_sha256) != 64 or expected_size_bytes <= 0:
        raise OnlineStudyError("Resume has no valid analysis-only index byte binding.")
    if observed_rows == committed_epochs:
        if (
            path.stat().st_size != expected_size_bytes
            or _sha256_file(path) != expected_sha256
        ):
            raise OnlineStudyError(
                "Committed analysis-only test index bytes changed."
            )
        return prefix

    # A crash may leave exactly one analysis-only row written after the last
    # optimizer-bearing resume commit. The resume stores the committed byte
    # length and hash, so remove the suffix byte-for-byte. Never parse and
    # rewrite committed float text: pandas round trips are not byte-stable
    # across all supported builds.
    payload = path.read_bytes()
    if expected_size_bytes >= len(payload):
        raise OnlineStudyError(
            "Uncommitted analysis-only row has no recoverable byte suffix."
        )
    committed = payload[:expected_size_bytes]
    if hashlib.sha256(committed).hexdigest() != expected_sha256:
        raise OnlineStudyError(
            "Analysis-only test index committed byte prefix changed."
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(committed)
    temporary.replace(path)
    restored = _test_index_prefix(
        path,
        run,
        committed_epochs,
        allow_one_uncommitted_row=False,
    )
    if path.stat().st_size != expected_size_bytes or _sha256_file(path) != expected_sha256:
        raise OnlineStudyError("Analysis-only test index byte rollback failed.")
    return restored


def _append_test_index_row(
    path: Path,
    run: SweepRun,
    committed_epochs: int,
    row: Mapping[str, Any],
) -> None:
    """Append one post-hoc row without carrying prior test metrics into training."""

    _test_index_prefix(
        path,
        run,
        committed_epochs,
        allow_one_uncommitted_row=False,
    )
    incoming = pd.DataFrame([dict(row)], columns=ONLINE_TEST_COLUMNS)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if committed_epochs == 0:
        incoming.to_csv(temporary, index=False)
    else:
        committed = path.read_bytes()
        if not committed.endswith(b"\n"):
            raise OnlineStudyError(
                "Committed analysis-only test index has no row boundary."
            )
        temporary.write_bytes(committed)
        incoming.to_csv(temporary, mode="a", header=False, index=False)
    temporary.replace(path)


def _manifest_hashes(
    config: Mapping[str, Any], train_manifest: Path, val_manifest: Path
) -> Dict[str, str]:
    try:
        train = validate_manifest_bundle(config, train_manifest, "candidate_train")
        val = validate_manifest_bundle(config, val_manifest, "biased_validation")
    except ManifestProvenanceError as exc:
        raise OnlineStudyError(str(exc)) from exc
    if train.bundle_sha256 != val.bundle_sha256:
        raise OnlineStudyError("Train and biased validation use different bundles.")
    return {
        "candidate_train": train.manifest_sha256,
        "biased_validation": val.manifest_sha256,
        "manifest_bundle": train.bundle_sha256,
        "original_metadata": train.original_metadata_sha256,
        "split_indices": train.split_indices_sha256,
        "split_summary": train.split_summary_sha256,
    }


def _validate_online_config(config: Mapping[str, Any]) -> None:
    epochs = int(config["training"]["epochs"])
    if candidate_epochs(config) != list(range(1, epochs + 1)):
        raise OnlineStudyError("Online study requires every epoch to be a candidate.")
    pool = config["candidate_pool"]
    if pool.get("unit") != "online_epoch_state":
        raise OnlineStudyError("candidate_pool.unit must be online_epoch_state.")
    if int(pool.get("expected_candidate_checkpoints", -1)) != (
        len(enumerate_sweep_runs(config)) * epochs
    ):
        raise OnlineStudyError("Expected online candidate count is inconsistent.")
    if pool.get("retained_checkpoint_selectors") != list(RETAINED_SELECTOR_SPECS):
        raise OnlineStudyError("Retained-selector order differs from the protocol.")
    if int(pool.get("max_retained_checkpoints_per_run", -1)) != 3:
        raise OnlineStudyError("Online study must cap retained checkpoints at three.")
    if int(pool.get("max_transient_retained_checkpoints_per_run", -1)) != 4:
        raise OnlineStudyError(
            "Crash-safe retention permits only one transient replacement checkpoint."
        )
    if (
        pool.get("post_freeze_checkpoint_retention")
        != "global_primary_selector_winners_only"
        or int(pool.get("max_final_retained_checkpoints", -1)) != 3
    ):
        raise OnlineStudyError(
            "Post-freeze retention must keep at most the three global primary winners."
        )
    forbidden = [
        column for column in ONLINE_VALIDATION_COLUMNS if column.startswith("test_")
    ]
    if forbidden:
        raise OnlineStudyError(f"Validation schema leaks test columns: {forbidden}")


def _probability_retention_ratio(score_path: Path, epsilon: float) -> float:
    frame = pd.read_csv(score_path)
    eligible = frame[frame["fcv_eligible"].astype(str).str.lower().isin({"true", "1"})]
    if eligible.empty:
        raise OnlineStudyError("FCV score has no eligible rows.")
    original = pd.to_numeric(eligible["p_y_original"], errors="raise").to_numpy(float)
    counterfactual = pd.to_numeric(
        eligible["p_y_counterfactual_mean"], errors="raise"
    ).to_numpy(float)
    ratio = counterfactual / np.maximum(original, epsilon)
    if not np.isfinite(ratio).all():
        raise OnlineStudyError("FCV probability-retention ratio is non-finite.")
    return float(ratio.mean())


def _prepare_online_intervention_plans(
    config: Mapping[str, Any],
    token_source: TokenBankSource,
    bank_dir: Path,
    candidate_id: str,
    checkpoint_sha256: str,
    donor_plan_path: Path,
    control_plan_path: Path,
) -> None:
    """Create either missing plan without coupling their commit boundaries."""

    donor_missing = not donor_plan_path.is_file()
    control_missing = not control_plan_path.is_file()
    if not donor_missing and not control_missing:
        return
    banks = {
        label: load_background_bank(
            config,
            bank_dir / f"{candidate_id}_{context_name}.pt",
            token_source,
            expected_label=label,
            expected_candidate_id=candidate_id,
            expected_checkpoint_sha256=checkpoint_sha256,
        )
        for label, context_name in CONTEXT_NAMES.items()
    }
    donor_plan = prepare_opposite_donor_plan(
        config, token_source, banks, donor_plan_path
    )
    prepare_control_plan(
        config,
        token_source,
        banks,
        donor_plan,
        donor_plan_path,
        control_plan_path,
        # If the donor artifact itself vanished, rebuild its dependent control
        # plan as a deterministic pair. In the ordinary interrupted case the
        # donor exists and only the missing control plan is created.
        overwrite=donor_missing and control_plan_path.is_file(),
    )
    del banks, donor_plan


def _local_winners(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Select exact per-run winners with the predeclared candidate-ID tie break."""

    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    result: Dict[str, str] = {}
    for selector, (column, direction) in RETAINED_SELECTOR_SPECS.items():
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise OnlineStudyError(f"Non-finite local selector: {selector}")
        best = values.max() if direction == "maximize" else values.min()
        tied = frame.loc[values == best].sort_values("candidate_id", kind="stable")
        result[selector] = str(tied.iloc[0]["candidate_id"])
    return result


def _retention_payload(
    rows: Sequence[Mapping[str, Any]], retained_dir: Path
) -> Dict[str, Any]:
    winners = _local_winners(rows)
    unique_ids = sorted(set(winners.values()))
    if len(unique_ids) > 3:
        raise OnlineStudyError("Local retention exceeded three unique checkpoints.")
    by_id = {str(row["candidate_id"]): row for row in rows}
    checkpoints: Dict[str, Dict[str, Any]] = {}
    for candidate_id in unique_ids:
        path = retained_dir / f"{candidate_id}.pt"
        expected_sha = str(by_id[candidate_id]["checkpoint_sha256"])
        if not path.is_file() or _sha256_file(path) != expected_sha:
            raise OnlineStudyError(
                f"Missing retained local winner {candidate_id}: {path}"
            )
        checkpoints[candidate_id] = {
            "path": str(path.resolve()),
            "sha256": expected_sha,
            "selectors": sorted(
                selector for selector, winner in winners.items() if winner == candidate_id
            ),
        }
    return {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_local_retention",
        "status": "complete",
        "selectors": winners,
        "unique_checkpoint_count": len(unique_ids),
        "checkpoints": checkpoints,
    }


def _stage_retention(
    rows: Sequence[Mapping[str, Any]],
    current_checkpoint: Path,
    retained_dir: Path,
) -> Dict[str, Any]:
    """Copy a new winner before the epoch commit; never delete an old winner here."""

    retained_dir.mkdir(parents=True, exist_ok=True)
    winners = _local_winners(rows)
    current_id = str(rows[-1]["candidate_id"])
    if current_id in set(winners.values()):
        destination = retained_dir / f"{current_id}.pt"
        if destination.exists() and _sha256_file(destination) != _sha256_file(
            current_checkpoint
        ):
            raise OnlineStudyError(f"Retained checkpoint collision: {destination}")
        if not destination.exists():
            shutil.copy2(current_checkpoint, destination)
    return _retention_payload(rows, retained_dir)


def _prune_retention(retention: Mapping[str, Any], retained_dir: Path) -> None:
    keep = {
        Path(str(details["path"])).resolve()
        for details in retention.get("checkpoints", {}).values()
    }
    for path in retained_dir.glob("*.pt"):
        if path.resolve() not in keep:
            path.unlink()


def _test_artifacts(
    config: Mapping[str, Any],
    run: SweepRun,
    epoch: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source: FinalTestSource,
    output_dir: Path,
    campaign_provenance: Mapping[str, Any],
    *,
    device: str,
) -> Dict[str, Any]:
    """Persist post-hoc-only test evidence without exposing it to retention code."""

    candidate_id = run.candidate_id(epoch)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / f"{candidate_id}_test_per_image.csv"
    summary_path = output_dir / f"{candidate_id}_test_summary.json"
    metrics, training_fingerprint, frame = evaluate_checkpoint_test_metrics(
        config, candidate_id, checkpoint_path, source, device=device
    )
    _atomic_csv(frame, per_image_path)
    metrics = recompute_test_metrics_from_frame(
        pd.read_csv(per_image_path), source, candidate_id=candidate_id
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_posthoc_test_summary",
        "status": "complete",
        "candidate_id": candidate_id,
        "run": asdict(run),
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "training_fingerprint": training_fingerprint,
        "campaign_provenance_path": campaign_provenance["artifact_path"],
        "campaign_provenance_sha256": campaign_provenance["artifact_sha256"],
        "campaign_bindings_sha256": campaign_provenance["bindings_sha256"],
        "pretrained_provenance_path": campaign_provenance["bindings"][
            "pretrained"
        ]["path"],
        "pretrained_provenance_sha256": campaign_provenance["bindings"][
            "pretrained"
        ]["sha256"],
        "pretrained_backbone_sha256": campaign_provenance["bindings"][
            "pretrained"
        ]["backbone_sha256"],
        "initial_model_state_sha256": campaign_provenance["bindings"][
            "initialization"
        ]["initial_model_state_sha256_by_seed"][str(run.seed)],
        "test_manifest_path": str(source.manifest_path),
        "test_manifest_sha256": source.manifest_sha256,
        "manifest_bundle_sha256": source.manifest_bundle_sha256,
        "metrics": metrics,
        "per_image_csv_path": str(per_image_path.resolve()),
        "per_image_csv_sha256": _sha256_file(per_image_path),
        "execution": {
            "batch_size": source.batch_size,
            "num_workers": source.num_workers,
            "precision": str(config["evaluation"]["final_test"]["precision"]),
        },
        "test_data_accessed": True,
        "posthoc_analysis_only": True,
        "eligible_for_model_selection": False,
        "test_metrics_available_to_training_or_retention": False,
        "test_metrics_affected_selection": False,
    }
    _atomic_json(summary, summary_path)
    del frame
    return {
        "run_index": run.run_index,
        "candidate_id": candidate_id,
        "epoch": epoch,
        "seed": run.seed,
        "learning_rate": run.learning_rate,
        "weight_decay": run.weight_decay,
        "checkpoint_sha256": checkpoint_sha256,
        "test_loss": float(metrics["loss"]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_balanced_group_accuracy": float(metrics["balanced_group_accuracy"]),
        "test_worst_group_accuracy": float(metrics["worst_group_accuracy"]),
        **{
            f"test_group_{group}_accuracy": float(metrics[f"group_{group}_accuracy"])
            for group in range(4)
        },
        **{
            f"test_group_{group}_count": int(metrics[f"group_{group}_count"])
            for group in range(4)
        },
        "per_image_path": str(per_image_path.resolve()),
        "per_image_sha256": _sha256_file(per_image_path),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256_file(summary_path),
    }


def _validation_row(
    config: Mapping[str, Any],
    run: SweepRun,
    epoch: int,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    lr_epoch_start: float,
    lr_epoch_end: float,
    checkpoint_sha256: str,
    train_seconds: float,
    score_seconds: float,
    fcv_summary: Mapping[str, Any],
    control_summary: Mapping[str, Any],
    oracle_summary: Mapping[str, Any],
    fcv_summary_path: Path,
    controls_summary_path: Path,
    oracle_summary_path: Path,
) -> Dict[str, Any]:
    controls = control_summary["controls"]
    same = controls["same_context"]
    token_means = fcv_summary["token_distribution_diagnostics"]["global_means"]
    real_swap = fcv_summary["real_swap_integrity_diagnostics"]
    opposite_drop = float(fcv_summary["mean_counterfactual_confidence_drop"])
    shortcut_sensitivity = opposite_drop - float(same["mean_confidence_drop"])
    selector_cfg = config["fcv"]["selector_analysis"]
    fcv_score_path = Path(str(fcv_summary["score_csv_path"]))
    row = {
        "run_index": run.run_index,
        "candidate_id": run.candidate_id(epoch),
        "epoch": epoch,
        "model_name": str(config["model"]["name"]),
        "seed": run.seed,
        "learning_rate": run.learning_rate,
        "weight_decay": run.weight_decay,
        "train_loss": float(train_metrics["loss"]),
        "train_accuracy": float(train_metrics["accuracy"]),
        # The serialized per-example FCV evidence is the canonical selector
        # reduction. The ordinary batch-reduced value remains diagnostic only;
        # score_candidate_fcv already requires the two to agree within 1e-6.
        "biased_val_loss": float(
            fcv_summary["biased_validation_loss_recomputed"]
        ),
        "biased_val_loss_batch_reduced_diagnostic": float(
            validation_metrics["loss"]
        ),
        "biased_val_accuracy": float(validation_metrics["accuracy"]),
        "lr_epoch_start": lr_epoch_start,
        "lr_epoch_end": lr_epoch_end,
        "checkpoint_sha256": checkpoint_sha256,
        "epoch_train_seconds": train_seconds,
        "epoch_online_score_seconds": score_seconds,
        "fcv_counterfactual_accuracy": float(
            fcv_summary["opposite_context_counterfactual_accuracy"]
        ),
        "fcv_counterfactual_majority_accuracy": float(
            fcv_summary["opposite_context_counterfactual_majority_accuracy"]
        ),
        "fcv_true_class_probability": float(
            fcv_summary["opposite_context_true_class_probability"]
        ),
        "fcv_probability_retention_ratio": _probability_retention_ratio(
            fcv_score_path,
            float(selector_cfg["probability_ratio_epsilon"]),
        ),
        "fcv_confidence_drop": opposite_drop,
        "primary_selector_score": float(fcv_summary["primary_selector_score"]),
        "same_context_counterfactual_accuracy": float(
            same["counterfactual_accuracy"]
        ),
        "same_context_mean_confidence_drop": float(same["mean_confidence_drop"]),
        "random_mask_counterfactual_accuracy": float(
            controls["random_mask"]["counterfactual_accuracy"]
        ),
        "shuffled_mask_counterfactual_accuracy": float(
            controls["shuffled_mask"]["counterfactual_accuracy"]
        ),
        "evidence_swap_counterfactual_accuracy": float(
            controls["evidence_swap"]["counterfactual_accuracy"]
        ),
        "control_diagnostic_warning_count": int(
            control_summary["diagnostic_warning_count"]
        ),
        "control_diagnostic_status": str(control_summary["diagnostic_status"]),
        "target_donor_cosine_similarity_mean": float(
            token_means["target_donor_cosine_similarity_mean"]
        ),
        "target_nearest_donor_cosine_mean": float(
            token_means["target_nearest_donor_cosine_mean"]
        ),
        "donor_unique_source_images_mean": float(
            token_means["donor_unique_source_images"]
        ),
        "donor_max_source_fraction_mean": float(
            token_means["donor_max_source_fraction"]
        ),
        "real_swap_replaced_token_changed_fraction": float(
            real_swap["replaced_token_changed_fraction"]
        ),
        "real_swap_replacement_delta_mean": float(
            real_swap["replacement_delta_mean"]
        ),
        "real_swap_replacement_delta_max": float(
            real_swap["replacement_delta_max"]
        ),
        "real_swap_foreground_token_max_abs_error": float(
            real_swap["foreground_token_max_abs_error"]
        ),
        "real_swap_donor_reconstruction_max_abs_error": float(
            real_swap["donor_reconstruction_max_abs_error"]
        ),
        "shortcut_sensitivity": shortcut_sensitivity,
        "control_normalized_fcv_score": float(
            validation_metrics["accuracy"]
            - float(selector_cfg["control_normalized_lambda"])
            * shortcut_sensitivity
        ),
        "oracle_validation_loss": float(oracle_summary["metrics"]["loss"]),
        "oracle_validation_accuracy": float(
            oracle_summary["metrics"]["accuracy"]
        ),
        "oracle_validation_balanced_group_accuracy": float(
            oracle_summary["metrics"]["balanced_group_accuracy"]
        ),
        "oracle_validation_worst_group_accuracy": float(
            oracle_summary["metrics"]["worst_group_accuracy"]
        ),
        **{
            f"oracle_group_{group}_accuracy": float(
                oracle_summary["metrics"][f"group_{group}_accuracy"]
            )
            for group in range(4)
        },
        "fcv_summary_path": str(fcv_summary_path.resolve()),
        "fcv_summary_sha256": _sha256_file(fcv_summary_path),
        "controls_summary_path": str(controls_summary_path.resolve()),
        "controls_summary_sha256": _sha256_file(controls_summary_path),
        "oracle_summary_path": str(oracle_summary_path.resolve()),
        "oracle_summary_sha256": _sha256_file(oracle_summary_path),
    }
    if any(str(key).startswith("test_") for key in row):
        raise OnlineStudyError("Validation row contains a test metric.")
    return row


def _require_hashed_file(path_value: Any, sha_value: Any, context: str) -> Path:
    path = Path(str(path_value)).expanduser().resolve()
    if (
        not path.is_file()
        or len(str(sha_value)) != 64
        or _sha256_file(path) != str(sha_value)
    ):
        raise OnlineStudyError(f"Missing or changed completed-run artifact: {context}")
    return path


def _validate_completed_detail_artifacts(
    run: SweepRun,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    campaign_provenance: Mapping[str, Any],
) -> None:
    """Validate every compact-row binding before declaring a run reusable."""

    for row in validation.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        checkpoint_sha = str(row.checkpoint_sha256)
        fcv_path = _require_hashed_file(
            row.fcv_summary_path, row.fcv_summary_sha256, f"{candidate_id}.fcv"
        )
        fcv = _load_json(fcv_path)
        fcv_csv = _require_hashed_file(
            fcv.get("score_csv_path"),
            fcv.get("score_csv_sha256"),
            f"{candidate_id}.fcv_per_image",
        )
        if (
            fcv.get("candidate_id") != candidate_id
            or fcv.get("checkpoint_sha256") != checkpoint_sha
            or fcv.get("campaign_provenance_path")
            != campaign_provenance["artifact_path"]
            or fcv.get("campaign_provenance_sha256")
            != campaign_provenance["artifact_sha256"]
            or fcv.get("campaign_bindings_sha256")
            != campaign_provenance["bindings_sha256"]
            or fcv.get("training_fingerprint")
            != campaign_provenance["bindings"]["training_fingerprint"]
            or fcv.get("software_versions")
            != campaign_provenance["bindings"]["software_versions"]
            or fcv.get("pretrained_provenance_path")
            != campaign_provenance["bindings"]["pretrained"]["path"]
            or fcv.get("pretrained_provenance_sha256")
            != campaign_provenance["bindings"]["pretrained"]["sha256"]
            or fcv.get("pretrained_backbone_sha256")
            != campaign_provenance["bindings"]["pretrained"]["backbone_sha256"]
            or fcv.get("initial_model_state_sha256")
            != campaign_provenance["bindings"]["initialization"][
                "initial_model_state_sha256_by_seed"
            ].get(str(run.seed))
            or fcv_csv.stat().st_size != int(fcv.get("score_csv_size_bytes", -1))
        ):
            raise OnlineStudyError(f"Invalid completed FCV binding: {candidate_id}")

        controls_path = _require_hashed_file(
            row.controls_summary_path,
            row.controls_summary_sha256,
            f"{candidate_id}.controls",
        )
        controls = _load_json(controls_path)
        files = controls.get("score_csvs")
        if (
            controls.get("candidate_id") != candidate_id
            or controls.get("checkpoint_sha256") != checkpoint_sha
            or controls.get("campaign_provenance_path")
            != campaign_provenance["artifact_path"]
            or controls.get("campaign_provenance_sha256")
            != campaign_provenance["artifact_sha256"]
            or controls.get("campaign_bindings_sha256")
            != campaign_provenance["bindings_sha256"]
            or controls.get("initial_model_state_sha256")
            != campaign_provenance["bindings"]["initialization"][
                "initial_model_state_sha256_by_seed"
            ].get(str(run.seed))
            or not isinstance(files, Mapping)
            or set(files) != set(CONTROL_NAMES)
        ):
            raise OnlineStudyError(f"Invalid completed control binding: {candidate_id}")
        for name in CONTROL_NAMES:
            details = files[name]
            if not isinstance(details, Mapping):
                raise OnlineStudyError(
                    f"Invalid completed {name} control binding: {candidate_id}"
                )
            path = _require_hashed_file(
                details.get("path"),
                details.get("sha256"),
                f"{candidate_id}.{name}_per_image",
            )
            if path.stat().st_size != int(details.get("size_bytes", -1)):
                raise OnlineStudyError(
                    f"Changed completed {name} control size: {candidate_id}"
                )

        oracle_path = _require_hashed_file(
            row.oracle_summary_path,
            row.oracle_summary_sha256,
            f"{candidate_id}.oracle",
        )
        oracle = _load_json(oracle_path)
        _require_hashed_file(
            oracle.get("per_image_csv_path"),
            oracle.get("per_image_csv_sha256"),
            f"{candidate_id}.oracle_per_image",
        )
        if (
            oracle.get("candidate_id") != candidate_id
            or oracle.get("checkpoint_sha256") != checkpoint_sha
            or oracle.get("campaign_provenance_path")
            != campaign_provenance["artifact_path"]
            or oracle.get("campaign_provenance_sha256")
            != campaign_provenance["artifact_sha256"]
            or oracle.get("campaign_bindings_sha256")
            != campaign_provenance["bindings_sha256"]
            or oracle.get("initial_model_state_sha256")
            != campaign_provenance["bindings"]["initialization"][
                "initial_model_state_sha256_by_seed"
            ].get(str(run.seed))
        ):
            raise OnlineStudyError(f"Invalid completed Oracle binding: {candidate_id}")

    for row in test.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        checkpoint_sha = str(row.checkpoint_sha256)
        per_image = _require_hashed_file(
            row.per_image_path, row.per_image_sha256, f"{candidate_id}.test_per_image"
        )
        summary_path = _require_hashed_file(
            row.summary_path, row.summary_sha256, f"{candidate_id}.test_summary"
        )
        summary = _load_json(summary_path)
        if (
            summary.get("candidate_id") != candidate_id
            or summary.get("checkpoint_sha256") != checkpoint_sha
            or summary.get("per_image_csv_path") != str(per_image)
            or summary.get("per_image_csv_sha256") != str(row.per_image_sha256)
            or summary.get("campaign_provenance_sha256")
            != campaign_provenance["artifact_sha256"]
            or summary.get("campaign_bindings_sha256")
            != campaign_provenance["bindings_sha256"]
            or summary.get("initial_model_state_sha256")
            != campaign_provenance["bindings"]["initialization"][
                "initial_model_state_sha256_by_seed"
            ].get(str(run.seed))
            or summary.get("posthoc_analysis_only") is not True
            or summary.get("eligible_for_model_selection") is not False
        ):
            raise OnlineStudyError(f"Invalid completed test binding: {candidate_id}")
    expected_epochs = list(range(1, len(validation) + 1))
    if (
        validation["run_index"].astype(int).tolist()
        != [run.run_index] * len(validation)
        or test["run_index"].astype(int).tolist() != [run.run_index] * len(test)
        or validation["epoch"].astype(int).tolist() != expected_epochs
        or test["epoch"].astype(int).tolist() != expected_epochs
    ):
        raise OnlineStudyError(f"Completed candidate identities changed: {run.run_id}")


def _invalidate_completed_run_for_bounded_repair(
    run_dir: Path, reason: str
) -> None:
    """Discard only bounded run state so all details can be regenerated online."""

    summary_path = run_dir / "run_summary.json"
    receipt_path = run_dir / "bounded_repair_receipt.json"
    receipt = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_online_bounded_repair",
        "status": "run_state_invalidated",
        "reason": reason,
        "run_summary_sha256": (
            _sha256_file(summary_path) if summary_path.is_file() else None
        ),
        "checkpoint_retention_expanded": False,
        "repair_strategy": "rerun_twenty_epochs_with_existing_bounded_retention",
    }
    _atomic_json(receipt, receipt_path)
    for path in (
        summary_path,
        run_dir / "validation_metrics.csv",
        run_dir / "test_metrics_analysis_only.csv",
        run_dir / "retention_state.json",
        run_dir / "resume_state.pt",
    ):
        if path.exists():
            path.unlink()
    for path in (run_dir / "retained_checkpoints").glob("*.pt"):
        path.unlink()


def _completed_run(
    config: Mapping[str, Any],
    run: SweepRun,
    run_dir: Path,
    output_root: Path,
    campaign_provenance: Mapping[str, Any],
    manifest_hashes: Mapping[str, str],
    pretrained: Mapping[str, Any],
) -> Dict[str, Any] | None:
    summary_path = run_dir / "run_summary.json"
    validation_path = run_dir / "validation_metrics.csv"
    test_path = run_dir / "test_metrics_analysis_only.csv"
    retention_path = run_dir / "retention_state.json"
    if not summary_path.is_file():
        return None
    summary = _load_json(summary_path)
    expected = int(config["training"]["epochs"])
    valid = (
        summary.get("artifact_type") == "fcv_vit_online_run_summary"
        and summary.get("status") == "complete"
        and summary.get("run") == asdict(run)
        and summary.get("training_fingerprint")
        == candidate_training_fingerprint(config)
        and summary.get("software_versions") == software_versions()
        and summary.get("software_fingerprint") == software_fingerprint()
        and summary.get("source_tree_sha256")
        == source_tree_provenance()["source_tree_sha256"]
        and summary.get("campaign_provenance_path")
        == campaign_provenance["artifact_path"]
        and summary.get("campaign_provenance_sha256")
        == campaign_provenance["artifact_sha256"]
        and summary.get("campaign_bindings_sha256")
        == campaign_provenance["bindings_sha256"]
        and summary.get("manifest_sha256") == dict(manifest_hashes)
        and summary.get("pretrained_provenance_path")
        == pretrained["artifact_path"]
        and summary.get("pretrained_provenance_sha256")
        == pretrained["artifact_sha256"]
        and summary.get("pretrained_backbone_sha256")
        == pretrained["pretrained_backbone_sha256"]
        and summary.get("initial_model_state_sha256")
        == campaign_provenance["bindings"]["initialization"][
            "initial_model_state_sha256_by_seed"
        ].get(str(run.seed))
        and int(summary.get("candidate_count", -1)) == expected
        and validation_path.is_file()
        and test_path.is_file()
        and retention_path.is_file()
        and summary.get("validation_metrics_sha256") == _sha256_file(validation_path)
        and summary.get("test_metrics_sha256") == _sha256_file(test_path)
        and summary.get("retention_state_sha256") == _sha256_file(retention_path)
    )
    selection_started = any(
        (output_root / "selection_results" / name).is_file()
        for name in (
            "online_unprivileged_freeze_receipt.json",
            "online_checkpoint_cleanup_plan.json",
            "online_selection_summary.json",
        )
    )
    try:
        if not valid:
            raise OnlineStudyError(f"Completed online run is stale: {run_dir}")
        validation = pd.read_csv(validation_path)
        test = pd.read_csv(test_path)
        if len(validation) != expected or len(test) != expected:
            raise OnlineStudyError(
                f"Completed online run has an incomplete matrix: {run_dir}"
            )
        _validate_completed_detail_artifacts(
            run, validation, test, campaign_provenance
        )
        if not selection_started:
            retention = _load_json(retention_path)
            checkpoints = retention.get("checkpoints")
            if not isinstance(checkpoints, Mapping):
                raise OnlineStudyError(f"Invalid completed retention: {run.run_id}")
            for candidate_id, details in checkpoints.items():
                if not isinstance(details, Mapping):
                    raise OnlineStudyError(
                        f"Invalid completed retained checkpoint: {candidate_id}"
                    )
                _require_hashed_file(
                    details.get("path"),
                    details.get("sha256"),
                    f"{candidate_id}.retained_checkpoint",
                )
    except (OSError, ValueError, TypeError, KeyError, OnlineStudyError) as exc:
        if selection_started:
            raise OnlineStudyError(
                "A frozen campaign artifact is stale and cannot be regenerated after "
                f"selection began: {run.run_id}: {exc}"
            ) from exc
        _invalidate_completed_run_for_bounded_repair(run_dir, str(exc))
        return None
    resume = run_dir / "resume_state.pt"
    if resume.exists():
        resume.unlink()
    return summary


def train_and_score_online_run(
    config: Mapping[str, Any],
    run: SweepRun,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    patch_masks: str | Path,
    oracle_manifest: str | Path,
    test_manifest: str | Path,
    output_root: str | Path,
    *,
    device_name: str = "cuda",
    pretrained_provenance_path: str | Path | None = None,
    stop_after_epoch: int | None = None,
) -> Dict[str, Any]:
    """Train one sweep run and score all 20 live epoch candidates online."""

    _validate_online_config(config)
    output_root = Path(output_root).expanduser().resolve()
    run_dir = output_root / "online_runs" / run.run_id
    retained_dir = run_dir / "retained_checkpoints"
    validation_path = run_dir / "validation_metrics.csv"
    test_path = run_dir / "test_metrics_analysis_only.csv"
    retention_path = run_dir / "retention_state.json"
    summary_path = run_dir / "run_summary.json"
    resume_path = run_dir / "resume_state.pt"
    plan_dir = run_dir / "plans"
    donor_plan_path = plan_dir / "opposite_donor_plan.pt"
    control_plan_path = plan_dir / "control_plan.pt"
    fcv_dir = output_root / "online_scores" / "fcv"
    control_dir = output_root / "online_scores" / "controls"
    oracle_dir = output_root / "online_scores" / "oracle"
    test_dir = output_root / "online_test_analysis_only"
    for path in (run_dir, retained_dir, plan_dir, fcv_dir, control_dir, oracle_dir, test_dir):
        path.mkdir(parents=True, exist_ok=True)

    # Validate the campaign trust root and current public/Oracle input bytes
    # before accepting either a completed run or a resume. Test image bytes
    # remain unopened here and are validated only inside analysis-only test
    # evaluation after the epoch's validation retention decision is staged.
    train_manifest = Path(train_manifest).expanduser().resolve()
    validation_manifest = Path(validation_manifest).expanduser().resolve()
    patch_masks = Path(patch_masks).expanduser().resolve()
    oracle_manifest = Path(oracle_manifest).expanduser().resolve()
    test_manifest = Path(test_manifest).expanduser().resolve()
    campaign = load_campaign_provenance_receipt(
        config, pretrained_path=pretrained_provenance_path
    )
    manifest_hashes = _manifest_hashes(config, train_manifest, validation_manifest)
    versions = software_versions()
    fingerprint = candidate_training_fingerprint(config)
    pretrained = load_pretrained_cache_provenance(
        config, campaign["bindings"]["pretrained"]["path"]
    )
    PublicManifestDataset(train_manifest, "candidate_train", None, check_images=True)
    token_source: TokenBankSource = prepare_token_bank_source(
        config, validation_manifest, patch_masks
    )
    oracle_source: OracleValidationSource = prepare_oracle_validation_source(
        config, oracle_manifest
    )

    completed = _completed_run(
        config,
        run,
        run_dir,
        output_root,
        campaign,
        manifest_hashes,
        pretrained,
    )
    if completed is not None:
        return {**completed, "invocation_status": "already_complete"}
    if not resume_path.exists():
        # A crash before the first atomic resume commit may leave only
        # uncommitted compact rows or a staged local winner. They carry no
        # recoverable model state, so restart the run from its deterministic
        # initialization while leaving candidate detail files to be overwritten.
        for path in (validation_path, test_path, retention_path):
            if path.exists():
                path.unlink()
        for path in retained_dir.glob("*.pt"):
            path.unlink()

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)
    reproducibility = config["reproducibility"]
    seed_everything(
        run.seed,
        deterministic_algorithms=bool(reproducibility["deterministic_algorithms"]),
        cudnn_benchmark=bool(reproducibility["cudnn_benchmark"]),
    )
    train_generator = torch.Generator().manual_seed(run.seed)
    loaders, _ = build_dataloaders(
        config, train_manifest, validation_manifest, train_generator
    )
    # Do not open the test manifest or construct its dataset before the first
    # validation-only retention decision has been staged.
    test_source: FinalTestSource | None = None

    if resume_path.is_file():
        resume_header = _load_trusted_checkpoint(resume_path)
        resume_header_valid = (
            resume_header.get("schema_version") == 2
            and resume_header.get("artifact_type")
            == "fcv_vit_online_resume_state"
            and resume_header.get("run") == asdict(run)
            and resume_header.get("training_fingerprint") == fingerprint
            and resume_header.get("software_versions") == versions
            and resume_header.get("source_tree_sha256")
            == source_tree_provenance()["source_tree_sha256"]
            and resume_header.get("manifest_sha256") == manifest_hashes
            and resume_header.get("campaign_provenance_path")
            == campaign["artifact_path"]
            and resume_header.get("campaign_provenance_sha256")
            == campaign["artifact_sha256"]
            and resume_header.get("campaign_bindings_sha256")
            == campaign["bindings_sha256"]
            and resume_header.get("pretrained_provenance_path")
            == pretrained["artifact_path"]
            and resume_header.get("pretrained_provenance_sha256")
            == pretrained["artifact_sha256"]
            and resume_header.get("pretrained_backbone_sha256")
            == pretrained["pretrained_backbone_sha256"]
            and resume_header.get("initial_model_state_sha256")
            == campaign["bindings"]["initialization"][
                "initial_model_state_sha256_by_seed"
            ].get(str(run.seed))
        )
        del resume_header
        if not resume_header_valid:
            selection_started = any(
                (output_root / "selection_results" / name).is_file()
                for name in (
                    "online_unprivileged_freeze_receipt.json",
                    "online_checkpoint_cleanup_plan.json",
                    "online_selection_summary.json",
                )
            )
            if selection_started:
                raise OnlineStudyError(
                    f"Stale resume cannot be repaired after selection began: {run.run_id}"
                )
            _invalidate_completed_run_for_bounded_repair(
                run_dir, "stale pre-campaign or incompatible online resume state"
            )

    resuming = resume_path.is_file()
    model = build_model(
        config, pretrained=bool(config["model"]["pretrained"]) and not resuming
    )
    initial_sha = ""
    backbone_sha = ""
    if not resuming:
        initial_sha = state_dict_sha256(model.state_dict())
        backbone_sha = pretrained_backbone_sha256(model) if pretrained else initial_sha
        if pretrained and backbone_sha != pretrained["pretrained_backbone_sha256"]:
            raise OnlineStudyError("Pretrained backbone differs from cache provenance.")
        expected_initial_sha = campaign["bindings"]["initialization"][
            "initial_model_state_sha256_by_seed"
        ].get(str(run.seed))
        if initial_sha != expected_initial_sha:
            raise OnlineStudyError(
                "Seeded model initialization differs from campaign provenance: "
                f"seed={run.seed}."
            )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=run.learning_rate, weight_decay=run.weight_decay
    )
    scheduler = build_scheduler(optimizer, config, len(loaders["train"]))
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["training"]["augmentation"]["label_smoothing"])
    )
    validation_rows: List[Dict[str, Any]] = []
    start_epoch = 1
    invocation_resumed_from_epoch: int | None = None
    if resuming:
        resume = _load_trusted_checkpoint(resume_path)
        if (
            resume.get("schema_version") != 2
            or resume.get("artifact_type") != "fcv_vit_online_resume_state"
            or resume.get("run") != asdict(run)
            or resume.get("training_fingerprint") != fingerprint
            or resume.get("software_versions") != versions
            or resume.get("source_tree_sha256")
            != source_tree_provenance()["source_tree_sha256"]
            or resume.get("manifest_sha256") != manifest_hashes
            or resume.get("campaign_provenance_path")
            != campaign["artifact_path"]
            or resume.get("campaign_provenance_sha256")
            != campaign["artifact_sha256"]
            or resume.get("campaign_bindings_sha256")
            != campaign["bindings_sha256"]
            or resume.get("pretrained_provenance_path")
            != pretrained["artifact_path"]
            or resume.get("pretrained_provenance_sha256")
            != pretrained["artifact_sha256"]
            or resume.get("pretrained_backbone_sha256")
            != pretrained["pretrained_backbone_sha256"]
            or resume.get("initial_model_state_sha256")
            != campaign["bindings"]["initialization"][
                "initial_model_state_sha256_by_seed"
            ].get(str(run.seed))
        ):
            raise OnlineStudyError("Online resume state is stale or incompatible.")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        initial_sha = str(resume["initial_model_state_sha256"])
        backbone_sha = str(resume["pretrained_backbone_sha256"])
        validation_rows = list(resume["validation_rows"])
        completed_epoch = int(resume["completed_epoch"])
        invocation_resumed_from_epoch = completed_epoch
        if len(validation_rows) != completed_epoch:
            raise OnlineStudyError("Resume rows do not match its committed epoch.")
        if [int(row["epoch"]) for row in validation_rows] != list(
            range(1, completed_epoch + 1)
        ):
            raise OnlineStudyError("Resume candidate epochs are non-contiguous.")
        if "test_rows" in resume:
            raise OnlineStudyError("Test metrics must not be stored in training resume state.")
        _atomic_csv(
            pd.DataFrame(validation_rows, columns=ONLINE_VALIDATION_COLUMNS),
            validation_path,
        )
        if _sha256_file(validation_path) != resume.get("validation_metrics_sha256"):
            raise OnlineStudyError("Committed validation index does not reproduce.")
        _restore_committed_test_index(
            test_path,
            run,
            completed_epoch,
            expected_sha256=str(
                resume.get("analysis_only_test_metrics_sha256", "")
            ),
            expected_size_bytes=int(
                resume.get("analysis_only_test_metrics_size_bytes", -1)
            ),
        )
        if (
            int(resume.get("analysis_only_test_row_count", -1)) != completed_epoch
            or _sha256_file(test_path)
            != resume.get("analysis_only_test_metrics_sha256")
        ):
            raise OnlineStudyError("Committed analysis-only test index does not reproduce.")
        retention = _retention_payload(validation_rows, retained_dir)
        _atomic_json(retention, retention_path)
        if _sha256_file(retention_path) != resume.get("retention_state_sha256"):
            raise OnlineStudyError("Committed local retention state does not reproduce.")
        _prune_retention(retention, retained_dir)
        _restore_rng_state(resume["rng_state"], train_generator)
        start_epoch = completed_epoch + 1

        # A restarted smoke may request an epoch that is already the committed
        # prefix. Validate and return it unchanged instead of silently advancing
        # another epoch and breaking exact-prefix validation.
        if stop_after_epoch is not None and completed_epoch >= stop_after_epoch:
            del resume
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "status": "paused",
                "run_id": run.run_id,
                "completed_epoch": completed_epoch,
                "resume_state": str(resume_path.resolve()),
                "invocation_status": "requested_prefix_already_committed",
            }

    scratch_parent = Path(
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or tempfile.gettempdir()
    )
    scratch_root = Path(
        tempfile.mkdtemp(prefix=f"fcv_online_{run.run_index:03d}_", dir=scratch_parent)
    )
    epochs = int(config["training"]["epochs"])
    reconstruction_contract = {
        "mode": "online_per_candidate_reconstruction_validation",
        "normal_vs_resumed_tolerance": 1.0e-5,
        "identity_and_real_swap_integrity_checked": True,
    }
    started = time.time()
    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_online_started = time.time()
            assert_storage_budget(config, output_root, stage=f"online_epoch:{run.candidate_id(epoch)}")
            train_started = time.time()
            lr_start = float(optimizer.param_groups[0]["lr"])
            train_metrics = _run_epoch(
                model,
                loaders["train"],
                criterion,
                device,
                str(config["training"]["precision"]),
                optimizer=optimizer,
                scheduler=scheduler,
            )
            validation_metrics = _run_epoch(
                model, loaders["biased_val"], criterion, device, "float32"
            )
            train_seconds = time.time() - train_started
            candidate_id = run.candidate_id(epoch)
            epoch_scratch = scratch_root / run.run_id
            checkpoint_dir = epoch_scratch / "checkpoints"
            bank_dir = epoch_scratch / "banks"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            bank_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            metric_row = {
                "run_index": run.run_index,
                "candidate_id": candidate_id,
                "epoch": epoch,
                "model_name": str(config["model"]["name"]),
                "seed": run.seed,
                "learning_rate": run.learning_rate,
                "weight_decay": run.weight_decay,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "biased_val_loss": float(validation_metrics["loss"]),
                "biased_val_accuracy": float(validation_metrics["accuracy"]),
                "lr_epoch_start": lr_start,
                "lr_epoch_end": float(optimizer.param_groups[0]["lr"]),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": "pending",
                "epoch_seconds": train_seconds,
            }
            payload = _candidate_checkpoint_payload(
                model,
                config,
                run,
                epoch,
                metric_row,
                manifest_hashes,
                fingerprint,
                versions,
                initial_sha,
                backbone_sha,
                pretrained,
                campaign,
            )
            _atomic_torch_save(payload, checkpoint_path)
            checkpoint_sha = _sha256_file(checkpoint_path)

            score_started = time.time()
            build_background_token_banks(
                config,
                checkpoint_path,
                token_source,
                bank_dir,
                reconstruction_reports=reconstruction_contract,
                device=device,
                overwrite=True,
            )
            _prepare_online_intervention_plans(
                config,
                token_source,
                bank_dir,
                candidate_id,
                checkpoint_sha,
                donor_plan_path,
                control_plan_path,
            )
            fcv_summary = score_candidate_fcv(
                config,
                checkpoint_path,
                token_source,
                bank_dir,
                donor_plan_path,
                fcv_dir,
                reconstruction_reports=reconstruction_contract,
                device=device,
                counterfactual_forward_batch_size=int(
                    config["execution"]["fcv_counterfactual_forward_batch_size"]
                ),
                overwrite=True,
            )
            control_summary = score_candidate_controls(
                config,
                checkpoint_path,
                token_source,
                bank_dir,
                donor_plan_path,
                control_plan_path,
                fcv_dir,
                control_dir,
                reconstruction_reports=reconstruction_contract,
                device=device,
                target_batch_size=int(config["execution"]["control_target_batch_size"]),
                counterfactual_forward_batch_size=int(
                    config["execution"]["control_counterfactual_forward_batch_size"]
                ),
                overwrite=True,
            )
            oracle_summary = evaluate_candidate_oracle(
                config,
                checkpoint_path,
                oracle_source,
                oracle_dir,
                device=device,
                overwrite=True,
            )
            score_seconds = time.time() - score_started
            fcv_summary_path = fcv_dir / f"{candidate_id}_summary.json"
            controls_summary_path = control_dir / f"{candidate_id}_controls_summary.json"
            oracle_summary_path = oracle_dir / f"{candidate_id}_oracle_summary.json"
            row = _validation_row(
                config,
                run,
                epoch,
                train_metrics,
                validation_metrics,
                lr_start,
                float(optimizer.param_groups[0]["lr"]),
                checkpoint_sha,
                train_seconds,
                score_seconds,
                fcv_summary,
                control_summary,
                oracle_summary,
                fcv_summary_path,
                controls_summary_path,
                oracle_summary_path,
            )
            validation_rows.append(row)

            # Freeze the current retention decision before test inference. The
            # post-hoc test helper below is therefore causally downstream of a
            # decision made from validation_rows alone.
            retention = _stage_retention(validation_rows, checkpoint_path, retained_dir)
            with _isolated_training_rng(train_generator) as pre_test_rng:
                if test_source is None:
                    test_source = prepare_final_test_source(config, test_manifest)
                test_row = _test_artifacts(
                    config,
                    run,
                    epoch,
                    checkpoint_path,
                    checkpoint_sha,
                    test_source,
                    test_dir,
                    campaign,
                    device=device_name,
                )
                test_row["epoch_online_total_seconds"] = (
                    time.time() - epoch_online_started
                )

            # The old winners are deleted only after the resume state atomically
            # commits the pre-test retention decision and an opaque binding to
            # the separate analysis-only index. Test metric values never enter
            # the optimizer-bearing resume payload.
            _atomic_csv(
                pd.DataFrame(validation_rows, columns=ONLINE_VALIDATION_COLUMNS),
                validation_path,
            )
            _append_test_index_row(test_path, run, epoch - 1, test_row)
            _atomic_json(retention, retention_path)
            resume_payload = {
                "schema_version": 2,
                "artifact_type": "fcv_vit_online_resume_state",
                "run": asdict(run),
                "completed_epoch": epoch,
                "invocation_start_epoch": start_epoch,
                "invocation_resumed_from_epoch": invocation_resumed_from_epoch,
                "training_fingerprint": fingerprint,
                "software_versions": versions,
                "source_tree_sha256": source_tree_provenance()["source_tree_sha256"],
                "initial_model_state_sha256": initial_sha,
                "pretrained_backbone_sha256": backbone_sha,
                "manifest_sha256": manifest_hashes,
                "campaign_provenance_path": campaign["artifact_path"],
                "campaign_provenance_sha256": campaign["artifact_sha256"],
                "campaign_bindings_sha256": campaign["bindings_sha256"],
                "pretrained_provenance_path": pretrained["artifact_path"],
                "pretrained_provenance_sha256": pretrained["artifact_sha256"],
                "model_state_dict": _recursive_cpu(model.state_dict()),
                "optimizer_state_dict": _recursive_cpu(optimizer.state_dict()),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": pre_test_rng,
                "validation_rows": validation_rows,
                "retention": retention,
                "validation_metrics_sha256": _sha256_file(validation_path),
                "analysis_only_test_row_count": epoch,
                "analysis_only_test_metrics_sha256": _sha256_file(test_path),
                "analysis_only_test_metrics_size_bytes": test_path.stat().st_size,
                "retention_state_sha256": _sha256_file(retention_path),
                "test_metric_values_stored_in_resume_state": False,
            }
            _atomic_torch_save(resume_payload, resume_path)
            _prune_retention(retention, retained_dir)
            del test_row

            print(
                f"[ONLINE] run={run.run_index:02d} epoch={epoch:02d}/{epochs} "
                f"biased_val={row['biased_val_accuracy']:.4f} "
                f"fcv={row['primary_selector_score']:.4f} "
                f"oracle_bal={row['oracle_validation_balanced_group_accuracy']:.4f} "
                f"control_warnings={row['control_diagnostic_warning_count']} "
                "test=analysis_only_complete",
                flush=True,
            )
            shutil.rmtree(epoch_scratch)
            assert_storage_budget(
                config,
                output_root,
                stage=f"online_epoch_post_commit:{candidate_id}",
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                return {
                    "status": "paused",
                    "run_id": run.run_id,
                    "completed_epoch": epoch,
                    "resume_state": str(resume_path.resolve()),
                }

        retention = _retention_payload(validation_rows, retained_dir)
        _atomic_json(retention, retention_path)
        summary = {
            "schema_version": 1,
            "artifact_type": "fcv_vit_online_run_summary",
            "status": "complete",
            "run": asdict(run),
            "run_id": run.run_id,
            "training_fingerprint": fingerprint,
            "software_versions": versions,
            "software_fingerprint": software_fingerprint(versions),
            "source_tree_sha256": source_tree_provenance()["source_tree_sha256"],
            "campaign_provenance_path": campaign["artifact_path"],
            "campaign_provenance_sha256": campaign["artifact_sha256"],
            "campaign_bindings_sha256": campaign["bindings_sha256"],
            "manifest_sha256": manifest_hashes,
            "pretrained_provenance_path": pretrained["artifact_path"],
            "pretrained_provenance_sha256": pretrained["artifact_sha256"],
            "pretrained_backbone_sha256": backbone_sha,
            "initial_model_state_sha256": initial_sha,
            "candidate_count": len(validation_rows),
            "candidate_epochs": list(range(1, epochs + 1)),
            "validation_metrics_path": str(validation_path.resolve()),
            "validation_metrics_sha256": _sha256_file(validation_path),
            "test_metrics_path": str(test_path.resolve()),
            "test_metrics_sha256": _sha256_file(test_path),
            "retention_state_path": str(retention_path.resolve()),
            "retention_state_sha256": _sha256_file(retention_path),
            "retained_checkpoint_count": retention["unique_checkpoint_count"],
            "test_metrics_affected_selection": False,
            "test_metrics_available_to_training_or_retention": False,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(summary, summary_path)
        if resume_path.exists():
            resume_path.unlink()
        return summary
    finally:
        if scratch_root.exists():
            shutil.rmtree(scratch_root)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
