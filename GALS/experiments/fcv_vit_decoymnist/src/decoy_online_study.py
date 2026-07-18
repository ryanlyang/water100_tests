"""Step-10 no-checkpoint online DecoyMNIST FCV training orchestration."""

from __future__ import annotations

import gc
import json
import random
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from decoy_candidate_training import (
    build_candidate_dataloaders,
    build_model,
    build_optimizer_and_scheduler,
    seed_everything,
    train_one_epoch,
)
from decoy_campaign_preflight import pretrained_backbone_sha256
from decoy_donor_plans import load_and_validate_donor_plan
from decoy_fcv_scoring import (
    CONTROL_NAMES,
    build_online_token_bank,
    prepare_exact_control_masks,
    score_online_fcv_and_controls,
    token_bank_classification_metrics,
)
from decoy_full_config import (
    CampaignRun,
    canonical_config_sha256,
    enumerate_runs,
    sha256_file,
)
from decoy_manifest_provenance import atomic_json
from decoy_online_evaluation import (
    build_analysis_loaders,
    evaluate_oracle_online,
    evaluate_test_analysis_only,
)
from decoy_online_schema import (
    CONTROL_METRIC_NAMES,
    atomic_write_namespace_rows,
    namespace_output_path,
    validate_namespace_row,
)
from decoy_teacher_masks import load_projected_teacher_masks


class OnlineStudyError(ValueError):
    """Raised when online execution violates the frozen no-checkpoint contract."""


@contextmanager
def isolated_evaluation_rng():
    """Make all online analysis causally inert to subsequent training RNG."""

    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _directory_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def assert_no_forbidden_persistence(
    config: Mapping[str, Any], output_root: str | Path
) -> Dict[str, Any]:
    """Fail if model, optimizer, resume, token, or embedding artifacts persist."""

    root = Path(output_root).expanduser().resolve()
    forbidden_suffixes = set(str(value) for value in config["storage"]["forbidden_suffixes"])
    forbidden_fragments = (
        "checkpoint",
        "optimizer_state",
        "resume_state",
        "token_bank",
        "patch_embedding",
        "model_state",
    )
    violations = []
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in forbidden_suffixes or any(
                fragment in path.name.lower() for fragment in forbidden_fragments
            ):
                violations.append(str(path))
    if violations:
        raise OnlineStudyError(
            f"Forbidden persistent training artifacts detected: {violations[:5]}"
        )
    allocated = _directory_bytes(root)
    budget = int(float(config["storage"]["persistent_output_budget_gib"]) * 1024**3)
    if allocated > budget:
        raise OnlineStudyError(
            f"Campaign output uses {allocated} bytes, exceeding {budget} bytes."
        )
    return {"allocated_bytes": allocated, "budget_bytes": budget, "violations": 0}


def _common_row(run: CampaignRun, epoch: int) -> Dict[str, Any]:
    return {
        "run_index": int(run.run_index),
        "run_id": run.run_id,
        "candidate_id": run.candidate_id(epoch),
        "epoch": int(epoch),
        "seed": int(run.seed),
        "learning_rate": float(run.learning_rate),
        "weight_decay": float(run.weight_decay),
        "crop_scale_min": float(run.crop_scale_min),
    }


def _class_accuracy_fields(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, float]:
    values = list(metrics["per_class_accuracy"])
    if len(values) != 10:
        raise OnlineStudyError(f"{prefix} must report ten class accuracies.")
    return {
        f"{prefix}_class_{label}_accuracy": float(value)
        for label, value in enumerate(values)
    }


def _biased_row(
    run: CampaignRun,
    epoch: int,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    *,
    lr_start: float,
    lr_end: float,
    train_seconds: float,
) -> Dict[str, Any]:
    row = {
        **_common_row(run, epoch),
        "train_loss": float(train_metrics["loss"]),
        "train_accuracy": float(train_metrics["accuracy"]),
        "lr_epoch_start": float(lr_start),
        "lr_epoch_end": float(lr_end),
        "biased_validation_loss": float(validation_metrics["loss"]),
        "biased_validation_accuracy": float(validation_metrics["accuracy"]),
        "biased_validation_balanced_class_accuracy": float(
            validation_metrics["balanced_class_accuracy"]
        ),
        "biased_validation_worst_class_accuracy": float(
            validation_metrics["worst_class_accuracy"]
        ),
        **_class_accuracy_fields("biased_validation", validation_metrics),
        "epoch_train_seconds": float(train_seconds),
    }
    validate_namespace_row("biased_validation", row)
    return row


def _fcv_row(
    run: CampaignRun,
    epoch: int,
    aggregate: Mapping[str, Any],
    *,
    seconds: float,
) -> Dict[str, Any]:
    primary = aggregate["primary_fcv"]
    row = {
        **_common_row(run, epoch),
        "original_biased_validation_accuracy": float(
            aggregate["original_biased_validation_accuracy"]
        ),
        "harmonic_fcv_score": float(aggregate["harmonic_fcv_score"]),
        "fcv_eligible_target_count": int(primary["eligible_target_count"]),
        "fcv_eligible_target_fraction": float(primary["eligible_target_fraction"]),
        "fcv_donor_draw_count": int(primary["donor_draw_count"]),
        "fcv_counterfactual_accuracy": float(primary["counterfactual_accuracy"]),
        "fcv_counterfactual_majority_accuracy": float(
            primary["counterfactual_majority_accuracy"]
        ),
        "fcv_mean_true_class_probability": float(
            primary["mean_true_class_counterfactual_probability"]
        ),
        "fcv_mean_confidence_drop": float(
            primary["mean_original_to_counterfactual_confidence_drop"]
        ),
        "fcv_mean_replaced_patch_count": float(primary["mean_replaced_patch_count"]),
        "fcv_changed_replacement_fraction": float(
            primary["changed_replacement_fraction"]
        ),
        "identity_forward_max_abs_error": float(
            aggregate["identity_forward"]["max_abs_error"]
        ),
        "epoch_fcv_seconds": float(seconds),
    }
    validate_namespace_row("fcv", row)
    return row


def _controls_row(
    run: CampaignRun, epoch: int, aggregate: Mapping[str, Any]
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        **_common_row(run, epoch),
        "control_diagnostics_warning_only": bool(
            aggregate["control_diagnostics_warning_only"]
        ),
        "control_warning_count": int(aggregate["control_warning_count"]),
        "control_warning_reason_counts_json": json.dumps(
            aggregate["control_warning_reason_counts"],
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    for control in CONTROL_NAMES:
        summary = aggregate["controls"][control]
        for metric in CONTROL_METRIC_NAMES:
            row[f"control_{control}_{metric}"] = summary.get(metric)
    validate_namespace_row("controls", row)
    return row


def _analysis_row(
    namespace: str,
    prefix: str,
    run: CampaignRun,
    epoch: int,
    result: Mapping[str, Any],
    *,
    seconds: float,
) -> Dict[str, Any]:
    metrics = result["metrics"]
    row = {
        **_common_row(run, epoch),
        f"{prefix}_loss": float(metrics["loss"]),
        f"{prefix}_accuracy": float(metrics["accuracy"]),
        f"{prefix}_balanced_class_accuracy": float(
            metrics["balanced_class_accuracy"]
        ),
        f"{prefix}_worst_class_accuracy": float(metrics["worst_class_accuracy"]),
        **_class_accuracy_fields(prefix, metrics),
        f"epoch_{'oracle' if prefix.startswith('oracle') else 'test'}_seconds": float(
            seconds
        ),
    }
    validate_namespace_row(namespace, row)
    return row


def _namespace_paths(output_root: Path, run: CampaignRun) -> Dict[str, Path]:
    return {
        namespace: namespace_output_path(output_root, namespace, run.run_id)
        for namespace in (
            "biased_validation",
            "fcv",
            "controls",
            "oracle_analysis_only",
            "test_analysis_only",
        )
    }


def _summary_path(output_root: Path, run: CampaignRun) -> Path:
    return output_root / "run_summaries" / f"{run.run_id}.json"


def _validate_completed_run(
    config: Mapping[str, Any],
    run: CampaignRun,
    output_root: Path,
    *,
    expected_epochs: int | None = None,
    artifact_type: str = "fcv_vit_decoymnist_online_run_summary",
    expected_preflight_receipt_sha256: str | None = None,
    expected_pretrained_backbone_sha256: str | None = None,
) -> Dict[str, Any] | None:
    expected_count = int(expected_epochs or config["training"]["epochs"])
    summary_path = _summary_path(output_root, run)
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("artifact_type") != artifact_type
        or summary.get("artifact_version") != 1
        or summary.get("status") != "complete"
        or summary.get("config_sha256") != canonical_config_sha256(config)
        or summary.get("run") != asdict(run)
        or int(summary.get("completed_candidate_count", -1))
        != expected_count
        or (
            expected_preflight_receipt_sha256 is not None
            and summary.get("preflight_receipt_sha256")
            != expected_preflight_receipt_sha256
        )
        or (
            expected_pretrained_backbone_sha256 is not None
            and summary.get("pretrained_backbone_sha256")
            != expected_pretrained_backbone_sha256
        )
    ):
        raise OnlineStudyError(f"Completed-run summary is stale: {summary_path}")
    paths = _namespace_paths(output_root, run)
    expected_epoch_values = list(range(1, expected_count + 1))
    for namespace, path in paths.items():
        record = summary.get("namespace_artifacts", {}).get(namespace, {})
        if (
            not path.is_file()
            or record.get("path") != str(path)
            or record.get("sha256") != sha256_file(path)
            or int(record.get("row_count", -1)) != expected_count
        ):
            raise OnlineStudyError(f"Completed {namespace} artifact is stale.")
        frame = pd.read_csv(path)
        if len(frame) != expected_count:
            raise OnlineStudyError(f"Completed {namespace} row count changed.")
        if frame["epoch"].astype(int).tolist() != expected_epoch_values:
            raise OnlineStudyError(f"Completed {namespace} epoch sequence changed.")
        for epoch, row in zip(expected_epoch_values, frame.to_dict("records")):
            validate_namespace_row(namespace, row)
            expected_common = _common_row(run, epoch)
            for key, expected in expected_common.items():
                observed = row[key]
                if isinstance(expected, float):
                    matches = bool(
                        np.isclose(float(observed), expected, rtol=0.0, atol=1.0e-15)
                    )
                else:
                    matches = observed == expected
                if not matches:
                    raise OnlineStudyError(
                        f"Completed {namespace} row changed at epoch {epoch}: "
                        f"{key}={observed!r}, expected {expected!r}."
                    )
    assert_no_forbidden_persistence(config, output_root)
    return summary


def run_online_study(
    config: Mapping[str, Any],
    run_index: int,
    *,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    oracle_manifest: str | Path,
    test_manifest: str | Path,
    mask_artifact: str | Path,
    donor_plan_path: str | Path,
    output_root: str | Path | None = None,
    device_name: str = "cuda",
    restart_partial: bool = False,
    epoch_limit: int | None = None,
    preflight_receipt_sha256: str | None = None,
    expected_pretrained_backbone_sha256: str | None = None,
) -> Dict[str, Any]:
    """Train and evaluate one ten-epoch run without persisting training state."""

    import torch

    runs = enumerate_runs(config)
    if not 0 <= int(run_index) < len(runs):
        raise OnlineStudyError(f"run_index must be in [0,{len(runs) - 1}].")
    run = runs[int(run_index)]
    production_epochs = int(config["training"]["epochs"])
    if epoch_limit is None:
        epochs_to_run = production_epochs
        artifact_type = "fcv_vit_decoymnist_online_run_summary"
        execution_mode = "production"
    else:
        if int(epoch_limit) != 1:
            raise OnlineStudyError("The only supported reduced execution is one-epoch smoke.")
        epochs_to_run = 1
        artifact_type = "fcv_vit_decoymnist_online_smoke_run_summary"
        execution_mode = "one_epoch_production_path_smoke"
    root = Path(output_root or config["paths"]["output_root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed = _validate_completed_run(
        config,
        run,
        root,
        expected_epochs=epochs_to_run,
        artifact_type=artifact_type,
        expected_preflight_receipt_sha256=preflight_receipt_sha256,
        expected_pretrained_backbone_sha256=expected_pretrained_backbone_sha256,
    )
    if completed is not None:
        return {**completed, "invocation_status": "already_complete"}

    paths = _namespace_paths(root, run)
    partial = [str(path) for path in paths.values() if path.exists()]
    if partial and not restart_partial:
        raise OnlineStudyError(
            "This no-checkpoint run has a partial aggregate prefix and must restart "
            f"from epoch 1 with restart_partial=True: {partial}"
        )
    assert_no_forbidden_persistence(config, root)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise OnlineStudyError("CUDA was requested but is unavailable.")
    device = torch.device(device_name)

    seed_everything(config, run.seed)
    candidate_loaders, candidate_datasets = build_candidate_dataloaders(
        config, run, train_manifest, validation_manifest
    )
    analysis_loaders, analysis_datasets = build_analysis_loaders(
        config, oracle_manifest, test_manifest, seed=run.seed
    )
    bundle_hashes = {
        candidate_datasets["train"].binding.bundle_sha256,
        candidate_datasets["biased_validation"].binding.bundle_sha256,
        analysis_datasets["oracle_analysis_only"].binding.bundle_sha256,
        analysis_datasets["test_analysis_only"].binding.bundle_sha256,
    }
    if len(bundle_hashes) != 1:
        raise OnlineStudyError("Training, validation, Oracle, and test use different bundles.")
    projected_masks, mask_binding = load_projected_teacher_masks(
        config, validation_manifest, mask_artifact
    )
    donor_plan = load_and_validate_donor_plan(
        config, validation_manifest, mask_artifact, donor_plan_path
    )
    if mask_binding.bundle_sha256 not in bundle_hashes:
        raise OnlineStudyError("Teacher masks use a different manifest bundle.")
    validation_frame = candidate_datasets["biased_validation"].frame
    exact_categories, decoy_positions = prepare_exact_control_masks(
        config, validation_frame
    )

    model = build_model(config, pretrained=bool(config["model"]["pretrained"]))
    observed_pretrained_backbone_sha256 = pretrained_backbone_sha256(model)
    if (
        expected_pretrained_backbone_sha256 is not None
        and observed_pretrained_backbone_sha256
        != str(expected_pretrained_backbone_sha256)
    ):
        raise OnlineStudyError(
            "Loaded pretrained backbone differs from the GH200 preflight cache."
        )
    model.to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, config, run, len(candidate_loaders["train"])
    )
    histories: Dict[str, List[Dict[str, Any]]] = {
        namespace: [] for namespace in paths
    }
    started = time.time()
    try:
        for epoch in range(1, epochs_to_run + 1):
            assert_no_forbidden_persistence(config, root)
            lr_start = float(optimizer.param_groups[0]["lr"])
            train_started = time.time()
            train_metrics = train_one_epoch(
                model,
                candidate_loaders["train"],
                optimizer,
                scheduler,
                device,
                str(config["training"]["precision"]),
            )
            train_seconds = time.time() - train_started
            token_bank = None
            with isolated_evaluation_rng():
                try:
                    fcv_started = time.time()
                    token_bank = build_online_token_bank(
                        model,
                        candidate_loaders["biased_validation"],
                        device,
                        precision=str(config["training"]["precision"]),
                    )
                    validation_metrics = token_bank_classification_metrics(token_bank)
                    fcv_aggregate = score_online_fcv_and_controls(
                        config,
                        model,
                        token_bank,
                        projected_masks,
                        donor_plan,
                        exact_categories,
                        decoy_positions,
                        device,
                    )
                    if not np.isclose(
                        float(validation_metrics["accuracy"]),
                        float(fcv_aggregate["original_biased_validation_accuracy"]),
                        rtol=0.0,
                        atol=1.0e-12,
                    ):
                        raise OnlineStudyError(
                            "Biased-validation and FCV original accuracies disagree."
                        )
                    fcv_seconds = time.time() - fcv_started

                    oracle_started = time.time()
                    oracle_result = evaluate_oracle_online(
                        model,
                        analysis_loaders["oracle_analysis_only"],
                        device,
                        precision=str(config["training"]["precision"]),
                        num_classes=int(config["data"]["num_classes"]),
                    )
                    oracle_seconds = time.time() - oracle_started

                    # Test is deliberately last and flows only to its own row.
                    test_started = time.time()
                    test_result = evaluate_test_analysis_only(
                        model,
                        analysis_loaders["test_analysis_only"],
                        device,
                        precision=str(config["training"]["precision"]),
                        num_classes=int(config["data"]["num_classes"]),
                    )
                    test_seconds = time.time() - test_started
                finally:
                    if token_bank is not None:
                        del token_bank.raw_patch_tokens
                        del token_bank.original_logits
                        del token_bank
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            rows = {
                "biased_validation": _biased_row(
                    run,
                    epoch,
                    train_metrics,
                    validation_metrics,
                    lr_start=lr_start,
                    lr_end=float(optimizer.param_groups[0]["lr"]),
                    train_seconds=train_seconds,
                ),
                "fcv": _fcv_row(
                    run, epoch, fcv_aggregate, seconds=fcv_seconds
                ),
                "controls": _controls_row(run, epoch, fcv_aggregate),
                "oracle_analysis_only": _analysis_row(
                    "oracle_analysis_only",
                    "oracle_validation",
                    run,
                    epoch,
                    oracle_result,
                    seconds=oracle_seconds,
                ),
                "test_analysis_only": _analysis_row(
                    "test_analysis_only",
                    "test",
                    run,
                    epoch,
                    test_result,
                    seconds=test_seconds,
                ),
            }
            for namespace, row in rows.items():
                histories[namespace].append(row)
                atomic_write_namespace_rows(namespace, histories[namespace], paths[namespace])
            assert_no_forbidden_persistence(config, root)
            print(
                f"[ONLINE] run={run.run_index:03d} epoch={epoch:02d}/"
                f"{epochs_to_run} "
                f"biased_val={rows['biased_validation']['biased_validation_accuracy']:.4f} "
                f"fcv={rows['fcv']['harmonic_fcv_score']:.4f} "
                f"oracle={rows['oracle_analysis_only']['oracle_validation_accuracy']:.4f} "
                "test=analysis_only_complete",
                flush=True,
            )
    finally:
        del model
        del optimizer
        del scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    epochs = epochs_to_run
    if any(len(rows) != epochs for rows in histories.values()):
        raise OnlineStudyError("Online run ended without all namespace rows.")
    artifact_records = {
        namespace: {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(histories[namespace]),
        }
        for namespace, path in paths.items()
    }
    storage = assert_no_forbidden_persistence(config, root)
    summary = {
        "artifact_type": artifact_type,
        "artifact_version": 1,
        "status": "complete",
        "execution_mode": execution_mode,
        "production_training_epochs": production_epochs,
        "config_sha256": canonical_config_sha256(config),
        "run": asdict(run),
        "completed_candidate_count": epochs,
        "manifest_bundle_sha256": next(iter(bundle_hashes)),
        "projected_teacher_masks_sha256": sha256_file(mask_artifact),
        "donor_plan_sha256": sha256_file(donor_plan_path),
        "namespace_artifacts": artifact_records,
        "seconds": float(time.time() - started),
        "storage": storage,
        "model_checkpoints_persisted": False,
        "optimizer_states_persisted": False,
        "resume_states_persisted": False,
        "token_banks_persisted": False,
        "test_metrics_used_for_training_or_selection": False,
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "pretrained_backbone_sha256": observed_pretrained_backbone_sha256,
    }
    atomic_json(summary, _summary_path(root, run))
    assert_no_forbidden_persistence(config, root)
    return summary
