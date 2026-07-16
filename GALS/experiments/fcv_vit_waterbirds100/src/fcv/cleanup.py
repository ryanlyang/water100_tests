"""Auditable cleanup of completed candidate-training resume states."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from .candidate_training import (
    aggregate_candidate_metrics,
    candidate_training_fingerprint,
    enumerate_sweep_runs,
)
from .config import candidate_epochs
from .storage import atomic_json, file_binding, validate_file_binding


class CleanupError(RuntimeError):
    """Raised when a cleanup boundary cannot be proven safe."""


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_resume_receipt(
    receipt: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    require_complete: bool,
) -> bool:
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type")
        != "fcv_vit_resume_state_cleanup_receipt"
        or receipt.get("training_fingerprint")
        != candidate_training_fingerprint(config)
        or receipt.get("candidate_epochs") != candidate_epochs(config)
        or (require_complete and receipt.get("status") != "complete")
    ):
        return False
    retained = receipt.get("retained_artifacts")
    deleted = receipt.get("deleted_resume_states")
    if not isinstance(retained, list) or not isinstance(deleted, list):
        return False
    if not all(validate_file_binding(item) for item in retained):
        return False
    for item in deleted:
        path = Path(str(item.get("path", "")))
        if receipt.get("status") == "complete" and path.exists():
            return False
        if len(str(item.get("sha256", ""))) != 64 or int(
            item.get("size_bytes", -1)
        ) <= 0:
            return False
    return len(deleted) == int(config["candidate_pool"]["expected_training_runs"])


def prune_completed_resume_states(
    config: Mapping[str, Any],
    candidate_root: str | Path,
    receipt_path: str | Path,
) -> Dict[str, Any]:
    """Delete 27 optimizer-bearing resume files only after strict pool validation."""

    if not config["storage"]["delete_completed_resume_states_after_pool_validation"]:
        raise CleanupError("Resume-state cleanup is disabled by configuration.")
    candidate_root = Path(candidate_root).expanduser().resolve()
    receipt_path = Path(receipt_path).expanduser().resolve()
    pool_csv = candidate_root / "candidate_metrics_biased_val.csv"
    pool_summary_path = candidate_root / "candidate_pool_summary.json"
    pool_summary = aggregate_candidate_metrics(
        config,
        candidate_root,
        pool_csv,
        pool_summary_path,
        allow_incomplete=False,
    )
    if (
        pool_summary.get("status") != "complete"
        or int(pool_summary.get("candidate_count", -1))
        != int(config["candidate_pool"]["expected_candidate_checkpoints"])
        or pool_summary.get("candidate_epochs") != candidate_epochs(config)
    ):
        raise CleanupError("Strict candidate-pool validation did not complete.")

    if receipt_path.is_file():
        existing = _load_json(receipt_path)
        if _validate_resume_receipt(existing, config=config, require_complete=True):
            return dict(existing)
        if not _validate_resume_receipt(existing, config=config, require_complete=False):
            raise CleanupError("Existing resume cleanup receipt is stale or malformed.")
        payload = dict(existing)
    else:
        retained = [file_binding(pool_csv), file_binding(pool_summary_path)]
        deleted = []
        for run in enumerate_sweep_runs(config):
            run_dir = candidate_root / run.run_id
            summary_path = run_dir / "run_summary.json"
            metrics_path = run_dir / "metrics.csv"
            summary = _load_json(summary_path)
            if (
                summary.get("status") != "complete"
                or summary.get("run_id") != run.run_id
                or summary.get("candidate_epochs") != candidate_epochs(config)
            ):
                raise CleanupError(f"Run is not safely complete: {run.run_id}")
            resume_path = Path(str(summary.get("resume_state_path", ""))).resolve()
            expected_resume = (run_dir / "resume_state.pt").resolve()
            if resume_path != expected_resume or not resume_path.is_file():
                raise CleanupError(
                    f"Missing unpruned resume state without a receipt: {expected_resume}"
                )
            retained.extend([file_binding(summary_path), file_binding(metrics_path)])
            deleted.append(file_binding(resume_path))
        payload = {
            "schema_version": 1,
            "artifact_type": "fcv_vit_resume_state_cleanup_receipt",
            "status": "prepared",
            "training_fingerprint": candidate_training_fingerprint(config),
            "candidate_epochs": candidate_epochs(config),
            "candidate_count": int(pool_summary["candidate_count"]),
            "deleted_resume_states": deleted,
            "retained_artifacts": retained,
            "prepared_unix_time": time.time(),
        }
        atomic_json(payload, receipt_path)

    # A prepared receipt is a durable recovery journal: existing files must
    # still match it; already-absent files are accepted as prior completed
    # deletions from an interrupted cleanup invocation.
    for item in payload["deleted_resume_states"]:
        path = Path(str(item["path"]))
        if path.exists():
            if not validate_file_binding(item):
                raise CleanupError(f"Resume state changed before cleanup: {path}")
            path.unlink()
    payload["status"] = "complete"
    payload["completed_unix_time"] = time.time()
    atomic_json(payload, receipt_path)
    if not _validate_resume_receipt(payload, config=config, require_complete=True):
        raise CleanupError("Completed resume-state cleanup receipt failed validation.")
    return payload
