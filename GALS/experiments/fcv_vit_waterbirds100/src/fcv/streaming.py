"""Candidate-by-candidate FCV/control scoring with safe token-bank deletion."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from .candidate_training import candidate_training_fingerprint
from .controls import score_candidate_controls
from .fcv_scoring import score_candidate_fcv
from .storage import (
    assert_storage_budget,
    atomic_json,
    file_binding,
    sha256_file,
    validate_cleanup_receipt,
)
from .token_banks import (
    CONTEXT_NAMES,
    TokenBankError,
    build_background_token_banks,
    candidate_checkpoints_for_run,
    prepare_token_bank_source,
)
from .vit_counterfactual_forward import validate_reconstruction_gate


class StreamingScoreError(RuntimeError):
    """Raised when bounded-storage scoring cannot prove a safe cleanup boundary."""


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _retained_bindings(
    fcv_summary_path: Path,
    control_summary_path: Path,
) -> list[Dict[str, Any]]:
    fcv_summary = _load_json(fcv_summary_path)
    control_summary = _load_json(control_summary_path)
    if fcv_summary.get("status") not in {"complete", "reused"}:
        raise StreamingScoreError("FCV summary is not complete at cleanup boundary.")
    if control_summary.get("status") not in {"complete", "reused"}:
        raise StreamingScoreError("Control summary is not complete at cleanup boundary.")
    paths = [
        fcv_summary_path,
        Path(str(fcv_summary["score_csv_path"])),
        control_summary_path,
    ]
    score_csvs = control_summary.get("score_csvs")
    if not isinstance(score_csvs, Mapping):
        raise StreamingScoreError("Control summary has no bound score CSV mapping.")
    paths.extend(Path(str(details["path"])) for details in score_csvs.values())
    bindings = [file_binding(path) for path in paths]
    if len({item["path"] for item in bindings}) != len(bindings):
        raise StreamingScoreError("Cleanup retained-artifact set contains duplicates.")
    return bindings


def _cleanup_candidate_banks(
    config: Mapping[str, Any],
    *,
    candidate_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    bank_summary: Mapping[str, Any],
    fcv_summary_path: Path,
    control_summary_path: Path,
    receipt_path: Path,
) -> Dict[str, Any]:
    banks = bank_summary.get("banks")
    if not isinstance(banks, Mapping) or set(banks) != set(CONTEXT_NAMES.values()):
        raise StreamingScoreError("Token-bank summary is incomplete at cleanup boundary.")
    retained = _retained_bindings(fcv_summary_path, control_summary_path)
    deleted: Dict[str, Dict[str, Any]] = {}
    for context_name, details in banks.items():
        path = Path(str(details["path"])).expanduser().resolve()
        expected_size = int(details["file_size_bytes"])
        expected_sha = str(details["sha256"])
        if not path.is_file():
            raise StreamingScoreError(f"Token bank vanished before cleanup: {path}")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
            raise StreamingScoreError(f"Token bank changed before cleanup: {path}")
        deleted[context_name] = {
            "path": str(path),
            "size_bytes": expected_size,
            "sha256": expected_sha,
        }
    payload = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_stream_cleanup_receipt",
        "status": "prepared",
        "candidate_id": candidate_id,
        "training_fingerprint": candidate_training_fingerprint(config),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "token_bank_summary": file_binding(
            Path(str(bank_summary["banks"][CONTEXT_NAMES[0]]["path"])).parent
            / f"{candidate_id}_summary.json"
        ),
        "deleted_token_banks": deleted,
        "retained_artifacts": retained,
        "prepared_unix_time": time.time(),
    }
    atomic_json(payload, receipt_path)
    for details in deleted.values():
        path = Path(details["path"])
        if path.exists():
            path.unlink()
    payload["status"] = "complete"
    payload["completed_unix_time"] = time.time()
    atomic_json(payload, receipt_path)
    validated = validate_cleanup_receipt(
        receipt_path,
        candidate_id=candidate_id,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        training_fingerprint=candidate_training_fingerprint(config),
        token_banks=deleted,
    )
    if validated is None:
        raise StreamingScoreError("Completed cleanup receipt failed validation.")
    return validated


def stream_score_run(
    config: Mapping[str, Any],
    *,
    run_index: int,
    candidate_root: str | Path,
    manifest: str | Path,
    patch_masks: str | Path,
    token_bank_dir: str | Path,
    donor_plan: str | Path,
    control_plan: str | Path,
    fcv_score_dir: str | Path,
    control_score_dir: str | Path,
    output_root: str | Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    if not config["storage"]["streaming_token_banks"]:
        raise StreamingScoreError("Streaming token banks are not enabled.")
    output_root = Path(output_root).expanduser().resolve()
    token_bank_dir = Path(token_bank_dir).expanduser().resolve()
    fcv_score_dir = Path(fcv_score_dir).expanduser().resolve()
    control_score_dir = Path(control_score_dir).expanduser().resolve()
    receipt_dir = token_bank_dir / "cleanup_receipts"
    source = prepare_token_bank_source(config, manifest, patch_masks)
    reconstruction_reports = validate_reconstruction_gate(config)
    checkpoints = candidate_checkpoints_for_run(config, candidate_root, run_index)
    results = []
    for checkpoint_path in checkpoints:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        candidate_id = str(checkpoint["candidate_id"])
        del checkpoint
        checkpoint_sha = sha256_file(checkpoint_path)
        receipt_path = receipt_dir / f"{candidate_id}.json"
        fcv_summary_path = fcv_score_dir / f"{candidate_id}_summary.json"
        control_summary_path = (
            control_score_dir / f"{candidate_id}_controls_summary.json"
        )
        fcv_summary = _load_json(fcv_summary_path) if fcv_summary_path.is_file() else {}
        recorded_banks = fcv_summary.get("token_banks", {})
        receipt = None
        if isinstance(recorded_banks, Mapping) and recorded_banks:
            receipt = validate_cleanup_receipt(
                receipt_path,
                candidate_id=candidate_id,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                training_fingerprint=candidate_training_fingerprint(config),
                token_banks=recorded_banks,
            )
        if receipt is not None:
            results.append({"candidate_id": candidate_id, "status": "reused_cleanup"})
            continue
        assert_storage_budget(config, output_root, stage=f"banks:{candidate_id}")
        try:
            bank_summary = build_background_token_banks(
                config,
                checkpoint_path,
                source,
                token_bank_dir,
                reconstruction_reports=reconstruction_reports,
                device=device,
                overwrite=False,
            )
        except TokenBankError:
            # Regeneration is automatic only when a cleanup journal proves
            # why a formerly complete bank is absent. Other stale-bank states
            # remain fatal instead of silently hiding corruption.
            if not receipt_path.is_file():
                raise
            bank_summary = build_background_token_banks(
                config,
                checkpoint_path,
                source,
                token_bank_dir,
                reconstruction_reports=reconstruction_reports,
                device=device,
                overwrite=True,
            )
        fcv_summary = score_candidate_fcv(
            config,
            checkpoint_path,
            source,
            token_bank_dir,
            donor_plan,
            fcv_score_dir,
            reconstruction_reports=reconstruction_reports,
            device=device,
            counterfactual_forward_batch_size=int(
                config["execution"]["fcv_counterfactual_forward_batch_size"]
            ),
        )
        control_summary = score_candidate_controls(
            config,
            checkpoint_path,
            source,
            token_bank_dir,
            donor_plan,
            control_plan,
            fcv_score_dir,
            control_score_dir,
            reconstruction_reports=reconstruction_reports,
            device=device,
            target_batch_size=int(config["execution"]["control_target_batch_size"]),
            counterfactual_forward_batch_size=int(
                config["execution"]["control_counterfactual_forward_batch_size"]
            ),
        )
        if fcv_summary.get("candidate_id") != candidate_id or control_summary.get(
            "candidate_id"
        ) != candidate_id:
            raise StreamingScoreError("Scoring returned a different candidate identity.")
        receipt = _cleanup_candidate_banks(
            config,
            candidate_id=candidate_id,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            bank_summary=bank_summary,
            fcv_summary_path=fcv_summary_path,
            control_summary_path=control_summary_path,
            receipt_path=receipt_path,
        )
        results.append({"candidate_id": candidate_id, "status": receipt["status"]})
    return {"run_index": run_index, "candidate_count": len(results), "results": results}
