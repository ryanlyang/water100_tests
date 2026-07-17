"""Bounded-storage guards and auditable token-bank cleanup receipts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


class StorageBudgetError(RuntimeError):
    """Raised before an artifact campaign can exceed its locked disk budget."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocated_bytes(root: Path) -> int:
    """Return allocated bytes without following symlinks outside the study root."""

    root = root.expanduser().resolve()
    if not root.exists():
        return 0
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                try:
                    stat = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += int(stat.st_blocks) * 512
    return total


def online_storage_breakdown(root: str | Path) -> Dict[str, int]:
    """Separate durable candidate evidence from bounded model-state storage."""

    root = Path(root).expanduser().resolve()
    online_runs = root / "online_runs"
    retained = sum(
        allocated_bytes(path)
        for path in online_runs.glob("*/retained_checkpoints")
        if path.is_dir()
    )
    plans = sum(
        allocated_bytes(path)
        for path in online_runs.glob("*/plans")
        if path.is_dir()
    )
    resumes = 0
    for path in online_runs.glob("*/resume_state.pt"):
        if path.is_file():
            resumes += int(path.stat().st_blocks) * 512
    online_runs_total = allocated_bytes(online_runs)
    online_run_other = max(0, online_runs_total - retained - plans - resumes)
    breakdown = {
        "fcv_evidence": allocated_bytes(root / "online_scores" / "fcv"),
        "control_evidence": allocated_bytes(root / "online_scores" / "controls"),
        "oracle_evidence": allocated_bytes(root / "online_scores" / "oracle"),
        "test_evidence": allocated_bytes(root / "online_test_analysis_only"),
        "online_run_other": online_run_other,
        "retained_checkpoints": retained,
        "resume_states": resumes,
        "intervention_plans": plans,
    }
    breakdown["candidate_evidence_total"] = sum(
        breakdown[key]
        for key in (
            "fcv_evidence",
            "control_evidence",
            "oracle_evidence",
            "test_evidence",
            "online_run_other",
        )
    )
    breakdown["model_state_total"] = retained + resumes
    breakdown["categorized_total"] = sum(
        breakdown[key]
        for key in (
            "candidate_evidence_total",
            "retained_checkpoints",
            "resume_states",
            "intervention_plans",
        )
    )
    return breakdown


def assert_storage_budget(
    config: Mapping[str, Any],
    output_root: str | Path,
    *,
    stage: str,
) -> Dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    used = allocated_bytes(output_root)
    gib = 1024 ** 3
    hard = float(config["storage"]["hard_budget_gib"])
    guard = float(config["storage"]["launch_guard_gib"])
    projected_growth = float(
        config["storage"]["worst_case_concurrent_growth_gib"]
    )
    used_gib = used / gib
    if used_gib + projected_growth > hard:
        raise StorageBudgetError(
            f"Projected hard storage budget exceeded before {stage}: "
            f"used={used_gib:.3f} GiB, projected_concurrent_growth="
            f"{projected_growth:.3f} GiB, hard_budget={hard:.3f} GiB. "
            "No new large artifact was started."
        )
    if used_gib >= guard:
        raise StorageBudgetError(
            f"Storage launch guard reached before {stage}: used={used_gib:.3f} GiB, "
            f"launch_guard={guard:.3f} GiB, hard_budget={hard:.3f} GiB. "
            "No new large artifact was started."
        )
    return {
        "stage": stage,
        "output_root": str(output_root),
        "allocated_bytes": used,
        "allocated_gib": used_gib,
        "launch_guard_gib": guard,
        "hard_budget_gib": hard,
        "projected_concurrent_growth_gib": projected_growth,
        "projected_peak_gib": used_gib + projected_growth,
        "storage_breakdown_bytes": online_storage_breakdown(output_root),
    }


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def file_binding(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_file_binding(binding: Mapping[str, Any]) -> bool:
    path = Path(str(binding.get("path", "")))
    return bool(
        path.is_file()
        and path.stat().st_size == int(binding.get("size_bytes", -1))
        and sha256_file(path) == str(binding.get("sha256", ""))
    )


def validate_cleanup_receipt(
    receipt_path: str | Path,
    *,
    candidate_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    training_fingerprint: str,
    token_banks: Mapping[str, Mapping[str, Any]],
    require_complete: bool = True,
) -> Dict[str, Any] | None:
    receipt_path = Path(receipt_path).expanduser().resolve()
    if not receipt_path.is_file():
        return None
    try:
        with receipt_path.open("r", encoding="utf-8") as handle:
            receipt = json.load(handle)
        if (
            receipt.get("schema_version") != 1
            or receipt.get("artifact_type") != "fcv_vit_stream_cleanup_receipt"
            or receipt.get("candidate_id") != candidate_id
            or receipt.get("training_fingerprint") != training_fingerprint
            or receipt.get("checkpoint_path") != str(checkpoint_path)
            or receipt.get("checkpoint_sha256") != checkpoint_sha256
            or (require_complete and receipt.get("status") != "complete")
            or not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != checkpoint_sha256
        ):
            return None
        recorded_banks = receipt.get("deleted_token_banks")
        if not isinstance(recorded_banks, Mapping) or set(recorded_banks) != set(
            token_banks
        ):
            return None
        for context_name, expected in token_banks.items():
            recorded = recorded_banks[context_name]
            if (
                recorded.get("path") != str(Path(str(expected["path"])).resolve())
                or recorded.get("sha256") != str(expected["sha256"])
                or int(recorded.get("size_bytes", -1))
                != int(expected.get("size_bytes", recorded.get("size_bytes", -1)))
            ):
                return None
            if receipt.get("status") == "complete" and Path(recorded["path"]).exists():
                return None
        retained = receipt.get("retained_artifacts")
        if not isinstance(retained, Sequence) or isinstance(retained, (str, bytes)):
            return None
        token_bank_summary = receipt.get("token_bank_summary")
        if not isinstance(token_bank_summary, Mapping) or not validate_file_binding(
            token_bank_summary
        ):
            return None
        with Path(str(token_bank_summary["path"])).open(
            "r", encoding="utf-8"
        ) as handle:
            bank_summary = json.load(handle)
        if (
            bank_summary.get("artifact_type") != "fcv_vit_token_bank_summary"
            or bank_summary.get("status") != "complete"
            or bank_summary.get("candidate_id") != candidate_id
            or bank_summary.get("checkpoint_path") != str(checkpoint_path)
            or bank_summary.get("checkpoint_sha256") != checkpoint_sha256
            or bank_summary.get("training_fingerprint") != training_fingerprint
        ):
            return None
        summary_banks = bank_summary.get("banks")
        if not isinstance(summary_banks, Mapping) or set(summary_banks) != set(
            recorded_banks
        ):
            return None
        for context_name, recorded in recorded_banks.items():
            summary_bank = summary_banks[context_name]
            if (
                str(Path(str(summary_bank.get("path", ""))).resolve())
                != recorded["path"]
                or summary_bank.get("sha256") != recorded["sha256"]
                or int(summary_bank.get("file_size_bytes", -1))
                != int(recorded["size_bytes"])
            ):
                return None
        if not all(validate_file_binding(item) for item in retained):
            return None
        return receipt
    except (OSError, ValueError, TypeError, KeyError):
        return None
