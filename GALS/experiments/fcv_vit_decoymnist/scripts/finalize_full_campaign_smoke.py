#!/usr/bin/env python3
"""Validate the real one-epoch smoke and issue the full-array launch gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_campaign_preflight import (  # noqa: E402
    source_tree_provenance,
    validate_preflight_receipt,
)
from decoy_full_config import (  # noqa: E402
    canonical_config_sha256,
    enumerate_runs,
    load_and_validate_config,
    sha256_file,
)
from decoy_manifest_provenance import atomic_json  # noqa: E402
from decoy_online_study import (  # noqa: E402
    _validate_completed_run,
    assert_no_forbidden_persistence,
)


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "decoymnist_vit_s16_fcv_full_online.yaml"
)


def _bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def smoke_projections(
    *,
    observed_seconds: float,
    observed_workspace_bytes: int,
    current_campaign_bytes: int,
    expected_candidates: int,
    production_epochs: int,
    runtime_safety_factor: float = 1.5,
    storage_safety_factor: float = 2.0,
    task_limit_seconds: float = 86400.0,
    storage_budget_bytes: int,
) -> tuple[dict, dict]:
    if observed_seconds <= 0 or observed_workspace_bytes <= 0:
        raise ValueError("Smoke must produce positive runtime and aggregate storage.")
    projected_run_seconds = observed_seconds * production_epochs * runtime_safety_factor
    fixed_campaign_bytes = max(0, current_campaign_bytes - observed_workspace_bytes)
    projected_campaign_bytes = int(
        fixed_campaign_bytes
        + observed_workspace_bytes * expected_candidates * storage_safety_factor
    )
    runtime = {
        "observed_one_epoch_seconds": float(observed_seconds),
        "production_epochs": int(production_epochs),
        "safety_factor": float(runtime_safety_factor),
        "projected_task_seconds": float(projected_run_seconds),
        "task_limit_seconds": float(task_limit_seconds),
        "within_task_limit": bool(projected_run_seconds <= task_limit_seconds),
    }
    storage = {
        "observed_smoke_workspace_bytes": int(observed_workspace_bytes),
        "fixed_campaign_bytes": int(fixed_campaign_bytes),
        "expected_candidate_rows": int(expected_candidates),
        "safety_factor": float(storage_safety_factor),
        "projected_campaign_bytes": projected_campaign_bytes,
        "budget_bytes": int(storage_budget_bytes),
        "within_budget": bool(projected_campaign_bytes <= storage_budget_bytes),
    }
    return runtime, storage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    campaign_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    workspace = campaign_root / "preflight" / "online_smoke_workspace"
    preflight = validate_preflight_receipt(
        config, campaign_root / "preflight" / "preflight_receipt.json"
    )
    run = enumerate_runs(config)[0]
    summary = _validate_completed_run(
        config,
        run,
        workspace,
        expected_epochs=1,
        artifact_type="fcv_vit_decoymnist_online_smoke_run_summary",
    )
    if summary is None:
        raise RuntimeError("The one-epoch production smoke is incomplete.")
    if summary.get("execution_mode") != "one_epoch_production_path_smoke":
        raise RuntimeError("The smoke summary has the wrong execution mode.")
    if summary.get("preflight_receipt_sha256") != preflight["artifact_sha256"]:
        raise RuntimeError("The smoke did not use the current preflight receipt.")
    if summary.get("pretrained_backbone_sha256") != preflight["pretrained_backbone_sha256"]:
        raise RuntimeError("The smoke did not use the preflighted pretrained backbone.")
    forbidden = assert_no_forbidden_persistence(config, campaign_root)

    workspace_bytes = _bytes(workspace)
    campaign_bytes = _bytes(campaign_root)
    runtime, storage = smoke_projections(
        observed_seconds=float(summary["seconds"]),
        observed_workspace_bytes=workspace_bytes,
        current_campaign_bytes=campaign_bytes,
        expected_candidates=int(config["candidate_pool"]["expected_candidate_states"]),
        production_epochs=int(config["training"]["epochs"]),
        storage_budget_bytes=int(
            float(config["storage"]["persistent_output_budget_gib"]) * 1024**3
        ),
    )
    if not runtime["within_task_limit"]:
        raise RuntimeError(f"Smoke runtime projection exceeds one day: {runtime}")
    if not storage["within_budget"]:
        raise RuntimeError(f"Smoke storage projection exceeds one GiB: {storage}")

    source = source_tree_provenance()
    smoke_summary_path = workspace / "run_summaries" / f"{run.run_id}.json"
    gate = {
        "artifact_type": "fcv_vit_decoymnist_launch_gate",
        "artifact_version": 1,
        "status": "PASS",
        "config_sha256": canonical_config_sha256(config),
        "source_tree_sha256": source["source_tree_sha256"],
        "preflight_receipt_path": preflight["artifact_path"],
        "preflight_receipt_sha256": preflight["artifact_sha256"],
        "pretrained_backbone_sha256": preflight["pretrained_backbone_sha256"],
        "smoke_summary_path": str(smoke_summary_path),
        "smoke_summary_sha256": sha256_file(smoke_summary_path),
        "runtime_projection": runtime,
        "storage_projection": storage,
        "no_checkpoint_artifacts_verified": forbidden["violations"] == 0,
        "test_metrics_used_for_training_or_selection": False,
    }
    gate_path = campaign_root / "preflight" / "launch_gate.json"
    atomic_json(gate, gate_path)
    print(json.dumps({**gate, "artifact_path": str(gate_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
