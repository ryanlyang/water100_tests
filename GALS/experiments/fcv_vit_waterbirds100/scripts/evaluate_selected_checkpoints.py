#!/usr/bin/env python3
"""Run Step 10 on the unique checkpoints frozen by Step 9."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.candidate_training import validate_runtime_software  # noqa: E402
from fcv.test_evaluation import (  # noqa: E402
    assemble_final_test_results,
    evaluate_selected_checkpoint,
    load_frozen_selection,
    prepare_final_test_source,
)


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Step 9 selections on Waterbirds100 test data."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection-table", type=Path, default=None)
    parser.add_argument("--selection-summary", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--candidate-output-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-selection-only",
        action="store_true",
        help="Validate frozen Step 9 artifacts without opening the test manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    validate_runtime_software(config)
    output_root = Path(config["paths"]["output_root"])
    selection_root = output_root / config["outputs"]["selection_results"]
    selection_table = args.selection_table or selection_root / "selection_table.csv"
    selection_summary = args.selection_summary or (
        selection_root / "selection_table_summary.json"
    )
    frozen = load_frozen_selection(config, selection_table, selection_summary)
    frozen_description = {
        "selection_table": str(frozen.selection_table_path),
        "selection_table_sha256": frozen.selection_table_sha256,
        "selector_count": len(frozen.table),
        "unique_selected_checkpoint_count": len(frozen.unique_checkpoints),
        "unique_selected_checkpoints": [
            {
                "candidate_id": item.candidate_id,
                "checkpoint_path": str(item.checkpoint_path),
                "selectors": list(item.selectors),
            }
            for item in frozen.unique_checkpoints
        ],
    }
    print(json.dumps(frozen_description, indent=2, sort_keys=True), flush=True)
    if args.validate_selection_only:
        print("[VALID] Step 9 selection is frozen; test manifest was not opened.")
        return

    test_manifest = args.test_manifest or (
        output_root
        / config["outputs"]["split_manifests"]
        / "metadata_test_analysis_only.csv"
    )
    candidate_output_dir = args.candidate_output_dir or (
        selection_root / "final_test_scores"
    )
    output_csv = args.output_csv or selection_root / "final_test_results.csv"
    output_summary = args.output_summary or (
        selection_root / "final_test_results_summary.json"
    )
    source = prepare_final_test_source(
        config,
        test_manifest,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(
        json.dumps(
            {
                "test_manifest": str(source.manifest_path),
                "test_manifest_sha256": source.manifest_sha256,
                "test_sample_count": source.sample_count,
                "device": args.device,
                "precision": "float32",
                "test_metrics_affect_selection": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    candidate_summaries = {}
    for index, selected in enumerate(frozen.unique_checkpoints, start=1):
        result = evaluate_selected_checkpoint(
            config,
            selected,
            source,
            candidate_output_dir,
            device=args.device,
            overwrite=args.overwrite,
        )
        candidate_summaries[selected.candidate_id] = result
        metrics = result["metrics"]
        print(
            f"[TEST] {index}/{len(frozen.unique_checkpoints)} "
            f"candidate={selected.candidate_id} "
            f"status={result['invocation_status']} "
            f"accuracy={metrics['accuracy']:.6f} "
            f"balanced_group={metrics['balanced_group_accuracy']:.6f} "
            f"worst_group={metrics['worst_group_accuracy']:.6f}",
            flush=True,
        )
    summary = assemble_final_test_results(
        config,
        frozen,
        source,
        candidate_summaries,
        output_csv,
        output_summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
