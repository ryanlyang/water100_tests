#!/usr/bin/env python3
"""Prove GH200 interrupted/resumed training matches an uninterrupted run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.candidate_training import (  # noqa: E402
    METRIC_COLUMNS,
    get_sweep_run,
    validate_runtime_software,
)
from fcv.config import load_and_validate_config  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)
EXACT_METRIC_COLUMNS = [
    column
    for column in METRIC_COLUMNS
    if column not in {"checkpoint_path", "checkpoint_sha256", "epoch_seconds"}
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare every epoch of resumed and uninterrupted GH200 training."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--resumed-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    versions = validate_runtime_software(config)
    run = get_sweep_run(config, args.run_index)
    epochs = int(config["training"]["epochs"])
    run_dirs = {
        "resumed": args.resumed_root.expanduser().resolve() / run.run_id,
        "reference": args.reference_root.expanduser().resolve() / run.run_id,
    }
    frames = {}
    for name, run_dir in run_dirs.items():
        metrics_path = run_dir / "metrics.csv"
        summary_path = run_dir / "run_summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Incomplete {name} run: {run_dir}")
        frame = pd.read_csv(metrics_path).sort_values("epoch").reset_index(drop=True)
        if len(frame) != epochs:
            raise RuntimeError(f"{name} has {len(frame)} epochs; expected {epochs}.")
        frames[name] = frame

    left = frames["resumed"][EXACT_METRIC_COLUMNS]
    right = frames["reference"][EXACT_METRIC_COLUMNS]
    if list(left.columns) != list(right.columns):
        raise RuntimeError("Metric schemas differ.")
    for column in EXACT_METRIC_COLUMNS:
        left_values = left[column].to_numpy()
        right_values = right[column].to_numpy()
        if np.issubdtype(left_values.dtype, np.number):
            equal = np.array_equal(left_values, right_values, equal_nan=True)
        else:
            equal = np.array_equal(left_values.astype(str), right_values.astype(str))
        if not equal:
            raise RuntimeError(f"Resume equivalence failed for metric {column!r}.")

    compared_tensor_count = 0
    for epoch in range(1, epochs + 1):
        checkpoints = {
            name: load_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
            for name, run_dir in run_dirs.items()
        }
        for key in (
            "artifact_type",
            "candidate_id",
            "run",
            "epoch",
            "model",
            "training_fingerprint",
            "software_versions",
            "manifest_sha256",
        ):
            if checkpoints["resumed"].get(key) != checkpoints["reference"].get(key):
                raise RuntimeError(
                    f"Resume equivalence failed for checkpoint field {key!r} at epoch {epoch}."
                )
        left_state = checkpoints["resumed"]["model_state_dict"]
        right_state = checkpoints["reference"]["model_state_dict"]
        if set(left_state) != set(right_state):
            raise RuntimeError(f"Model-state keys differ at epoch {epoch}.")
        for key in left_state:
            if not torch.equal(left_state[key], right_state[key]):
                raise RuntimeError(
                    f"Model tensor {key!r} differs after resume at epoch {epoch}."
                )
            compared_tensor_count += 1

    report = {
        "schema_version": 1,
        "artifact_type": "fcv_vit_gh200_resume_equivalence",
        "status": "passed",
        "run_index": run.run_index,
        "run_id": run.run_id,
        "epochs_compared": epochs,
        "metric_columns_compared": EXACT_METRIC_COLUMNS,
        "model_tensors_compared": compared_tensor_count,
        "resumed_root": str(args.resumed_root.expanduser().resolve()),
        "reference_root": str(args.reference_root.expanduser().resolve()),
        "software_versions": versions,
    }
    output = args.output_report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
