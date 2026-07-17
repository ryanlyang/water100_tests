#!/usr/bin/env python3
"""Aggregate the nine unmodified-DecoyMNIST ViT susceptibility runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = "/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility"
EXPECTED_RUNS = 9
EXPECTED_EPOCHS = 10


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise RuntimeError("susceptibility_gate_passed contains non-boolean values")
    return normalized.eq("true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-runs", type=int, default=EXPECTED_RUNS)
    parser.add_argument("--expected-epochs", type=int, default=EXPECTED_EPOCHS)
    args = parser.parse_args()
    root = Path(args.output_root).expanduser().resolve()
    metric_paths = sorted((root / "runs").glob("*/metrics.csv"))
    if len(metric_paths) != args.expected_runs:
        raise RuntimeError(
            f"Expected {args.expected_runs} metrics files, found {len(metric_paths)}"
        )
    frames = [pd.read_csv(path) for path in metric_paths]
    matrix = pd.concat(frames, ignore_index=True)
    expected_rows = args.expected_runs * args.expected_epochs
    if len(matrix) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(matrix)}")
    counts = matrix.groupby("run_index")["epoch"].nunique()
    if set(counts.tolist()) != {args.expected_epochs}:
        raise RuntimeError(f"Incomplete run epoch counts: {counts.to_dict()}")
    if matrix["split_sha256"].nunique() != 1:
        raise RuntimeError("The nine jobs did not use one identical train holdout")
    if matrix.duplicated(["run_index", "epoch"]).any():
        raise RuntimeError("Duplicate run/epoch rows found")

    matrix = matrix.sort_values(["run_index", "epoch"]).reset_index(drop=True)
    final = matrix.groupby("run_index", as_index=False).tail(1).copy()
    matrix["susceptibility_gate_passed"] = _boolean_series(
        matrix["susceptibility_gate_passed"]
    )
    run_pass = matrix.groupby("run_index")["susceptibility_gate_passed"].max()
    seed_pass = matrix.groupby("seed")["susceptibility_gate_passed"].max()
    best_gap_index = matrix.groupby("run_index")[
        "biased_val_to_reversed_test_gap"
    ].idxmax()
    best_by_run = matrix.loc[best_gap_index].sort_values("run_index").copy()
    best_overall = matrix.loc[matrix["biased_val_to_reversed_test_gap"].idxmax()]
    transfer_supported = bool(seed_pass.astype(bool).all())

    root.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(root / "all_epoch_metrics.csv", index=False)
    final.to_csv(root / "final_epoch_metrics.csv", index=False)
    best_by_run.to_csv(root / "max_gap_by_run.csv", index=False)
    report = {
        "artifact_type": "decoymnist_vit_susceptibility_pilot_summary",
        "complete": True,
        "unmodified_decoymnist": True,
        "training_runs": int(matrix["run_index"].nunique()),
        "candidate_epochs": int(len(matrix)),
        "seeds": sorted(int(value) for value in matrix["seed"].unique()),
        "learning_rates": sorted(
            float(value) for value in matrix["learning_rate"].unique()
        ),
        "weight_decays": sorted(
            float(value) for value in matrix["weight_decay"].unique()
        ),
        "split_sha256": str(matrix["split_sha256"].iloc[0]),
        "candidate_epochs_passing_gate": int(
            matrix["susceptibility_gate_passed"].astype(bool).sum()
        ),
        "runs_with_passing_epoch": int(run_pass.astype(bool).sum()),
        "seeds_with_passing_epoch": int(seed_pass.astype(bool).sum()),
        "all_seeds_show_shortcut_susceptibility": transfer_supported,
        "recommended_decision": (
            "proceed_to_fcv_transfer_on_standard_decoymnist"
            if transfer_supported
            else "do_not_launch_full_fcv_transfer_without_reviewing_diagnostics"
        ),
        "strongest_observed_gap": {
            "run_id": str(best_overall["run_id"]),
            "epoch": int(best_overall["epoch"]),
            "biased_validation_accuracy": float(
                best_overall["biased_val_original_accuracy"]
            ),
            "reversed_test_accuracy": float(
                best_overall["reversed_test_original_accuracy"]
            ),
            "gap": float(best_overall["biased_val_to_reversed_test_gap"]),
            "biased_validation_digit_only_accuracy": float(
                best_overall["biased_val_digit_only_accuracy"]
            ),
            "biased_validation_patch_only_accuracy": float(
                best_overall["biased_val_patch_only_accuracy"]
            ),
        },
        "checkpoint_files_saved": 0,
    }
    with (root / "pilot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("\n=== DecoyMNIST ViT susceptibility pilot ===")
    print(
        f"complete candidates: {len(matrix)} "
        f"({report['training_runs']} runs x {args.expected_epochs} epochs)"
    )
    print(
        f"passing candidates: {report['candidate_epochs_passing_gate']} | "
        f"passing runs: {report['runs_with_passing_epoch']}/{args.expected_runs} | "
        f"passing seeds: {report['seeds_with_passing_epoch']}/3"
    )
    print(
        "strongest gap: "
        f"biased={report['strongest_observed_gap']['biased_validation_accuracy']*100:.2f}% "
        f"reversed_test={report['strongest_observed_gap']['reversed_test_accuracy']*100:.2f}% "
        f"gap={report['strongest_observed_gap']['gap']*100:.2f}pp"
    )
    print(f"decision: {report['recommended_decision']}")
    print(f"[DONE] {root / 'pilot_summary.json'}")


if __name__ == "__main__":
    main()
