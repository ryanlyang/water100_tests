#!/usr/bin/env python3
"""Run AFR's native validation-selected grid and final IN-9 evaluation for five seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from imagenet9_final_utils import atomic_json, parse_seeds, write_method_tables


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-test-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["0,1,2,3,4"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-hours-per-seed", type=float, default=70.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        handle.write("[COMMAND] " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
    if completed.returncode:
        tail = log_path.read_text(errors="replace").splitlines()[-60:]
        print("\n".join(tail), flush=True)
        raise RuntimeError(f"Command failed with code {completed.returncode}: {log_path}")


def validate_source(args: argparse.Namespace) -> Mapping[str, object]:
    if not args.source_summary.is_file() or not args.source_contract.is_file():
        raise FileNotFoundError(
            f"Missing completed AFR source files: {args.source_summary}, {args.source_contract}"
        )
    summary = json.loads(args.source_summary.read_text())
    contract = json.loads(args.source_contract.read_text())
    if int(summary.get("completed_stage2_configurations", 0)) != 165:
        raise RuntimeError(f"AFR source grid is incomplete: {args.source_summary}")
    if summary.get("official_variants_used_for_selection") is not False:
        raise RuntimeError("AFR source does not certify held-out official variants")
    if contract.get("objective") != "val_macro_class_accuracy":
        raise RuntimeError(f"Unexpected AFR objective: {contract.get('objective')}")
    return contract


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    source = validate_source(args)
    args.run_root.mkdir(parents=True, exist_ok=True)
    final_contract = {
        "method": "afr",
        "source_summary": str(args.source_summary.resolve()),
        "source_contract": source,
        "seeds": seeds,
        "selection": "native 33x5 AFR grid independently within each seed",
        "official_variants_used_for_selection": False,
    }
    contract_path = args.run_root / "run_contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != final_contract:
        raise RuntimeError(f"Refusing to resume changed final AFR contract: {contract_path}")
    if not contract_path.is_file():
        atomic_json(contract_path, final_contract)

    script_root = Path(__file__).resolve().parent
    evaluations: List[Path] = []
    for seed in seeds:
        seed_root = args.run_root / f"seed_{seed}"
        afr_root = seed_root / "afr"
        summary_path = afr_root / "summary.json"
        complete = False
        if summary_path.is_file():
            stored = json.loads(summary_path.read_text())
            complete = int(stored.get("completed_stage2_configurations", 0)) == 165
            if complete:
                best_checkpoint = Path(stored["best"]["classifier_checkpoint"])
                complete = (afr_root / "stage1_final.pt").is_file() and best_checkpoint.is_file()
        if not complete:
            command = [
                args.python,
                "-u",
                str(script_root / "run_imagenet9_afr.py"),
                "--manifest",
                str(args.manifest),
                "--run-root",
                str(afr_root),
                "--seed",
                str(seed),
                "--split-seed",
                str(source["split_seed"]),
                "--stage1-prop",
                str(source["stage1_prop"]),
                "--stage1-epochs",
                str(source["stage1_epochs"]),
                "--stage1-lr",
                str(source["stage1_lr"]),
                "--stage1-weight-decay",
                str(source["stage1_weight_decay"]),
                "--stage1-momentum",
                str(source["stage1_momentum"]),
                "--stage2-epochs",
                str(source["stage2_epochs"]),
                "--stage2-lr",
                str(source["stage2_lr"]),
                "--batch-size",
                str(args.batch_size),
                "--embedding-batch-size",
                str(args.embedding_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--max-hours",
                str(args.max_hours_per_seed),
                "--device",
                args.device,
            ]
            run_logged(command, seed_root / "afr.log")
            if not summary_path.is_file():
                raise RuntimeError(f"AFR did not produce {summary_path}")
            stored = json.loads(summary_path.read_text())
            if int(stored.get("completed_stage2_configurations", 0)) != 165:
                raise RuntimeError(
                    f"AFR seed {seed} remains incomplete; resubmit the same final job"
                )
        else:
            print(f"[RESUME] method=afr seed={seed} native grid", flush=True)

        stored = json.loads(summary_path.read_text())
        best = stored["best"]
        if int(stored.get("completed_stage2_configurations", 0)) != 165:
            raise RuntimeError(f"AFR seed {seed} summary is not complete")
        evaluation_json = seed_root / "official_evaluation.json"
        if not evaluation_json.is_file():
            command = [
                args.python,
                "-u",
                str(script_root / "evaluate_imagenet9_final_checkpoint.py"),
                "--method",
                "afr",
                "--seed",
                str(seed),
                "--checkpoint",
                str(afr_root / "stage1_final.pt"),
                "--afr-classifier-checkpoint",
                str(best["classifier_checkpoint"]),
                "--official-manifest",
                str(args.official_manifest),
                "--official-test-root",
                str(args.official_test_root),
                "--output-json",
                str(evaluation_json),
                "--selection-value",
                str(best["best_val_macro_class_accuracy"]),
                "--batch-size",
                str(args.eval_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--skip-file-checks",
            ]
            run_logged(command, seed_root / "evaluation.log")
        evaluations.append(evaluation_json)
        write_method_tables("afr", args.run_root, evaluations)
        print(
            f"[SEED DONE] method=afr seed={seed} gamma={best['gamma']} "
            f"reg_coeff={best['reg_coeff']}",
            flush=True,
        )
    print(f"[DONE] {args.run_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
