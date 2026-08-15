#!/usr/bin/env python3
"""Run final five-seed ImageNet-9 evaluation for one Optuna-tuned CNN baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from imagenet9_final_utils import atomic_json, parse_seeds, write_method_tables


METHODS = ("erm", "upweight", "abn", "elrep")
RESULT_PREFIX = "[RESULT] "


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-test-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["0,1,2,3,4"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--abn-checkpoint", type=Path)
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


def parse_training_result(log_path: Path) -> Mapping[str, object]:
    result = None
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
    if result is None:
        raise RuntimeError(f"Missing {RESULT_PREFIX.strip()} in {log_path}")
    return result


def load_selection(args: argparse.Namespace) -> Dict[str, object]:
    if not args.sweep_summary.is_file():
        raise FileNotFoundError(args.sweep_summary)
    summary = json.loads(args.sweep_summary.read_text())
    if summary.get("method") != args.method:
        raise RuntimeError(
            f"Sweep method mismatch: expected={args.method} stored={summary.get('method')}"
        )
    if int(summary.get("complete_trials", 0)) < int(summary.get("target_complete_trials", 50)):
        raise RuntimeError(f"Sweep is incomplete: {args.sweep_summary}")
    if summary.get("objective") != "val_macro_class_accuracy":
        raise RuntimeError(f"Unexpected sweep objective: {summary.get('objective')}")
    if summary.get("official_variants_used_for_selection") is not False:
        raise RuntimeError("Sweep summary does not certify held-out official variants")
    params = dict(summary["best_params"])
    contract = summary["contract"]
    params.setdefault("momentum", contract.get("fixed_momentum", 0.9))
    params.setdefault("abn_cls_weight", 1.0)
    params.setdefault("theta1", 1e-4)
    params.setdefault("theta2", 1e-5)
    return {
        "method": args.method,
        "source_summary": str(args.sweep_summary.resolve()),
        "best_trial": summary["best_trial"],
        "best_value": summary["best_value"],
        "best_params": params,
        "fixed": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": contract.get("weight_decay", 1e-5),
            "nesterov": False,
            "pretrained": True,
        },
        "official_variants_used_for_selection": False,
    }


def ensure_contract(path: Path, contract: Mapping[str, object]) -> None:
    if path.is_file():
        if json.loads(path.read_text()) != contract:
            raise RuntimeError(f"Refusing to resume changed final contract: {path}")
    else:
        atomic_json(path, contract)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    selection = load_selection(args)
    selection["seeds"] = seeds
    args.run_root.mkdir(parents=True, exist_ok=True)
    ensure_contract(args.run_root / "run_contract.json", selection)
    params = selection["best_params"]
    fixed = selection["fixed"]
    script_root = Path(__file__).resolve().parent
    evaluations: List[Path] = []

    for seed in seeds:
        seed_root = args.run_root / f"seed_{seed}"
        checkpoint = seed_root / "best_checkpoint.pt"
        training_json = seed_root / "training_result.json"
        training_log = seed_root / "training.log"
        evaluation_json = seed_root / "official_evaluation.json"
        if not (checkpoint.is_file() and training_json.is_file()):
            evaluation_json.unlink(missing_ok=True)
            command = [
                args.python,
                "-u",
                str(script_root / "train_imagenet9_baseline.py"),
                "--method",
                args.method,
                "--manifest",
                str(args.manifest),
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--base-lr",
                str(params["base_lr"]),
                "--classifier-lr",
                str(params["classifier_lr"]),
                "--momentum",
                str(params["momentum"]),
                "--weight-decay",
                str(fixed["weight_decay"]),
                "--abn-cls-weight",
                str(params["abn_cls_weight"]),
                "--theta1",
                str(params["theta1"]),
                "--theta2",
                str(params["theta2"]),
                "--checkpoint",
                str(checkpoint),
                "--device",
                args.device,
                "--skip-file-checks",
            ]
            if args.method == "abn":
                if args.abn_checkpoint is None or not args.abn_checkpoint.is_file():
                    raise FileNotFoundError(args.abn_checkpoint)
                command.extend(["--abn-checkpoint", str(args.abn_checkpoint)])
            run_logged(command, training_log)
            atomic_json(training_json, parse_training_result(training_log))
        else:
            print(f"[RESUME] method={args.method} seed={seed} training", flush=True)

        training = json.loads(training_json.read_text())
        if training.get("method") != args.method or int(training.get("seed", -1)) != seed:
            raise RuntimeError(f"Stored training result does not match {args.method} seed {seed}")
        if Path(training.get("checkpoint", "")).resolve() != checkpoint.resolve():
            raise RuntimeError(f"Stored checkpoint path does not match {checkpoint}")
        if not evaluation_json.is_file():
            command = [
                args.python,
                "-u",
                str(script_root / "evaluate_imagenet9_final_checkpoint.py"),
                "--method",
                args.method,
                "--seed",
                str(seed),
                "--checkpoint",
                str(checkpoint),
                "--official-manifest",
                str(args.official_manifest),
                "--official-test-root",
                str(args.official_test_root),
                "--output-json",
                str(evaluation_json),
                "--selection-value",
                str(training["best_val_macro_class_accuracy"]),
                "--batch-size",
                str(args.eval_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--skip-file-checks",
            ]
            run_logged(command, seed_root / "evaluation.log")
        else:
            print(f"[RESUME] method={args.method} seed={seed} evaluation", flush=True)
        evaluated = json.loads(evaluation_json.read_text())
        if evaluated.get("method") != args.method or int(evaluated.get("seed", -1)) != seed:
            raise RuntimeError(f"Stored evaluation does not match {args.method} seed {seed}")
        evaluations.append(evaluation_json)
        write_method_tables(args.method, args.run_root, evaluations)
        print(f"[SEED DONE] method={args.method} seed={seed}", flush=True)

    write_method_tables(args.method, args.run_root, evaluations)
    print(f"[DONE] {args.run_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
