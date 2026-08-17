#!/usr/bin/env python3
"""Run final five-seed ImageNet-9 evaluation for one tuned R4RR loss."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from imagenet9_final_utils import atomic_json, parse_seeds, write_method_tables
from run_imagenet9_final_baseline import (
    ensure_contract,
    parse_training_result,
    run_logged,
)


ALIGNMENT_LOSSES = (
    "forward_kl",
    "reverse_kl",
    "jensen_shannon",
    "squared_l2",
    "cosine",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-loss", choices=ALIGNMENT_LOSSES, required=True)
    parser.add_argument(
        "--trial-number",
        type=int,
        help="Evaluate this completed sweep trial instead of the validation winner.",
    )
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-map-root", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-test-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["0,1,2,3,4"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def load_selection(args: argparse.Namespace) -> Dict[str, object]:
    if not args.sweep_summary.is_file():
        raise FileNotFoundError(args.sweep_summary)
    summary = json.loads(args.sweep_summary.read_text())
    if summary.get("method") != "r4rr":
        raise RuntimeError(f"Unexpected sweep method: {summary.get('method')}")
    if int(summary.get("complete_trials", 0)) < int(
        summary.get("target_complete_trials", 50)
    ):
        raise RuntimeError(f"R4RR sweep is incomplete: {args.sweep_summary}")
    if summary.get("objective") != "val_macro_class_accuracy":
        raise RuntimeError(f"Unexpected sweep objective: {summary.get('objective')}")
    if summary.get("official_variants_used_for_selection") is not False:
        raise RuntimeError("Sweep summary does not certify held-out official variants")
    contract = summary["contract"]
    if contract.get("alignment_loss") != args.alignment_loss:
        raise RuntimeError(
            f"Alignment mismatch: requested={args.alignment_loss} "
            f"stored={contract.get('alignment_loss')}"
        )
    teacher_contract = contract.get("r4rr_teacher_maps", {})
    selected_root = Path(str(teacher_contract.get("root", ""))).resolve()
    if args.teacher_map_root.resolve() != selected_root:
        raise RuntimeError(
            f"Teacher root differs from sweep contract: "
            f"{args.teacher_map_root.resolve()} != {selected_root}"
        )
    trial_number = getattr(args, "trial_number", None)
    if trial_number is None:
        selection_mode = "validation_best"
        selected_trial = int(summary["best_trial"])
        selected_value = float(summary["best_value"])
        selected_params = dict(summary["best_params"])
        result_method = f"r4rr_{args.alignment_loss}"
    else:
        trials_path = args.sweep_summary.with_name("trials.csv")
        if not trials_path.is_file():
            raise FileNotFoundError(trials_path)
        with trials_path.open(newline="") as handle:
            matches = [
                row
                for row in csv.DictReader(handle)
                if int(row["trial"]) == trial_number
            ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one row for trial {trial_number}, found {len(matches)}"
            )
        row = matches[0]
        if row["state"] != "COMPLETE":
            raise RuntimeError(
                f"Requested trial {trial_number} is not complete: {row['state']}"
            )
        if row["alignment_loss"] != args.alignment_loss:
            raise RuntimeError(
                f"Trial {trial_number} alignment mismatch: "
                f"{row['alignment_loss']} != {args.alignment_loss}"
            )
        selection_mode = "fixed_completed_trial"
        selected_trial = trial_number
        selected_value = float(row["objective"])
        selected_params = {
            "attention_epoch": int(row["attention_epoch"]),
            "kl_lambda": float(row["kl_lambda"]),
            "base_lr": float(row["base_lr"]),
            "classifier_lr": float(row["classifier_lr"]),
            "lr2_mult": float(row["lr2_mult"]),
        }
        result_method = f"r4rr_{args.alignment_loss}_trial{trial_number}"
    return {
        "method": "r4rr",
        "result_method": result_method,
        "alignment_loss": args.alignment_loss,
        "selection_mode": selection_mode,
        "source_summary": str(args.sweep_summary.resolve()),
        "sweep_best_trial": summary["best_trial"],
        "sweep_best_value": summary["best_value"],
        "selected_trial": selected_trial,
        "selected_value": selected_value,
        "best_params": selected_params,
        "teacher_map_root": str(args.teacher_map_root.resolve()),
        "fixed": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": contract.get("weight_decay", 1e-5),
            "momentum": contract.get("fixed_momentum", 0.9),
            "nesterov": False,
            "pretrained": True,
            "kl_increment": "kl_lambda/10_per_align_epoch",
        },
        "official_variants_used_for_selection": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    selection = load_selection(args)
    selection["seeds"] = seeds
    args.run_root.mkdir(parents=True, exist_ok=True)
    ensure_contract(args.run_root / "run_contract.json", selection)
    params = selection["best_params"]
    fixed = selection["fixed"]
    result_method = str(selection["result_method"])
    script_root = Path(__file__).resolve().parent
    evaluations: List[Path] = []

    for seed in seeds:
        seed_root = args.run_root / f"seed_{seed}"
        checkpoint = seed_root / "best_checkpoint.pt"
        training_json = seed_root / "training_result.json"
        evaluation_json = seed_root / "official_evaluation.json"
        if not (checkpoint.is_file() and training_json.is_file()):
            evaluation_json.unlink(missing_ok=True)
            command = [
                args.python,
                "-u",
                str(script_root / "train_imagenet9_r4rr.py"),
                "--method", "r4rr",
                "--manifest", str(args.manifest),
                "--teacher-map-root", str(args.teacher_map_root),
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--attention-epoch", str(params["attention_epoch"]),
                "--kl-lambda", str(params["kl_lambda"]),
                "--base-lr", str(params["base_lr"]),
                "--classifier-lr", str(params["classifier_lr"]),
                "--lr2-mult", str(params["lr2_mult"]),
                "--alignment-loss", args.alignment_loss,
                "--momentum", str(fixed["momentum"]),
                "--weight-decay", str(fixed["weight_decay"]),
                "--checkpoint", str(checkpoint),
                "--device", args.device,
                "--skip-file-checks",
            ]
            run_logged(command, seed_root / "training.log")
            atomic_json(training_json, parse_training_result(seed_root / "training.log"))
        else:
            print(f"[RESUME] method={result_method} seed={seed} training", flush=True)

        training = json.loads(training_json.read_text())
        if training.get("method") != "r4rr" or int(training.get("seed", -1)) != seed:
            raise RuntimeError(f"Stored R4RR training result does not match seed {seed}")
        if training.get("alignment_loss") != args.alignment_loss:
            raise RuntimeError(f"Stored R4RR alignment loss does not match seed {seed}")
        if Path(str(training.get("checkpoint", ""))).resolve() != checkpoint.resolve():
            raise RuntimeError(f"Stored checkpoint path does not match {checkpoint}")

        if not evaluation_json.is_file():
            command = [
                args.python,
                "-u",
                str(script_root / "evaluate_imagenet9_final_checkpoint.py"),
                "--method", result_method,
                "--seed", str(seed),
                "--checkpoint", str(checkpoint),
                "--official-manifest", str(args.official_manifest),
                "--official-test-root", str(args.official_test_root),
                "--output-json", str(evaluation_json),
                "--selection-value", str(training["best_val_macro_class_accuracy"]),
                "--batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.num_workers),
                "--device", args.device,
                "--skip-file-checks",
            ]
            run_logged(command, seed_root / "evaluation.log")
        else:
            print(f"[RESUME] method={result_method} seed={seed} evaluation", flush=True)
        evaluated = json.loads(evaluation_json.read_text())
        if evaluated.get("method") != result_method or int(
            evaluated.get("seed", -1)
        ) != seed:
            raise RuntimeError(
                f"Stored evaluation does not match {result_method} seed {seed}"
            )
        evaluations.append(evaluation_json)
        write_method_tables(result_method, args.run_root, evaluations)
        print(f"[SEED DONE] method={result_method} seed={seed}", flush=True)

    write_method_tables(result_method, args.run_root, evaluations)
    print(f"[DONE] {args.run_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
