#!/usr/bin/env python3
"""Transfer one WB95-optimized R4RR alignment loss to ImageNet-9."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from imagenet9_final_utils import atomic_json, parse_seeds, write_method_tables
from run_imagenet9_final_baseline import ensure_contract, parse_training_result, run_logged
from run_r4rr_alignment_best5 import resolve_sweep_csv, select_best_row


LOSSES = ("reverse_kl", "jensen_shannon", "squared_l2", "cosine")
SOURCE_IMAGES = 4795
SOURCE_EPOCHS = 200
TARGET_IMAGES = 45405
TARGET_EPOCHS = int(SOURCE_IMAGES * SOURCE_EPOCHS / TARGET_IMAGES + 0.5)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-loss", choices=LOSSES, required=True)
    parser.add_argument("--sweep-csv", type=Path)
    parser.add_argument("--sweep-log-dir", type=Path, required=True)
    parser.add_argument("--min-sweep-trials", type=int, default=50)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-map-root", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-test-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["0,1,2,3,4"])
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exposure_scaled_epoch(source_epoch: int) -> int:
    scaled = int(source_epoch * SOURCE_IMAGES / TARGET_IMAGES + 0.5)
    return max(0, min(TARGET_EPOCHS - 1, scaled))


def result_method(loss: str) -> str:
    return f"r4rr_wb95_transfer_{loss}_klincr0"


def resolve_source(args: argparse.Namespace) -> Path:
    contract_path = args.run_root / "run_contract.json"
    if args.sweep_csv is not None:
        return args.sweep_csv.resolve()
    if contract_path.is_file():
        stored = json.loads(contract_path.read_text())
        source = stored.get("source_sweep_csv")
        if source:
            return Path(str(source)).resolve()
    return Path(
        resolve_sweep_csv(
            str(args.sweep_log_dir),
            "wb95",
            args.alignment_loss,
            args.min_sweep_trials,
        )
    ).resolve()


def load_selection(args: argparse.Namespace) -> Dict[str, object]:
    source = resolve_source(args)
    if not source.is_file():
        raise FileNotFoundError(source)
    best, row_count = select_best_row(
        str(source), args.alignment_loss, args.min_sweep_trials
    )
    source_attention = int(best["attention_epoch"])
    target_attention = exposure_scaled_epoch(source_attention)
    source_increment = float(best.get("kl_incr", 0.0))
    return {
        "method": "r4rr",
        "result_method": result_method(args.alignment_loss),
        "alignment_loss": args.alignment_loss,
        "hyperparameter_selection": "waterbirds95_validation_transfer",
        "source_sweep_csv": str(source),
        "source_sweep_sha256": sha256(source),
        "source_valid_trials": row_count,
        "source_best_trial": int(best["trial"]),
        "source_best_validation_value": float(best["best_balanced_val_acc"]),
        "source_hparams": {
            "attention_epoch": source_attention,
            "kl_lambda": float(best["kl_lambda"]),
            "kl_increment": source_increment,
            "base_lr": float(best["base_lr"]),
            "classifier_lr": float(best["classifier_lr"]),
            "lr2_mult": float(best["lr2_mult"]),
        },
        "target_hparams": {
            "epochs": TARGET_EPOCHS,
            "attention_epoch": target_attention,
            "kl_lambda": float(best["kl_lambda"]),
            "kl_increment": 0.0,
            "base_lr": float(best["base_lr"]),
            "classifier_lr": float(best["classifier_lr"]),
            "lr2_mult": float(best["lr2_mult"]),
        },
        "exposure_scaling": {
            "source_train_images": SOURCE_IMAGES,
            "source_epochs": SOURCE_EPOCHS,
            "source_total_exposures": SOURCE_IMAGES * SOURCE_EPOCHS,
            "target_train_images": TARGET_IMAGES,
            "target_epochs": TARGET_EPOCHS,
            "target_total_exposures": TARGET_IMAGES * TARGET_EPOCHS,
        },
        "fixed": {
            "momentum": 0.9,
            "weight_decay": 1e-5,
            "pretrained": True,
        },
        "teacher_map_root": str(args.teacher_map_root.resolve()),
        "imagenet9_validation_use": "checkpoint_selection_only",
        "official_variants_used_for_hparam_selection": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    for path in (
        args.manifest,
        args.official_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.sweep_log_dir,
        args.teacher_map_root,
        args.official_test_root,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)

    args.run_root.mkdir(parents=True, exist_ok=True)
    selection = load_selection(args)
    selection["seeds"] = seeds
    selection["manifest"] = str(args.manifest.resolve())
    ensure_contract(args.run_root / "run_contract.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)

    params = selection["target_hparams"]
    fixed = selection["fixed"]
    label = str(selection["result_method"])
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
                args.python, "-u", str(script_root / "train_imagenet9_r4rr.py"),
                "--method", "r4rr",
                "--manifest", str(args.manifest),
                "--teacher-map-root", str(args.teacher_map_root),
                "--seed", str(seed),
                "--epochs", str(params["epochs"]),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--attention-epoch", str(params["attention_epoch"]),
                "--kl-lambda", str(params["kl_lambda"]),
                "--kl-increment", "0",
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
            print(f"[RESUME] method={label} seed={seed} training", flush=True)

        training = json.loads(training_json.read_text())
        if training.get("method") != "r4rr" or int(training.get("seed", -1)) != seed:
            raise RuntimeError(f"Stored training result does not match {label} seed {seed}")
        if training.get("alignment_loss") != args.alignment_loss:
            raise RuntimeError(f"Stored alignment loss does not match {args.alignment_loss}")
        if float(training.get("kl_increment", -1)) != 0.0:
            raise RuntimeError("Stored transferred run did not use kl_increment=0")
        if Path(str(training.get("checkpoint", ""))).resolve() != checkpoint.resolve():
            raise RuntimeError(f"Stored checkpoint does not match {checkpoint}")

        if not evaluation_json.is_file():
            command = [
                args.python, "-u",
                str(script_root / "evaluate_imagenet9_final_checkpoint.py"),
                "--method", label,
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
        evaluated = json.loads(evaluation_json.read_text())
        if evaluated.get("method") != label or int(evaluated.get("seed", -1)) != seed:
            raise RuntimeError(f"Stored evaluation does not match {label} seed {seed}")
        evaluations.append(evaluation_json)
        write_method_tables(label, args.run_root, evaluations)
        print(f"[SEED DONE] method={label} seed={seed}", flush=True)

    write_method_tables(label, args.run_root, evaluations)
    print(f"[DONE] {args.run_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
