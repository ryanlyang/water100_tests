#!/usr/bin/env python3
"""Run five seeds for one ImageNet-9 systematic teacher-corruption condition."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from imagenet9_final_utils import atomic_json, parse_seeds, write_method_tables
from imagenet9_systematic_corruption import (
    CONDITIONS,
    CLASS_COUNT,
    prepare_manifest,
    sha256_file,
)
from run_imagenet9_final_baseline import ensure_contract, parse_training_result, run_logged
from run_imagenet9_final_r4rr import load_selection


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--corruption-manifest-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sweep-summary", type=Path)
    source.add_argument("--transfer-config", type=Path)
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
    parser.add_argument("--kl-increment", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def read_metric_summary(path: Path) -> Dict[str, Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["metric"]: {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "n": int(row["n"]),
            }
            for row in csv.DictReader(handle)
        }


def build_selection_args(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt corruption-run arguments to the final R4RR selection contract."""
    return argparse.Namespace(
        sweep_summary=args.sweep_summary,
        alignment_loss="forward_kl",
        teacher_map_root=args.teacher_map_root,
        trial_number=None,
        kl_increment=args.kl_increment,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


def load_hyperparameter_selection(
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[str, object], str]:
    transfer_config = getattr(args, "transfer_config", None)
    if transfer_config is None:
        selection = load_selection(build_selection_args(args))
        source_fields: Dict[str, object] = {
            "source_sweep_summary_sha256": sha256_file(args.sweep_summary),
        }
        return selection, source_fields, "imagenet9_optuna"

    from run_imagenet9_wb95_transfer_5seed import load_config

    config = load_config(argparse.Namespace(method="r4rr", config=transfer_config))
    experiment = config["experiment"]
    fixed_config = config["fixed"]
    params = config["params"]
    if int(args.epochs) != int(experiment["target_standard_epochs"]):
        raise RuntimeError(
            f"Transfer protocol requires {experiment['target_standard_epochs']} epochs, "
            f"got {args.epochs}"
        )
    if int(args.batch_size) != int(experiment["target_batch_size"]):
        raise RuntimeError(
            f"Transfer protocol requires batch size {experiment['target_batch_size']}, "
            f"got {args.batch_size}"
        )
    if params["alignment_loss"] != "forward_kl":
        raise RuntimeError("WB95 transfer corruption requires forward KL")
    if float(params["kl_increment"]) != float(args.kl_increment):
        raise RuntimeError("WB95 transfer corruption requires the configured KL increment")

    selection = {
        "method": "r4rr",
        "result_method": "r4rr_wb95_transfer_systematic_klincr0",
        "alignment_loss": "forward_kl",
        "selection_mode": "waterbirds95_validation_transfer",
        "source_config": str(transfer_config.resolve()),
        "selected_trial": None,
        "selected_value": None,
        "best_params": {
            key: params[key]
            for key in (
                "attention_epoch",
                "kl_lambda",
                "base_lr",
                "classifier_lr",
                "lr2_mult",
            )
        },
        "teacher_map_root": str(args.teacher_map_root.resolve()),
        "fixed": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "weight_decay": fixed_config["weight_decay"],
            "momentum": fixed_config["momentum"],
            "nesterov": fixed_config["nesterov"],
            "pretrained": fixed_config["pretrained"],
            "kl_increment": float(params["kl_increment"]),
        },
        "hyperparameter_selection": "waterbirds95_validation_transfer",
        "imagenet9_validation_use": "checkpoint_selection_only",
        "official_variants_used_for_selection": False,
    }
    source_fields = {
        "source_transfer_config_sha256": sha256_file(transfer_config),
    }
    return selection, source_fields, "waterbirds95_transfer"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.kl_increment != 0.0:
        raise ValueError("The systematic corruption protocol locks --kl-increment=0")
    seeds = parse_seeds(args.seeds)
    if tuple(seeds) != (0, 1, 2, 3, 4):
        raise RuntimeError(f"The locked protocol requires seeds 0..4, got {seeds}")

    manifest, indices_path, manifest_sha256 = prepare_manifest(
        args.condition,
        args.manifest,
        args.corruption_manifest_root / args.condition,
        args.corruption_seed,
    )
    indices_sha256 = sha256_file(indices_path)
    selection, source_fields, hyperparameter_protocol = load_hyperparameter_selection(args)
    if hyperparameter_protocol == "imagenet9_optuna":
        result_method = f"r4rr_systematic_{args.condition}_klincr0"
    else:
        result_method = f"r4rr_wb95_transfer_systematic_{args.condition}_klincr0"
    selection.update(
        {
            "result_method": result_method,
            "study": "imagenet9_systematic_teacher_corruption",
            "condition": args.condition,
            "corruption_seed": args.corruption_seed,
            "corruption_manifest": manifest,
            "corruption_manifest_sha256": manifest_sha256,
            "corruption_indices": str(indices_path.resolve()),
            "corruption_indices_sha256": indices_sha256,
            **source_fields,
            "seeds": list(seeds),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "official_variants_used_for_selection": False,
        }
    )
    if hyperparameter_protocol == "waterbirds95_transfer":
        selection["hyperparameter_protocol"] = hyperparameter_protocol
    args.run_root.mkdir(parents=True, exist_ok=True)
    ensure_contract(args.run_root / "run_contract.json", selection)
    params = selection["best_params"]
    fixed = selection["fixed"]
    script_root = Path(__file__).resolve().parent
    evaluations: List[Path] = []

    print(
        f"[SETUP] condition={args.condition} corrupted={CLASS_COUNT}/45405 "
        f"seed={args.corruption_seed} manifest={indices_path}",
        flush=True,
    )
    print(f"[SETUP] corrupted_class_counts={manifest['corrupted_class_counts']}", flush=True)
    print(f"[LOCKED] best_params={params} kl_increment=0", flush=True)

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
                "--corruption-indices", str(indices_path),
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--attention-epoch", str(params["attention_epoch"]),
                "--kl-lambda", str(params["kl_lambda"]),
                "--kl-increment", "0",
                "--base-lr", str(params["base_lr"]),
                "--classifier-lr", str(params["classifier_lr"]),
                "--lr2-mult", str(params["lr2_mult"]),
                "--alignment-loss", "forward_kl",
                "--momentum", str(fixed["momentum"]),
                "--weight-decay", str(fixed["weight_decay"]),
                "--checkpoint", str(checkpoint),
                "--device", args.device,
                "--skip-file-checks",
            ]
            run_logged(command, seed_root / "training.log")
            atomic_json(training_json, parse_training_result(seed_root / "training.log"))
        else:
            print(f"[RESUME] condition={args.condition} seed={seed} training", flush=True)

        training = json.loads(training_json.read_text(encoding="utf-8"))
        expected = {
            "method": "r4rr",
            "seed": seed,
            "attention_epoch": int(params["attention_epoch"]),
            "alignment_loss": "forward_kl",
            "corrupted_examples_per_epoch": CLASS_COUNT,
            "corrupted_examples_seen": CLASS_COUNT * args.epochs,
            "corruption_indices_sha256": indices_sha256,
        }
        for key, value in expected.items():
            if training.get(key) != value:
                raise RuntimeError(
                    f"Stored training result mismatch for seed={seed} {key}: "
                    f"{training.get(key)!r} != {value!r}"
                )
        if float(training.get("kl_increment", -1)) != 0.0:
            raise RuntimeError(f"Stored seed {seed} does not use kl_increment=0")
        if Path(str(training.get("corruption_indices", ""))).resolve() != indices_path.resolve():
            raise RuntimeError(f"Stored seed {seed} uses different corruption indices")
        if Path(str(training.get("checkpoint", ""))).resolve() != checkpoint.resolve():
            raise RuntimeError(f"Stored checkpoint path differs for seed {seed}")

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
            print(f"[RESUME] condition={args.condition} seed={seed} evaluation", flush=True)
        evaluated = json.loads(evaluation_json.read_text(encoding="utf-8"))
        if evaluated.get("method") != result_method or int(evaluated.get("seed", -1)) != seed:
            raise RuntimeError(f"Stored official evaluation differs for seed {seed}")
        evaluations.append(evaluation_json)
        write_method_tables(result_method, args.run_root, evaluations)
        print(f"[SEED DONE] condition={args.condition} seed={seed}", flush=True)

    write_method_tables(result_method, args.run_root, evaluations)
    metrics = read_metric_summary(args.run_root / "summary.csv")
    corruption_summary: Dict[str, object] = {
        "protocol_version": manifest["protocol_version"],
        "dataset": "imagenet9",
        "study": "systematic_teacher_corruption",
        "condition": args.condition,
        "condition_type": manifest["condition_type"],
        "target_class": manifest["target_class"],
        "corruption_seed": args.corruption_seed,
        "corrupted_example_count": manifest["corrupted_example_count"],
        "corrupted_fraction_of_training": manifest["corrupted_fraction_of_training"],
        "corrupted_class_counts": manifest["corrupted_class_counts"],
        "corruption_operation": manifest["corruption_operation"],
        "corruption_manifest_sha256": manifest_sha256,
        "corruption_indices_sha256": indices_sha256,
        "selected_trial": selection["selected_trial"],
        "selected_value": selection["selected_value"],
        "best_params": params,
        "kl_increment": 0.0,
        "completed_seeds": list(seeds),
        "n_completed": len(seeds),
        "standard_deviation": "population",
        "metrics": metrics,
        "official_variants_used_for_selection": False,
    }
    if hyperparameter_protocol == "imagenet9_optuna":
        corruption_summary["source_sweep_summary"] = str(args.sweep_summary.resolve())
    else:
        corruption_summary.update(
            {
                "hyperparameter_protocol": "waterbirds95_transfer",
                "hyperparameter_selection": "waterbirds95_validation_transfer",
                "source_transfer_config": str(args.transfer_config.resolve()),
                "source_transfer_config_sha256": sha256_file(args.transfer_config),
                "imagenet9_validation_use": "checkpoint_selection_only",
            }
        )
    atomic_json(args.run_root / "corruption_summary.json", corruption_summary)
    print(f"[DONE] {args.run_root / 'corruption_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
