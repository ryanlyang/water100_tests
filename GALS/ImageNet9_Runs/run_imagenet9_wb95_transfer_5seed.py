#!/usr/bin/env python3
"""Run one five-seed ImageNet-9 study with Waterbirds-95 hyperparameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml

from imagenet9_final_utils import atomic_json, parse_seeds, write_method_tables
from run_imagenet9_final_baseline import ensure_contract, parse_training_result, run_logged


METHODS = ("erm", "upweight", "abn", "elrep", "gals", "afr", "clip_lr", "r4rr")
STANDARD_METHODS = ("erm", "upweight", "abn", "elrep", "gals")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--official-test-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["0,1,2,3,4"])
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--abn-checkpoint", type=Path)
    parser.add_argument("--gals-map-root", type=Path)
    parser.add_argument("--r4rr-map-root", type=Path)
    parser.add_argument("--clip-feature-cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(args: argparse.Namespace) -> Dict[str, object]:
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    payload = yaml.safe_load(args.config.read_text())
    experiment = payload["experiment"]
    fixed = payload["fixed"]
    methods = payload["methods"]
    params = dict(methods[args.method])

    source_images = int(experiment["source_train_images"])
    source_epochs = int(experiment["source_standard_epochs"])
    target_images = int(experiment["target_train_images"])
    expected_epochs = int(source_images * source_epochs / target_images + 0.5)
    if int(experiment["target_standard_epochs"]) != expected_epochs:
        raise RuntimeError("Configured target epochs do not match exposure scaling")
    if int(experiment["target_standard_exposures"]) != target_images * expected_epochs:
        raise RuntimeError("Configured target exposure count is inconsistent")
    if args.method == "r4rr":
        expected_attention = int(
            int(params["source_attention_epoch"]) * source_images / target_images + 0.5
        )
        if int(params["attention_epoch"]) != expected_attention:
            raise RuntimeError("R4RR attention epoch does not match exposure scaling")
        if float(params["kl_increment"]) != 0.0:
            raise RuntimeError("The WB95 transfer contract requires kl_increment=0")
    if args.method == "afr":
        stage1_images = int(target_images * float(params["target_stage1_fraction"]))
        expected_stage1 = int(
            source_images * int(params["source_stage1_epochs"]) / stage1_images + 0.5
        )
        if int(params["target_stage1_epochs"]) != expected_stage1:
            raise RuntimeError("AFR stage-one epochs do not match exposure scaling")

    return {
        "experiment": experiment,
        "fixed": fixed,
        "params": params,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
    }


def result_method(method: str) -> str:
    if method == "r4rr":
        # The evaluator intentionally recognizes R4RR checkpoints by prefix.
        return "r4rr_wb95_transfer_klincr0"
    return f"wb95_transfer_{method}"


def relabel_evaluation(path: Path, expected_native: str, label: str) -> None:
    payload = json.loads(path.read_text())
    stored = payload.get("method")
    if stored == label:
        return
    if stored != expected_native:
        raise RuntimeError(f"Unexpected evaluation method in {path}: {stored}")
    payload["native_method"] = expected_native
    payload["method"] = label
    payload["hyperparameter_selection"] = "waterbirds95_validation_transfer"
    atomic_json(path, payload)


def evaluate_checkpoint(
    args: argparse.Namespace,
    native_method: str,
    label: str,
    seed: int,
    checkpoint: Path,
    selection_value: float,
    output: Path,
    afr_classifier: Optional[Path] = None,
) -> None:
    if not output.is_file():
        command = [
            args.python,
            "-u",
            str(Path(__file__).resolve().parent / "evaluate_imagenet9_final_checkpoint.py"),
            "--method", native_method,
            "--seed", str(seed),
            "--checkpoint", str(checkpoint),
            "--official-manifest", str(args.official_manifest),
            "--official-test-root", str(args.official_test_root),
            "--output-json", str(output),
            "--selection-value", str(selection_value),
            "--batch-size", str(args.eval_batch_size),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--skip-file-checks",
        ]
        if afr_classifier is not None:
            command.extend(["--afr-classifier-checkpoint", str(afr_classifier)])
        run_logged(command, output.parent / "evaluation.log")
    relabel_evaluation(output, native_method, label)


def run_standard_or_r4rr(
    args: argparse.Namespace,
    config: Mapping[str, object],
    seeds: Sequence[int],
) -> None:
    params = config["params"]
    fixed = config["fixed"]
    experiment = config["experiment"]
    assert isinstance(params, Mapping) and isinstance(fixed, Mapping)
    epochs = int(experiment["target_standard_epochs"])
    native_method = "r4rr" if args.method == "r4rr" else str(params["trainer"])
    label = result_method(args.method)
    evaluations: List[Path] = []
    script_root = Path(__file__).resolve().parent

    for seed in seeds:
        seed_root = args.run_root / f"seed_{seed}"
        checkpoint = seed_root / "best_checkpoint.pt"
        training_json = seed_root / "training_result.json"
        evaluation_json = seed_root / "official_evaluation.json"
        if not (checkpoint.is_file() and training_json.is_file()):
            evaluation_json.unlink(missing_ok=True)
            if args.method == "r4rr":
                if args.r4rr_map_root is None or not args.r4rr_map_root.is_dir():
                    raise FileNotFoundError(args.r4rr_map_root)
                command = [
                    args.python, "-u", str(script_root / "train_imagenet9_r4rr.py"),
                    "--method", "r4rr",
                    "--manifest", str(args.manifest),
                    "--teacher-map-root", str(args.r4rr_map_root),
                    "--seed", str(seed),
                    "--epochs", str(epochs),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--attention-epoch", str(params["attention_epoch"]),
                    "--kl-lambda", str(params["kl_lambda"]),
                    "--kl-increment", str(params["kl_increment"]),
                    "--base-lr", str(params["base_lr"]),
                    "--classifier-lr", str(params["classifier_lr"]),
                    "--lr2-mult", str(params["lr2_mult"]),
                    "--alignment-loss", str(params["alignment_loss"]),
                    "--momentum", str(fixed["momentum"]),
                    "--weight-decay", str(fixed["weight_decay"]),
                    "--checkpoint", str(checkpoint),
                    "--device", args.device,
                    "--skip-file-checks",
                ]
            else:
                command = [
                    args.python, "-u", str(script_root / "train_imagenet9_baseline.py"),
                    "--method", native_method,
                    "--manifest", str(args.manifest),
                    "--seed", str(seed),
                    "--epochs", str(epochs),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--base-lr", str(params["base_lr"]),
                    "--classifier-lr", str(params["classifier_lr"]),
                    "--momentum", str(params.get("momentum", fixed["momentum"])),
                    "--weight-decay", str(fixed["weight_decay"]),
                    "--abn-cls-weight", str(params.get("abn_cls_weight", 1.0)),
                    "--theta1", str(params.get("theta1", 1e-4)),
                    "--theta2", str(params.get("theta2", 1e-5)),
                    "--checkpoint", str(checkpoint),
                    "--device", args.device,
                    "--skip-file-checks",
                ]
                if args.method == "abn":
                    if args.abn_checkpoint is None or not args.abn_checkpoint.is_file():
                        raise FileNotFoundError(args.abn_checkpoint)
                    command.extend(["--abn-checkpoint", str(args.abn_checkpoint)])
                if args.method == "gals":
                    if args.gals_map_root is None or not args.gals_map_root.is_dir():
                        raise FileNotFoundError(args.gals_map_root)
                    command.extend(
                        [
                            "--gals-map-root", str(args.gals_map_root),
                            "--grad-weight", str(params["grad_weight"]),
                            "--grad-criterion", str(params["grad_criterion"]),
                        ]
                    )
            run_logged(command, seed_root / "training.log")
            atomic_json(training_json, parse_training_result(seed_root / "training.log"))
        else:
            print(f"[RESUME] method={label} seed={seed} training", flush=True)

        training = json.loads(training_json.read_text())
        if training.get("method") != native_method or int(training.get("seed", -1)) != seed:
            raise RuntimeError(f"Stored training result does not match {label} seed {seed}")
        if Path(str(training.get("checkpoint", ""))).resolve() != checkpoint.resolve():
            raise RuntimeError(f"Stored checkpoint does not match {checkpoint}")
        if args.method == "r4rr" and float(training.get("kl_increment", -1)) != 0.0:
            raise RuntimeError("Stored transferred R4RR run did not use kl_increment=0")
        eval_native = label if args.method == "r4rr" else native_method
        evaluate_checkpoint(
            args,
            eval_native,
            label,
            seed,
            checkpoint,
            float(training["best_val_macro_class_accuracy"]),
            evaluation_json,
        )
        evaluations.append(evaluation_json)
        write_method_tables(label, args.run_root, evaluations)
        print(f"[SEED DONE] method={label} seed={seed}", flush=True)


def run_afr(
    args: argparse.Namespace,
    config: Mapping[str, object],
    seeds: Sequence[int],
) -> None:
    params = config["params"]
    assert isinstance(params, Mapping)
    label = result_method("afr")
    evaluations: List[Path] = []
    script_root = Path(__file__).resolve().parent
    for seed in seeds:
        seed_root = args.run_root / f"seed_{seed}"
        afr_root = seed_root / "afr"
        summary_path = afr_root / "summary.json"
        complete = False
        if summary_path.is_file():
            stored = json.loads(summary_path.read_text())
            complete = int(stored.get("completed_stage2_configurations", 0)) == 1
            if complete:
                complete = (
                    (afr_root / "stage1_final.pt").is_file()
                    and Path(stored["best"]["classifier_checkpoint"]).is_file()
                )
        if not complete:
            command = [
                args.python, "-u", str(script_root / "run_imagenet9_afr.py"),
                "--manifest", str(args.manifest),
                "--run-root", str(afr_root),
                "--seed", str(seed),
                "--split-seed", str(params["split_seed"]),
                "--stage1-prop", str(params["target_stage1_fraction"]),
                "--stage1-epochs", str(params["target_stage1_epochs"]),
                "--stage1-lr", str(params["stage1_lr"]),
                "--stage1-weight-decay", str(params["stage1_weight_decay"]),
                "--stage1-momentum", str(params["stage1_momentum"]),
                "--stage2-epochs", str(params["stage2_epochs"]),
                "--stage2-lr", str(params["stage2_lr"]),
                "--fixed-gamma", str(params["gamma"]),
                "--fixed-reg-coeff", str(params["reg_coeff"]),
                "--batch-size", str(args.batch_size),
                "--embedding-batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.num_workers),
                "--device", args.device,
            ]
            run_logged(command, seed_root / "training.log")
        stored = json.loads(summary_path.read_text())
        if int(stored.get("completed_stage2_configurations", 0)) != 1:
            raise RuntimeError(f"Transferred AFR seed {seed} is incomplete")
        best = stored["best"]
        if float(best["gamma"]) != float(params["gamma"]) or float(
            best["reg_coeff"]
        ) != float(params["reg_coeff"]):
            raise RuntimeError("Stored AFR head does not match WB95 transfer settings")
        evaluation_json = seed_root / "official_evaluation.json"
        evaluate_checkpoint(
            args,
            "afr",
            label,
            seed,
            afr_root / "stage1_final.pt",
            float(best["best_val_macro_class_accuracy"]),
            evaluation_json,
            Path(best["classifier_checkpoint"]),
        )
        evaluations.append(evaluation_json)
        write_method_tables(label, args.run_root, evaluations)
        print(f"[SEED DONE] method={label} seed={seed}", flush=True)


def atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def run_clip_lr(
    args: argparse.Namespace,
    config: Mapping[str, object],
    seeds: Sequence[int],
) -> None:
    if args.clip_feature_cache is None or not args.clip_feature_cache.is_file():
        raise FileNotFoundError(args.clip_feature_cache)
    params = config["params"]
    assert isinstance(params, Mapping)
    from run_imagenet9_final_clip_lr import build_official_feature_cache
    from sweep_imagenet9_clip_lr import _macro_accuracy
    from sklearn.linear_model import LogisticRegression

    with np.load(args.clip_feature_cache) as payload:
        train_features = payload["train_features"]
        train_labels = payload["train_labels"]
    official_cache = args.run_root / "official_features.npz"
    build_official_feature_cache(args, official_cache)
    official = np.load(official_cache)
    from imagenet9_data import FORBIDDEN_SELECTION_VARIANTS

    label = result_method("clip_lr")
    evaluations: List[Path] = []
    for seed in seeds:
        seed_root = args.run_root / f"seed_{seed}"
        checkpoint = seed_root / "classifier.pkl"
        evaluation_json = seed_root / "official_evaluation.json"
        if not evaluation_json.is_file():
            random.seed(seed)
            np.random.seed(seed)
            classifier = LogisticRegression(
                C=float(params["c"]),
                penalty=str(params["penalty"]),
                solver=str(params["solver"]),
                fit_intercept=bool(params["fit_intercept"]),
                tol=float(params["tol"]),
                max_iter=int(params["max_iter"]),
                random_state=seed,
                multi_class="auto",
            )
            classifier.fit(train_features, train_labels)
            atomic_pickle(checkpoint, classifier)
            variant_results = {}
            for variant in FORBIDDEN_SELECTION_VARIANTS:
                features = official[f"{variant}_features"]
                labels = official[f"{variant}_labels"]
                predictions = classifier.predict(features)
                macro, per_class = _macro_accuracy(labels, predictions)
                variant_results[variant] = {
                    "accuracy": float(np.mean(predictions == labels)),
                    "macro_class_accuracy": macro,
                    "per_class_accuracy": per_class,
                    "class_support": [int(np.sum(labels == index)) for index in range(9)],
                    "samples": int(labels.shape[0]),
                }
            atomic_json(
                evaluation_json,
                {
                    "method": label,
                    "native_method": "clip_lr",
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "selection_objective": "waterbirds95_validation_transfer",
                    "selection_value": float(params["c"]),
                    "official_variants_used_for_selection": False,
                    "variant_results": variant_results,
                },
            )
        evaluations.append(evaluation_json)
        write_method_tables(label, args.run_root, evaluations)
        print(f"[SEED DONE] method={label} seed={seed}", flush=True)
    official.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    config = load_config(args)
    args.run_root.mkdir(parents=True, exist_ok=True)
    contract = {
        "method": args.method,
        "result_method": result_method(args.method),
        "hyperparameter_selection": "waterbirds95_validation_transfer",
        "config": config,
        "seeds": list(seeds),
        "manifest": str(args.manifest.resolve()),
        "official_variants_used_for_hparam_selection": False,
        "imagenet9_validation_use": "checkpoint_selection_only",
    }
    ensure_contract(args.run_root / "run_contract.json", contract)
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)

    if args.method in STANDARD_METHODS or args.method == "r4rr":
        run_standard_or_r4rr(args, config, seeds)
    elif args.method == "afr":
        run_afr(args, config, seeds)
    else:
        run_clip_lr(args, config, seeds)
    print(f"[DONE] {args.run_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
