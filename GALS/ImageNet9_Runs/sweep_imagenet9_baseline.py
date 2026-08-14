#!/usr/bin/env python3
"""Resumable Optuna sweeps for ImageNet-9 baselines."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


NON_TEACHER_METHODS = ("erm", "upweight", "abn", "elrep")
METHODS = NON_TEACHER_METHODS + ("gals",)
OBJECTIVE_NAME = "val_macro_class_accuracy"
RESULT_PREFIX = "[RESULT] "


SEARCH_SPACES: Mapping[str, Mapping[str, object]] = {
    "erm": {
        "base_lr": (1e-5, 5e-2, "log"),
        "classifier_lr": (1e-5, 5e-2, "log"),
        "momentum": (0.85, 0.95, "linear"),
    },
    "upweight": {
        "base_lr": (5e-5, 1e-1, "log"),
        "classifier_lr": (5e-5, 1e-1, "log"),
    },
    "abn": {
        "base_lr": (5e-5, 1e-1, "log"),
        "classifier_lr": (5e-5, 1e-1, "log"),
        "abn_cls_weight": (1e-2, 1e2, "log"),
    },
    "elrep": {
        "base_lr": (1e-5, 5e-2, "log"),
        "classifier_lr": (1e-5, 5e-2, "log"),
        "theta1": (1e-5, 1e-2, "log"),
        "theta2": (1e-6, 1e-3, "log"),
    },
    "gals": {
        "base_lr": (1e-5, 5e-2, "log"),
        "classifier_lr": (1e-5, 5e-2, "log"),
        "grad_weight": (1e3, 1e5, "log"),
        "grad_criterion": ("L1", "L2", "categorical"),
    },
}


FIXED_HPARAMS = {
    "epochs": 20,
    "batch_size": 96,
    "weight_decay": 1e-5,
    "momentum": 0.9,
    "nesterov": False,
    "pretrained": True,
    "student": "torchvision_resnet50_imagenet",
    "objective": OBJECTIVE_NAME,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--study-db", type=Path, required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--trial-logs", type=Path, required=True)
    parser.add_argument("--target-complete-trials", type=int, default=50)
    parser.add_argument("--sweep-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fixed-momentum", type=float, default=0.9)
    parser.add_argument("--max-hours", type=float, default=94.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--abn-checkpoint", type=Path)
    parser.add_argument("--gals-map-root", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--trainer", type=Path, default=Path(__file__).with_name("train_imagenet9_baseline.py"))
    parser.add_argument("--no-enqueue-default", action="store_true")
    return parser.parse_args(argv)


def _contract(args: argparse.Namespace) -> Dict[str, object]:
    contract = {
        "method": args.method,
        "search_space": SEARCH_SPACES[args.method],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "fixed_momentum": args.fixed_momentum,
        "train_seed": args.train_seed,
        "manifest": str(args.manifest.resolve()),
        "objective": OBJECTIVE_NAME,
        "official_variants_used": False,
    }
    if args.method == "gals":
        contract["gals_maps"] = _validate_gals_maps(args)
    return contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_gals_maps(args: argparse.Namespace) -> Dict[str, object]:
    if args.gals_map_root is None:
        raise ValueError("GALS requires --gals-map-root")
    contract_path = args.gals_map_root / "map_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    map_contract = json.loads(contract_path.read_text())
    if map_contract.get("map_tensor_key") != "unnormalized_attentions":
        raise RuntimeError(f"Unexpected GALS map tensor contract: {contract_path}")
    if map_contract.get("expected_map_shape") != [2, 1, 7, 7]:
        raise RuntimeError(f"Unexpected GALS map shape contract: {contract_path}")
    if map_contract.get("manifest_sha256") != _sha256(args.manifest):
        raise RuntimeError("GALS maps were generated from a different ImageNet-9 manifest")
    map_count = sum(1 for _ in (args.gals_map_root / "maps").glob("*/*.pth"))
    if map_count != 45405:
        raise RuntimeError(f"Expected 45,405 GALS maps, found {map_count}")
    return {
        "root": str(args.gals_map_root.resolve()),
        "map_count": map_count,
        "map_contract_sha256": _sha256(contract_path),
        "model": map_contract.get("model"),
        "method": map_contract.get("method"),
    }


def _contract_hash(contract: Mapping[str, object]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _suggest(trial, method: str, fixed_momentum: float) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for name, specification in SEARCH_SPACES[method].items():
        low, high, scale = specification
        if scale == "categorical":
            params[name] = trial.suggest_categorical(name, list(specification[:-1]))
        else:
            params[name] = float(
                trial.suggest_float(name, float(low), float(high), log=(scale == "log"))
            )
    params.setdefault("momentum", fixed_momentum)
    params.setdefault("abn_cls_weight", 1.0)
    params.setdefault("theta1", 1e-4)
    params.setdefault("theta2", 1e-5)
    params.setdefault("grad_weight", 1e4)
    params.setdefault("grad_criterion", "L1")
    return params


def _default_trial(method: str) -> Dict[str, object]:
    defaults = {
        "erm": {"base_lr": 1e-2, "classifier_lr": 1e-3, "momentum": 0.9},
        "upweight": {"base_lr": 1e-2, "classifier_lr": 1e-3},
        "abn": {"base_lr": 1e-2, "classifier_lr": 1e-3, "abn_cls_weight": 1.0},
        "elrep": {
            "base_lr": 1e-2,
            "classifier_lr": 1e-3,
            "theta1": 1e-4,
            "theta2": 1e-5,
        },
        "gals": {
            "base_lr": 5e-3,
            "classifier_lr": 1e-4,
            "grad_weight": 1e4,
            "grad_criterion": "L1",
        },
    }
    return defaults[method]


def _trainer_command(args: argparse.Namespace, params: Mapping[str, object]) -> List[str]:
    command = [
        args.python,
        "-u",
        str(args.trainer),
        "--method", args.method,
        "--manifest", str(args.manifest),
        "--seed", str(args.train_seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--base-lr", str(params["base_lr"]),
        "--classifier-lr", str(params["classifier_lr"]),
        "--momentum", str(params["momentum"]),
        "--weight-decay", str(args.weight_decay),
        "--abn-cls-weight", str(params["abn_cls_weight"]),
        "--theta1", str(params["theta1"]),
        "--theta2", str(params["theta2"]),
        "--device", args.device,
        "--skip-file-checks",
    ]
    if args.method == "abn":
        if args.abn_checkpoint is None:
            raise ValueError("ABN sweep requires --abn-checkpoint")
        command.extend(["--abn-checkpoint", str(args.abn_checkpoint)])
    elif args.method == "gals":
        if args.gals_map_root is None:
            raise ValueError("GALS sweep requires --gals-map-root")
        command.extend(
            [
                "--gals-map-root", str(args.gals_map_root),
                "--grad-weight", str(params["grad_weight"]),
                "--grad-criterion", str(params["grad_criterion"]),
            ]
        )
    return command


def _parse_result(log_path: Path) -> Dict[str, object]:
    result = None
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(RESULT_PREFIX):
                result = json.loads(line[len(RESULT_PREFIX):])
    if result is None:
        raise RuntimeError(f"Trial log did not contain {RESULT_PREFIX.strip()}: {log_path}")
    return result


def _run_trial(trial, args: argparse.Namespace) -> float:
    params = _suggest(trial, args.method, args.fixed_momentum)
    log_path = args.trial_logs / f"trial_{trial.number:04d}.log"
    command = _trainer_command(args, params)
    start = time.time()
    with log_path.open("w") as log_handle:
        log_handle.write(f"[COMMAND] {' '.join(command)}\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
    seconds = time.time() - start
    trial.set_user_attr("log_path", str(log_path.resolve()))
    trial.set_user_attr("seconds", seconds)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        print(f"[TRIAL {trial.number}] failed rc={completed.returncode}: {log_path}", flush=True)
        for line in tail:
            print(line, flush=True)
        raise RuntimeError(f"Trial process failed with exit code {completed.returncode}")

    result = _parse_result(log_path)
    objective = float(result["best_val_macro_class_accuracy"])
    trial.set_user_attr("best_epoch", int(result["best_epoch"]))
    trial.set_user_attr("best_val_accuracy", float(result["best_val_accuracy"]))
    trial.set_user_attr("best_val_per_class_accuracy", result["best_val_per_class_accuracy"])
    trial.set_user_attr("class_weights", result["class_weights"])
    print(
        f"[TRIAL {trial.number}] complete objective={objective:.6f} "
        f"best_epoch={result['best_epoch']} seconds={seconds:.1f} params={params}",
        flush=True,
    )
    gc.collect()
    return objective


def _write_study_csv(study, path: Path) -> None:
    import optuna

    fieldnames = [
        "trial", "state", "objective", "base_lr", "classifier_lr", "momentum",
        "abn_cls_weight", "theta1", "theta2", "grad_weight", "grad_criterion",
        "best_epoch", "best_val_accuracy",
        "best_val_per_class_accuracy", "class_weights", "seconds", "log_path",
    ]
    rows = []
    for trial in study.get_trials(deepcopy=False):
        rows.append(
            {
                "trial": trial.number,
                "state": trial.state.name,
                "objective": trial.value if trial.state == optuna.trial.TrialState.COMPLETE else "",
                "base_lr": trial.params.get("base_lr", ""),
                "classifier_lr": trial.params.get("classifier_lr", ""),
                "momentum": trial.params.get("momentum", study.user_attrs.get("fixed_momentum", "")),
                "abn_cls_weight": trial.params.get("abn_cls_weight", ""),
                "theta1": trial.params.get("theta1", ""),
                "theta2": trial.params.get("theta2", ""),
                "grad_weight": trial.params.get("grad_weight", ""),
                "grad_criterion": trial.params.get("grad_criterion", ""),
                "best_epoch": trial.user_attrs.get("best_epoch", ""),
                "best_val_accuracy": trial.user_attrs.get("best_val_accuracy", ""),
                "best_val_per_class_accuracy": json.dumps(trial.user_attrs.get("best_val_per_class_accuracy", [])),
                "class_weights": json.dumps(trial.user_attrs.get("class_weights", [])),
                "seconds": trial.user_attrs.get("seconds", ""),
                "log_path": trial.user_attrs.get("log_path", ""),
            }
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_summary(study, args: argparse.Namespace, contract: Mapping[str, object]) -> None:
    import optuna

    trials = study.get_trials(deepcopy=False)
    complete = [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE]
    failed = [trial for trial in trials if trial.state == optuna.trial.TrialState.FAIL]
    running = [trial for trial in trials if trial.state == optuna.trial.TrialState.RUNNING]
    summary: Dict[str, object] = {
        "study_name": study.study_name,
        "storage": str(args.study_db.resolve()),
        "method": args.method,
        "objective": OBJECTIVE_NAME,
        "direction": "maximize",
        "target_complete_trials": args.target_complete_trials,
        "complete_trials": len(complete),
        "failed_trials": len(failed),
        "running_trials": len(running),
        "contract": contract,
        "official_variants_used_for_selection": False,
    }
    if complete:
        summary["best_trial"] = study.best_trial.number
        summary["best_value"] = study.best_value
        summary["best_params"] = study.best_params
        summary["best_user_attrs"] = study.best_trial.user_attrs
    temporary = args.summary_json.with_suffix(args.summary_json.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.summary_json)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.target_complete_trials <= 0:
        raise ValueError("--target-complete-trials must be positive")
    for path in (args.study_db.parent, args.output_csv.parent, args.summary_json.parent, args.trial_logs):
        path.mkdir(parents=True, exist_ok=True)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.trainer.is_file():
        raise FileNotFoundError(args.trainer)

    import optuna

    contract = _contract(args)
    contract_digest = _contract_hash(contract)
    storage_url = f"sqlite:///{args.study_db.resolve()}"
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 60}},
        heartbeat_interval=60,
        grace_period=600,
    )
    sampler = optuna.samplers.TPESampler(seed=args.sweep_seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    if hasattr(optuna.storages, "fail_stale_trials"):
        optuna.storages.fail_stale_trials(study)

    existing_digest = study.user_attrs.get("contract_sha256")
    if existing_digest is not None and existing_digest != contract_digest:
        raise RuntimeError(
            "Refusing to resume an Optuna study with a changed experiment contract. "
            f"stored={existing_digest} requested={contract_digest}"
        )
    if existing_digest is None:
        study.set_user_attr("contract_sha256", contract_digest)
        study.set_user_attr("contract", contract)
        study.set_user_attr("fixed_momentum", args.fixed_momentum)
        if not args.no_enqueue_default and not study.get_trials(deepcopy=False):
            study.enqueue_trial(_default_trial(args.method))

    stopped_for_failures = {"value": False}

    def persist_callback(current_study, _trial) -> None:
        _write_study_csv(current_study, args.output_csv)
        _write_summary(current_study, args, contract)
        recent = current_study.get_trials(deepcopy=False)[-3:]
        if len(recent) == 3 and all(
            trial.state == optuna.trial.TrialState.FAIL for trial in recent
        ):
            stopped_for_failures["value"] = True
            print("[STOP] Three consecutive trials failed; stopping for inspection.", flush=True)
            current_study.stop()

    complete_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.get_trials(deepcopy=False)
    )
    remaining = max(args.target_complete_trials - complete_before, 0)
    print(
        f"[STUDY] name={study.study_name} method={args.method} "
        f"complete={complete_before}/{args.target_complete_trials} remaining={remaining}",
        flush=True,
    )
    print(f"[STUDY] storage={args.study_db.resolve()}", flush=True)
    print(f"[STUDY] contract_sha256={contract_digest}", flush=True)
    print(f"[STUDY] search_space={SEARCH_SPACES[args.method]}", flush=True)

    if remaining:
        study.optimize(
            lambda trial: _run_trial(trial, args),
            n_trials=remaining,
            timeout=args.max_hours * 3600.0,
            callbacks=[persist_callback],
            catch=(RuntimeError,),
            gc_after_trial=True,
        )

    _write_study_csv(study, args.output_csv)
    _write_summary(study, args, contract)
    complete_after = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.get_trials(deepcopy=False)
    )
    print(
        f"[DONE] complete={complete_after}/{args.target_complete_trials} "
        f"best_trial={study.best_trial.number if complete_after else 'NONE'} "
        f"best_value={study.best_value if complete_after else 'NONE'}",
        flush=True,
    )
    if stopped_for_failures["value"]:
        return 2
    if complete_after < args.target_complete_trials:
        print("[INCOMPLETE] Wall-clock budget ended; resubmit the same study to continue.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
