#!/usr/bin/env python3
"""Train one validation-selected Waterbirds checkpoint for Pointing Game.

This is the seed-level worker used by the five-seed Pointing Game Slurm jobs.
It keeps each method's original training implementation and fixed optimized
hyperparameters, then writes a small JSON manifest consumed by the evaluator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional

import yaml


GALS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = GALS_ROOT.parent

DATASET_DEFAULTS = {
    "95": {
        "config_key": "waterbirds95_optimized_hparams",
        "data_path": "/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2",
        "teacher_maps": "/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap",
        "gals_config": "configs/waterbirds_95_gals_vit.yaml",
        "upweight_config": "configs/waterbirds_95_upweight.yaml",
        "abn_config": "configs/waterbirds_95_abn.yaml",
    },
    "100": {
        "config_key": "waterbirds100_optimized_hparams",
        "data_path": "/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2",
        "teacher_maps": "/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap",
        "gals_config": "configs/waterbirds_100_gals_vit.yaml",
        "upweight_config": "configs/waterbirds_100_upweight.yaml",
        "abn_config": "configs/waterbirds_100_abn.yaml",
    },
}

METHODS = ("vanilla", "elrep", "upweight", "abn", "gals", "afr", "r4rr")


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar values before writing a manifest."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_guided_script(explicit: str) -> Path:
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        PROJECT_ROOT / "run_guided_waterbird.py",
        PROJECT_ROOT / "old_stuff" / "run_guided_waterbird.py",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find run_guided_waterbird.py. Pass --guided-script explicitly."
    )


def _resolve_checkpoint(path: str, base: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Training reported a missing checkpoint: {p}")
    return p


def _load_hparams(dataset: str, config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    root_key = DATASET_DEFAULTS[dataset]["config_key"]
    if root_key not in obj or not isinstance(obj[root_key], dict):
        raise KeyError(f"Missing {root_key} in {config_path}")
    return obj[root_key]


def _common_result(
    *, dataset: str, method: str, seed: int, checkpoint: Path,
    hparams: Dict[str, Any], metrics: Dict[str, Any],
    stage1_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "stage1_checkpoint": "" if stage1_checkpoint is None else str(stage1_checkpoint),
        "hparams": hparams,
        "metrics": metrics,
    }


def _train_vanilla(args: argparse.Namespace, hparams: Dict[str, Any]) -> Dict[str, Any]:
    import run_vanilla_waterbird_clip as trainer

    hp = dict(hparams["vanilla"])
    run_args = argparse.Namespace(
        data_path=args.data_path,
        seed=args.seed,
        model="resnet50",
        clip_model="RN50",
        tune_mode="full",
        pretrained=True,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=float(hp["base_lr"]),
        base_lr=float(hp["base_lr"]),
        classifier_lr=float(hp["classifier_lr"]),
        momentum=float(hp["momentum"]),
        weight_decay=args.weight_decay,
        nesterov=False,
        num_workers=args.num_workers,
        checkpoint_dir=str(args.checkpoint_dir),
    )
    best_val, test_acc, per_group, worst_group, ckpt = trainer.run_single(run_args)
    checkpoint = _resolve_checkpoint(ckpt, GALS_ROOT)
    return _common_result(
        dataset=args.dataset, method=args.method, seed=args.seed, checkpoint=checkpoint,
        hparams=hp,
        metrics={"best_balanced_val_acc": best_val, "test_acc": test_acc,
                 "per_group": per_group, "worst_group": worst_group},
    )


def _train_elrep(args: argparse.Namespace, hparams: Dict[str, Any]) -> Dict[str, Any]:
    import run_elrep_waterbird as trainer

    hp = dict(hparams["elrep"])
    run_args = argparse.Namespace(
        data_path=args.data_path,
        seed=args.seed,
        model="resnet50",
        clip_model="RN50",
        tune_mode="full",
        pretrained=True,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=float(hp["base_lr"]),
        base_lr=float(hp["base_lr"]),
        classifier_lr=float(hp["classifier_lr"]),
        theta1=float(hp["theta1"]),
        theta2=float(hp["theta2"]),
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=False,
        num_workers=args.num_workers,
        checkpoint_dir=str(args.checkpoint_dir),
    )
    best_val, test_acc, per_group, worst_group, ckpt = trainer.run_single(run_args)
    checkpoint = _resolve_checkpoint(ckpt, GALS_ROOT)
    return _common_result(
        dataset=args.dataset, method=args.method, seed=args.seed, checkpoint=checkpoint,
        hparams=hp,
        metrics={"best_balanced_val_acc": best_val, "test_acc": test_acc,
                 "per_group": per_group, "worst_group": worst_group},
    )


def _train_r4rr(args: argparse.Namespace, hparams: Dict[str, Any]) -> Dict[str, Any]:
    guided_path = _resolve_guided_script(args.guided_script)
    trainer = _load_module(guided_path, f"pointing_guided_seed_{args.seed}")
    hp = dict(hparams["r4rr_optimized"])

    trainer.SEED = args.seed
    trainer.base_lr = float(hp["base_lr"])
    trainer.classifier_lr = float(hp["classifier_lr"])
    trainer.lr2_mult = float(hp["lr2_mult"])
    trainer.num_epochs = args.num_epochs
    trainer.checkpoint_dir = str(args.checkpoint_dir)

    run_args = SimpleNamespace(
        data_path=args.data_path,
        gt_path=args.teacher_maps,
        model_name="resnet50",
        pretrained=True,
    )
    best_val, test_acc, per_group, worst_group, ckpt = trainer.run_single(
        run_args,
        int(hp["attention_epoch"]),
        float(hp["kl_lambda"]),
        0.0,
    )
    checkpoint = _resolve_checkpoint(ckpt, GALS_ROOT)
    return _common_result(
        dataset=args.dataset, method=args.method, seed=args.seed, checkpoint=checkpoint,
        hparams=hp,
        metrics={"best_balanced_val_acc": best_val, "test_acc": test_acc,
                 "per_group": per_group, "worst_group": worst_group},
    )


def _train_gals_family(args: argparse.Namespace, hparams: Dict[str, Any]) -> Dict[str, Any]:
    import run_gals_sweep as sweep

    defaults = DATASET_DEFAULTS[args.dataset]
    if args.method == "gals":
        hp = dict(hparams["gals_rrr_vit_maps"])
        config = defaults["gals_config"]
        sweep_method = "gals"
        extra = {
            "grad_weight": float(hp["grad_weight"]),
            "grad_criterion": str(hp["grad_criterion"]),
        }
    elif args.method == "upweight":
        hp = dict(hparams["upweight"])
        config = defaults["upweight_config"]
        sweep_method = "upweight"
        extra = {}
    else:
        hp = dict(hparams["abn"])
        config = defaults["abn_config"]
        sweep_method = "abn_cls"
        extra = {"abn_cls_weight": float(hp["abn_cls_weight"])}

    run_name = f"pointing5_{args.dataset}_{args.method}_seed{args.seed}"
    kwargs: Dict[str, Any] = dict(
        trial_id=args.seed,
        run_name=run_name,
        method=sweep_method,
        config=str(GALS_ROOT / config),
        data_root=str(Path(args.data_path).resolve().parent),
        waterbirds_dir=Path(args.data_path).name,
        dataset_name="waterbirds",
        train_seed=args.seed,
        base_lr=float(hp["base_lr"]),
        classifier_lr=float(hp["classifier_lr"]),
        grad_weight=None,
        grad_criterion=None,
        cam_weight=None,
        abn_cls_weight=None,
        abn_att_weight=None,
        weight_decay=args.weight_decay,
        python_exe=sys.executable,
        logs_dir=str(args.output_dir / "training_logs"),
        extra_overrides=[
            f"EXP.NUM_EPOCHS={args.num_epochs}",
            "LOGGING.SAVE_BEST=True",
            "LOGGING.SAVE_LAST=False",
        ],
    )
    kwargs.update(extra)
    with _pushd(GALS_ROOT):
        row = sweep.run_one_trial(**kwargs)
    checkpoint = _resolve_checkpoint(str(row.get("checkpoint", "")), GALS_ROOT)
    metrics = {
        key: row.get(key)
        for key in ("best_balanced_val_acc", "test_acc", "per_group", "worst_group")
    }
    return _common_result(
        dataset=args.dataset, method=args.method, seed=args.seed, checkpoint=checkpoint,
        hparams=hp, metrics=metrics,
    )


def _resolve_afr_root(explicit: str) -> Path:
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        PROJECT_ROOT / "afr",
        PROJECT_ROOT / "old_stuff" / "afr",
        GALS_ROOT / "RightForTheRightRegions" / "repro_runs" / "third_party" / "afr",
    ]
    for candidate in candidates:
        if (
            candidate is not None
            and (candidate / "train_supervised.py").is_file()
            and (candidate / "data" / "datasets.py").is_file()
        ):
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate a complete AFR repository containing "
        "train_supervised.py and data/datasets.py. Pass --afr-root."
    )


def _train_afr(args: argparse.Namespace, hparams: Dict[str, Any]) -> Dict[str, Any]:
    import pandas as pd
    import run_afr_waterbirds_repro as afr

    hp = dict(hparams["afr"])
    afr_root = _resolve_afr_root(args.afr_root)
    output_root = args.checkpoint_dir / "afr_run"
    logs_root = args.output_dir / "training_logs"
    run_args = argparse.Namespace(
        afr_root=str(afr_root),
        data_dir=args.data_path,
        output_root=str(output_root),
        logs_root=str(logs_root),
        python_exe=sys.executable,
        seeds=str(args.seed),
        stage1_epochs=50,
        stage1_eval_freq=10,
        stage1_save_freq=10,
        stage1_scheduler="constant_lr_scheduler",
        stage2_epochs=500,
        stage2_lr=1e-2,
        full_paper_grid=False,
        gammas=str(hp["gamma"]),
        reg_coeffs=str(hp.get("reg", hp.get("reg_coeff", 0.0))),
        force_stage1=False,
        force_stage2=False,
    )
    afr.run(run_args)

    best_csv = output_root / "afr_waterbirds_best_by_seed.csv"
    if not best_csv.is_file():
        raise FileNotFoundError(f"AFR did not write {best_csv}")
    rows = pd.read_csv(best_csv)
    rows = rows[rows["seed"].astype(int) == args.seed]
    if rows.empty:
        raise RuntimeError(f"AFR best CSV has no row for seed {args.seed}")
    row = rows.iloc[-1]
    stage1 = Path(str(row["stage1_dir"])) / "final_checkpoint.pt"
    stage2 = Path(str(row["stage2_dir"])) / "final_checkpoint.pt"
    stage1 = _resolve_checkpoint(str(stage1), GALS_ROOT)
    stage2 = _resolve_checkpoint(str(stage2), GALS_ROOT)
    metrics = {
        key: row.get(key)
        for key in ("best_val_wga", "best_test_at_val", "best_test_wga",
                    "best_val_mean", "best_test_mean")
        if key in row
    }
    return _common_result(
        dataset=args.dataset, method=args.method, seed=args.seed,
        checkpoint=stage2, stage1_checkpoint=stage1, hparams=hp, metrics=metrics,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--result-json", type=Path, required=True)
    p.add_argument("--data-path", default="")
    p.add_argument("--teacher-maps", default="")
    p.add_argument("--hparams-config", type=Path, default=None)
    p.add_argument("--guided-script", default="")
    p.add_argument("--afr-root", default="")
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument(
        "--existing-checkpoint",
        default="",
        help="Register an already-trained validation-selected checkpoint instead of training.",
    )
    p.add_argument(
        "--existing-stage1-checkpoint",
        default="",
        help="AFR stage-1 checkpoint paired with --existing-checkpoint (the stage-2 head).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    args.output_dir = args.output_dir.expanduser().resolve()
    args.result_json = args.result_json.expanduser().resolve()
    args.checkpoint_dir = args.output_dir / "checkpoints" / f"seed_{args.seed}"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.data_path = str(Path(args.data_path or defaults["data_path"]).expanduser().resolve())
    args.teacher_maps = str(Path(args.teacher_maps or defaults["teacher_maps"]).expanduser().resolve())
    if args.hparams_config is None:
        args.hparams_config = (
            GALS_ROOT / "RightForTheRightRegions" / "configs"
            / f"waterbirds{args.dataset}_optimized_hparams.yaml"
        )
    args.hparams_config = args.hparams_config.expanduser().resolve()

    if not Path(args.data_path).is_dir():
        raise FileNotFoundError(f"Missing Waterbirds data path: {args.data_path}")
    if args.method == "r4rr" and not Path(args.teacher_maps).is_dir():
        raise FileNotFoundError(f"Missing R4RR teacher maps: {args.teacher_maps}")

    os.environ["SAVE_CHECKPOINTS"] = "1"
    os.environ.setdefault("WANDB_DISABLED", "true")
    hparams = _load_hparams(args.dataset, args.hparams_config)

    if args.existing_checkpoint:
        checkpoint = _resolve_checkpoint(args.existing_checkpoint, GALS_ROOT)
        stage1 = None
        if args.existing_stage1_checkpoint:
            stage1 = _resolve_checkpoint(args.existing_stage1_checkpoint, GALS_ROOT)
        if args.method == "afr" and stage1 is None:
            raise ValueError(
                "AFR requires --existing-stage1-checkpoint when importing a stage-2 checkpoint."
            )
        method_key = {
            "gals": "gals_rrr_vit_maps",
            "r4rr": "r4rr_optimized",
        }.get(args.method, args.method)
        result = _common_result(
            dataset=args.dataset,
            method=args.method,
            seed=args.seed,
            checkpoint=checkpoint,
            stage1_checkpoint=stage1,
            hparams=dict(hparams[method_key]),
            metrics={"source": "imported_existing_checkpoint"},
        )
        result["data_path"] = args.data_path
        result["teacher_maps"] = args.teacher_maps if args.method == "r4rr" else ""
        result["hparams_config"] = str(args.hparams_config)
        _atomic_json(args.result_json, result)
        print(f"[DONE] imported checkpoint manifest: {args.result_json}", flush=True)
        print(f"[DONE] checkpoint: {result['checkpoint']}", flush=True)
        return

    if args.method == "vanilla":
        result = _train_vanilla(args, hparams)
    elif args.method == "elrep":
        result = _train_elrep(args, hparams)
    elif args.method == "r4rr":
        result = _train_r4rr(args, hparams)
    elif args.method in ("gals", "upweight", "abn"):
        result = _train_gals_family(args, hparams)
    else:
        result = _train_afr(args, hparams)

    result["data_path"] = args.data_path
    result["teacher_maps"] = args.teacher_maps if args.method == "r4rr" else ""
    result["hparams_config"] = str(args.hparams_config)
    _atomic_json(args.result_json, result)
    print(f"[DONE] wrote training manifest: {args.result_json}", flush=True)
    print(f"[DONE] checkpoint: {result['checkpoint']}", flush=True)
    if result.get("stage1_checkpoint"):
        print(f"[DONE] stage1 checkpoint: {result['stage1_checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
