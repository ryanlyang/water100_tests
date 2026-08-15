#!/usr/bin/env python3
"""Train one validation-selected RedMeat checkpoint for RISE Pointing Game.

The script delegates to each method's existing trainer, applies the finalized
RedMeat hyperparameters, and writes one common JSON manifest for the shared
RISE evaluator. It is intentionally seed-level so the Slurm worker can resume
without repeating completed training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml


THIS_DIR = Path(__file__).resolve().parent
GALS_ROOT = THIS_DIR.parent
PROJECT_ROOT = GALS_ROOT.parent
CLASS_NAMES = (
    "prime_rib",
    "pork_chop",
    "steak",
    "baby_back_ribs",
    "filet_mignon",
)
METHODS = ("vanilla", "elrep", "upweight", "abn", "gals", "afr", "r4rr")
DEFAULT_DATA_PATH = "/home/ryreu/guided_cnn/Food101/data/food-101-redmeat"
DEFAULT_TEACHER_MAPS = (
    "/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/"
    "results_redmeat_openclip_laion_dinovit/val/prediction_cmap"
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def resolve_checkpoint(path: object, base: Path = GALS_ROOT) -> Path:
    checkpoint = Path(str(path)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = base / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training reported a missing checkpoint: {checkpoint}")
    return checkpoint


def load_hparams(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    key = "redmeat_optimized_hparams"
    if not isinstance(payload, dict) or not isinstance(payload.get(key), dict):
        raise KeyError(f"Missing mapping {key!r} in {path}")
    return payload[key]


def common_result(
    *,
    method: str,
    seed: int,
    checkpoint: Path,
    hparams: Dict[str, Any],
    metrics: Dict[str, Any],
    stage1_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    return {
        "dataset": "redmeat",
        "method": method,
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "stage1_checkpoint": (
            "" if stage1_checkpoint is None else str(stage1_checkpoint)
        ),
        "hparams": hparams,
        "metrics": metrics,
    }


def train_vanilla(args: argparse.Namespace, hparams: Dict[str, Any]) -> Dict[str, Any]:
    from RedMeat_Runs import run_vanilla_redmeat as trainer

    hp = dict(hparams["vanilla"])
    hp["momentum"] = float(hp.get("momentum", 0.9))
    run_args = argparse.Namespace(
        data_path=str(args.data_path),
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
        classes=",".join(CLASS_NAMES),
    )
    best_val, test_acc, per_group, worst_group, checkpoint_text = trainer.run_single(
        run_args
    )
    checkpoint = resolve_checkpoint(checkpoint_text)
    return common_result(
        method=args.method,
        seed=args.seed,
        checkpoint=checkpoint,
        hparams=hp,
        metrics={
            "best_balanced_val_acc": best_val,
            "test_acc": test_acc,
            "per_group": per_group,
            "worst_group": worst_group,
        },
    )


def train_elrep(args: argparse.Namespace, hparams: Dict[str, Any]) -> Dict[str, Any]:
    from RedMeat_Runs import run_elrep_redmeat as trainer

    hp = dict(hparams["elrep"])
    run_args = argparse.Namespace(
        data_path=str(args.data_path),
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
        classes=",".join(CLASS_NAMES),
    )
    best_val, test_acc, per_group, worst_group, checkpoint_text = trainer.run_single(
        run_args
    )
    checkpoint = resolve_checkpoint(checkpoint_text)
    return common_result(
        method=args.method,
        seed=args.seed,
        checkpoint=checkpoint,
        hparams=hp,
        metrics={
            "best_balanced_val_acc": best_val,
            "test_acc": test_acc,
            "per_group": per_group,
            "worst_group": worst_group,
        },
    )


def train_r4rr(args: argparse.Namespace, hparams: Dict[str, Any]) -> Dict[str, Any]:
    from RedMeat_Runs import run_guided_redmeat as trainer

    hp = dict(hparams["r4rr_optimized"])
    trainer.SEED = int(args.seed)
    trainer.batch_size = int(args.batch_size)
    trainer.num_epochs = int(args.num_epochs)
    trainer.base_lr = float(hp["base_lr"])
    trainer.classifier_lr = float(hp["classifier_lr"])
    trainer.lr2_mult = float(hp["lr2_mult"])
    trainer.checkpoint_dir = str(args.checkpoint_dir)

    run_args = argparse.Namespace(
        data_path=str(args.data_path),
        gt_path=str(args.teacher_maps),
        split_col="split",
        label_col="label",
        path_col="abs_file_path",
        classes=list(CLASS_NAMES),
        model_name="resnet50",
        clip_model="RN50",
        tune_mode="full",
        pretrained=True,
    )
    best_val, test_acc, per_group, worst_group, checkpoint_text = trainer.run_single(
        run_args,
        int(hp["attention_epoch"]),
        float(hp["kl_lambda"]),
        0.0,
    )
    checkpoint = resolve_checkpoint(checkpoint_text)
    hp["kl_incr"] = 0.0
    return common_result(
        method=args.method,
        seed=args.seed,
        checkpoint=checkpoint,
        hparams=hp,
        metrics={
            "best_balanced_val_acc": best_val,
            "test_acc": test_acc,
            "per_group": per_group,
            "worst_group": worst_group,
        },
    )


def train_gals_family(
    args: argparse.Namespace,
    hparams: Dict[str, Any],
) -> Dict[str, Any]:
    from RedMeat_Runs import run_gals_sweep_redmeat as sweep

    if args.method == "gals":
        hp = dict(hparams["gals_rrr_rn50_maps"])
        config = GALS_ROOT / "RedMeat_Runs/configs/redmeat_gals_rn50.yaml"
        sweep_method = "gals"
        method_values: Dict[str, Any] = {
            "grad_weight": float(hp["grad_weight"]),
            "grad_criterion": str(hp["grad_criterion"]),
        }
    elif args.method == "upweight":
        hp = dict(hparams["upweight"])
        config = GALS_ROOT / "RedMeat_Runs/configs/redmeat_upweight.yaml"
        sweep_method = "upweight"
        method_values = {}
    else:
        hp = dict(hparams["abn"])
        config = GALS_ROOT / "RedMeat_Runs/configs/redmeat_abn.yaml"
        sweep_method = "abn_cls"
        method_values = {"abn_cls_weight": float(hp["abn_cls_weight"])}

    run_name = f"pointing5_redmeat_{args.method}_seed{args.seed}"
    kwargs: Dict[str, Any] = {
        "trial_id": args.seed,
        "run_name": run_name,
        "method": sweep_method,
        "config": str(config),
        "data_root": str(args.data_path.parent),
        "dataset_dir": args.data_path.name,
        "dataset_name": "food_subset",
        "train_seed": args.seed,
        "base_lr": float(hp["base_lr"]),
        "classifier_lr": float(hp["classifier_lr"]),
        "grad_weight": None,
        "grad_criterion": None,
        "cam_weight": None,
        "abn_cls_weight": None,
        "abn_att_weight": None,
        "weight_decay": args.weight_decay,
        "python_exe": sys.executable,
        "logs_dir": str(args.output_dir / "training_logs"),
        "extra_overrides": [
            f"DATA.BATCH_SIZE={args.batch_size}",
            f"EXP.NUM_EPOCHS={args.num_epochs}",
            "LOGGING.SAVE_BEST=True",
            "LOGGING.SAVE_LAST=False",
        ],
    }
    kwargs.update(method_values)
    with pushd(GALS_ROOT):
        row = sweep.run_one_trial(**kwargs)
    checkpoint = resolve_checkpoint(row.get("checkpoint", ""))
    metrics = {
        key: row.get(key)
        for key in ("best_balanced_val_acc", "test_acc", "per_group", "worst_group")
    }
    return common_result(
        method=args.method,
        seed=args.seed,
        checkpoint=checkpoint,
        hparams=hp,
        metrics=metrics,
    )


def resolve_afr_root(explicit: str) -> Path:
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        PROJECT_ROOT / "afr",
        PROJECT_ROOT / "old_stuff" / "afr",
        GALS_ROOT / "RightForTheRightRegions/repro_runs/third_party/afr",
    ]
    for candidate in candidates:
        if (
            candidate is not None
            and (candidate / "train_supervised.py").is_file()
            and (candidate / "data/datasets.py").is_file()
        ):
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate AFR with train_supervised.py and data/datasets.py; "
        "pass --afr-root."
    )


def train_afr(args: argparse.Namespace, hparams: Dict[str, Any]) -> Dict[str, Any]:
    import pandas as pd
    import run_afr_redmeat_repro as trainer

    hp = dict(hparams["afr"])
    afr_root = resolve_afr_root(args.afr_root)
    output_root = args.checkpoint_dir / "afr_run"
    run_args = argparse.Namespace(
        afr_root=str(afr_root),
        data_dir=str(args.data_path),
        metadata_input="",
        output_root=str(output_root),
        logs_root=str(args.output_dir / "training_logs"),
        python_exe=sys.executable,
        seeds=str(args.seed),
        path_col="abs_file_path",
        label_col="label",
        split_col="split",
        classes=",".join(CLASS_NAMES),
        place_mode="auto",
        place_col="",
        place_candidates=(
            "place,spurious,group,confounder,background,bg,environment,env,"
            "smoke,context"
        ),
        check_images=True,
        stage1_epochs=50,
        stage1_eval_freq=10,
        stage1_save_freq=10,
        stage1_scheduler="cosine_lr_scheduler",
        stage2_epochs=500,
        stage2_lr=1e-2,
        full_paper_grid=False,
        gammas=str(hp["gamma"]),
        reg_coeffs=str(hp.get("reg", hp.get("reg_coeff", 0.0))),
        force_stage1=False,
        force_stage2=False,
    )
    trainer.run(run_args)

    best_csv = output_root / "afr_redmeat_best_by_seed.csv"
    if not best_csv.is_file():
        raise FileNotFoundError(f"AFR did not write {best_csv}")
    frame = pd.read_csv(best_csv)
    frame = frame[frame["seed"].astype(int) == int(args.seed)]
    if frame.empty:
        raise RuntimeError(f"AFR best CSV has no row for seed={args.seed}")
    row = frame.iloc[-1]
    stage1 = resolve_checkpoint(Path(str(row["stage1_dir"])) / "final_checkpoint.pt")
    stage2 = resolve_checkpoint(Path(str(row["stage2_dir"])) / "final_checkpoint.pt")
    metrics = {
        key: row.get(key)
        for key in (
            "best_val_wga",
            "best_test_at_val",
            "best_test_wga",
            "best_val_mean",
            "best_test_mean",
            "best_test_mean_at_val",
        )
        if key in row
    }
    return common_result(
        method=args.method,
        seed=args.seed,
        checkpoint=stage2,
        stage1_checkpoint=stage1,
        hparams=hp,
        metrics=metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument("--teacher-maps", type=Path, default=Path(DEFAULT_TEACHER_MAPS))
    parser.add_argument("--hparams-config", type=Path)
    parser.add_argument("--afr-root", default="")
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--existing-checkpoint", default="")
    parser.add_argument("--existing-stage1-checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_path = args.data_path.expanduser().resolve()
    args.teacher_maps = args.teacher_maps.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.result_json = args.result_json.expanduser().resolve()
    args.checkpoint_dir = args.output_dir / "checkpoints" / f"seed_{args.seed}"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.hparams_config is None:
        args.hparams_config = (
            GALS_ROOT / "RightForTheRightRegions/configs/redmeat_optimized_hparams.yaml"
        )
    args.hparams_config = args.hparams_config.expanduser().resolve()

    if not args.data_path.is_dir():
        raise FileNotFoundError(f"Missing RedMeat data: {args.data_path}")
    if args.method == "r4rr" and not args.teacher_maps.is_dir():
        raise FileNotFoundError(f"Missing R4RR teacher maps: {args.teacher_maps}")
    if not args.hparams_config.is_file():
        raise FileNotFoundError(f"Missing hyperparameter config: {args.hparams_config}")

    os.environ["SAVE_CHECKPOINTS"] = "1"
    os.environ.setdefault("WANDB_DISABLED", "true")
    hparams = load_hparams(args.hparams_config)

    if args.existing_checkpoint:
        checkpoint = resolve_checkpoint(args.existing_checkpoint)
        stage1 = (
            resolve_checkpoint(args.existing_stage1_checkpoint)
            if args.existing_stage1_checkpoint
            else None
        )
        if args.method == "afr" and stage1 is None:
            raise ValueError("AFR imports require --existing-stage1-checkpoint")
        method_key = {
            "gals": "gals_rrr_rn50_maps",
            "r4rr": "r4rr_optimized",
        }.get(args.method, args.method)
        result = common_result(
            method=args.method,
            seed=args.seed,
            checkpoint=checkpoint,
            stage1_checkpoint=stage1,
            hparams=dict(hparams[method_key]),
            metrics={"source": "imported_existing_checkpoint"},
        )
    elif args.method == "vanilla":
        result = train_vanilla(args, hparams)
    elif args.method == "elrep":
        result = train_elrep(args, hparams)
    elif args.method == "r4rr":
        result = train_r4rr(args, hparams)
    elif args.method in ("gals", "upweight", "abn"):
        result = train_gals_family(args, hparams)
    else:
        result = train_afr(args, hparams)

    result["data_path"] = str(args.data_path)
    result["teacher_maps"] = str(args.teacher_maps) if args.method == "r4rr" else ""
    result["hparams_config"] = str(args.hparams_config)
    result["num_epochs"] = int(args.num_epochs)
    result["batch_size"] = int(args.batch_size)
    atomic_json(args.result_json, result)
    print(f"[DONE] wrote training manifest: {args.result_json}", flush=True)
    print(f"[DONE] checkpoint: {result['checkpoint']}", flush=True)
    if result.get("stage1_checkpoint"):
        print(f"[DONE] stage1 checkpoint: {result['stage1_checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
