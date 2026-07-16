#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None


CLASS_NAMES = ["prime_rib", "pork_chop", "steak", "baby_back_ribs", "filet_mignon"]
TEST_METRIC_RE = re.compile(r"^([A-Za-z0-9_]+):\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")


@dataclass(frozen=True)
class TrainSpec:
    method: str
    config: str


TRAIN_SPECS_RN50 = [
    TrainSpec("gals", "RedMeat_Runs/configs/redmeat_gals_rn50.yaml"),
    TrainSpec("abn", "RedMeat_Runs/configs/redmeat_abn.yaml"),
    TrainSpec("upweight", "RedMeat_Runs/configs/redmeat_upweight.yaml"),
    TrainSpec("vanilla", "RedMeat_Runs/configs/redmeat_vanilla.yaml"),
]

TRAIN_SPECS_VIT = [
    TrainSpec("gals_vit", "RedMeat_Runs/configs/redmeat_gals_vit.yaml"),
    TrainSpec("abn", "RedMeat_Runs/configs/redmeat_abn.yaml"),
    TrainSpec("upweight", "RedMeat_Runs/configs/redmeat_upweight.yaml"),
    TrainSpec("vanilla", "RedMeat_Runs/configs/redmeat_vanilla.yaml"),
]


def _to_pct(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if -1.0 <= x <= 1.0:
        return 100.0 * x
    return x


def _write_row(csv_path: str, row: Dict[str, object], header: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _tail_text(path: str, n: int = 80) -> str:
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        return "".join(lines[-n:]).rstrip()
    except Exception:
        return ""


def _required_attention_dir(config_path: str) -> Optional[str]:
    if OmegaConf is None:
        return None
    cfg = OmegaConf.load(config_path)
    if not hasattr(cfg, "DATA") or not hasattr(cfg.DATA, "ATTENTION_DIR"):
        return None
    att_dir = str(cfg.DATA.ATTENTION_DIR)
    if att_dir.upper() == "NONE":
        return None
    return att_dir


def _extract_hparams_from_config(config_path: str, method: str) -> Dict[str, object]:
    if OmegaConf is None:
        return {"config": config_path, "note": "omegaconf_not_available"}

    cfg = OmegaConf.load(config_path)
    hp: Dict[str, object] = {
        "config": config_path,
        "num_epochs": int(cfg.EXP.NUM_EPOCHS),
        "optimizer": str(cfg.EXP.OPTIMIZER),
        "base_lr": float(cfg.EXP.BASE.LR),
        "classifier_lr": float(cfg.EXP.CLASSIFIER.LR),
        "weight_decay": float(cfg.EXP.WEIGHT_DECAY),
        "momentum": float(cfg.EXP.MOMENTUM),
        "batch_size": int(cfg.DATA.BATCH_SIZE),
    }
    losses = cfg.EXP.LOSSES
    if method in ("gals", "gals_vit"):
        hp["grad_outside_weight"] = float(losses.GRADIENT_OUTSIDE.WEIGHT)
        hp["grad_outside_criterion"] = str(losses.GRADIENT_OUTSIDE.CRITERION)
        hp["grad_outside_gt"] = str(losses.GRADIENT_OUTSIDE.GT)
    elif method == "abn":
        hp["abn_cls_weight"] = float(losses.ABN_CLASSIFICATION.WEIGHT)
        hp["abn_sup_weight"] = float(losses.ABN_SUPERVISION.WEIGHT)
        hp["abn_sup_compute"] = bool(losses.ABN_SUPERVISION.COMPUTE)
    elif method == "upweight":
        hp["use_class_weights"] = bool(cfg.DATA.USE_CLASS_WEIGHTS)
    return hp


def _metric_for_class(test_metrics: Dict[str, float], class_name: str) -> Optional[float]:
    for key in (f"{class_name}_test_acc", f"test_acc_{class_name}"):
        if key in test_metrics:
            return _to_pct(test_metrics[key])
    return None


def _run_main_single(
    *,
    python_exe: str,
    data_root: str,
    dataset_dir: str,
    config_path: str,
    run_name: str,
    logs_dir: str,
    class_names: List[str],
    aux_losses_on_val: bool,
) -> Dict[str, object]:
    cmd = [
        python_exe,
        "-u",
        "main.py",
        "--dryrun",
        "--config",
        config_path,
        "--name",
        run_name,
        f"DATA.ROOT={data_root}",
        f"DATA.FOOD_SUBSET_DIR={dataset_dir}",
        f"DATA.SUBDIR={dataset_dir}",
        "EXP.NUM_TRIALS=1",
        f"EXP.AUX_LOSSES_ON_VAL={'True' if aux_losses_on_val else 'False'}",
    ]

    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")

    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{run_name}.log")
    t0 = time.time()
    checkpoint: Optional[str] = None
    test_metrics: Dict[str, float] = {}

    with open(log_path, "w") as lf:
        lf.write("[CMD] " + " ".join(cmd) + "\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            if line.startswith("TEST SET RESULTS FOR CHECKPOINT"):
                checkpoint = line.strip().split("CHECKPOINT", 1)[-1].strip()

            m = TEST_METRIC_RE.match(line.strip())
            if not m:
                continue
            key = m.group(1)
            val = float(m.group(2))
            if key.endswith("_test_acc") or key.startswith("test_acc_") or key in ("test_acc", "balanced_test_acc"):
                test_metrics[key] = val

        rc = proc.wait()
        if rc != 0:
            tail = _tail_text(log_path, n=120)
            if tail:
                raise RuntimeError(
                    f"Run failed (exit {rc}): {run_name}. See {log_path}\n"
                    f"--- tail of failing log ---\n{tail}"
                )
            raise RuntimeError(f"Run failed (exit {rc}): {run_name}. See {log_path}")

    class_accs: Dict[str, Optional[float]] = {}
    class_vals: List[float] = []
    for cls in class_names:
        v = _metric_for_class(test_metrics, cls)
        class_accs[f"{cls}_test_acc"] = v
        if v is not None:
            class_vals.append(float(v))

    per_group = float(np.mean(class_vals)) if class_vals else None
    worst_group = float(np.min(class_vals)) if class_vals else None
    out: Dict[str, object] = {
        "checkpoint": checkpoint,
        "log_path": log_path,
        "seconds": int(time.time() - t0),
        "test_acc": _to_pct(test_metrics.get("test_acc")),
        "balanced_test_acc": _to_pct(test_metrics.get("balanced_test_acc")),
        "per_group": per_group,
        "worst_group": worst_group,
    }
    out.update(class_accs)
    return out


def _run_clip_lr_fixed(
    *,
    dataset_path: str,
    class_names: List[str],
    clip_model: str,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    c_value: float,
    penalty: str,
    solver: str,
    fit_intercept: bool,
    max_iter: int,
) -> Dict[str, object]:
    import RedMeat_Runs.run_clip_lr_sweep_redmeat as clip_lr
    from sklearn.linear_model import LogisticRegression

    np.random.seed(seed)
    classes, train_samples, val_samples, test_samples = clip_lr._build_splits(
        dataset_path=dataset_path,
        split_col="split",
        label_col="label",
        path_col="abs_file_path",
        train_value="train",
        val_value="val",
        test_value="test",
        classes=class_names,
    )
    num_classes = len(classes)

    try:
        model, preprocess = clip_lr._try_import_clip().load(clip_model, device=device, jit=False)
    except TypeError:
        model, preprocess = clip_lr._try_import_clip().load(clip_model, device=device)

    x_train, y_train = clip_lr._extract_features(train_samples, model, preprocess, device, batch_size, num_workers)
    x_val, y_val = clip_lr._extract_features(val_samples, model, preprocess, device, batch_size, num_workers)
    x_test, y_test = clip_lr._extract_features(test_samples, model, preprocess, device, batch_size, num_workers)

    x_train = clip_lr._l2_normalize(x_train).astype(np.float32, copy=False)
    x_val = clip_lr._l2_normalize(x_val).astype(np.float32, copy=False)
    x_test = clip_lr._l2_normalize(x_test).astype(np.float32, copy=False)

    clf = LogisticRegression(
        random_state=seed,
        C=c_value,
        penalty=penalty,
        solver=solver,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        n_jobs=1,
        verbose=0,
    )
    clf.fit(x_train, y_train)

    val_pred = clf.predict(x_val)
    val_acc = float(np.mean((val_pred == y_val).astype(np.float64)) * 100.0)
    val_class = clip_lr._class_acc(y_val, val_pred, num_classes=num_classes)
    val_avg_group = float(np.nanmean(val_class))
    val_worst_group = float(np.nanmin(val_class))

    test_pred = clf.predict(x_test)
    test_acc = float(np.mean((test_pred == y_test).astype(np.float64)) * 100.0)
    test_class = clip_lr._class_acc(y_test, test_pred, num_classes=num_classes)
    test_avg_group = float(np.nanmean(test_class))
    test_worst_group = float(np.nanmin(test_class))

    out: Dict[str, object] = {
        "test_acc": test_acc,
        "balanced_test_acc": test_avg_group,
        "per_group": test_avg_group,
        "worst_group": test_worst_group,
        "val_acc": val_acc,
        "val_per_group": val_avg_group,
        "val_worst_group": val_worst_group,
    }
    for i, cls in enumerate(classes):
        out[f"{cls}_test_acc"] = float(test_class[i])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-run RedMeat reference check using paper-config hyperparameters for "
            "GALS (RN50 by default), ABN, UpWeight, Vanilla, plus fixed CLIP+LR."
        )
    )
    parser.add_argument("--data-root", default="/home/ryreu/guided_cnn/Food101/data")
    parser.add_argument("--dataset-dir", default="food-101-redmeat")
    parser.add_argument("--logs-dir", default="/home/ryreu/guided_cnn/logsRedMeat/paper_single_runs")
    parser.add_argument("--output-csv", default="/home/ryreu/guided_cnn/logsRedMeat/paper_single_runs_summary.csv")
    parser.add_argument("--run-prefix", default="paper_ref_redmeat")
    parser.add_argument(
        "--classes",
        default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon",
        help="Comma-separated class list order used for reporting class-wise test accuracy.",
    )
    parser.add_argument("--clip-model", default="RN50")
    parser.add_argument("--clip-device", default="cuda")
    parser.add_argument("--clip-batch-size", type=int, default=256)
    parser.add_argument("--clip-num-workers", type=int, default=4)
    parser.add_argument("--clip-seed", type=int, default=0)
    parser.add_argument("--clip-C", type=float, default=1.0)
    parser.add_argument("--clip-penalty", default="l2")
    parser.add_argument("--clip-solver", default="lbfgs")
    parser.add_argument("--clip-fit-intercept", action="store_true", default=True)
    parser.add_argument("--clip-no-fit-intercept", action="store_false", dest="clip_fit_intercept")
    parser.add_argument("--clip-max-iter", type=int, default=5000)
    parser.add_argument(
        "--aux-losses-on-val",
        action="store_true",
        default=False,
        help="Compute aux guidance losses on validation (default: off, matches sweep jobs).",
    )
    parser.add_argument(
        "--no-aux-losses-on-val",
        action="store_false",
        dest="aux_losses_on_val",
    )
    parser.add_argument(
        "--gals-mode",
        choices=("rn50", "vit", "both"),
        default="rn50",
        help="Which GALS variant(s) to run. Default 'rn50' matches paper-style GALS setup.",
    )
    args = parser.parse_args()

    dataset_path = os.path.join(args.data_root, args.dataset_dir)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Missing dataset path: {dataset_path}")

    class_names = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    if not class_names:
        class_names = CLASS_NAMES[:]

    os.makedirs(args.logs_dir, exist_ok=True)
    header = [
        "dataset",
        "method",
        "config",
        "hparam_source",
        "hyperparams",
        "test_acc",
        "balanced_test_acc",
        "per_group",
        "worst_group",
    ] + [f"{cls}_test_acc" for cls in class_names] + [
        "checkpoint",
        "log_path",
        "seconds",
    ]

    ts = time.strftime("%Y%m%d_%H%M%S")
    python_exe = sys.executable
    rows: List[Dict[str, object]] = []

    if args.gals_mode == "rn50":
        train_specs = TRAIN_SPECS_RN50
    elif args.gals_mode == "vit":
        train_specs = TRAIN_SPECS_VIT
    else:
        train_specs = TRAIN_SPECS_RN50 + [
            TrainSpec("gals_vit", "RedMeat_Runs/configs/redmeat_gals_vit.yaml")
        ]

    for spec in train_specs:
        config_path = spec.config
        run_name = f"{args.run_prefix}_{spec.method}_{ts}"
        print(f"[RUN] redmeat {spec.method} ({config_path})", flush=True)
        required_att = _required_attention_dir(config_path)
        if required_att is not None:
            att_path = os.path.join(args.data_root, args.dataset_dir, required_att)
            if not os.path.isdir(att_path):
                raise FileNotFoundError(
                    f"Missing required attention directory for {spec.method}: {att_path}"
                )
        hp = _extract_hparams_from_config(config_path, spec.method)
        hp["aux_losses_on_val_override"] = bool(args.aux_losses_on_val)
        result = _run_main_single(
            python_exe=python_exe,
            data_root=args.data_root,
            dataset_dir=args.dataset_dir,
            config_path=config_path,
            run_name=run_name,
            logs_dir=args.logs_dir,
            class_names=class_names,
            aux_losses_on_val=bool(args.aux_losses_on_val),
        )

        row: Dict[str, object] = {
            "dataset": "redmeat",
            "method": spec.method,
            "config": config_path,
            "hparam_source": "config_yaml (paper reproduction config)",
            "hyperparams": json.dumps(hp, sort_keys=True),
            "test_acc": result.get("test_acc"),
            "balanced_test_acc": result.get("balanced_test_acc"),
            "per_group": result.get("per_group"),
            "worst_group": result.get("worst_group"),
            "checkpoint": result.get("checkpoint"),
            "log_path": result.get("log_path"),
            "seconds": result.get("seconds"),
        }
        for cls in class_names:
            row[f"{cls}_test_acc"] = result.get(f"{cls}_test_acc")
        _write_row(args.output_csv, row, header)
        rows.append(row)
        print(
            f"[DONE] redmeat {spec.method}: per_group={row['per_group']:.2f} "
            f"worst_group={row['worst_group']:.2f}",
            flush=True,
        )

    print("[RUN] redmeat clip_lr (fixed)", flush=True)
    t0 = time.time()
    clip_result = _run_clip_lr_fixed(
        dataset_path=dataset_path,
        class_names=class_names,
        clip_model=args.clip_model,
        device=args.clip_device,
        batch_size=args.clip_batch_size,
        num_workers=args.clip_num_workers,
        seed=args.clip_seed,
        c_value=args.clip_C,
        penalty=args.clip_penalty,
        solver=args.clip_solver,
        fit_intercept=args.clip_fit_intercept,
        max_iter=args.clip_max_iter,
    )
    clip_hp = {
        "clip_model": args.clip_model,
        "seed": args.clip_seed,
        "C": args.clip_C,
        "penalty": args.clip_penalty,
        "solver": args.clip_solver,
        "fit_intercept": args.clip_fit_intercept,
        "max_iter": args.clip_max_iter,
    }
    clip_row: Dict[str, object] = {
        "dataset": "redmeat",
        "method": "clip_lr",
        "config": "",
        "hparam_source": "fixed_single_run (not specified in GALS paper tables)",
        "hyperparams": json.dumps(clip_hp, sort_keys=True),
        "test_acc": clip_result.get("test_acc"),
        "balanced_test_acc": clip_result.get("balanced_test_acc"),
        "per_group": clip_result.get("per_group"),
        "worst_group": clip_result.get("worst_group"),
        "checkpoint": "",
        "log_path": "",
        "seconds": int(time.time() - t0),
    }
    for cls in class_names:
        clip_row[f"{cls}_test_acc"] = clip_result.get(f"{cls}_test_acc")
    _write_row(args.output_csv, clip_row, header)
    rows.append(clip_row)
    print(
        f"[DONE] redmeat clip_lr: per_group={clip_row['per_group']:.2f} "
        f"worst_group={clip_row['worst_group']:.2f}",
        flush=True,
    )

    print(f"[SUMMARY] wrote {len(rows)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
