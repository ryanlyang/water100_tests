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
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None


GROUP_NAMES = ["Land_on_Land", "Land_on_Water", "Water_on_Land", "Water_on_Water"]
TEST_METRIC_RE = re.compile(r"^([A-Za-z0-9_]+):\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")


@dataclass(frozen=True)
class TrainSpec:
    method: str
    config_95: str
    config_100: str


TRAIN_SPECS_RN50 = [
    TrainSpec("gals", "configs/waterbirds_95_gals.yaml", "configs/waterbirds_100_gals.yaml"),
    TrainSpec("abn", "configs/waterbirds_95_abn.yaml", "configs/waterbirds_100_abn.yaml"),
    TrainSpec("upweight", "configs/waterbirds_95_upweight.yaml", "configs/waterbirds_100_upweight.yaml"),
    TrainSpec("vanilla", "configs/waterbirds_95_vanilla.yaml", "configs/waterbirds_100_vanilla.yaml"),
]

TRAIN_SPECS_VIT = [
    TrainSpec("gals_vit", "configs/waterbirds_95_gals_vit.yaml", "configs/waterbirds_100_gals_vit.yaml"),
    TrainSpec("abn", "configs/waterbirds_95_abn.yaml", "configs/waterbirds_100_abn.yaml"),
    TrainSpec("upweight", "configs/waterbirds_95_upweight.yaml", "configs/waterbirds_100_upweight.yaml"),
    TrainSpec("vanilla", "configs/waterbirds_95_vanilla.yaml", "configs/waterbirds_100_vanilla.yaml"),
]

WB_DATASET_MAP = {
    "wb95": "waterbird_complete95_forest2water2",
    "wb100": "waterbird_1.0_forest2water2",
}


def _to_pct(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if -1.0 <= x <= 1.0:
        return 100.0 * x
    return x


def _write_row(csv_path: str, row: Dict, header: List[str]) -> None:
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
        hp["separate_classes"] = bool(cfg.DATA.SEPARATE_CLASSES)
        hp["use_class_weights"] = bool(cfg.DATA.USE_CLASS_WEIGHTS)
    return hp


def _run_main_single(
    *,
    python_exe: str,
    data_root: str,
    dataset_dir: str,
    config_path: str,
    run_name: str,
    logs_dir: str,
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
        f"DATA.WATERBIRDS_DIR={dataset_dir}",
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
    test_metrics: Dict[str, float] = {}
    checkpoint: Optional[str] = None

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
            if m:
                key = m.group(1)
                val = float(m.group(2))
                if key == "test_acc" or key == "balanced_test_acc" or key.endswith("_test_acc"):
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

    group_accs: Dict[str, float] = {}
    for k, v in test_metrics.items():
        if k.endswith("_test_acc") and k != "balanced_test_acc":
            group_accs[k] = _to_pct(v)  # type: ignore[arg-type]

    per_group = float(np.mean(list(group_accs.values()))) if group_accs else None
    worst_group = float(np.min(list(group_accs.values()))) if group_accs else None
    out = {
        "checkpoint": checkpoint,
        "log_path": log_path,
        "seconds": int(time.time() - t0),
        "test_acc": _to_pct(test_metrics.get("test_acc")),
        "balanced_test_acc": _to_pct(test_metrics.get("balanced_test_acc")),
        "per_group": per_group,
        "worst_group": worst_group,
    }
    out.update(group_accs)
    return out


def _run_clip_lr_fixed(
    *,
    dataset_path: str,
    clip_model: str,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    C: float,
    penalty: str,
    solver: str,
    fit_intercept: bool,
    max_iter: int,
) -> Dict[str, object]:
    # Reuse helper utilities from the existing CLIP+LR script to avoid divergence.
    import run_clip_lr_sweep as clip_lr
    from sklearn.linear_model import LogisticRegression

    np.random.seed(seed)

    try:
        model, preprocess = clip_lr._try_import_clip().load(clip_model, device=device, jit=False)
    except TypeError:
        model, preprocess = clip_lr._try_import_clip().load(clip_model, device=device)

    root, cfg, Waterbirds = clip_lr._load_waterbirds(dataset_path)
    train_ds = Waterbirds(root=root, cfg=cfg, split="train", transform=preprocess)
    val_ds = Waterbirds(root=root, cfg=cfg, split="val", transform=preprocess)
    test_ds = Waterbirds(root=root, cfg=cfg, split="test", transform=preprocess)

    X_train, y_train, _ = clip_lr._extract_features(train_ds, model, device, batch_size, num_workers)
    X_val, y_val, g_val = clip_lr._extract_features(val_ds, model, device, batch_size, num_workers)
    X_test, y_test, g_test = clip_lr._extract_features(test_ds, model, device, batch_size, num_workers)

    X_train = clip_lr._l2_normalize(X_train).astype(np.float32, copy=False)
    X_val = clip_lr._l2_normalize(X_val).astype(np.float32, copy=False)
    X_test = clip_lr._l2_normalize(X_test).astype(np.float32, copy=False)

    clf = LogisticRegression(
        random_state=seed,
        C=C,
        penalty=penalty,
        solver=solver,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        n_jobs=1,
        verbose=0,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    val_acc = float(np.mean((val_pred == y_val).astype(np.float64)) * 100.0)
    val_group = clip_lr._group_acc(y_val, val_pred, g_val, num_groups=4)
    val_avg_group = float(np.nanmean(val_group))
    val_worst_group = float(np.nanmin(val_group))

    test_pred = clf.predict(X_test)
    test_acc = float(np.mean((test_pred == y_test).astype(np.float64)) * 100.0)
    test_group = clip_lr._group_acc(y_test, test_pred, g_test, num_groups=4)
    test_avg_group = float(np.nanmean(test_group))
    test_worst_group = float(np.nanmin(test_group))

    out = {
        "test_acc": test_acc,
        "balanced_test_acc": test_avg_group,
        "per_group": test_avg_group,
        "worst_group": test_worst_group,
        "val_acc": val_acc,
        "val_per_group": val_avg_group,
        "val_worst_group": val_worst_group,
    }
    for i, name in enumerate(GROUP_NAMES):
        out[f"{name}_test_acc"] = float(test_group[i])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-run Waterbirds reference check using paper-config hyperparameters for "
            "GALS (RN50 by default), ABN, UpWeight, Vanilla, plus fixed CLIP+LR."
        )
    )
    parser.add_argument("--data-root", default="/home/ryreu/guided_cnn/waterbirds")
    parser.add_argument("--wb95-dir", default=WB_DATASET_MAP["wb95"])
    parser.add_argument("--wb100-dir", default=WB_DATASET_MAP["wb100"])
    parser.add_argument("--logs-dir", default="/home/ryreu/guided_cnn/logsWaterbird/paper_single_runs")
    parser.add_argument("--output-csv", default="/home/ryreu/guided_cnn/logsWaterbird/paper_single_runs_summary.csv")
    parser.add_argument("--run-prefix", default="paper_ref")
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
        help="Which GALS variant(s) to run. Default 'rn50' matches paper GALS config.",
    )
    args = parser.parse_args()

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
        "Land_on_Land_test_acc",
        "Land_on_Water_test_acc",
        "Water_on_Land_test_acc",
        "Water_on_Water_test_acc",
        "checkpoint",
        "log_path",
        "seconds",
    ]

    datasets: List[Tuple[str, str]] = [("wb95", args.wb95_dir), ("wb100", args.wb100_dir)]
    ts = time.strftime("%Y%m%d_%H%M%S")
    python_exe = sys.executable
    rows: List[Dict[str, object]] = []
    if args.gals_mode == "rn50":
        train_specs = TRAIN_SPECS_RN50
    elif args.gals_mode == "vit":
        train_specs = TRAIN_SPECS_VIT
    else:
        train_specs = TRAIN_SPECS_RN50 + [
            TrainSpec("gals_vit", "configs/waterbirds_95_gals_vit.yaml", "configs/waterbirds_100_gals_vit.yaml")
        ]

    for dataset_tag, dataset_dir in datasets:
        for spec in train_specs:
            config_path = spec.config_95 if dataset_tag == "wb95" else spec.config_100
            run_name = f"{args.run_prefix}_{spec.method}_{dataset_tag}_{ts}"
            print(f"[RUN] {dataset_tag} {spec.method} ({config_path})", flush=True)
            required_att = _required_attention_dir(config_path)
            if required_att is not None:
                att_path = os.path.join(args.data_root, dataset_dir, required_att)
                if not os.path.isdir(att_path):
                    raise FileNotFoundError(
                        f"Missing required attention directory for {spec.method} ({dataset_tag}): {att_path}"
                    )
            hp = _extract_hparams_from_config(config_path, spec.method)
            hp["aux_losses_on_val_override"] = bool(args.aux_losses_on_val)
            result = _run_main_single(
                python_exe=python_exe,
                data_root=args.data_root,
                dataset_dir=dataset_dir,
                config_path=config_path,
                run_name=run_name,
                logs_dir=args.logs_dir,
                aux_losses_on_val=bool(args.aux_losses_on_val),
            )
            row = {
                "dataset": dataset_tag,
                "method": spec.method,
                "config": config_path,
                "hparam_source": "config_yaml (paper reproduction config)",
                "hyperparams": json.dumps(hp, sort_keys=True),
                "test_acc": result.get("test_acc"),
                "balanced_test_acc": result.get("balanced_test_acc"),
                "per_group": result.get("per_group"),
                "worst_group": result.get("worst_group"),
                "Land_on_Land_test_acc": result.get("Land_on_Land_test_acc"),
                "Land_on_Water_test_acc": result.get("Land_on_Water_test_acc"),
                "Water_on_Land_test_acc": result.get("Water_on_Land_test_acc"),
                "Water_on_Water_test_acc": result.get("Water_on_Water_test_acc"),
                "checkpoint": result.get("checkpoint"),
                "log_path": result.get("log_path"),
                "seconds": result.get("seconds"),
            }
            _write_row(args.output_csv, row, header)
            rows.append(row)
            print(
                f"[DONE] {dataset_tag} {spec.method}: per_group={row['per_group']:.2f} "
                f"worst_group={row['worst_group']:.2f}",
                flush=True,
            )

        # CLIP + LR fixed run (single deterministic setup).
        dataset_path = os.path.join(args.data_root, dataset_dir)
        print(f"[RUN] {dataset_tag} clip_lr (fixed)", flush=True)
        t0 = time.time()
        clip_result = _run_clip_lr_fixed(
            dataset_path=dataset_path,
            clip_model=args.clip_model,
            device=args.clip_device,
            batch_size=args.clip_batch_size,
            num_workers=args.clip_num_workers,
            seed=args.clip_seed,
            C=args.clip_C,
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
        row = {
            "dataset": dataset_tag,
            "method": "clip_lr",
            "config": "",
            "hparam_source": "fixed_single_run (not specified in GALS paper tables)",
            "hyperparams": json.dumps(clip_hp, sort_keys=True),
            "test_acc": clip_result.get("test_acc"),
            "balanced_test_acc": clip_result.get("balanced_test_acc"),
            "per_group": clip_result.get("per_group"),
            "worst_group": clip_result.get("worst_group"),
            "Land_on_Land_test_acc": clip_result.get("Land_on_Land_test_acc"),
            "Land_on_Water_test_acc": clip_result.get("Land_on_Water_test_acc"),
            "Water_on_Land_test_acc": clip_result.get("Water_on_Land_test_acc"),
            "Water_on_Water_test_acc": clip_result.get("Water_on_Water_test_acc"),
            "checkpoint": "",
            "log_path": "",
            "seconds": int(time.time() - t0),
        }
        _write_row(args.output_csv, row, header)
        rows.append(row)
        print(
            f"[DONE] {dataset_tag} clip_lr: per_group={row['per_group']:.2f} "
            f"worst_group={row['worst_group']:.2f}",
            flush=True,
        )

    print(f"[SUMMARY] wrote {len(rows)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
