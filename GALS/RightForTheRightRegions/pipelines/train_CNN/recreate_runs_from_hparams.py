#!/usr/bin/env python3
"""Recreate one-pass training runs from dataset hyperparameter YAMLs.

This script powers dataset-specific wrappers:
- recreate_waterbirds95_runs.py
- recreate_waterbirds100_runs.py
- recreate_redmeat_runs.py
- recreate_decoymnist_runs.py

Each run executes each listed method once (no Optuna sweep), logs stdout, and
writes summary CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install with `pip install pyyaml`. "
        f"Import error: {exc}"
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
GALS_ROOT = REPO_ROOT / "repro_runs" / "third_party" / "GALS"

WB_GALS_SWEEP = REPO_ROOT / "repro_runs" / "other_models" / "waterbirds" / "sweeps" / "gals_waterbirds_sweep.py"
WB_CLIP_LR_SWEEP = REPO_ROOT / "repro_runs" / "other_models" / "waterbirds" / "sweeps" / "clip_lr_waterbirds_sweep.py"
WB_CLIP_ZS = REPO_ROOT / "repro_runs" / "other_models" / "waterbirds" / "baselines" / "clip_zeroshot_waterbirds.py"
WB_AFR_SWEEP = REPO_ROOT / "repro_runs" / "other_models" / "waterbirds" / "sweeps" / "afr_waterbirds_sweep.py"

RM_GALS_SWEEP = REPO_ROOT / "repro_runs" / "other_models" / "redmeat" / "sweeps" / "gals_redmeat_sweep.py"
RM_CLIP_LR_SWEEP = REPO_ROOT / "repro_runs" / "other_models" / "redmeat" / "sweeps" / "clip_lr_redmeat_sweep.py"
RM_CLIP_ZS = REPO_ROOT / "repro_runs" / "other_models" / "redmeat" / "baselines" / "clip_zeroshot_redmeat.py"
RM_AFR_SWEEP = REPO_ROOT / "repro_runs" / "other_models" / "redmeat" / "sweeps" / "afr_redmeat_sweep.py"

R4RR_WB_SWEEP = REPO_ROOT / "repro_runs" / "r4rr" / "sweeps" / "r4rr_waterbirds_sweep.py"
R4RR_RM_SWEEP = REPO_ROOT / "repro_runs" / "r4rr" / "sweeps" / "r4rr_redmeat_sweep.py"
R4RR_WB_INVERT = REPO_ROOT / "repro_runs" / "r4rr" / "ablations" / "r4rr_waterbirds_invert.py"
R4RR_WB_JOINT = REPO_ROOT / "repro_runs" / "r4rr" / "ablations" / "r4rr_waterbirds_joint.py"

DEC_UPWEIGHT = REPO_ROOT / "repro_runs" / "other_models" / "decoymnist" / "train" / "upweight_decoy_fixed.py"
DEC_ABN = REPO_ROOT / "repro_runs" / "other_models" / "decoymnist" / "train" / "abn_decoy_fixed.py"
DEC_GALS = REPO_ROOT / "repro_runs" / "other_models" / "decoymnist" / "train" / "gals_decoy_fixed.py"
DEC_AFR = REPO_ROOT / "repro_runs" / "other_models" / "decoymnist" / "train" / "afr_decoy_fixed.py"
DEC_CLIP_LR = REPO_ROOT / "repro_runs" / "other_models" / "decoymnist" / "baselines" / "clip_lr_decoy_fixed.py"
DEC_CLIP_ZS = REPO_ROOT / "repro_runs" / "other_models" / "decoymnist" / "baselines" / "clip_zeroshot_decoy.py"
DEC_R4RR = REPO_ROOT / "repro_runs" / "r4rr" / "train" / "r4rr_decoy_fixed.py"


def _float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing config YAML: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _read_first_csv_row(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    raise RuntimeError(f"CSV had no data rows: {path}")


def _run_cmd(cmd: List[str], log_path: Path, *, env: Dict[str, str], cwd: Optional[Path], dry_run: bool) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("[DRY RUN]\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("[CMD] " + " ".join(cmd) + "\n")
        lf.flush()
        print(f"[RUN] {' '.join(cmd)}", flush=True)
        print(f"[LOG] {log_path}", flush=True)

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            print(line, end="", flush=True)
        returncode = proc.wait()

    if returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={returncode}): {' '.join(cmd)}\n"
            f"See log: {log_path}"
        )
    return subprocess.CompletedProcess(cmd, returncode, "", "")


def _extract_decoy_stdout_metrics(stdout: str) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "best_val": None,
        "test_acc": None,
        "test_balanced_class_acc": None,
        "test_worst_group": None,
    }

    # Per-seed lines (n-seeds=1 expected)
    m = re.search(r"best_val_acc=([0-9.]+)%.*test_acc=([0-9.]+)%", stdout)
    if m:
        metrics["best_val"] = float(m.group(1))
        metrics["test_acc"] = float(m.group(2))

    # Balanced class accuracy (preferred per-class aggregate metric for DecoyMNIST).
    m_bal = re.search(r"test_balanced_class_acc(?:@val(?:_worst)?)?=([0-9.]+)", stdout)
    if m_bal:
        metrics["test_balanced_class_acc"] = float(m_bal.group(1))

    # ABN / AFR variants may print explicit worst-class/group values.
    m_worst = re.search(r"test_worst(?:_class)?_acc(?:@val(?:_worst)?)?=([0-9.]+)", stdout)
    if m_worst:
        metrics["test_worst_group"] = float(m_worst.group(1))

    # Fallback summary lines.
    if metrics["best_val"] is None:
        m = re.search(r"best_val_acc\s+mean=([0-9.]+)%", stdout)
        if m:
            metrics["best_val"] = float(m.group(1))
    if metrics["test_acc"] is None:
        m = re.search(r"test_acc\s+mean=([0-9.]+)%", stdout)
        if m:
            metrics["test_acc"] = float(m.group(1))
    if metrics["test_balanced_class_acc"] is None:
        m = re.search(r"test_balanced_class_acc\s+mean=([0-9.]+)%", stdout)
        if m:
            metrics["test_balanced_class_acc"] = float(m.group(1))

    return metrics


def _base_env() -> Dict[str, str]:
    env = os.environ.copy()
    cur_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{GALS_ROOT}:{cur_py}" if cur_py else str(GALS_ROOT)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("SAVE_CHECKPOINTS", "0")
    # Hard-disable wandb to keep recreate runs fully local/non-interactive.
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    return env


def _summary_row(method: str, status: str, *,
                 best_val: Optional[float] = None,
                 test_acc: Optional[float] = None,
                 per_group: Optional[float] = None,
                 worst_group: Optional[float] = None,
                 notes: str = "",
                 log_path: Optional[Path] = None,
                 artifacts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    return {
        "method": method,
        "status": status,
        "best_val": best_val,
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "notes": notes,
        "log_path": str(log_path) if log_path else "",
        "artifacts": json.dumps(artifacts or {}, sort_keys=True),
    }


def _attention_override(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return f"DATA.ATTENTION_DIR={Path(path).expanduser().resolve()}"


def _default_decoy_weclip_maps(repo_root: Path) -> Optional[Path]:
    train_only = repo_root / "WeCLIPPlus" / "results_decoy_r4rr_trainonly" / "val" / "prediction_cmap"
    legacy = repo_root / "WeCLIPPlus" / "results_decoy_r4rr" / "val" / "prediction_cmap"
    if train_only.is_dir():
        return train_only
    if legacy.is_dir():
        return legacy
    return None


def _write_summary(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    header = [
        "method",
        "status",
        "best_val",
        "test_acc",
        "per_group",
        "worst_group",
        "notes",
        "log_path",
        "artifacts",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


class Runner:
    def __init__(self, *, python_bin: str, output_dir: Path, seed: int, dry_run: bool) -> None:
        self.python_bin = python_bin
        self.output_dir = output_dir
        self.seed = seed
        self.dry_run = dry_run
        self.env = _base_env()

    def run(self, method: str, cmd: List[str], *, cwd: Optional[Path] = None) -> Path:
        method_dir = self.output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        log_path = method_dir / "stdout.log"
        _run_cmd(cmd, log_path, env=self.env, cwd=cwd, dry_run=self.dry_run)
        return log_path


# ------------------------- Waterbirds -------------------------

def _wb_dataset_key(split: str) -> str:
    if split == "95":
        return "waterbirds95_optimized_hparams"
    if split == "100":
        return "waterbirds100_optimized_hparams"
    raise ValueError(f"Unsupported split: {split}")


def _wb_default_dir(split: str) -> str:
    return "waterbird_complete95_forest2water2" if split == "95" else "waterbird_1.0_forest2water2"


def _wb_run(split: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfg_path = Path(args.config_path) if args.config_path else (CONFIG_DIR / f"waterbirds{split}_optimized_hparams.yaml")
    data = _load_yaml(cfg_path)
    hp = data[_wb_dataset_key(split)]

    data_root = Path(args.data_root).expanduser().resolve()
    dataset_dir = args.dataset_dir or _wb_default_dir(split)
    dataset_path = data_root / dataset_dir
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Missing dataset path: {dataset_path}")

    run_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        REPO_ROOT / "logs" / "recreate" / f"waterbirds{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    runner = Runner(python_bin=args.python_bin, output_dir=run_root, seed=args.seed, dry_run=args.dry_run)

    rn50_maps = Path(args.r4rr_rn50_map_path).expanduser().resolve() if args.r4rr_rn50_map_path else None
    vit_maps = Path(args.r4rr_vit_map_path).expanduser().resolve() if args.r4rr_vit_map_path else None
    weclip_maps = (
        Path(args.weclip_prediction_cmap_dir).expanduser().resolve()
        if args.weclip_prediction_cmap_dir
        else None
    )
    our_masks = (
        Path(args.gals_our_masks_path).expanduser().resolve()
        if args.gals_our_masks_path
        else weclip_maps
    )
    r4rr_opt_maps = (
        Path(args.r4rr_optimized_map_path).expanduser().resolve()
        if args.r4rr_optimized_map_path
        else weclip_maps if weclip_maps is not None else rn50_maps
    )
    r4rr_ablation_maps = (
        Path(args.r4rr_ablation_map_path).expanduser().resolve()
        if args.r4rr_ablation_map_path
        else weclip_maps if weclip_maps is not None else r4rr_opt_maps
    )
    rn50_attention_override = _attention_override(rn50_maps) if rn50_maps else None
    vit_attention_override = _attention_override(vit_maps) if vit_maps else None

    rows: List[Dict[str, Any]] = []

    def run_gals_method(method_name: str, *, method_flag: str, config_name: str, h: Dict[str, Any], overrides: Optional[List[str]] = None) -> None:
        out_csv = run_root / method_name / "result.csv"
        trial_logs = run_root / method_name / "trial_logs"

        cmd = [
            runner.python_bin,
            str(WB_GALS_SWEEP),
            "--method",
            method_flag,
            "--config",
            f"configs/{config_name}",
            "--data-root",
            str(data_root),
            "--waterbirds-dir",
            dataset_dir,
            "--n-trials",
            "1",
            "--seed",
            str(runner.seed),
            "--train-seed",
            str(runner.seed),
            "--sampler",
            "random",
            "--keep",
            "none",
            "--post-seeds",
            "0",
            "--output-csv",
            str(out_csv),
            "--logs-dir",
            str(trial_logs),
            "--base-lr-min",
            str(h["base_lr"]),
            "--base-lr-max",
            str(h["base_lr"]),
            "--cls-lr-min",
            str(h["classifier_lr"]),
            "--cls-lr-max",
            str(h["classifier_lr"]),
            "--run-name-prefix",
            f"recreate_wb{split}_{method_name}",
        ]

        if "momentum" in h:
            cmd.append(f"EXP.MOMENTUM={h['momentum']}")
        if "grad_weight" in h:
            cmd.extend(["--weight-min", str(h["grad_weight"]), "--weight-max", str(h["grad_weight"])])
        if "grad_criterion" in h:
            cmd.extend(["--grad-criteria", str(h["grad_criterion"])])
        if "abn_cls_weight" in h:
            cmd.extend(["--abn-cls-weight-min", str(h["abn_cls_weight"]), "--abn-cls-weight-max", str(h["abn_cls_weight"])])
        if "abn_att_weight" in h:
            cmd.extend(["--abn-att-weight-min", str(h["abn_att_weight"]), "--abn-att-weight-max", str(h["abn_att_weight"])])
        if "cam_weight" in h:
            cmd.extend(["--cam-weight-min", str(h["cam_weight"]), "--cam-weight-max", str(h["cam_weight"])])
        if overrides:
            cmd.extend(overrides)

        log_path = runner.run(method_name, cmd, cwd=GALS_ROOT)
        row = _read_first_csv_row(out_csv)
        rows.append(
            _summary_row(
                method_name,
                "ok",
                best_val=_float(row.get("best_balanced_val_acc")),
                test_acc=_float(row.get("test_acc") or row.get("balanced_test_acc")),
                per_group=_float(row.get("per_group")),
                worst_group=_float(row.get("worst_group")),
                log_path=log_path,
                artifacts={"csv": str(out_csv)},
            )
        )

    # Baselines + GALS variants
    run_gals_method("vanilla", method_flag="upweight", config_name=f"waterbirds_{split}_vanilla.yaml", h=hp["vanilla"])
    run_gals_method("abn", method_flag="abn_cls", config_name=f"waterbirds_{split}_abn.yaml", h=hp["abn"])
    run_gals_method("upweight", method_flag="upweight", config_name=f"waterbirds_{split}_upweight.yaml", h=hp["upweight"])
    run_gals_method(
        "gals_rrr_rn50_maps",
        method_flag="gals",
        config_name=f"waterbirds_{split}_gals.yaml",
        h=hp["gals_rrr_rn50_maps"],
        overrides=[rn50_attention_override] if rn50_attention_override else None,
    )
    run_gals_method(
        "gals_rrr_vit_maps",
        method_flag="gals",
        config_name=f"waterbirds_{split}_gals_vit.yaml",
        h=hp["gals_rrr_vit_maps"],
        overrides=[vit_attention_override] if vit_attention_override else None,
    )
    run_gals_method(
        "gals_abn_vit_maps",
        method_flag="abn_att",
        config_name=f"waterbirds_{split}_abn_vit.yaml",
        h=hp["gals_abn_vit_maps"],
        overrides=[vit_attention_override] if vit_attention_override else None,
    )
    run_gals_method(
        "gals_gradcam_vit_maps",
        method_flag="gradcam",
        config_name=f"waterbirds_{split}_gradcam_vit.yaml",
        h=hp["gals_gradcam_vit_maps"],
        overrides=[vit_attention_override] if vit_attention_override else None,
    )

    if our_masks is None:
        rows.append(
            _summary_row(
                "gals_our_masks",
                "skipped",
                notes="Missing --gals-segmentation-dir or --weclip-prediction-cmap-dir",
            )
        )
    else:
        run_gals_method(
            "gals_our_masks",
            method_flag="gals",
            config_name=f"waterbirds_{split}_gals_ourmasks.yaml",
            h=hp["gals_our_masks"],
            overrides=[f"DATA.SEGMENTATION_DIR={our_masks}"],
        )

    # CLIP+LR
    clip_csv = run_root / "clip_lr" / "result.csv"
    clip_cmd = [
        runner.python_bin,
        str(WB_CLIP_LR_SWEEP),
        str(dataset_path),
        "--n-trials",
        "1",
        "--seed",
        str(runner.seed),
        "--sampler",
        "random",
        "--C-min",
        str(hp["clip_lr"]["c"]),
        "--C-max",
        str(hp["clip_lr"]["c"]),
        "--tol-min",
        "1e-4",
        "--tol-max",
        "1e-4",
        "--penalty-solvers",
        args.clip_lr_penalty_solvers,
        "--feature-modes",
        args.clip_lr_feature_modes,
        "--class-weight-options",
        args.clip_lr_class_weight_options,
        "--post-seeds",
        "0",
        "--output-csv",
        str(clip_csv),
    ]
    clip_log = runner.run("clip_lr", clip_cmd)
    clip_row = _read_first_csv_row(clip_csv)
    rows.append(
        _summary_row(
            "clip_lr",
            "ok",
            best_val=_float(clip_row.get("val_avg_group_acc") or clip_row.get("val_worst_group_acc")),
            test_acc=_float(clip_row.get("test_acc")),
            per_group=_float(clip_row.get("test_avg_group_acc")),
            worst_group=_float(clip_row.get("test_worst_group_acc")),
            log_path=clip_log,
            artifacts={"csv": str(clip_csv)},
        )
    )

    # CLIP zero-shot
    clip_zs_csv = run_root / "clip_zs" / "result.csv"
    clip_zs_cmd = [
        runner.python_bin,
        str(WB_CLIP_ZS),
        str(dataset_path),
        "--clip-model",
        args.clip_zs_model,
        "--batch-size",
        str(args.clip_zs_batch_size),
        "--num-workers",
        str(args.clip_zs_num_workers),
        "--seeds",
        str(runner.seed),
        "--output-csv",
        str(clip_zs_csv),
    ]
    if args.clip_zs_device:
        clip_zs_cmd.extend(["--device", args.clip_zs_device])
    clip_zs_log = runner.run("clip_zs", clip_zs_cmd)
    clip_zs_row = _read_first_csv_row(clip_zs_csv)
    rows.append(
        _summary_row(
            "clip_zs",
            "ok",
            best_val=_float(clip_zs_row.get("val_balanced_group_acc") or clip_zs_row.get("val_worst_group_acc") or clip_zs_row.get("val_acc")),
            test_acc=_float(clip_zs_row.get("test_acc")),
            per_group=_float(clip_zs_row.get("test_balanced_group_acc")),
            worst_group=_float(clip_zs_row.get("test_worst_group_acc")),
            log_path=clip_zs_log,
            artifacts={"csv": str(clip_zs_csv)},
        )
    )

    # AFR
    afr_out = run_root / "afr" / "out"
    afr_logs = run_root / "afr" / "logs"
    afr_cmd = [
        runner.python_bin,
        str(WB_AFR_SWEEP),
        "--data-dir",
        str(dataset_path),
        "--output-root",
        str(afr_out),
        "--logs-root",
        str(afr_logs),
        "--python-exe",
        runner.python_bin,
        "--seeds",
        str(runner.seed),
        "--gammas",
        str(hp["afr"]["gamma"]),
        "--reg-coeffs",
        str(hp["afr"]["reg"]),
        "--stage1-epochs",
        str(args.afr_stage1_epochs),
        "--stage2-epochs",
        str(args.afr_stage2_epochs),
    ]
    afr_log = runner.run("afr", afr_cmd)
    afr_best_csv = afr_out / "afr_waterbirds_best_by_seed.csv"
    afr_row = _read_first_csv_row(afr_best_csv)
    rows.append(
        _summary_row(
            "afr",
            "ok",
            best_val=_float(afr_row.get("best_val_wga")),
            test_acc=_float(afr_row.get("best_test_at_val")),
            per_group=_float(afr_row.get("best_test_mean_at_val") or afr_row.get("best_test_mean")),
            worst_group=_float(afr_row.get("best_test_at_val")),
            log_path=afr_log,
            artifacts={"csv": str(afr_best_csv)},
        )
    )

    def run_r4rr_sweep(method_name: str, h: Dict[str, Any], teacher_maps: Optional[Path]) -> None:
        if teacher_maps is None:
            rows.append(_summary_row(method_name, "skipped", notes=f"Missing teacher-map path for {method_name}"))
            return
        out_csv = run_root / method_name / "result.csv"
        cmd = [
            runner.python_bin,
            str(R4RR_WB_SWEEP),
            str(dataset_path),
            str(teacher_maps),
            "--n-trials",
            "1",
            "--seed",
            str(runner.seed),
            "--sampler",
            "random",
            "--output-csv",
            str(out_csv),
            "--attn-min",
            str(h["attention_epoch"]),
            "--attn-max",
            str(h["attention_epoch"]),
            "--kl-min",
            str(h["kl_lambda"]),
            "--kl-max",
            str(h["kl_lambda"]),
            "--base-lr-min",
            str(h["base_lr"]),
            "--base-lr-max",
            str(h["base_lr"]),
            "--cls-lr-min",
            str(h["classifier_lr"]),
            "--cls-lr-max",
            str(h["classifier_lr"]),
            "--lr2-mult-min",
            str(h.get("lr2_mult", 1.0)),
            "--lr2-mult-max",
            str(h.get("lr2_mult", 1.0)),
        ]
        log_path = runner.run(method_name, cmd, cwd=GALS_ROOT)
        row = _read_first_csv_row(out_csv)
        rows.append(
            _summary_row(
                method_name,
                "ok",
                best_val=_float(row.get("best_balanced_val_acc")),
                test_acc=_float(row.get("test_acc")),
                per_group=_float(row.get("per_group")),
                worst_group=_float(row.get("worst_group")),
                log_path=log_path,
                artifacts={"csv": str(out_csv)},
            )
        )

    run_r4rr_sweep("r4rr_gals_rn50_maps", hp["r4rr_gals_rn50_maps"], rn50_maps)
    run_r4rr_sweep("r4rr_gals_vit_maps", hp["r4rr_gals_vit_maps"], vit_maps)
    if "r4rr_optimized" in hp:
        run_r4rr_sweep("r4rr_optimized", hp["r4rr_optimized"], r4rr_opt_maps)

    # Ablations, if present in YAML.
    if "r4rr_invert" in hp:
        if r4rr_ablation_maps is None:
            rows.append(
                _summary_row(
                    "r4rr_invert",
                    "skipped",
                    notes="Missing --r4rr-ablation-teacher-maps-dir or --weclip-prediction-cmap-dir",
                )
            )
        else:
            h = hp["r4rr_invert"]
            cmd = [
                runner.python_bin,
                str(R4RR_WB_INVERT),
                str(dataset_path),
                str(r4rr_ablation_maps),
                "--seed",
                str(runner.seed),
                "--attention_epoch",
                str(h["attention_epoch"]),
                "--kl_lambda",
                str(h["kl_lambda"]),
                "--base_lr",
                str(h["base_lr"]),
                "--classifier_lr",
                str(h["classifier_lr"]),
                "--lr2_mult",
                str(h.get("lr2_mult", 1.0)),
            ]
            log_path = runner.run("r4rr_invert", cmd)
            stdout = (run_root / "r4rr_invert" / "stdout.log").read_text(encoding="utf-8", errors="ignore")
            best_val = _float(re.search(r"best_balanced_val_acc=([0-9.]+)", stdout).group(1) if re.search(r"best_balanced_val_acc=([0-9.]+)", stdout) else None)
            test_acc = _float(re.search(r"test_acc=([0-9.]+)%", stdout).group(1) if re.search(r"test_acc=([0-9.]+)%", stdout) else None)
            m_pg = re.search(r"Per Group:\s*([0-9.]+)%\s+Worst Group:\s*([0-9.]+)%", stdout)
            per_group = _float(m_pg.group(1)) if m_pg else None
            worst_group = _float(m_pg.group(2)) if m_pg else None
            rows.append(_summary_row("r4rr_invert", "ok", best_val=best_val, test_acc=test_acc, per_group=per_group, worst_group=worst_group, log_path=log_path))

    if "r4rr_joint" in hp:
        if r4rr_ablation_maps is None:
            rows.append(
                _summary_row(
                    "r4rr_joint",
                    "skipped",
                    notes="Missing --r4rr-ablation-teacher-maps-dir or --weclip-prediction-cmap-dir",
                )
            )
        else:
            h = hp["r4rr_joint"]
            cmd = [
                runner.python_bin,
                str(R4RR_WB_JOINT),
                str(dataset_path),
                str(r4rr_ablation_maps),
                "--seed",
                str(runner.seed),
                "--kl_lambda",
                str(h["kl_lambda"]),
                "--base_lr",
                str(h["base_lr"]),
                "--classifier_lr",
                str(h["classifier_lr"]),
            ]
            log_path = runner.run("r4rr_joint", cmd)
            stdout = (run_root / "r4rr_joint" / "stdout.log").read_text(encoding="utf-8", errors="ignore")
            best_val = _float(re.search(r"best_balanced_val_acc=([0-9.]+)", stdout).group(1) if re.search(r"best_balanced_val_acc=([0-9.]+)", stdout) else None)
            test_acc = _float(re.search(r"test_acc=([0-9.]+)%", stdout).group(1) if re.search(r"test_acc=([0-9.]+)%", stdout) else None)
            m_pg = re.search(r"Per Group:\s*([0-9.]+)%\s+Worst Group:\s*([0-9.]+)%", stdout)
            per_group = _float(m_pg.group(1)) if m_pg else None
            worst_group = _float(m_pg.group(2)) if m_pg else None
            rows.append(_summary_row("r4rr_joint", "ok", best_val=best_val, test_acc=test_acc, per_group=per_group, worst_group=worst_group, log_path=log_path))

    _write_summary(rows, run_root)
    return rows


# ------------------------- RedMeat -------------------------

def _rm_run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfg_path = Path(args.config_path) if args.config_path else (CONFIG_DIR / "redmeat_optimized_hparams.yaml")
    data = _load_yaml(cfg_path)
    hp = data["redmeat_optimized_hparams"]

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Missing dataset path: {dataset_path}")

    run_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        REPO_ROOT / "logs" / "recreate" / f"redmeat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    runner = Runner(python_bin=args.python_bin, output_dir=run_root, seed=args.seed, dry_run=args.dry_run)

    rn50_maps = Path(args.r4rr_rn50_map_path).expanduser().resolve() if args.r4rr_rn50_map_path else None
    vit_maps = Path(args.r4rr_vit_map_path).expanduser().resolve() if args.r4rr_vit_map_path else None
    weclip_maps = (
        Path(args.weclip_prediction_cmap_dir).expanduser().resolve()
        if args.weclip_prediction_cmap_dir
        else None
    )
    our_masks = (
        Path(args.gals_our_masks_path).expanduser().resolve()
        if args.gals_our_masks_path
        else weclip_maps
    )
    r4rr_opt_maps = (
        Path(args.r4rr_optimized_map_path).expanduser().resolve()
        if args.r4rr_optimized_map_path
        else weclip_maps if weclip_maps is not None else rn50_maps
    )
    if rn50_maps is None:
        default_rn50 = dataset_path / "clip_rn50_attention_gradcam"
        legacy_rn50 = dataset_path / "meat_clip_rn50_attention_gradcam"
        if default_rn50.is_dir():
            rn50_maps = default_rn50
        elif legacy_rn50.is_dir():
            rn50_maps = legacy_rn50
    if vit_maps is None:
        default_vit = dataset_path / "clip_vit_attention"
        if default_vit.is_dir():
            vit_maps = default_vit

    rn50_attention_override = _attention_override(rn50_maps) if rn50_maps else None
    vit_attention_override = _attention_override(vit_maps) if vit_maps else rn50_attention_override

    rows: List[Dict[str, Any]] = []

    # Config defaults can be overridden from CLI for missing RedMeat repo-specific configs.
    cfg = {
        "vanilla": args.cfg_vanilla,
        "abn": args.cfg_abn,
        "upweight": args.cfg_upweight,
        "gals_rn50": args.cfg_gals_rn50,
        "gals_vit": args.cfg_gals_vit,
        "abn_vit": args.cfg_abn_vit,
        "gradcam_vit": args.cfg_gradcam_vit,
        "gals_our_masks": args.cfg_gals_our_masks,
    }

    def run_gals_method(method_name: str, *, method_flag: str, config_rel: str, h: Dict[str, Any], overrides: Optional[List[str]] = None) -> None:
        out_csv = run_root / method_name / "result.csv"
        trial_logs = run_root / method_name / "trial_logs"
        cmd = [
            runner.python_bin,
            str(RM_GALS_SWEEP),
            "--method",
            method_flag,
            "--config",
            config_rel,
            "--data-root",
            str(dataset_path.parent),
            "--dataset-dir",
            dataset_path.name,
            "--n-trials",
            "1",
            "--seed",
            str(runner.seed),
            "--train-seed",
            str(runner.seed),
            "--sampler",
            "random",
            "--keep",
            "none",
            "--post-seeds",
            "0",
            "--output-csv",
            str(out_csv),
            "--logs-dir",
            str(trial_logs),
            "--base-lr-min",
            str(h["base_lr"]),
            "--base-lr-max",
            str(h["base_lr"]),
            "--cls-lr-min",
            str(h["classifier_lr"]),
            "--cls-lr-max",
            str(h["classifier_lr"]),
            "--run-name-prefix",
            f"recreate_rm_{method_name}",
        ]
        if "grad_weight" in h:
            cmd.extend(["--weight-min", str(h["grad_weight"]), "--weight-max", str(h["grad_weight"])])
        if "grad_criterion" in h:
            cmd.extend(["--grad-criteria", str(h["grad_criterion"])])
        if "abn_cls_weight" in h:
            cmd.extend(["--abn-cls-weight-min", str(h["abn_cls_weight"]), "--abn-cls-weight-max", str(h["abn_cls_weight"])])
        if "abn_att_weight" in h:
            cmd.extend(["--abn-att-weight-min", str(h["abn_att_weight"]), "--abn-att-weight-max", str(h["abn_att_weight"])])
        if "cam_weight" in h:
            cmd.extend(["--cam-weight-min", str(h["cam_weight"]), "--cam-weight-max", str(h["cam_weight"])])
        if overrides:
            cmd.extend(overrides)

        log_path = runner.run(method_name, cmd, cwd=GALS_ROOT)
        row = _read_first_csv_row(out_csv)
        rows.append(
            _summary_row(
                method_name,
                "ok",
                best_val=_float(row.get("best_balanced_val_acc")),
                test_acc=_float(row.get("test_acc") or row.get("balanced_test_acc")),
                per_group=_float(row.get("per_group")),
                worst_group=_float(row.get("worst_group")),
                log_path=log_path,
                artifacts={"csv": str(out_csv)},
            )
        )

    run_gals_method("vanilla", method_flag="upweight", config_rel=cfg["vanilla"], h=hp["vanilla"])
    run_gals_method("abn", method_flag="abn_cls", config_rel=cfg["abn"], h=hp["abn"])
    run_gals_method(
        "upweight",
        method_flag="upweight",
        config_rel=cfg["upweight"],
        h=hp["upweight"],
        overrides=["DATA.USE_CLASS_WEIGHTS=true"],
    )
    run_gals_method(
        "gals_rrr_rn50_maps",
        method_flag="gals",
        config_rel=cfg["gals_rn50"],
        h=hp["gals_rrr_rn50_maps"],
        overrides=[rn50_attention_override] if rn50_attention_override else None,
    )
    run_gals_method(
        "gals_rrr_vit_maps",
        method_flag="gals",
        config_rel=cfg["gals_vit"],
        h=hp["gals_rrr_vit_maps"],
        overrides=[vit_attention_override] if vit_attention_override else None,
    )
    run_gals_method(
        "gals_abn_vit_maps",
        method_flag="abn_att",
        config_rel=cfg["abn_vit"],
        h=hp["gals_abn_vit_maps"],
        overrides=[vit_attention_override] if vit_attention_override else None,
    )
    run_gals_method(
        "gals_gradcam_vit_maps",
        method_flag="gradcam",
        config_rel=cfg["gradcam_vit"],
        h=hp["gals_gradcam_vit_maps"],
        overrides=[vit_attention_override] if vit_attention_override else None,
    )

    if our_masks is None:
        rows.append(
            _summary_row(
                "gals_our_masks",
                "skipped",
                notes="Missing --gals-segmentation-dir or --weclip-prediction-cmap-dir",
            )
        )
    else:
        run_gals_method(
            "gals_our_masks",
            method_flag="gals",
            config_rel=cfg["gals_our_masks"],
            h=hp["gals_our_masks"],
            overrides=([f"DATA.SEGMENTATION_DIR={our_masks}"] + ([rn50_attention_override] if rn50_attention_override else [])),
        )

    # CLIP+LR
    clip_csv = run_root / "clip_lr" / "result.csv"
    clip_cmd = [
        runner.python_bin,
        str(RM_CLIP_LR_SWEEP),
        str(dataset_path),
        "--n-trials",
        "1",
        "--seed",
        str(runner.seed),
        "--sampler",
        "random",
        "--C-min",
        str(hp["clip_lr"]["c"]),
        "--C-max",
        str(hp["clip_lr"]["c"]),
        "--penalty-solvers",
        args.clip_lr_penalty_solvers,
        "--post-seeds",
        "0",
        "--output-csv",
        str(clip_csv),
    ]
    clip_log = runner.run("clip_lr", clip_cmd)
    clip_row = _read_first_csv_row(clip_csv)
    rows.append(
        _summary_row(
            "clip_lr",
            "ok",
            best_val=_float(clip_row.get("val_avg_group_acc") or clip_row.get("val_worst_group_acc")),
            test_acc=_float(clip_row.get("test_acc")),
            per_group=_float(clip_row.get("test_avg_group_acc")),
            worst_group=_float(clip_row.get("test_worst_group_acc")),
            log_path=clip_log,
            artifacts={"csv": str(clip_csv)},
        )
    )

    # CLIP zero-shot
    clip_zs_csv = run_root / "clip_zs" / "result.csv"
    clip_zs_cmd = [
        runner.python_bin,
        str(RM_CLIP_ZS),
        str(dataset_path),
        "--clip-model",
        args.clip_zs_model,
        "--batch-size",
        str(args.clip_zs_batch_size),
        "--num-workers",
        str(args.clip_zs_num_workers),
        "--seeds",
        str(runner.seed),
        "--output-csv",
        str(clip_zs_csv),
    ]
    if args.clip_zs_device:
        clip_zs_cmd.extend(["--device", args.clip_zs_device])
    clip_zs_log = runner.run("clip_zs", clip_zs_cmd)
    clip_zs_row = _read_first_csv_row(clip_zs_csv)
    rows.append(
        _summary_row(
            "clip_zs",
            "ok",
            best_val=_float(clip_zs_row.get("val_balanced_class_acc") or clip_zs_row.get("val_worst_class_acc") or clip_zs_row.get("val_acc")),
            test_acc=_float(clip_zs_row.get("test_acc")),
            per_group=_float(clip_zs_row.get("test_balanced_class_acc")),
            worst_group=_float(clip_zs_row.get("test_worst_class_acc")),
            log_path=clip_zs_log,
            artifacts={"csv": str(clip_zs_csv)},
        )
    )

    # AFR
    afr_out = run_root / "afr" / "out"
    afr_logs = run_root / "afr" / "logs"
    afr_cmd = [
        runner.python_bin,
        str(RM_AFR_SWEEP),
        "--data-dir",
        str(dataset_path),
        "--output-root",
        str(afr_out),
        "--logs-root",
        str(afr_logs),
        "--python-exe",
        runner.python_bin,
        "--seeds",
        str(runner.seed),
        "--gammas",
        str(hp["afr"]["gamma"]),
        "--reg-coeffs",
        str(hp["afr"]["reg"]),
        "--stage1-epochs",
        str(args.afr_stage1_epochs),
        "--stage2-epochs",
        str(args.afr_stage2_epochs),
    ]
    afr_log = runner.run("afr", afr_cmd)
    afr_best_csv = afr_out / "afr_redmeat_best_by_seed.csv"
    afr_row = _read_first_csv_row(afr_best_csv)
    rows.append(
        _summary_row(
            "afr",
            "ok",
            best_val=_float(afr_row.get("best_val_wga")),
            test_acc=_float(afr_row.get("best_test_at_val")),
            per_group=_float(afr_row.get("best_test_mean_at_val") or afr_row.get("best_test_mean")),
            worst_group=_float(afr_row.get("best_test_at_val")),
            log_path=afr_log,
            artifacts={"csv": str(afr_best_csv)},
        )
    )

    def run_r4rr(method_name: str, h: Dict[str, Any], maps: Optional[Path]) -> None:
        if maps is None:
            rows.append(_summary_row(method_name, "skipped", notes=f"Missing teacher-map path for {method_name}"))
            return
        out_csv = run_root / method_name / "result.csv"
        cmd = [
            runner.python_bin,
            str(R4RR_RM_SWEEP),
            str(dataset_path),
            str(maps),
            "--n-trials",
            "1",
            "--seed",
            str(runner.seed),
            "--sampler",
            "random",
            "--output-csv",
            str(out_csv),
            "--attn-min",
            str(h["attention_epoch"]),
            "--attn-max",
            str(h["attention_epoch"]),
            "--kl-min",
            str(h["kl_lambda"]),
            "--kl-max",
            str(h["kl_lambda"]),
            "--base-lr-min",
            str(h["base_lr"]),
            "--base-lr-max",
            str(h["base_lr"]),
            "--cls-lr-min",
            str(h["classifier_lr"]),
            "--cls-lr-max",
            str(h["classifier_lr"]),
            "--lr2-mult-min",
            str(h.get("lr2_mult", 1.0)),
            "--lr2-mult-max",
            str(h.get("lr2_mult", 1.0)),
        ]
        log_path = runner.run(method_name, cmd)
        row = _read_first_csv_row(out_csv)
        rows.append(
            _summary_row(
                method_name,
                "ok",
                best_val=_float(row.get("best_balanced_val_acc")),
                test_acc=_float(row.get("test_acc")),
                per_group=_float(row.get("per_group")),
                worst_group=_float(row.get("worst_group")),
                log_path=log_path,
                artifacts={"csv": str(out_csv)},
            )
        )

    run_r4rr("r4rr_gals_rn50_maps", hp["r4rr_gals_rn50_maps"], rn50_maps)
    run_r4rr("r4rr_gals_vit_maps", hp["r4rr_gals_vit_maps"], vit_maps)
    if "r4rr_optimized" in hp:
        run_r4rr("r4rr_optimized", hp["r4rr_optimized"], r4rr_opt_maps)

    _write_summary(rows, run_root)
    return rows


# ------------------------- DecoyMNIST -------------------------

def _decoy_run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfg_path = Path(args.config_path) if args.config_path else (CONFIG_DIR / "decoymnist_hparams.yaml")
    data = _load_yaml(cfg_path)
    hp = data["decoymnist_hparams"]

    png_root = Path(args.png_root).expanduser().resolve() if args.png_root else (
        REPO_ROOT / "repro_runs" / "third_party" / "CDEP" / "data" / "DecoyMNIST_png"
    )
    if not png_root.is_dir():
        raise FileNotFoundError(f"Missing PNG root: {png_root}")

    vit_teacher_maps = (
        Path(args.decoy_gals_vit_map_path).expanduser().resolve()
        if args.decoy_gals_vit_map_path
        else (REPO_ROOT / "repro_runs" / "third_party" / "CDEP" / "data" / "DecoyMNIST_png" / "clip_vit_attention")
    )
    r4rr_teacher_maps = (
        Path(args.decoy_r4rr_teacher_map_path).expanduser().resolve()
        if args.decoy_r4rr_teacher_map_path
        else (
            Path(args.weclip_prediction_cmap_dir).expanduser().resolve()
            if args.weclip_prediction_cmap_dir
            else _default_decoy_weclip_maps(REPO_ROOT)
        )
    )

    run_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        REPO_ROOT / "logs" / "recreate" / f"decoymnist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    runner = Runner(python_bin=args.python_bin, output_dir=run_root, seed=args.seed, dry_run=args.dry_run)

    epochs = int(hp.get("shared_training", {}).get("epochs", args.decoy_epochs))
    lr = float(hp.get("shared_training", {}).get("learning_rate", args.decoy_lr))

    rows: List[Dict[str, Any]] = []

    if not vit_teacher_maps.is_dir():
        raise FileNotFoundError(
            "Missing Decoy ViT attention maps directory. "
            f"Checked: {vit_teacher_maps}\n"
            "Pass --gals-vit-teacher-maps-dir."
        )

    def run_and_parse_stdout(method: str, cmd: List[str]) -> None:
        log_path = runner.run(method, cmd)
        stdout = (run_root / method / "stdout.log").read_text(encoding="utf-8", errors="ignore")
        m = _extract_decoy_stdout_metrics(stdout)
        rows.append(
            _summary_row(
                method,
                "ok",
                best_val=m.get("best_val"),
                test_acc=m.get("test_acc"),
                per_group=m.get("test_balanced_class_acc"),
                worst_group=m.get("test_worst_group"),
                log_path=log_path,
            )
        )

    # Vanilla via GALS runner with grad_weight=0.
    run_and_parse_stdout(
        "vanilla",
        [
            runner.python_bin,
            str(DEC_GALS),
            "--png-root",
            str(png_root),
            "--mask-root",
            str(vit_teacher_maps),
            "--loss-mode",
            "rrr",
            "--grad-weight",
            "0.0",
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--n-seeds",
            "1",
            "--seed-start",
            str(runner.seed),
            "--num-workers",
            str(args.num_workers),
            "--print-every",
            "1",
            "--no-progress-bar",
        ],
    )

    run_and_parse_stdout(
        "abn",
        [
            runner.python_bin,
            str(DEC_ABN),
            "--png-root",
            str(png_root),
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--abn-cls-weight",
            str(hp["abn"]["abn_cls_weight"]),
            "--n-seeds",
            "1",
            "--seed-start",
            str(runner.seed),
            "--num-workers",
            str(args.num_workers),
            "--print-every",
            "1",
        ],
    )

    run_and_parse_stdout(
        "upweight",
        [
            runner.python_bin,
            str(DEC_UPWEIGHT),
            "--png-root",
            str(png_root),
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--n-seeds",
            "1",
            "--seed-start",
            str(runner.seed),
            "--num-workers",
            str(args.num_workers),
            "--print-every",
            "1",
        ],
    )

    gals_rrr_cfg = hp.get("gals_rrr_vit_maps", hp.get("gals_rrr_rn50_maps"))
    if gals_rrr_cfg is None:
        rows.append(
            _summary_row(
                "gals_rrr_vit_maps",
                "skipped",
                notes="Missing gals_rrr_vit_maps (or legacy gals_rrr_rn50_maps) in decoymnist_hparams.yaml",
            )
        )
    elif not vit_teacher_maps.is_dir():
        rows.append(
            _summary_row(
                "gals_rrr_vit_maps",
                "skipped",
                notes=f"Missing --gals-vit-teacher-maps-dir: {vit_teacher_maps}",
            )
        )
    else:
        run_and_parse_stdout(
            "gals_rrr_vit_maps",
            [
                runner.python_bin,
                str(DEC_GALS),
                "--png-root",
                str(png_root),
                "--mask-root",
                str(vit_teacher_maps),
                "--loss-mode",
                "rrr",
                "--grad-weight",
                str(gals_rrr_cfg["grad_weight"]),
                "--grad-criterion",
                str(gals_rrr_cfg["grad_criterion"]),
                "--epochs",
                str(epochs),
                "--lr",
                str(lr),
                "--n-seeds",
                "1",
                "--seed-start",
                str(runner.seed),
                "--num-workers",
                str(args.num_workers),
                "--print-every",
                "1",
                "--no-progress-bar",
            ],
        )

    clip_csv = run_root / "clip_lr" / "result.csv"
    clip_cmd = [
        runner.python_bin,
        str(DEC_CLIP_LR),
        "--png-root",
        str(png_root),
        "--C",
        str(hp["clip_lr"]["c"]),
        "--seeds",
        str(runner.seed),
        "--output-csv",
        str(clip_csv),
    ]
    clip_log = runner.run("clip_lr", clip_cmd)
    clip_row = _read_first_csv_row(clip_csv)
    rows.append(
        _summary_row(
            "clip_lr",
            "ok",
            best_val=_float(
                clip_row.get("val_balanced_class_acc")
                or clip_row.get("val_avg_group_acc")
                or clip_row.get("val_worst_group_acc")
            ),
            test_acc=_float(clip_row.get("test_acc")),
            per_group=_float(clip_row.get("test_balanced_class_acc") or clip_row.get("test_avg_group_acc")),
            worst_group=_float(clip_row.get("test_worst_class_acc") or clip_row.get("test_worst_group_acc")),
            log_path=clip_log,
            artifacts={"csv": str(clip_csv)},
        )
    )

    clip_zs_csv = run_root / "clip_zs" / "result.csv"
    clip_zs_cmd = [
        runner.python_bin,
        str(DEC_CLIP_ZS),
        "--png-root",
        str(png_root),
        "--clip-model",
        args.clip_zs_model,
        "--batch-size",
        str(args.clip_zs_batch_size),
        "--num-workers",
        str(args.clip_zs_num_workers),
        "--seeds",
        str(runner.seed),
        "--output-csv",
        str(clip_zs_csv),
    ]
    if args.clip_zs_device:
        clip_zs_cmd.extend(["--device", args.clip_zs_device])
    clip_zs_log = runner.run("clip_zs", clip_zs_cmd)
    clip_zs_row = _read_first_csv_row(clip_zs_csv)
    rows.append(
        _summary_row(
            "clip_zs",
            "ok",
            test_acc=_float(clip_zs_row.get("test_acc")),
            per_group=_float(clip_zs_row.get("test_balanced_class_acc")),
            worst_group=_float(clip_zs_row.get("test_worst_class_acc")),
            log_path=clip_zs_log,
            artifacts={"csv": str(clip_zs_csv)},
        )
    )

    afr_csv = run_root / "afr" / "result.csv"
    afr_cmd = [
        runner.python_bin,
        str(DEC_AFR),
        "--png-root",
        str(png_root),
        "--seeds",
        str(runner.seed),
        "--stage1-epochs",
        str(epochs),
        "--stage2-epochs",
        str(args.afr_stage2_epochs),
        "--gamma",
        str(hp["afr"]["gamma"]),
        "--reg-coeff",
        str(hp["afr"]["reg"]),
        "--output-csv",
        str(afr_csv),
        "--num-workers",
        str(args.num_workers),
    ]
    afr_log = runner.run("afr", afr_cmd)
    afr_row = _read_first_csv_row(afr_csv)
    rows.append(
        _summary_row(
            "afr",
            "ok",
            best_val=_float(afr_row.get("val_worst_class_acc") or afr_row.get("best_val_wga")),
            test_acc=_float(afr_row.get("test_acc_at_val_worst") or afr_row.get("best_test_at_val")),
            per_group=_float(afr_row.get("test_balanced_acc_at_val_worst")),
            worst_group=_float(afr_row.get("test_worst_class_acc_at_val_worst")),
            log_path=afr_log,
            artifacts={"csv": str(afr_csv)},
        )
    )

    if r4rr_teacher_maps is None or not r4rr_teacher_maps.is_dir():
        rows.append(
            _summary_row(
                "r4rr_optimized",
                "skipped",
                notes=(
                    "Missing Decoy R4RR teacher maps. Pass --decoy-r4rr-teacher-maps-dir "
                    "(or --weclip-prediction-cmap-dir)."
                ),
            )
        )
    else:
        run_and_parse_stdout(
            "r4rr_optimized",
            [
                runner.python_bin,
                str(DEC_R4RR),
                "--png-root",
                str(png_root),
                "--teacher-map-path",
                str(r4rr_teacher_maps),
                "--epochs",
                str(epochs),
                "--lr",
                str(lr),
                "--attention-epoch",
                str(hp["r4rr_optimized"]["attention_epoch"]),
                "--kl-lambda",
                str(hp["r4rr_optimized"]["kl_lambda"]),
                "--kl-incr",
                "0.0",
                "--n-seeds",
                "1",
                "--seed-start",
                str(runner.seed),
                "--num-workers",
                str(args.num_workers),
                "--print-every",
                "1",
            ],
        )

    _write_summary(rows, run_root)
    return rows


def _print_rows(rows: List[Dict[str, Any]]) -> None:
    print("method,status,best_val,test_acc,per_group,worst_group")
    for r in rows:
        best_val = "" if r["best_val"] is None else f"{r['best_val']:.4f}"
        test_acc = "" if r["test_acc"] is None else f"{r['test_acc']:.4f}"
        per_group = "" if r["per_group"] is None else f"{r['per_group']:.4f}"
        worst_group = "" if r["worst_group"] is None else f"{r['worst_group']:.4f}"
        print(
            f"{r['method']},{r['status']},"
            f"{best_val},"
            f"{test_acc},"
            f"{per_group},"
            f"{worst_group}"
        )


def build_parser(default_dataset: Optional[str] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recreate one-pass runs from optimized hparam YAMLs")
    p.add_argument("--dataset", choices=["waterbirds95", "waterbirds100", "redmeat", "decoymnist"], default=default_dataset)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="")
    p.add_argument("--config-path", default="")
    p.add_argument("--dry-run", action="store_true")

    # Waterbirds options
    p.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    p.add_argument("--dataset-dir", default="")
    p.add_argument(
        "--gals-segmentation-dir",
        "--gals-our-masks-path",
        dest="gals_our_masks_path",
        default="",
        help=(
            "Directory of segmentation masks for `gals_our_masks` runs "
            "(typically WeCLIP+ prediction_cmap output). If omitted, "
            "falls back to --weclip-prediction-cmap-dir."
        ),
    )
    p.add_argument(
        "--r4rr-rn50-teacher-maps-dir",
        "--r4rr-rn50-map-path",
        dest="r4rr_rn50_map_path",
        default="",
        help=(
            "Teacher map directory for `r4rr_gals_rn50_maps` "
            "(typically GALS RN50 attention maps)."
        ),
    )
    p.add_argument(
        "--r4rr-vit-teacher-maps-dir",
        "--r4rr-vit-map-path",
        dest="r4rr_vit_map_path",
        default="",
        help=(
            "Teacher map directory for `r4rr_gals_vit_maps` "
            "(typically GALS ViT attention maps)."
        ),
    )
    p.add_argument(
        "--weclip-prediction-cmap-dir",
        default="",
        help=(
            "Shared WeCLIP+ `prediction_cmap` directory. "
            "When set, this is used by default for `gals_our_masks`, "
            "`r4rr_optimized`, Waterbirds R4RR ablations, and Decoy R4RR unless "
            "those specific flags are explicitly provided."
        ),
    )
    p.add_argument(
        "--r4rr-optimized-teacher-maps-dir",
        "--r4rr-optimized-map-path",
        dest="r4rr_optimized_map_path",
        default="",
        help=(
            "Teacher map directory for `r4rr_optimized` "
            "(typically WeCLIP+ `prediction_cmap`). "
            "If omitted, falls back to --weclip-prediction-cmap-dir, then "
            "--r4rr-rn50-teacher-maps-dir."
        ),
    )
    p.add_argument(
        "--r4rr-ablation-teacher-maps-dir",
        "--r4rr-ablation-map-path",
        dest="r4rr_ablation_map_path",
        default="",
        help=(
            "Teacher map directory for Waterbirds R4RR ablations "
            "(`r4rr_invert`, `r4rr_joint`). "
            "If omitted, falls back to --weclip-prediction-cmap-dir, then "
            "--r4rr-optimized-teacher-maps-dir."
        ),
    )
    p.add_argument("--clip-lr-penalty-solvers", default="l2:lbfgs")
    p.add_argument("--clip-lr-feature-modes", default="l2")
    p.add_argument("--clip-lr-class-weight-options", default="balanced")
    p.add_argument("--clip-zs-model", default="RN50")
    p.add_argument(
        "--clip-zs-device",
        default="",
        help='Optional torch device override for clip_zs (e.g. "cuda" or "cpu"). Default: use each script\'s own default.',
    )
    p.add_argument("--clip-zs-batch-size", type=int, default=256)
    p.add_argument("--clip-zs-num-workers", type=int, default=4)
    p.add_argument("--afr-stage1-epochs", type=int, default=50)
    p.add_argument("--afr-stage2-epochs", type=int, default=500)

    # RedMeat options
    p.add_argument("--dataset-path", default="")
    p.add_argument("--cfg-vanilla", default="configs/food_vanilla.yaml")
    p.add_argument("--cfg-abn", default="configs/food_abn.yaml")
    p.add_argument("--cfg-upweight", default="configs/food_vanilla.yaml")
    p.add_argument("--cfg-gals-rn50", default="configs/food_gals.yaml")
    p.add_argument("--cfg-gals-vit", default="configs/food_gals.yaml")
    p.add_argument("--cfg-abn-vit", default="configs/food_abn.yaml")
    p.add_argument("--cfg-gradcam-vit", default="configs/food_gals.yaml")
    p.add_argument("--cfg-gals-our-masks", default="configs/food_gals.yaml")

    # Decoy options
    p.add_argument("--png-root", default="")
    p.add_argument(
        "--gals-vit-teacher-maps-dir",
        "--gals-rn50-teacher-maps-dir",
        dest="decoy_gals_vit_map_path",
        default="",
        help=(
            "ViT attention map directory for Decoy GALS map-based runs "
            "(expects .pth maps, e.g. clip_vit_attention). "
            "Legacy alias --gals-rn50-teacher-maps-dir is still accepted."
        ),
    )
    p.add_argument(
        "--decoy-r4rr-teacher-maps-dir",
        "--decoy-weclip-prediction-cmap-dir",
        "--r4rr-teacher-maps-dir",
        "--teacher-map-path",
        dest="decoy_r4rr_teacher_map_path",
        default="",
        help=(
            "Teacher map directory for Decoy R4RR "
            "(typically WeCLIP+ prediction_cmap image masks). "
            "If omitted, falls back to --weclip-prediction-cmap-dir."
        ),
    )
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--decoy-epochs", type=int, default=19)
    p.add_argument("--decoy-lr", type=float, default=0.001)

    return p


def main(default_dataset: Optional[str] = None) -> int:
    parser = build_parser(default_dataset=default_dataset)
    args = parser.parse_args()

    if not args.dataset:
        parser.error("--dataset is required")

    if args.dataset == "waterbirds95":
        rows = _wb_run("95", args)
    elif args.dataset == "waterbirds100":
        rows = _wb_run("100", args)
    elif args.dataset == "redmeat":
        if not args.dataset_path:
            parser.error("--dataset-path is required for redmeat")
        rows = _rm_run(args)
    elif args.dataset == "decoymnist":
        rows = _decoy_run(args)
    else:
        raise RuntimeError(f"Unsupported dataset: {args.dataset}")

    _print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
