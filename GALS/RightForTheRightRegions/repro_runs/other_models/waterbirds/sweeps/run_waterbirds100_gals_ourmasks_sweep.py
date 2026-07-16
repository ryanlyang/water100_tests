#!/usr/bin/env python3
"""Run Waterbirds-100 GALS our-masks sweep without Slurm/sbatch."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_gals_root() -> Path:
    return _repo_root() / "repro_runs" / "third_party" / "GALS"


def _default_data_root() -> Path:
    return _repo_root() / "data"


def _default_logs_dir() -> Path:
    return _repo_root() / "logs" / "waterbirds"


def _default_job_tag() -> str:
    return os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")


def _strip_quotes(text: str) -> str:
    return str(text).strip().strip('"').strip("'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--sweep-script", default=str(Path(__file__).resolve().parent / "gals_waterbirds_sweep.py"))
    parser.add_argument("--gals-root", default=str(_default_gals_root()))
    parser.add_argument("--data-root", default=str(_default_data_root()))
    parser.add_argument("--waterbirds-dir", default="waterbird_1.0_forest2water2")
    parser.add_argument("--config", default="configs/waterbirds_100_gals_ourmasks.yaml")
    parser.add_argument("--mask-dir", required=True, help="Teacher/attention map directory")

    parser.add_argument("--post-mask-dirs", default="", help="Comma-separated extra mask dirs")
    parser.add_argument("--post-mask-labels", default="", help="Comma-separated labels for --post-mask-dirs")

    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--sweep-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--keep", choices=["best", "all", "none"], default="best")
    parser.add_argument("--max-hours", type=float, default=None)

    parser.add_argument("--tune-weight-decay", action="store_true")
    parser.add_argument("--base-lr-min", type=float, default=1e-5)
    parser.add_argument("--base-lr-max", type=float, default=5e-2)
    parser.add_argument("--cls-lr-min", type=float, default=1e-5)
    parser.add_argument("--cls-lr-max", type=float, default=5e-2)

    parser.add_argument("--post-seeds", type=int, default=5)
    parser.add_argument("--post-seed-start", type=int, default=0)
    parser.add_argument("--post-keep", choices=["all", "none"], default="all")

    parser.add_argument("--logs-dir", default=str(_default_logs_dir()))
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--trial-logs", default="")

    parser.add_argument("--install-optuna", dest="install_optuna", action="store_true")
    parser.add_argument("--no-install-optuna", dest="install_optuna", action="store_false")
    parser.set_defaults(install_optuna=True)

    parser.add_argument("--override", action="append", default=[], help="Extra override, e.g. EXP.NUM_EPOCHS=200")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gals_root = Path(args.gals_root).expanduser().resolve()
    sweep_script = Path(args.sweep_script).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    dataset_root = data_root / args.waterbirds_dir
    logs_dir = Path(args.logs_dir).expanduser().resolve()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (gals_root / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")

    mask_dir = Path(_strip_quotes(args.mask_dir)).expanduser().resolve()
    if not sweep_script.is_file():
        raise FileNotFoundError(f"Missing sweep script: {sweep_script}")
    if not gals_root.is_dir():
        raise FileNotFoundError(f"Missing GALS root: {gals_root}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset dir: {dataset_root}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

    post_dirs = [Path(_strip_quotes(x)).expanduser().resolve() for x in args.post_mask_dirs.split(",") if x.strip()]
    for p in post_dirs:
        if not p.is_dir():
            raise FileNotFoundError(f"Missing post mask dir: {p}")

    logs_dir.mkdir(parents=True, exist_ok=True)
    job_tag = _default_job_tag()
    out_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else logs_dir / f"gals100_ourmasks_sweep_{job_tag}.csv"
    trial_logs = Path(args.trial_logs).expanduser().resolve() if args.trial_logs else logs_dir / f"gals100_ourmasks_sweep_logs_{job_tag}"

    if args.install_optuna:
        try:
            __import__("optuna")
        except Exception:
            print("[INFO] Installing optuna...")
            subprocess.run([args.python_bin, "-m", "pip", "install", "-q", "optuna"], check=True)

    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{gals_root}:{py_path}" if py_path else str(gals_root)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cmd = [
        args.python_bin,
        "-u",
        str(sweep_script),
        "--config",
        str(config_path),
        "--data-root",
        str(data_root),
        "--waterbirds-dir",
        args.waterbirds_dir,
        "--n-trials",
        str(args.n_trials),
        "--seed",
        str(args.sweep_seed),
        "--train-seed",
        str(args.train_seed),
        "--sampler",
        args.sampler,
        "--keep",
        args.keep,
        "--output-csv",
        str(out_csv),
        "--logs-dir",
        str(trial_logs),
        "--base-lr-min",
        str(args.base_lr_min),
        "--base-lr-max",
        str(args.base_lr_max),
        "--cls-lr-min",
        str(args.cls_lr_min),
        "--cls-lr-max",
        str(args.cls_lr_max),
        "--post-seeds",
        str(args.post_seeds),
        "--post-seed-start",
        str(args.post_seed_start),
        "--post-keep",
        args.post_keep,
    ]

    if args.post_mask_dirs:
        cmd.extend(["--post-segmentation-dirs", args.post_mask_dirs])
    if args.post_mask_labels:
        cmd.extend(["--post-segmentation-labels", args.post_mask_labels])
    if args.tune_weight_decay:
        cmd.append("--tune-weight-decay")
    if args.max_hours is not None:
        cmd.extend(["--max-hours", str(args.max_hours)])

    cmd.append(f"DATA.SEGMENTATION_DIR={mask_dir}")
    cmd.extend(args.override)

    print(f"[INFO] gals_root={gals_root}")
    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] waterbirds_dir={args.waterbirds_dir}")
    print(f"[INFO] config={config_path}")
    print(f"[INFO] mask_dir={mask_dir}")
    print(f"[RUN] {' '.join(cmd)}")

    if not args.dry_run:
        subprocess.run(cmd, cwd=gals_root, env=env, check=True)


if __name__ == "__main__":
    main()
