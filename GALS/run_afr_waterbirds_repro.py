#!/usr/bin/env python3
"""Two-stage AFR Waterbirds reproduction runner.

This script orchestrates:
1) Stage-1 ERM training via afr/train_supervised.py
2) Stage-2 AFR last-layer retraining sweeps via afr/train_embeddings.py

Selection rule follows the AFR paper:
- choose hyperparameters by best validation worst-group accuracy
- report test worst-group accuracy at that validation-selected epoch
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _parse_float_list(text: str) -> List[float]:
    vals = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        vals.append(float(piece))
    return vals


def _parse_int_list(text: str) -> List[int]:
    vals = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        vals.append(int(piece))
    return vals


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _float_or_nan(x: Optional[float]) -> float:
    return float("nan") if x is None else float(x)


def _metric_max(metrics: Dict[int, Dict[str, float]], key: str) -> Optional[float]:
    best = None
    for _, row in metrics.items():
        if key not in row:
            continue
        val = row[key]
        if val is None:
            continue
        val = float(val)
        if not math.isfinite(val):
            continue
        if best is None or val > best:
            best = val
    return best


def _row_value(row: Dict[str, float], key: str) -> Optional[float]:
    if key not in row:
        return None
    val = row[key]
    if val is None:
        return None
    try:
        out = float(val)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _argmax_epoch(metrics: Dict[int, Dict[str, float]], key: str) -> Optional[int]:
    best_epoch: Optional[int] = None
    best_val: Optional[float] = None
    for epoch, row in metrics.items():
        v = _row_value(row, key)
        if v is None:
            continue
        if best_val is None or v > best_val:
            best_val = v
            best_epoch = int(epoch)
    return best_epoch


def _load_metrics(metrics_path: Path) -> Dict[int, Dict[str, float]]:
    with metrics_path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected metrics format: {metrics_path}")
    return obj


def _extract_stage2_stats(metrics_path: Path) -> Dict[str, float]:
    m = _load_metrics(metrics_path)
    best_val_wga = _metric_max(m, "best_val_wga")
    best_test_at_val = _metric_max(m, "best_test_at_val")
    best_test_wga = _metric_max(m, "best_test_wga")
    best_val_mean = _metric_max(m, "best_val_mean")
    best_test_mean = _metric_max(m, "best_test_mean")
    selected_epoch = _argmax_epoch(m, "val_worst_accuracy")
    selected_row = m[selected_epoch] if selected_epoch is not None else {}

    out = {
        "best_val_wga": _float_or_nan(best_val_wga),
        "best_test_at_val": _float_or_nan(best_test_at_val),
        "best_test_wga": _float_or_nan(best_test_wga),
        "best_val_mean": _float_or_nan(best_val_mean),
        "best_test_mean": _float_or_nan(best_test_mean),
        "selected_epoch": float("nan") if selected_epoch is None else float(selected_epoch),
    }

    # Group-wise accuracies at the checkpoint selected by validation WGA.
    for split in ("val", "test"):
        for group in ("0_0", "1_0", "2_0", "3_0"):
            key = f"{split}_accuracy_{group}"
            out[f"{split}_acc_{group}"] = _float_or_nan(_row_value(selected_row, key))

    return out


def _run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> None:
    _ensure_dir(log_path.parent)
    # AFR scripts (notably train_embeddings.py) save auxiliary plots to ./logs/.
    # Ensure it exists in the command working directory to avoid late-stage crashes.
    _ensure_dir(cwd / "logs")
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("[CMD] " + " ".join(cmd) + "\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            print(line, end="")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}. See {log_path}")


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], header: List[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _maybe_skip(existing: Path, force: bool) -> bool:
    return existing.exists() and not force


def _format_gamma(gamma: float) -> str:
    return f"{gamma:.6g}".replace(".", "p")


def _format_reg(reg: float) -> str:
    return f"{reg:.6g}".replace(".", "p")


def run(args: argparse.Namespace) -> None:
    afr_root = Path(args.afr_root).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    logs_root = Path(args.logs_root).expanduser().resolve()
    python_exe = args.python_exe

    if not afr_root.is_dir():
        raise FileNotFoundError(f"Missing AFR root: {afr_root}")
    if not (afr_root / "train_supervised.py").exists():
        raise FileNotFoundError(f"AFR script missing: {afr_root / 'train_supervised.py'}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")
    if not (data_dir / "metadata.csv").exists():
        raise FileNotFoundError(
            f"Missing metadata.csv in {data_dir}. AFR expects official Waterbirds metadata format."
        )

    stage1_root = output_root / "stage1"
    stage2_root = output_root / "stage2"
    run_logs = logs_root / f"afr_waterbirds_repro_{time.strftime('%Y%m%d_%H%M%S')}"
    _ensure_dir(stage1_root)
    _ensure_dir(stage2_root)
    _ensure_dir(run_logs)

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds is empty")

    if args.full_paper_grid:
        gammas = [4.0 + 0.5 * i for i in range(33)]
        regs = [0.0, 0.1, 0.2, 0.3, 0.4]
    else:
        gammas = _parse_float_list(args.gammas)
        regs = _parse_float_list(args.reg_coeffs)

    if not gammas:
        raise ValueError("No gammas configured")
    if not regs:
        raise ValueError("No reg_coeffs configured")

    print(f"[INFO] AFR_ROOT={afr_root}")
    print(f"[INFO] DATA_DIR={data_dir}")
    print(f"[INFO] OUTPUT_ROOT={output_root}")
    print(f"[INFO] LOGS_ROOT={run_logs}")
    print(f"[INFO] Seeds={seeds}")
    print(f"[INFO] Stage1 scheduler={args.stage1_scheduler}")
    print(f"[INFO] Stage2 gammas ({len(gammas)}): {gammas[:5]}{' ...' if len(gammas) > 5 else ''}")
    print(f"[INFO] Stage2 reg_coeffs ({len(regs)}): {regs}")

    stage2_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []

    for seed in seeds:
        print(f"\n===== [SEED {seed}] Stage 1 =====")
        s1_dir = stage1_root / f"seed_{seed}"
        s1_ckpt = s1_dir / "final_checkpoint.pt"
        s1_log = run_logs / f"seed{seed}_stage1.log"

        if _maybe_skip(s1_ckpt, args.force_stage1):
            print(f"[SKIP] Stage-1 checkpoint exists: {s1_ckpt}")
        else:
            s1_cmd = [
                python_exe,
                "train_supervised.py",
                f"--output_dir={s1_dir}",
                "--project=afr_waterbirds_repro",
                f"--seed={seed}",
                f"--eval_freq={args.stage1_eval_freq}",
                f"--save_freq={args.stage1_save_freq}",
                f"--data_dir={data_dir}",
                "--dataset=SpuriousDataset",
                "--data_transform=AugWaterbirdsCelebATransform",
                "--model=imagenet_resnet50_pretrained",
                "--max_prop=1.0",
                "--train_prop=80",
                f"--num_epochs={args.stage1_epochs}",
                "--batch_size=32",
                "--optimizer=sgd_optimizer",
                f"--scheduler={args.stage1_scheduler}",
                "--init_lr=0.003",
                "--weight_decay=1e-4",
            ]
            _run_cmd(s1_cmd, cwd=afr_root, log_path=s1_log)

        if not s1_ckpt.exists():
            raise RuntimeError(f"Stage-1 checkpoint not found after training: {s1_ckpt}")

        seed_best: Optional[Dict[str, object]] = None

        print(f"===== [SEED {seed}] Stage 2 Sweep ({len(gammas) * len(regs)} runs) =====")
        for idx, (gamma, reg) in enumerate(itertools.product(gammas, regs), start=1):
            run_name = f"seed{seed}_g{_format_gamma(gamma)}_r{_format_reg(reg)}"
            s2_dir = stage2_root / f"seed_{seed}" / run_name
            s2_metrics = s2_dir / "metrics.pkl"
            s2_log = run_logs / f"{run_name}.log"

            print(
                f"[RUN] seed={seed} {idx}/{len(gammas)*len(regs)} gamma={gamma} reg={reg} -> {s2_dir}",
                flush=True,
            )

            if not _maybe_skip(s2_metrics, args.force_stage2):
                s2_cmd = [
                    python_exe,
                    "train_embeddings.py",
                    f"--output_dir={s2_dir}",
                    "--project=afr_waterbirds_repro",
                    f"--seed={seed}",
                    f"--base_model_dir={s1_dir}",
                    "--checkpoint=final_checkpoint.pt",
                    "--model=imagenet_resnet50_pretrained",
                    f"--data_dir={data_dir}",
                    "--dataset=SpuriousDataset",
                    "--data_transform=NoAugWaterbirdsCelebATransform",
                    "--num_augs=1",
                    f"--num_epochs={args.stage2_epochs}",
                    "--batch_size=128",
                    "--emb_batch_size=-1",
                    "--optimizer=sgd_optimizer",
                    "--scheduler=constant_lr_scheduler",
                    f"--init_lr={args.stage2_lr}",
                    "--momentum=0.0",
                    "--weight_decay=0.0",
                    "--grad_norm=1.0",
                    "--loss=afr",
                    "--tune_on=train",
                    "--train_prop=-20",
                    f"--gamma={gamma}",
                    f"--reg_coeff={reg}",
                ]
                _run_cmd(s2_cmd, cwd=afr_root, log_path=s2_log)
            else:
                print(f"[SKIP] Stage-2 metrics exist: {s2_metrics}")

            if not s2_metrics.exists():
                raise RuntimeError(f"Missing stage-2 metrics after run: {s2_metrics}")

            stats = _extract_stage2_stats(s2_metrics)
            row: Dict[str, object] = {
                "seed": seed,
                "gamma": gamma,
                "reg_coeff": reg,
                "stage1_dir": str(s1_dir),
                "stage2_dir": str(s2_dir),
                "metrics_path": str(s2_metrics),
                **stats,
            }
            stage2_rows.append(row)

            if seed_best is None:
                seed_best = row
            else:
                best_val = float(seed_best["best_val_wga"])
                cur_val = float(row["best_val_wga"])
                if cur_val > best_val:
                    seed_best = row

        assert seed_best is not None
        best_rows.append(seed_best)
        print(
            "[BEST] "
            f"seed={seed} gamma={seed_best['gamma']} reg={seed_best['reg_coeff']} "
            f"best_val_wga={float(seed_best['best_val_wga']):.4f} "
            f"best_test_at_val={float(seed_best['best_test_at_val']):.4f} "
            f"(test groups: "
            f"{float(seed_best['test_acc_0_0']):.4f}, "
            f"{float(seed_best['test_acc_1_0']):.4f}, "
            f"{float(seed_best['test_acc_2_0']):.4f}, "
            f"{float(seed_best['test_acc_3_0']):.4f})"
        )

    stage2_csv = output_root / "afr_waterbirds_stage2_all.csv"
    best_csv = output_root / "afr_waterbirds_best_by_seed.csv"
    _write_csv(
        stage2_csv,
        stage2_rows,
        [
            "seed",
            "gamma",
            "reg_coeff",
            "best_val_wga",
            "best_test_at_val",
            "best_test_wga",
            "best_val_mean",
            "best_test_mean",
            "selected_epoch",
            "val_acc_0_0",
            "val_acc_1_0",
            "val_acc_2_0",
            "val_acc_3_0",
            "test_acc_0_0",
            "test_acc_1_0",
            "test_acc_2_0",
            "test_acc_3_0",
            "stage1_dir",
            "stage2_dir",
            "metrics_path",
        ],
    )
    _write_csv(
        best_csv,
        best_rows,
        [
            "seed",
            "gamma",
            "reg_coeff",
            "best_val_wga",
            "best_test_at_val",
            "best_test_wga",
            "best_val_mean",
            "best_test_mean",
            "selected_epoch",
            "val_acc_0_0",
            "val_acc_1_0",
            "val_acc_2_0",
            "val_acc_3_0",
            "test_acc_0_0",
            "test_acc_1_0",
            "test_acc_2_0",
            "test_acc_3_0",
            "stage1_dir",
            "stage2_dir",
            "metrics_path",
        ],
    )

    best_test_vals = [float(r["best_test_at_val"]) for r in best_rows if math.isfinite(float(r["best_test_at_val"]))]
    if best_test_vals:
        mean_best = statistics.mean(best_test_vals)
        std_best = statistics.pstdev(best_test_vals) if len(best_test_vals) > 1 else 0.0
    else:
        mean_best = float("nan")
        std_best = float("nan")

    group_stats: Dict[str, Tuple[float, float]] = {}
    for group in ("0_0", "1_0", "2_0", "3_0"):
        key = f"test_acc_{group}"
        vals = [float(r[key]) for r in best_rows if key in r and math.isfinite(float(r[key]))]
        if not vals:
            group_stats[group] = (float("nan"), float("nan"))
            continue
        g_mean = statistics.mean(vals)
        g_std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        group_stats[group] = (g_mean, g_std)

    summary_txt = output_root / "afr_waterbirds_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"AFR Waterbirds reproduction summary\n")
        f.write(f"AFR_ROOT: {afr_root}\n")
        f.write(f"DATA_DIR: {data_dir}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Stage1 epochs: {args.stage1_epochs}\n")
        f.write(f"Stage1 scheduler: {args.stage1_scheduler}\n")
        f.write(f"Stage2 epochs: {args.stage2_epochs}\n")
        f.write(f"Gammas: {gammas}\n")
        f.write(f"Reg coeffs: {regs}\n")
        f.write(f"Mean best test@val WGA: {mean_best:.4f}\n")
        f.write(f"Std best test@val WGA: {std_best:.4f}\n")
        for group in ("0_0", "1_0", "2_0", "3_0"):
            g_mean, g_std = group_stats[group]
            f.write(f"Mean test@val group {group}: {g_mean:.4f} +/- {g_std:.4f}\n")
        f.write(f"Best-per-seed CSV: {best_csv}\n")
        f.write(f"All stage2 CSV: {stage2_csv}\n")

    print("\n===== [DONE] =====")
    print(f"All stage-2 results: {stage2_csv}")
    print(f"Best by seed:        {best_csv}")
    print(f"Summary:             {summary_txt}")
    print(f"Mean best test@val WGA across seeds: {mean_best:.4f} +/- {std_best:.4f}")
    for group in ("0_0", "1_0", "2_0", "3_0"):
        g_mean, g_std = group_stats[group]
        print(f"Mean test@val group {group}: {g_mean:.4f} +/- {g_std:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AFR Waterbirds reproduction runner")
    parser.add_argument(
        "--afr-root",
        default=str((Path(__file__).resolve().parent.parent / "afr").resolve()),
        help="Path to AFR repository root",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2",
        help="Waterbirds dataset directory (expects metadata.csv)",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ryreu/guided_cnn/logsWaterbird/afr_repro",
        help="Directory where AFR run outputs and CSVs are stored",
    )
    parser.add_argument(
        "--logs-root",
        default="/home/ryreu/guided_cnn/logsWaterbird/afr_repro_logs",
        help="Directory for command logs",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to launch AFR scripts",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated list of seeds for stage1+stage2",
    )

    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage1-eval-freq", type=int, default=10)
    parser.add_argument("--stage1-save-freq", type=int, default=10)
    parser.add_argument(
        "--stage1-scheduler",
        default="constant_lr_scheduler",
        choices=["constant_lr_scheduler", "cosine_lr_scheduler"],
        help="Stage-1 LR scheduler; paper Waterbirds setup uses constant_lr_scheduler.",
    )
    parser.add_argument("--stage2-epochs", type=int, default=500)
    parser.add_argument("--stage2-lr", type=float, default=1e-2)

    parser.add_argument(
        "--full-paper-grid",
        action="store_true",
        help="Use paper Waterbirds grid: gamma in linspace(4,20,33), reg in {0,0.1,0.2,0.3,0.4}",
    )
    parser.add_argument(
        "--gammas",
        default="4,6,8,10,12,14,16,18,20",
        help="Comma-separated gamma values (ignored if --full-paper-grid)",
    )
    parser.add_argument(
        "--reg-coeffs",
        default="0,0.1,0.2,0.3,0.4",
        help="Comma-separated reg_coeff values (ignored if --full-paper-grid)",
    )

    parser.add_argument(
        "--force-stage1",
        action="store_true",
        help="Re-run stage1 even if final_checkpoint.pt already exists",
    )
    parser.add_argument(
        "--force-stage2",
        action="store_true",
        help="Re-run stage2 configs even if metrics.pkl already exists",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
