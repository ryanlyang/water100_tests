#!/usr/bin/env python3
"""Run one R4RR training run per dataset using configs/r4rr_optimized_hparams.yaml.

Datasets covered:
- waterbirds95
- waterbirds100
- redmeat
- decoymnist

Notes:
- Waterbirds/RedMeat run the standard R4RR ResNet50 training scripts.
- DecoyMNIST runs the CDEP-style R4RR script (LeNet-style backbone), i.e.
  a different CNN setup by design.
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
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install with `pip install pyyaml`. "
        f"Import error: {exc}"
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
R4RR_TRAIN = REPO_ROOT / "repro_runs" / "r4rr" / "train"


def _default_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "logs" / "recreate" / f"r4rr_all_{ts}"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing YAML: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _run_cmd(
    cmd: List[str],
    *,
    env: Dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_str = " ".join(cmd)
    if dry_run:
        log_path.write_text(f"[DRY RUN]\n{cmd_str}\n", encoding="utf-8")
        return f"[DRY RUN] {cmd_str}\n"

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    stdout = proc.stdout or ""
    log_path.write_text(stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {cmd_str}\n"
            f"See log: {log_path}"
        )
    return stdout


def _parse_wb_redmeat_stdout(stdout: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "best_val": None,
        "test_acc": None,
        "per_group": None,
        "worst_group": None,
    }

    m_best = re.search(r"best_balanced_val_acc=([0-9.]+)", stdout)
    if m_best:
        out["best_val"] = float(m_best.group(1))

    m_test = re.search(r"test_acc=([0-9.]+)%", stdout)
    if m_test:
        out["test_acc"] = float(m_test.group(1))

    m_groups = re.search(r"Per Group:\s*([0-9.]+)%\s+Worst Group:\s*([0-9.]+)%", stdout)
    if m_groups:
        out["per_group"] = float(m_groups.group(1))
        out["worst_group"] = float(m_groups.group(2))

    return out


def _parse_decoy_stdout(stdout: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "best_val": None,
        "test_acc": None,
        "per_group": None,
        "worst_group": None,
    }

    m_best = re.search(r"best_val_acc\s+mean=([0-9.]+)%", stdout)
    if m_best:
        out["best_val"] = float(m_best.group(1))

    m_test = re.search(r"test_acc\s+mean=([0-9.]+)%", stdout)
    if m_test:
        out["test_acc"] = float(m_test.group(1))

    m_bal = re.search(r"test_balanced_class_acc\s+mean=([0-9.]+)%", stdout)
    if m_bal:
        out["per_group"] = float(m_bal.group(1))

    m_worst = re.search(r"test_worst_class_acc\s+mean=([0-9.]+)%", stdout)
    if m_worst:
        out["worst_group"] = float(m_worst.group(1))

    return out


def _summary_header() -> List[str]:
    return [
        "dataset",
        "status",
        "best_val",
        "test_acc",
        "per_group",
        "worst_group",
        "notes",
        "log_path",
        "command",
    ]


def _write_summary(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_summary_header())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _norm(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()


def _default_decoy_maps(repo_root: Path) -> Path:
    train_only = repo_root / "WeCLIPPlus" / "results_decoy_r4rr_trainonly" / "val" / "prediction_cmap"
    legacy = repo_root / "WeCLIPPlus" / "results_decoy_r4rr" / "val" / "prediction_cmap"
    if train_only.is_dir():
        return train_only
    return legacy


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run one R4RR training run per dataset using "
            "configs/r4rr_optimized_hparams.yaml."
        )
    )
    p.add_argument("--python-bin", default=sys.executable, help="Python executable for launched runs.")
    p.add_argument(
        "--config-path",
        default=str(REPO_ROOT / "configs" / "r4rr_optimized_hparams.yaml"),
        help="Path to r4rr_optimized_hparams.yaml",
    )
    p.add_argument("--output-dir", default="", help="Directory for logs + summary outputs.")
    p.add_argument("--seed", type=int, default=0, help="Training seed for all runs.")
    p.add_argument("--dry-run", action="store_true", help="Print commands and write dry-run logs without executing.")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining datasets even if one run fails.",
    )

    # Dataset roots.
    p.add_argument("--waterbirds95-dir", default=str(REPO_ROOT / "data" / "waterbird_complete95_forest2water2"))
    p.add_argument("--waterbirds100-dir", default=str(REPO_ROOT / "data" / "waterbird_1.0_forest2water2"))
    p.add_argument("--redmeat-dir", default=str(REPO_ROOT / "data" / "food-101-redmeat"))
    p.add_argument("--decoy-png-root", default=str(REPO_ROOT / "data" / "DecoyMNIST_png"))

    # Teacher-map roots.
    p.add_argument(
        "--wb95-teacher-maps",
        default=str(REPO_ROOT / "WeCLIPPlus" / "results_waterbirds95_r4rr" / "val" / "prediction_cmap"),
    )
    p.add_argument(
        "--wb100-teacher-maps",
        default=str(REPO_ROOT / "WeCLIPPlus" / "results_waterbirds100_r4rr" / "val" / "prediction_cmap"),
    )
    p.add_argument(
        "--redmeat-teacher-maps",
        default=str(REPO_ROOT / "WeCLIPPlus" / "results_redmeat_r4rr" / "val" / "prediction_cmap"),
    )
    p.add_argument(
        "--decoy-teacher-maps",
        default=str(_default_decoy_maps(REPO_ROOT)),
        help=(
            "Decoy teacher-map root. Defaults to results_decoy_r4rr_trainonly if present, "
            "else results_decoy_r4rr."
        ),
    )

    # Decoy run setup.
    p.add_argument("--decoy-epochs", type=int, default=19)
    p.add_argument("--decoy-num-workers", type=int, default=1)
    p.add_argument("--decoy-no-cuda", action="store_true", help="Pass --no-cuda to r4rr_decoy_fixed.py.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    output_dir = _norm(args.output_dir) if args.output_dir else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_yaml(_norm(args.config_path))
    hp_all = config.get("r4rr_optimized_hparams")
    if not isinstance(hp_all, dict):
        raise KeyError("Expected top-level key 'r4rr_optimized_hparams' in config.")

    needed = ["waterbirds95", "waterbirds100", "redmeat", "decoymnist"]
    for key in needed:
        if key not in hp_all or not isinstance(hp_all[key], dict):
            raise KeyError(f"Missing dataset hparams for '{key}' in {args.config_path}")

    wb95_dir = _norm(args.waterbirds95_dir)
    wb100_dir = _norm(args.waterbirds100_dir)
    redmeat_dir = _norm(args.redmeat_dir)
    decoy_png_root = _norm(args.decoy_png_root)

    wb95_maps = _norm(args.wb95_teacher_maps)
    wb100_maps = _norm(args.wb100_teacher_maps)
    redmeat_maps = _norm(args.redmeat_teacher_maps)
    decoy_maps = _norm(args.decoy_teacher_maps)

    if not args.dry_run:
        _require_dir(wb95_dir, "waterbirds95 dataset")
        _require_dir(wb100_dir, "waterbirds100 dataset")
        _require_dir(redmeat_dir, "redmeat dataset")
        _require_dir(decoy_png_root, "decoy PNG root")
        _require_dir(wb95_maps, "waterbirds95 teacher maps")
        _require_dir(wb100_maps, "waterbirds100 teacher maps")
        _require_dir(redmeat_maps, "redmeat teacher maps")
        _require_dir(decoy_maps, "decoy teacher maps")

    env = os.environ.copy()
    env.setdefault("SAVE_CHECKPOINTS", "0")

    rows: List[Dict[str, Any]] = []
    failures = 0

    def run_one(
        dataset: str,
        cmd: List[str],
        parser_fn,
        notes: str = "",
    ) -> None:
        nonlocal failures
        ds_dir = output_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        log_path = ds_dir / "stdout.log"
        cmd_str = " ".join(cmd)
        try:
            stdout = _run_cmd(cmd, env=env, log_path=log_path, dry_run=args.dry_run)
            metrics = parser_fn(stdout)
            rows.append(
                {
                    "dataset": dataset,
                    "status": "ok",
                    "best_val": metrics.get("best_val"),
                    "test_acc": metrics.get("test_acc"),
                    "per_group": metrics.get("per_group"),
                    "worst_group": metrics.get("worst_group"),
                    "notes": notes,
                    "log_path": str(log_path),
                    "command": cmd_str,
                }
            )
        except Exception as exc:
            failures += 1
            rows.append(
                {
                    "dataset": dataset,
                    "status": "failed",
                    "best_val": None,
                    "test_acc": None,
                    "per_group": None,
                    "worst_group": None,
                    "notes": str(exc),
                    "log_path": str(log_path),
                    "command": cmd_str,
                }
            )
            if not args.continue_on_error:
                raise

    wb95 = hp_all["waterbirds95"]
    run_one(
        "waterbirds95",
        [
            args.python_bin,
            str(R4RR_TRAIN / "r4rr_waterbirds.py"),
            str(wb95_dir),
            str(wb95_maps),
            "--seed",
            str(args.seed),
            "--attention_epoch",
            str(wb95["attention_epoch"]),
            "--kl_lambda",
            str(wb95["kl_lambda"]),
            "--base_lr",
            str(wb95["base_lr"]),
            "--classifier_lr",
            str(wb95["classifier_lr"]),
            "--lr2_mult",
            str(wb95["lr2_mult"]),
            "--kl_increment",
            "0.0",
        ],
        _parse_wb_redmeat_stdout,
    )

    wb100 = hp_all["waterbirds100"]
    run_one(
        "waterbirds100",
        [
            args.python_bin,
            str(R4RR_TRAIN / "r4rr_waterbirds.py"),
            str(wb100_dir),
            str(wb100_maps),
            "--seed",
            str(args.seed),
            "--attention_epoch",
            str(wb100["attention_epoch"]),
            "--kl_lambda",
            str(wb100["kl_lambda"]),
            "--base_lr",
            str(wb100["base_lr"]),
            "--classifier_lr",
            str(wb100["classifier_lr"]),
            "--lr2_mult",
            str(wb100["lr2_mult"]),
            "--kl_increment",
            "0.0",
        ],
        _parse_wb_redmeat_stdout,
    )

    redmeat = hp_all["redmeat"]
    run_one(
        "redmeat",
        [
            args.python_bin,
            str(R4RR_TRAIN / "r4rr_redmeat.py"),
            str(redmeat_dir),
            str(redmeat_maps),
            "--seed",
            str(args.seed),
            "--attention-epoch",
            str(redmeat["attention_epoch"]),
            "--kl-lambda",
            str(redmeat["kl_lambda"]),
            "--base_lr",
            str(redmeat["base_lr"]),
            "--classifier_lr",
            str(redmeat["classifier_lr"]),
            "--lr2-mult",
            str(redmeat["lr2_mult"]),
            "--kl-increment",
            "0.0",
            "--model-name",
            "resnet50",
            "--tune-mode",
            "full",
            "--pretrained",
        ],
        _parse_wb_redmeat_stdout,
    )

    decoy = hp_all["decoymnist"]
    decoy_lr = float(decoy["base_lr"])
    decoy_cls_lr = float(decoy["classifier_lr"])
    decoy_notes = "Decoy uses CDEP-style LeNet/single-LR runner."
    if abs(decoy_lr - decoy_cls_lr) > 1e-12:
        decoy_notes += (
            f" classifier_lr ({decoy_cls_lr}) ignored; script uses single lr={decoy_lr}."
        )

    decoy_cmd = [
        args.python_bin,
        str(R4RR_TRAIN / "r4rr_decoy_fixed.py"),
        "--png-root",
        str(decoy_png_root),
        "--teacher-map-path",
        str(decoy_maps),
        "--epochs",
        str(args.decoy_epochs),
        "--lr",
        str(decoy_lr),
        "--attention-epoch",
        str(decoy["attention_epoch"]),
        "--kl-lambda",
        str(decoy["kl_lambda"]),
        "--kl-incr",
        "0.0",
        "--n-seeds",
        "1",
        "--seed-start",
        str(args.seed),
        "--num-workers",
        str(args.decoy_num_workers),
        "--print-every",
        "1",
    ]
    if args.decoy_no_cuda:
        decoy_cmd.append("--no-cuda")

    run_one("decoymnist", decoy_cmd, _parse_decoy_stdout, notes=decoy_notes)

    _write_summary(rows, output_dir)

    print(f"Summary written to: {output_dir}")
    print("dataset,status,best_val,test_acc,per_group,worst_group")
    for r in rows:
        def _fmt(v: Any) -> str:
            if v is None:
                return ""
            try:
                return f"{float(v):.4f}"
            except Exception:
                return str(v)

        print(
            f"{r['dataset']},{r['status']},"
            f"{_fmt(r['best_val'])},"
            f"{_fmt(r['test_acc'])},"
            f"{_fmt(r['per_group'])},"
            f"{_fmt(r['worst_group'])}"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
