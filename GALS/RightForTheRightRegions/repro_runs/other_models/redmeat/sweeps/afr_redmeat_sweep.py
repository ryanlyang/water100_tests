#!/usr/bin/env python3
"""Two-stage AFR runner for RedMeat (Food-101 subset).

This script mirrors the Waterbirds AFR runner, but first prepares an AFR-style
`metadata.csv` from RedMeat metadata (`all_images.csv`) when needed.

AFR expects columns:
  - img_filename: image path
  - y: class id
  - place: spurious attribute id
  - split: 0=train, 1=val, 2=test
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

import pandas as pd


DEFAULT_CLASSES = [
    "prime_rib",
    "pork_chop",
    "steak",
    "baby_back_ribs",
    "filet_mignon",
]

SPLIT_TO_INT = {
    "train": 0,
    "tr": 0,
    "0": 0,
    "val": 1,
    "valid": 1,
    "validation": 1,
    "dev": 1,
    "1": 1,
    "test": 2,
    "te": 2,
    "2": 2,
}

REPRO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AFR_ROOT = (REPRO_ROOT / "third_party" / "afr").resolve()


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


def _parse_str_list(text: str) -> List[str]:
    vals = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if piece:
            vals.append(piece)
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
    # AFR logs use "best_test_mean_val" in run output text ("Best test mean val").
    # Keep robust fallbacks for potential key variants.
    best_test_mean_at_val = _metric_max(m, "best_test_mean_val")
    if best_test_mean_at_val is None:
        best_test_mean_at_val = _metric_max(m, "best_test_mean_at_val")
    if best_test_mean_at_val is None:
        best_test_mean_at_val = best_test_mean
    return {
        "best_val_wga": _float_or_nan(best_val_wga),
        "best_test_at_val": _float_or_nan(best_test_at_val),
        "best_test_wga": _float_or_nan(best_test_wga),
        "best_val_mean": _float_or_nan(best_val_mean),
        "best_test_mean": _float_or_nan(best_test_mean),
        "best_test_mean_at_val": _float_or_nan(best_test_mean_at_val),
    }


def _run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> None:
    _ensure_dir(log_path.parent)
    # AFR train_embeddings.py writes plots to ./logs/.
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


def _pick_metadata_source(data_dir: Path, metadata_input: str) -> Path:
    candidates = []
    if metadata_input:
        candidates.append(Path(metadata_input).expanduser())
    candidates.extend(
        [
            data_dir / "metadata.csv",
            data_dir / "all_images.csv",
            data_dir / "meta" / "all_images.csv",
        ]
    )
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        "Could not find metadata source. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _to_abs_image_path(path_text: object, data_dir: Path) -> str:
    p = os.path.expanduser(str(path_text).strip())
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(str(data_dir), p))


def _normalize_split_value(v: object) -> int:
    if pd.isna(v):
        raise ValueError("Encountered NaN split value")
    text = str(v).strip().lower()
    if text in SPLIT_TO_INT:
        return SPLIT_TO_INT[text]
    try:
        # Handles values like "1.0".
        iv = int(float(text))
        if iv in (0, 1, 2):
            return iv
    except Exception:
        pass
    raise ValueError(f"Unrecognized split value: {v!r}")


def _encode_to_int(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    clean = series.astype(str).str.strip()
    uniq = sorted(clean.dropna().unique().tolist())
    mapping = {k: i for i, k in enumerate(uniq)}
    return clean.map(mapping).astype(int), mapping


def _prepare_afr_metadata(args: argparse.Namespace, data_dir: Path, output_root: Path) -> Tuple[Path, Dict[str, object]]:
    source_csv = _pick_metadata_source(data_dir, args.metadata_input)
    df = pd.read_csv(source_csv)
    if len(df) == 0:
        raise RuntimeError(f"Metadata file has no rows: {source_csv}")

    required = {"img_filename", "y", "place", "split"}
    notes: List[str] = []
    class_map: Dict[str, int] = {}
    place_map: Dict[str, int] = {}

    if required.issubset(set(df.columns)):
        out = df[["img_filename", "y", "place", "split"]].copy()
        out["img_filename"] = out["img_filename"].map(lambda p: _to_abs_image_path(p, data_dir))

        if not pd.api.types.is_numeric_dtype(out["y"]):
            out["y"], class_map = _encode_to_int(out["y"])
            notes.append("Converted non-numeric y to integer ids.")
        else:
            out["y"] = out["y"].astype(int)

        if not pd.api.types.is_numeric_dtype(out["place"]):
            out["place"], place_map = _encode_to_int(out["place"])
            notes.append("Converted non-numeric place to integer ids.")
        else:
            out["place"] = out["place"].astype(int)

        out["split"] = out["split"].map(_normalize_split_value).astype(int)
    else:
        for col in (args.path_col, args.label_col, args.split_col):
            if col not in df.columns:
                raise KeyError(f"Missing required column {col!r} in {source_csv}")

        working = df[[args.path_col, args.label_col, args.split_col] + [c for c in df.columns if c not in (args.path_col, args.label_col, args.split_col)]].copy()

        labels = working[args.label_col].astype(str).str.strip()
        classes = _parse_str_list(args.classes)
        if classes:
            class_set = set(classes)
            mask = labels.isin(class_set)
            dropped = int((~mask).sum())
            if dropped > 0:
                notes.append(f"Dropped {dropped} rows outside requested classes.")
            working = working[mask].copy()
            labels = working[args.label_col].astype(str).str.strip()
        else:
            classes = sorted(labels.unique().tolist())

        if len(working) == 0:
            raise RuntimeError("No rows left after class filtering.")

        class_map = {name: idx for idx, name in enumerate(classes)}
        y = labels.map(class_map).astype(int)

        # place handling:
        place_series: pd.Series
        place_source = ""
        if args.place_mode == "constant":
            place_series = pd.Series([0] * len(working), index=working.index, dtype=int)
            place_source = "constant(0)"
        else:
            candidates = []
            if args.place_col:
                candidates.append(args.place_col)
            candidates.extend(_parse_str_list(args.place_candidates))
            chosen = None
            for c in candidates:
                if c in working.columns:
                    chosen = c
                    break
            if chosen is None:
                place_series = pd.Series([0] * len(working), index=working.index, dtype=int)
                place_source = "constant(0, no place column found)"
                notes.append(
                    "No place-like column found; place was set to 0 for all rows."
                )
            else:
                raw = working[chosen]
                if pd.api.types.is_numeric_dtype(raw):
                    place_series = raw.astype(int)
                else:
                    place_series, place_map = _encode_to_int(raw)
                place_source = f"column:{chosen}"

        split = working[args.split_col].map(_normalize_split_value).astype(int)
        img = working[args.path_col].map(lambda p: _to_abs_image_path(p, data_dir))

        out = pd.DataFrame(
            {
                "img_filename": img,
                "y": y,
                "place": place_series,
                "split": split,
            }
        )
        notes.append(f"place_source={place_source}")

    # Optional image existence check.
    if args.check_images:
        missing = []
        for p in out["img_filename"].tolist():
            if not os.path.exists(p):
                missing.append(p)
                if len(missing) >= 20:
                    break
        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                "Prepared metadata points to missing images. First examples:\n" + preview
            )

    prepared_dir = output_root / "prepared_data"
    _ensure_dir(prepared_dir)
    metadata_path = prepared_dir / "metadata.csv"
    out.to_csv(metadata_path, index=False)

    split_counts = out["split"].value_counts().to_dict()
    class_counts = out["y"].value_counts().sort_index().to_dict()
    place_counts = out["place"].value_counts().sort_index().to_dict()
    n_groups = int(out["y"].nunique() * out["place"].nunique())
    active_groups = int(out[["y", "place"]].drop_duplicates().shape[0])

    info: Dict[str, object] = {
        "source_csv": str(source_csv),
        "prepared_data_dir": str(prepared_dir),
        "metadata_csv": str(metadata_path),
        "rows": int(len(out)),
        "n_classes": int(out["y"].nunique()),
        "n_place": int(out["place"].nunique()),
        "n_groups_possible": n_groups,
        "n_groups_active": active_groups,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "place_counts": place_counts,
        "notes": notes,
        "class_map": class_map,
        "place_map": place_map,
    }
    return prepared_dir, info


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

    _ensure_dir(output_root)
    _ensure_dir(logs_root)

    prepared_data_dir, prep_info = _prepare_afr_metadata(args, data_dir, output_root)

    stage1_root = output_root / "stage1"
    stage2_root = output_root / "stage2"
    run_logs = logs_root / f"afr_redmeat_repro_{time.strftime('%Y%m%d_%H%M%S')}"
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
    print(f"[INFO] PREPARED_DATA_DIR={prepared_data_dir}")
    print(f"[INFO] METADATA_SOURCE={prep_info['source_csv']}")
    print(f"[INFO] OUTPUT_ROOT={output_root}")
    print(f"[INFO] LOGS_ROOT={run_logs}")
    print(
        "[INFO] Prepared metadata stats: "
        f"rows={prep_info['rows']} n_classes={prep_info['n_classes']} "
        f"n_place={prep_info['n_place']} active_groups={prep_info['n_groups_active']}/"
        f"{prep_info['n_groups_possible']}"
    )
    print(f"[INFO] Split counts: {prep_info['split_counts']}")
    print(f"[INFO] Class counts: {prep_info['class_counts']}")
    print(f"[INFO] Place counts: {prep_info['place_counts']}")
    for note in prep_info.get("notes", []):
        print(f"[INFO] {note}")

    print(f"[INFO] Seeds={seeds}")
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
                "--project=afr_redmeat_repro",
                f"--seed={seed}",
                f"--eval_freq={args.stage1_eval_freq}",
                f"--save_freq={args.stage1_save_freq}",
                f"--data_dir={prepared_data_dir}",
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
        total_runs = len(gammas) * len(regs)
        print(f"===== [SEED {seed}] Stage 2 Sweep ({total_runs} runs) =====")
        for idx, (gamma, reg) in enumerate(itertools.product(gammas, regs), start=1):
            run_name = f"seed{seed}_g{_format_gamma(gamma)}_r{_format_reg(reg)}"
            s2_dir = stage2_root / f"seed_{seed}" / run_name
            s2_metrics = s2_dir / "metrics.pkl"
            s2_log = run_logs / f"{run_name}.log"

            print(
                f"[RUN] seed={seed} {idx}/{total_runs} gamma={gamma} reg={reg} -> {s2_dir}",
                flush=True,
            )

            if not _maybe_skip(s2_metrics, args.force_stage2):
                s2_cmd = [
                    python_exe,
                    "train_embeddings.py",
                    f"--output_dir={s2_dir}",
                    "--project=afr_redmeat_repro",
                    f"--seed={seed}",
                    f"--base_model_dir={s1_dir}",
                    "--checkpoint=final_checkpoint.pt",
                    "--model=imagenet_resnet50_pretrained",
                    f"--data_dir={prepared_data_dir}",
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
            f"best_test_at_val={float(seed_best['best_test_at_val']):.4f}"
        )

    stage2_csv = output_root / "afr_redmeat_stage2_all.csv"
    best_csv = output_root / "afr_redmeat_best_by_seed.csv"
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
            "best_test_mean_at_val",
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
            "best_test_mean_at_val",
            "stage1_dir",
            "stage2_dir",
            "metrics_path",
        ],
    )

    best_test_vals = [
        float(r["best_test_at_val"]) for r in best_rows if math.isfinite(float(r["best_test_at_val"]))
    ]
    if best_test_vals:
        mean_best = statistics.mean(best_test_vals)
        std_best = statistics.pstdev(best_test_vals) if len(best_test_vals) > 1 else 0.0
    else:
        mean_best = float("nan")
        std_best = float("nan")

    best_test_mean_at_val_vals = [
        float(r["best_test_mean_at_val"])
        for r in best_rows
        if math.isfinite(float(r["best_test_mean_at_val"]))
    ]
    if best_test_mean_at_val_vals:
        mean_best_test_mean_at_val = statistics.mean(best_test_mean_at_val_vals)
        std_best_test_mean_at_val = (
            statistics.pstdev(best_test_mean_at_val_vals)
            if len(best_test_mean_at_val_vals) > 1
            else 0.0
        )
    else:
        mean_best_test_mean_at_val = float("nan")
        std_best_test_mean_at_val = float("nan")

    summary_txt = output_root / "afr_redmeat_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("AFR RedMeat reproduction summary\n")
        f.write(f"AFR_ROOT: {afr_root}\n")
        f.write(f"DATA_DIR: {data_dir}\n")
        f.write(f"PREPARED_DATA_DIR: {prepared_data_dir}\n")
        f.write(f"METADATA_SOURCE: {prep_info['source_csv']}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Stage1 epochs: {args.stage1_epochs}\n")
        f.write(f"Stage1 scheduler: {args.stage1_scheduler}\n")
        f.write(f"Stage2 epochs: {args.stage2_epochs}\n")
        f.write(f"Gammas: {gammas}\n")
        f.write(f"Reg coeffs: {regs}\n")
        f.write(f"Prepared stats: rows={prep_info['rows']} n_classes={prep_info['n_classes']} ")
        f.write(f"n_place={prep_info['n_place']} active_groups={prep_info['n_groups_active']}/{prep_info['n_groups_possible']}\n")
        f.write(f"Split counts: {prep_info['split_counts']}\n")
        f.write(f"Class counts: {prep_info['class_counts']}\n")
        f.write(f"Place counts: {prep_info['place_counts']}\n")
        for note in prep_info.get("notes", []):
            f.write(f"Note: {note}\n")
        f.write(f"Mean best test@val test_acc: {mean_best_test_mean_at_val:.4f}\n")
        f.write(f"Std best test@val test_acc: {std_best_test_mean_at_val:.4f}\n")
        f.write(f"Mean best test@val worst_group_acc: {mean_best:.4f}\n")
        f.write(f"Std best test@val worst_group_acc: {std_best:.4f}\n")
        f.write(f"Best-per-seed CSV: {best_csv}\n")
        f.write(f"All stage2 CSV: {stage2_csv}\n")

    print("\n===== [DONE] =====")
    print(f"All stage-2 results: {stage2_csv}")
    print(f"Best by seed:        {best_csv}")
    print(f"Summary:             {summary_txt}")
    print(
        f"Mean best test@val test_acc across seeds: "
        f"{mean_best_test_mean_at_val:.4f} +/- {std_best_test_mean_at_val:.4f}"
    )
    print(
        f"Mean best test@val worst_group_acc across seeds: "
        f"{mean_best:.4f} +/- {std_best:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AFR RedMeat reproduction runner")
    parser.add_argument(
        "--afr-root",
        default=str(DEFAULT_AFR_ROOT),
        help="Path to AFR repository root",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/ryreu/guided_cnn/Food101/data/food-101-redmeat",
        help="RedMeat dataset directory",
    )
    parser.add_argument(
        "--metadata-input",
        default="",
        help="Optional explicit metadata CSV path. If empty, auto-detect under --data-dir.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ryreu/guided_cnn/logsRedMeat/afr_repro",
        help="Directory where AFR run outputs and CSVs are stored",
    )
    parser.add_argument(
        "--logs-root",
        default="/home/ryreu/guided_cnn/logsRedMeat/afr_repro_logs",
        help="Directory for command logs",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to launch AFR scripts",
    )
    parser.add_argument(
        "--seeds",
        default="0,21,42",
        help="Comma-separated list of seeds for stage1+stage2",
    )

    # Metadata conversion settings for all_images.csv-like inputs.
    parser.add_argument("--path-col", default="abs_file_path")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--split-col", default="split")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class names to keep; empty means all classes in metadata.",
    )
    parser.add_argument(
        "--place-mode",
        choices=["auto", "constant"],
        default="auto",
        help="How to construct place: auto uses a place-like column if available, else constant 0.",
    )
    parser.add_argument(
        "--place-col",
        default="",
        help="Optional explicit place column name used when --place-mode=auto.",
    )
    parser.add_argument(
        "--place-candidates",
        default="place,spurious,group,confounder,background,bg,environment,env,smoke,context",
        help="Comma-separated fallback place-like column names for --place-mode=auto.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Validate that image paths referenced by prepared metadata exist.",
    )

    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage1-eval-freq", type=int, default=10)
    parser.add_argument("--stage1-save-freq", type=int, default=10)
    parser.add_argument(
        "--stage1-scheduler",
        default="cosine_lr_scheduler",
        help="AFR stage-1 scheduler passed to train_supervised.py",
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
