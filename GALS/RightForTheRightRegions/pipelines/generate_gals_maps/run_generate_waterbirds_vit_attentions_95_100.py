#!/usr/bin/env python3
"""Generate Waterbirds-95/100 GALS ViT attentions (no Slurm required)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_gals_root() -> Path:
    return _repo_root() / "repro_runs" / "third_party" / "GALS"


def _default_data_root() -> Path:
    return _repo_root() / "data"


def _infer_num_images(metadata_csv: Path) -> int:
    with metadata_csv.open("r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _resolve_config(gals_root: Path, config_arg: str, default_rel: str) -> Path:
    if config_arg:
        cfg = Path(config_arg).expanduser()
        if not cfg.is_absolute():
            cfg = gals_root / cfg
    else:
        cfg = gals_root / default_rel
    if not cfg.is_file():
        raise FileNotFoundError(f"Config not found: {cfg}")
    return cfg


def _run_dataset(
    *,
    name: str,
    gals_root: Path,
    data_root: Path,
    dataset_dir: str,
    config: Path,
    chunk_size: int,
    explicit_n: int,
    python_bin: str,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    dataset_root = data_root / dataset_dir
    metadata_csv = dataset_root / "metadata.csv"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset dir for {name}: {dataset_root}")
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Missing metadata CSV for {name}: {metadata_csv}")

    total_n = explicit_n if explicit_n > 0 else _infer_num_images(metadata_csv)
    if total_n <= 0:
        raise ValueError(f"Could not infer positive image count for {name} from {metadata_csv}")

    print(f"[GEN] {name}: dataset_dir={dataset_dir}, N={total_n}, chunk_size={chunk_size}")
    for start in range(0, total_n, chunk_size):
        end = min(start + chunk_size, total_n)
        cmd = [
            python_bin,
            "-u",
            "extract_attention.py",
            "--config",
            str(config),
            f"DATA.ROOT={data_root}",
            f"DATA.WATERBIRDS_DIR={dataset_dir}",
            "DISABLE_VIS=true",
            "SKIP_EXISTING=true",
            f"START_IDX={start}",
            f"END_IDX={end}",
        ]
        print("[RUN]", " ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, cwd=gals_root, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gals-root", type=str, default=str(_default_gals_root()))
    parser.add_argument("--data-root", type=str, default=str(_default_data_root()))
    parser.add_argument("--wb95-dir", type=str, default="waterbird_complete95_forest2water2")
    parser.add_argument("--wb100-dir", type=str, default="waterbird_1.0_forest2water2")
    parser.add_argument("--config-wb95", type=str, default="configs/waterbirds_95_attention_vit.yaml")
    parser.add_argument("--config-wb100", type=str, default="configs/waterbirds_100_attention_vit.yaml")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--wb95-n", type=int, default=11788)
    parser.add_argument("--wb100-n", type=int, default=11788)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--run-wb95", dest="run_wb95", action="store_true")
    parser.add_argument("--no-run-wb95", dest="run_wb95", action="store_false")
    parser.add_argument("--run-wb100", dest="run_wb100", action="store_true")
    parser.add_argument("--no-run-wb100", dest="run_wb100", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(run_wb95=True, run_wb100=True)
    args = parser.parse_args()

    gals_root = Path(args.gals_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()

    if not gals_root.is_dir():
        raise FileNotFoundError(f"Missing GALS root: {gals_root}")
    if not args.run_wb95 and not args.run_wb100:
        raise ValueError("Both --run-wb95 and --run-wb100 are disabled.")

    cfg95 = _resolve_config(gals_root, args.config_wb95, "configs/waterbirds_95_attention_vit.yaml")
    cfg100 = _resolve_config(gals_root, args.config_wb100, "configs/waterbirds_100_attention_vit.yaml")

    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{gals_root}:{py_path}" if py_path else str(gals_root)
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

    print(f"[INFO] gals_root={gals_root}")
    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] wb95_dir={args.wb95_dir} run={args.run_wb95}")
    print(f"[INFO] wb100_dir={args.wb100_dir} run={args.run_wb100}")
    print(f"[INFO] chunk_size={args.chunk_size}")

    if args.run_wb95:
        _run_dataset(
            name="WB95 ViT",
            gals_root=gals_root,
            data_root=data_root,
            dataset_dir=args.wb95_dir,
            config=cfg95,
            chunk_size=args.chunk_size,
            explicit_n=args.wb95_n,
            python_bin=args.python_bin,
            env=env,
            dry_run=args.dry_run,
        )

    if args.run_wb100:
        _run_dataset(
            name="WB100 ViT",
            gals_root=gals_root,
            data_root=data_root,
            dataset_dir=args.wb100_dir,
            config=cfg100,
            chunk_size=args.chunk_size,
            explicit_n=args.wb100_n,
            python_bin=args.python_bin,
            env=env,
            dry_run=args.dry_run,
        )

    print("[DONE] Waterbirds ViT attention generation complete.")


if __name__ == "__main__":
    main()
