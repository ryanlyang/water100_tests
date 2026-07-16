#!/usr/bin/env python3
"""Generate RedMeat GALS attention maps with CLIP RN50 (no Slurm required)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


BPE_URL = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
BPE_REL = Path("CLIP/clip/bpe_simple_vocab_16e6.txt.gz")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_gals_root() -> Path:
    return _repo_root() / "repro_runs" / "third_party" / "GALS"


def _default_data_root() -> Path:
    return _repo_root() / "data"


def _infer_total_images(meta_csv: Path) -> int:
    with meta_csv.open("r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _resolve_config(gals_root: Path, config_arg: str, candidates: list[str]) -> Path:
    if config_arg:
        cfg = Path(config_arg).expanduser()
        if not cfg.is_absolute():
            cfg = gals_root / cfg
        if not cfg.is_file():
            raise FileNotFoundError(f"Config not found: {cfg}")
        return cfg

    for rel in candidates:
        cfg = gals_root / rel
        if cfg.is_file():
            return cfg
    raise FileNotFoundError(
        "Could not find a RedMeat attention config. Checked: " + ", ".join(candidates)
    )


def _ensure_bpe_vocab(gals_root: Path) -> None:
    bpe_path = gals_root / BPE_REL
    if bpe_path.is_file():
        return
    bpe_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading CLIP BPE vocab to {bpe_path}")
    urllib.request.urlretrieve(BPE_URL, bpe_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gals-root", type=str, default=str(_default_gals_root()))
    parser.add_argument("--data-root", type=str, default=str(_default_data_root()))
    parser.add_argument("--dataset-dir", type=str, default="food-101-redmeat")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--save-folder", type=str, default="clip_rn50_attention_gradcam")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--total-images", type=int, default=-1)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download-bpe-vocab", dest="download_bpe_vocab", action="store_true")
    parser.add_argument("--no-download-bpe-vocab", dest="download_bpe_vocab", action="store_false")
    parser.set_defaults(download_bpe_vocab=True)
    args = parser.parse_args()

    gals_root = Path(args.gals_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    dataset_root = data_root / args.dataset_dir
    meta_csv = dataset_root / "all_images.csv"

    if not gals_root.is_dir():
        raise FileNotFoundError(f"Missing GALS root: {gals_root}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
    if not meta_csv.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {meta_csv}")

    config = _resolve_config(
        gals_root,
        args.config,
        [
            "RedMeat_Runs/configs/redmeat_attention_rn50.yaml",
            "configs/redmeat_attention_rn50.yaml",
            "configs/food_attention.yaml",
        ],
    )

    if args.download_bpe_vocab:
        _ensure_bpe_vocab(gals_root)

    total_images = args.total_images if args.total_images > 0 else _infer_total_images(meta_csv)
    if total_images <= 0:
        raise ValueError(f"Could not infer a positive image count from {meta_csv}")

    start_idx = max(0, args.start_idx)
    stop_idx = total_images if args.end_idx < 0 else min(args.end_idx, total_images)
    if start_idx >= stop_idx:
        raise ValueError(
            f"Empty index range: start_idx={start_idx}, stop_idx={stop_idx}, total={total_images}"
        )

    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{gals_root}:{py_path}" if py_path else str(gals_root)
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

    print(f"[INFO] gals_root={gals_root}")
    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] dataset_dir={args.dataset_dir}")
    print(f"[INFO] config={config}")
    print(f"[INFO] total_images={total_images}")
    print(f"[INFO] range=[{start_idx}, {stop_idx}) chunk_size={args.chunk_size}")

    for start in range(start_idx, stop_idx, args.chunk_size):
        end = min(start + args.chunk_size, stop_idx)
        cmd = [
            args.python_bin,
            "-u",
            "extract_attention.py",
            "--config",
            str(config),
            f"DATA.ROOT={data_root}",
            f"DATA.FOOD_SUBSET_DIR={args.dataset_dir}",
            f"SAVE_FOLDER={args.save_folder}",
            "DISABLE_VIS=true",
            "SKIP_EXISTING=true",
            f"START_IDX={start}",
            f"END_IDX={end}",
        ]
        print("[RUN]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=gals_root, env=env, check=True)

    print("[DONE] RedMeat RN50 attention generation complete.")


if __name__ == "__main__":
    main()
