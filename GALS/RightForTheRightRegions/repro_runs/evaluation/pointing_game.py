#!/usr/bin/env python3
"""Run Pointing Game from a manifest and saliency maps."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    if "mask_path" not in rows[0]:
        raise ValueError("Manifest must include a mask_path column.")
    return rows


def sample_id(row: Dict[str, str], index: int) -> str:
    raw = row.get("sample_id") or row.get("id") or Path(row.get("image_path", f"sample_{index}")).stem
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(raw))
    return f"{index:04d}_{safe}"


def load_saliency(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return np.asarray(arr, dtype=np.float32)


def load_mask(path: Path, size_hw: tuple[int, int], threshold: float) -> np.ndarray:
    h, w = size_hw
    with Image.open(path) as im:
        im = im.convert("L").resize((w, h), Image.NEAREST)
        arr = np.asarray(im, dtype=np.float32)
    if threshold <= 1.0:
        threshold = threshold * 255.0
    return arr > threshold


def resolve_saliency_path(row: Dict[str, str], saliency_dir: Path | None, index: int) -> Path:
    if row.get("saliency_path"):
        return Path(row["saliency_path"]).expanduser()
    if saliency_dir is None:
        raise ValueError("Manifest lacks saliency_path; pass --saliency-dir.")
    sid = sample_id(row, index)
    candidates = [
        saliency_dir / sid / "rise.npy",
        saliency_dir / sid / "rise.png",
        saliency_dir / f"{sid}.npy",
        saliency_dir / f"{sid}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No saliency map found for {sid}. Checked: {candidates}")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV with mask_path and saliency_path or sample IDs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--saliency-dir", default=None, help="Directory written by rise_saliency.py if manifest lacks saliency_path.")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--group-column", default="group")
    args = parser.parse_args()

    rows = read_manifest(Path(args.manifest))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saliency_dir = Path(args.saliency_dir).expanduser() if args.saliency_dir else None

    per_image = []
    group_counts = defaultdict(lambda: {"hits": 0, "total": 0})
    total_hits = 0
    total = 0
    errors = 0

    for idx, row in enumerate(rows):
        sid = row.get("sample_id") or sample_id(row, idx)
        try:
            sal_path = resolve_saliency_path(row, saliency_dir, idx)
            sal = load_saliency(sal_path)
            mask = load_mask(Path(row["mask_path"]).expanduser(), sal.shape[:2], args.mask_threshold)
            y, x = np.unravel_index(int(np.argmax(sal)), sal.shape)
            hit = bool(mask[y, x])
            group = row.get(args.group_column, "") or "all"
            total += 1
            total_hits += int(hit)
            group_counts[group]["hits"] += int(hit)
            group_counts[group]["total"] += 1
            per_image.append(
                {
                    "sample_id": sid,
                    "image_path": row.get("image_path", ""),
                    "mask_path": row["mask_path"],
                    "saliency_path": str(sal_path),
                    "argmax_x": int(x),
                    "argmax_y": int(y),
                    "hit": int(hit),
                    "group": group,
                    "error": "",
                }
            )
        except Exception as exc:
            errors += 1
            per_image.append(
                {
                    "sample_id": sid,
                    "image_path": row.get("image_path", ""),
                    "mask_path": row.get("mask_path", ""),
                    "saliency_path": row.get("saliency_path", ""),
                    "argmax_x": "",
                    "argmax_y": "",
                    "hit": "",
                    "group": row.get(args.group_column, ""),
                    "error": str(exc),
                }
            )

    summary = [
        {
            "scope": "overall",
            "hits": total_hits,
            "total": total,
            "pointing_game_acc": total_hits / total if total else float("nan"),
            "errors": errors,
        }
    ]
    for group, counts in sorted(group_counts.items()):
        summary.append(
            {
                "scope": f"group:{group}",
                "hits": counts["hits"],
                "total": counts["total"],
                "pointing_game_acc": counts["hits"] / counts["total"] if counts["total"] else float("nan"),
                "errors": 0,
            }
        )

    write_csv(
        out_dir / "pointing_game_per_image.csv",
        per_image,
        ["sample_id", "image_path", "mask_path", "saliency_path", "argmax_x", "argmax_y", "hit", "group", "error"],
    )
    write_csv(out_dir / "pointing_game_summary.csv", summary, ["scope", "hits", "total", "pointing_game_acc", "errors"])

    print(f"[DONE] total={total} hits={total_hits} acc={total_hits / total if total else float('nan'):.6f} errors={errors}")
    print(f"[DONE] wrote {out_dir / 'pointing_game_summary.csv'}")
    print(f"[DONE] wrote {out_dir / 'pointing_game_per_image.csv'}")


if __name__ == "__main__":
    main()

