#!/usr/bin/env python3
"""
Sample RedMeat prediction_cmaps and copy each cmap + matching source image
into its own folder for quick visual inspection.
"""

import argparse
import csv
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageOps


def _find_metadata_csv(data_root: str) -> str:
    candidates = [
        os.path.join(data_root, "all_images.csv"),
        os.path.join(data_root, "meta", "all_images.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Could not find all_images.csv. Checked:\n  - " + "\n  - ".join(candidates)
    )


def _sanitize_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "item"


def _resolve_image_path(data_root: str, rel_or_abs: str) -> str:
    p = str(rel_or_abs)
    if os.path.isabs(p):
        return p
    return os.path.join(data_root, p.lstrip("/"))


def _cmap_candidates(cmap_root: str, img_rel_path: str) -> List[str]:
    # Mirrors dataset naming logic for WeCLIPPlus-style prediction_cmap masks.
    img_rel_path = img_rel_path.strip().lstrip(os.sep)
    rel_no_ext = os.path.splitext(img_rel_path)[0]
    base = os.path.basename(rel_no_ext)
    parent = os.path.basename(os.path.dirname(rel_no_ext)).replace(".", "_")
    rel_without_images = rel_no_ext[len("images/") :] if rel_no_ext.startswith("images/") else rel_no_ext

    rel_candidates = [
        f"{parent}_{base}.png",
        f"{base}.png",
        rel_no_ext + ".png",
        rel_without_images + ".png",
        os.path.join(parent, f"{base}.png"),
        f"{parent}_{base}.jpg",
        f"{base}.jpg",
        rel_no_ext + ".jpg",
        rel_without_images + ".jpg",
        os.path.join(parent, f"{base}.jpg"),
        f"{parent}_{base}.jpeg",
        f"{base}.jpeg",
        rel_no_ext + ".jpeg",
        rel_without_images + ".jpeg",
        os.path.join(parent, f"{base}.jpeg"),
    ]
    return [os.path.join(cmap_root, c) for c in rel_candidates]


def _first_existing(paths: Iterable[str]) -> Optional[str]:
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p
    return None


def _load_rows(meta_csv: str, split_col: str, split_value: str) -> List[Dict[str, str]]:
    out = []
    with open(meta_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get(split_col, "")).strip() == str(split_value):
                out.append(row)
    return out


def _write_index_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_overlay(image_path: str, cmap_path: str, out_path: str, alpha: float) -> None:
    image = Image.open(image_path).convert("RGB")
    cmap = Image.open(cmap_path).convert("L").resize(image.size, resample=Image.BILINEAR)

    # Normalize to full [0, 255] for stable coloring.
    lo, hi = cmap.getextrema()
    if hi > lo:
        cmap = cmap.point(lambda v: int(255.0 * (float(v - lo) / float(hi - lo))))

    heat = ImageOps.colorize(cmap, black="#1f77b4", white="#d62728").convert("RGB")
    blend = Image.blend(image, heat, alpha=float(alpha))
    blend.save(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Save sampled RedMeat prediction_cmap + image pairs.")
    p.add_argument(
        "--cmap-root",
        required=True,
        help="Path to prediction_cmap directory (e.g., .../results_redmeat_openclip_dinovit/val/prediction_cmap).",
    )
    p.add_argument(
        "--data-root",
        default="/home/ryreu/guided_cnn/Food101/data/food-101-redmeat",
        help="RedMeat dataset root containing all_images.csv and images/...",
    )
    p.add_argument("--split-col", default="split")
    p.add_argument("--split", default="val")
    p.add_argument("--label-col", default="label")
    p.add_argument("--path-col", default="abs_file_path")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: logsRedMeat/redmeat_cmap_pairs_<timestamp>",
    )
    p.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Alpha for optional image+cmap overlay.",
    )
    p.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable writing overlay.png for each pair.",
    )
    args = p.parse_args()

    cmap_root = os.path.abspath(os.path.expanduser(args.cmap_root))
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    if not os.path.isdir(cmap_root):
        raise FileNotFoundError(f"Missing cmap root: {cmap_root}")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Missing data root: {data_root}")

    meta_csv = _find_metadata_csv(data_root)
    rows = _load_rows(meta_csv, split_col=args.split_col, split_value=args.split)
    if not rows:
        raise RuntimeError(f"No rows found for split='{args.split}' using split_col='{args.split_col}' in {meta_csv}")

    matched: List[Tuple[Dict[str, str], str, str]] = []
    missing_preview: List[str] = []
    for row in rows:
        img_rel = str(row.get(args.path_col, "")).strip()
        if not img_rel:
            continue
        image_path = _resolve_image_path(data_root, img_rel)
        if not os.path.isfile(image_path):
            continue
        cmap_path = _first_existing(_cmap_candidates(cmap_root, img_rel))
        if cmap_path is None:
            if len(missing_preview) < 10:
                missing_preview.append(img_rel)
            continue
        matched.append((row, image_path, cmap_path))

    if not matched:
        msg = [
            f"No cmap/image matches found.",
            f"cmap_root={cmap_root}",
            f"data_root={data_root}",
            f"metadata={meta_csv}",
            f"split={args.split}",
        ]
        if missing_preview:
            msg.append("Example image rel paths with no cmap:\n  - " + "\n  - ".join(missing_preview))
        raise RuntimeError("\n".join(msg))

    rng = random.Random(args.seed)
    k = min(int(args.num_samples), len(matched))
    sampled = rng.sample(matched, k=k)

    if args.output_dir:
        output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/ryreu/guided_cnn/logsRedMeat/redmeat_cmap_pairs_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    index_rows = []
    for i, (row, image_path, cmap_path) in enumerate(sampled):
        label = str(row.get(args.label_col, "unknown"))
        img_rel = str(row.get(args.path_col, ""))
        stem = os.path.splitext(os.path.basename(image_path))[0]
        pair_name = f"{i:03d}_{_sanitize_name(label)}_{_sanitize_name(stem)}"
        pair_dir = os.path.join(output_dir, pair_name)
        os.makedirs(pair_dir, exist_ok=True)

        img_ext = os.path.splitext(image_path)[1].lower() or ".jpg"
        cmap_ext = os.path.splitext(cmap_path)[1].lower() or ".png"
        out_img = os.path.join(pair_dir, f"image{img_ext}")
        out_cmap = os.path.join(pair_dir, f"prediction_cmap{cmap_ext}")

        shutil.copy2(image_path, out_img)
        shutil.copy2(cmap_path, out_cmap)

        overlay_path = ""
        if not args.no_overlay:
            overlay_path = os.path.join(pair_dir, "overlay.png")
            try:
                _save_overlay(out_img, out_cmap, overlay_path, alpha=args.overlay_alpha)
            except Exception:
                overlay_path = ""

        index_rows.append(
            {
                "sample_id": i,
                "pair_dir": pair_dir,
                "label": label,
                "split": args.split,
                "image_rel": img_rel,
                "image_src": image_path,
                "cmap_src": cmap_path,
                "image_out": out_img,
                "cmap_out": out_cmap,
                "overlay_out": overlay_path,
            }
        )

    index_csv = os.path.join(output_dir, "sample_index.csv")
    _write_index_csv(index_csv, index_rows)

    print(f"[DONE] Saved {k} pairs to: {output_dir}")
    print(f"[DONE] Index CSV: {index_csv}")
    print(f"[INFO] Matched pool size for split='{args.split}': {len(matched)}")


if __name__ == "__main__":
    main()
