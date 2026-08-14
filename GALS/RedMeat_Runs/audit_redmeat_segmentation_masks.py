#!/usr/bin/env python3
"""Rank RedMeat segmentation masks for targeted visual quality review."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--sheet-size", type=int, default=20)
    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-12:
        return np.zeros_like(values)
    return 0.67448975 * (values - median) / mad


def connected_component_features(mask: np.ndarray) -> Dict[str, float]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:]
    total = int(mask.sum())
    if total <= 0:
        raise RuntimeError("Encountered an empty mask during audit.")
    largest = int(areas.max()) if areas.size else 0
    tiny_cutoff = max(20, int(math.ceil(mask.size * 0.001)))
    tiny_area = int(areas[areas < tiny_cutoff].sum()) if areas.size else 0
    tiny_count = int((areas < tiny_cutoff).sum()) if areas.size else 0

    filled = ndimage.binary_fill_holes(mask)
    hole_pixels = int(np.count_nonzero(filled & ~mask))
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    boundary = mask & ~eroded
    perimeter = int(boundary.sum())

    ys, xs = np.nonzero(mask)
    bbox_area = int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    sides_touched = int(mask[0].any()) + int(mask[-1].any())
    sides_touched += int(mask[:, 0].any()) + int(mask[:, -1].any())
    border_pixels = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())

    return {
        "component_count": float(count),
        "largest_component_ratio": largest / total,
        "tiny_component_count": float(tiny_count),
        "tiny_component_fraction": tiny_area / total,
        "hole_fraction": hole_pixels / total,
        "boundary_pixels": float(perimeter),
        "boundary_complexity": perimeter / math.sqrt(total),
        "bbox_fill": total / bbox_area,
        "sides_touched": float(sides_touched),
        "border_pixel_fraction": border_pixels / total,
        "boundary_mask": boundary,
    }


def image_edge_alignment(source: np.ndarray, boundary: np.ndarray) -> float:
    gray = (
        0.2126 * source[..., 0].astype(np.float32)
        + 0.7152 * source[..., 1].astype(np.float32)
        + 0.0722 * source[..., 2].astype(np.float32)
    )
    gx = ndimage.sobel(gray, axis=1, mode="reflect")
    gy = ndimage.sobel(gray, axis=0, mode="reflect")
    gradient = np.hypot(gx, gy)
    boundary_values = gradient[boundary]
    if boundary_values.size == 0:
        return 0.0
    global_mean = float(gradient.mean())
    return float(boundary_values.mean()) / max(global_mean, 1e-6)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reasons(row: Dict[str, object]) -> List[str]:
    result = []
    area = float(row["foreground_fraction"])
    if area < 0.04:
        result.append("very small mask")
    elif area > 0.90:
        result.append("very large mask")
    if abs(float(row["class_area_robust_z"])) > 3.0:
        result.append("class area outlier")
    if int(row["component_count"]) >= 5:
        result.append("many components")
    if float(row["largest_component_ratio"]) < 0.65:
        result.append("fragmented foreground")
    if float(row["tiny_component_fraction"]) > 0.01:
        result.append("tiny islands")
    if float(row["boundary_complexity_robust_z"]) > 3.0:
        result.append("complex boundary")
    if float(row["edge_alignment"]) < 1.0:
        result.append("weak edge alignment")
    if int(row["sides_touched"]) >= 3:
        result.append("touches 3+ borders")
    return result


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def make_sheets(
    root: Path,
    output_root: Path,
    rows: Sequence[Dict[str, object]],
    sheet_size: int,
) -> List[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    font = load_font(22)
    small_font = load_font(18)
    card_width, image_height, text_height = 760, 380, 92
    columns = 2
    rows_per_sheet = int(math.ceil(sheet_size / columns))
    sheet_paths = []
    for offset in range(0, len(rows), sheet_size):
        chunk = rows[offset : offset + sheet_size]
        sheet = Image.new(
            "RGB",
            (columns * card_width, rows_per_sheet * (image_height + text_height)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for local_index, row in enumerate(chunk):
            column = local_index % columns
            row_index = local_index // columns
            x = column * card_width
            y = row_index * (image_height + text_height)
            with Image.open(root / str(row["review_path"])) as opened:
                review = opened.convert("RGB")
            review.thumbnail((card_width, image_height), Image.Resampling.LANCZOS)
            review = ImageOps.pad(review, (card_width, image_height), color="#20242b")
            sheet.paste(review, (x, y))
            title = (
                f"#{offset + local_index + 1}  {row['image_id']}  {row['class_name']}  "
                f"score={float(row['anomaly_score']):.2f}"
            )
            metrics = (
                f"area={100 * float(row['foreground_fraction']):.1f}%  "
                f"components={int(row['component_count'])}  "
                f"largest={100 * float(row['largest_component_ratio']):.0f}%  "
                f"edge={float(row['edge_alignment']):.2f}"
            )
            why = str(row["reasons"])
            draw.text((x + 8, y + image_height + 6), title, fill="black", font=font)
            draw.text((x + 8, y + image_height + 35), metrics, fill="#222", font=small_font)
            draw.text((x + 8, y + image_height + 60), why, fill="#9b2c20", font=small_font)
        path = output_root / f"candidate_sheet_{offset // sheet_size + 1:02d}.jpg"
        sheet.save(path, quality=90, optimize=True)
        sheet_paths.append(path)
    return sheet_paths


def unique_ranked(rows: Sequence[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
    selectors = (
        ("anomaly_score", True, top_k),
        ("foreground_fraction", False, 20),
        ("foreground_fraction", True, 20),
        ("component_count", True, 25),
        ("edge_alignment", False, 25),
        ("boundary_complexity", True, 25),
        ("tiny_component_fraction", True, 20),
    )
    chosen: Dict[str, Dict[str, object]] = {}
    for key, reverse, count in selectors:
        for row in sorted(rows, key=lambda item: float(item[key]), reverse=reverse)[:count]:
            chosen[str(row["image_id"])] = row
    return sorted(chosen.values(), key=lambda row: float(row["anomaly_score"]), reverse=True)[
        :top_k
    ]


def main() -> None:
    args = parse_args()
    root = args.review_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest = read_manifest(root / "review_manifest.csv")
    rows: List[Dict[str, object]] = []

    for index, source_row in enumerate(manifest):
        mask_path = root / source_row["binary_mask_path"]
        source_path = Path(source_row["source_path"])
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened.convert("L")) > 0
        with Image.open(source_path) as opened:
            source = np.asarray(opened.convert("RGB"))
        if source.shape[:2] != mask.shape:
            raise RuntimeError(f"Shape mismatch for {source_row['image_id']}")

        features = connected_component_features(mask)
        boundary = features.pop("boundary_mask")
        row: Dict[str, object] = dict(source_row)
        row.update(features)
        row["edge_alignment"] = image_edge_alignment(source, boundary)
        rows.append(row)
        if (index + 1) % 250 == 0:
            print(f"[FEATURES] {index + 1}/{len(manifest)}", flush=True)

    classes = sorted({str(row["class_name"]) for row in rows})
    for class_name in classes:
        indices = [i for i, row in enumerate(rows) if row["class_name"] == class_name]
        area = np.asarray([float(rows[i]["foreground_fraction"]) for i in indices])
        complexity = np.asarray([float(rows[i]["boundary_complexity"]) for i in indices])
        area_z = robust_z(np.log(np.clip(area, 1e-6, 1 - 1e-6) / np.clip(1 - area, 1e-6, 1)))
        complexity_z = robust_z(complexity)
        for local, global_index in enumerate(indices):
            rows[global_index]["class_area_robust_z"] = float(area_z[local])
            rows[global_index]["boundary_complexity_robust_z"] = float(complexity_z[local])

    for row in rows:
        area_z = abs(float(row["class_area_robust_z"]))
        complexity_z = max(0.0, float(row["boundary_complexity_robust_z"]))
        components = int(row["component_count"])
        largest_ratio = float(row["largest_component_ratio"])
        tiny_fraction = float(row["tiny_component_fraction"])
        edge_alignment = float(row["edge_alignment"])
        sides = int(row["sides_touched"])
        area = float(row["foreground_fraction"])
        score = min(area_z, 6.0) * 1.2
        score += math.log1p(max(0, components - 1)) * 1.8
        score += max(0.0, 0.75 - largest_ratio) * 5.0
        score += min(tiny_fraction, 0.15) * 25.0
        score += min(complexity_z, 6.0) * 0.8
        score += max(0.0, 1.0 - edge_alignment) * 3.0
        score += max(0, sides - 2) * 0.6
        score += 2.0 if area < 0.03 or area > 0.94 else 0.0
        row["anomaly_score"] = score
        row["reasons"] = "; ".join(reasons(row)) or "combined heuristic rank"

    ranked = sorted(rows, key=lambda row: float(row["anomaly_score"]), reverse=True)
    candidates = unique_ranked(rows, int(args.top_k))
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "all_mask_features.csv", ranked)
    write_csv(output_root / "candidate_masks.csv", candidates)
    sheets = make_sheets(root, output_root / "sheets", candidates, int(args.sheet_size))
    summary = {
        "images": len(rows),
        "candidates": len(candidates),
        "sheets": [str(path) for path in sheets],
        "note": "Candidates require visual review; heuristics do not imply annotation error.",
    }
    (output_root / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] candidates: {output_root / 'candidate_masks.csv'}")
    print(f"[DONE] sheets: {output_root / 'sheets'}")


if __name__ == "__main__":
    main()
