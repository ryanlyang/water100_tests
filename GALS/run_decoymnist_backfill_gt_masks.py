#!/usr/bin/env python3
"""Backfill GT-path mask visualizations into existing DecoyMNIST sample folders.

This script is intended for folders produced by:
  run_decoymnist_multimodel_saliency.py

It reads each sample directory, resolves the corresponding mask file from
--mask-root (supports .pth/.pt payloads and image masks), and writes:
  - gt_mask_grayscale_white_black.png
  - gt_mask_heatmap_blue_red.png
  - gt_mask_overlay_on_image.png
  - gt_mask_binary_white_black.png
  - gt_mask_contours_on_image.png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


_VALID_EXT = {".pth", ".pt", ".png", ".jpg", ".jpeg"}
_SPLIT_NAMES = {"train", "test", "val", "valid", "validation"}


def _norm01(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32)
    mn = float(out.min())
    mx = float(out.max())
    if mx > mn:
        out = (out - mn) / (mx - mn)
    else:
        out = np.zeros_like(out, dtype=np.float32)
    return out


def _to_u8(arr01: np.ndarray) -> np.ndarray:
    return np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _heatmap_rgb(arr01: np.ndarray) -> np.ndarray:
    u8 = _to_u8(arr01)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 255).astype(np.uint8)


def _contours(rgb: np.ndarray, arr01: np.ndarray, thresh: float = 0.75) -> np.ndarray:
    canvas = rgb.copy()
    binary = ((arr01 >= thresh).astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(canvas, cnts, -1, (255, 255, 0), 1)
    return canvas


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_attention_payload(path: Path) -> np.ndarray:
    payload = _torch_load(path)
    arr = None
    if isinstance(payload, dict):
        for key in ("attentions", "unnormalized_attentions", "attention", "cam", "saliency", "mask"):
            if key in payload:
                arr = payload[key]
                break
    if arr is None:
        arr = payload

    t = torch.as_tensor(arr, dtype=torch.float32)
    if t.ndim > 2:
        t = t.reshape(-1, t.shape[-2], t.shape[-1]).max(dim=0).values
    if t.ndim != 2:
        raise RuntimeError(f"Expected 2D mask after reduction, got shape={tuple(t.shape)} from {path}")
    return _norm01(t.detach().cpu().numpy())


def _load_mask_any(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".pth", ".pt"}:
        return _load_attention_payload(path)
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return _norm01(arr)


def _extract_sample_hint(sample_dir: Path, info: Dict[str, object]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Path]]:
    image_path = None
    split = None
    cls = None
    stem = None

    image_path_text = info.get("image_path")
    if isinstance(image_path_text, str) and image_path_text.strip():
        ip = Path(image_path_text).expanduser()
        image_path = ip
        stem = ip.stem
        cls = ip.parent.name if ip.parent else None
        if ip.parent and ip.parent.parent:
            p2 = ip.parent.parent.name.lower()
            if p2 in _SPLIT_NAMES:
                split = p2

    if stem is None:
        m = re.match(r"^\d+_digit\d+_(.+)$", sample_dir.name)
        if m:
            stem = m.group(1)
    return split, cls, stem, image_path


@dataclass
class MaskIndex:
    by_triplet: Dict[Tuple[str, str, str], List[Path]]
    by_pair: Dict[Tuple[str, str], List[Path]]
    by_stem: Dict[str, List[Path]]


def _append_index(d: Dict, key, value: Path) -> None:
    if key not in d:
        d[key] = []
    d[key].append(value)


def _build_mask_index(mask_root: Path) -> MaskIndex:
    by_triplet: Dict[Tuple[str, str, str], List[Path]] = {}
    by_pair: Dict[Tuple[str, str], List[Path]] = {}
    by_stem: Dict[str, List[Path]] = {}

    for p in mask_root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in _VALID_EXT:
            continue

        stem = p.stem
        _append_index(by_stem, stem, p)

        try:
            rel = p.relative_to(mask_root)
            parts = rel.parts
        except Exception:
            parts = ()

        split = None
        cls = None
        if len(parts) >= 3 and parts[0].lower() in _SPLIT_NAMES:
            split = parts[0].lower()
            cls = parts[1]
        elif len(parts) >= 2:
            cls = parts[-2]

        if cls is not None:
            _append_index(by_pair, (cls, stem), p)
        if split is not None and cls is not None:
            _append_index(by_triplet, (split, cls, stem), p)

    return MaskIndex(by_triplet=by_triplet, by_pair=by_pair, by_stem=by_stem)


def _path_priority(p: Path) -> Tuple[int, int, int]:
    ext = p.suffix.lower()
    # Prefer raw .pth attention maps.
    ext_rank = {".pth": 0, ".pt": 1, ".png": 2, ".jpg": 3, ".jpeg": 4}.get(ext, 9)
    return (ext_rank, len(p.parts), len(str(p)))


def _pick_best(cands: Sequence[Path]) -> Optional[Path]:
    if not cands:
        return None
    return sorted(cands, key=_path_priority)[0]


def _resolve_mask_path(
    idx: MaskIndex,
    split: Optional[str],
    cls: Optional[str],
    stem: Optional[str],
) -> Optional[Path]:
    if stem is None:
        return None
    if split is not None and cls is not None:
        p = _pick_best(idx.by_triplet.get((split, cls, stem), []))
        if p is not None:
            return p
    if cls is not None:
        p = _pick_best(idx.by_pair.get((cls, stem), []))
        if p is not None:
            return p
    return _pick_best(idx.by_stem.get(stem, []))


def _load_base_rgb(sample_dir: Path, image_path: Optional[Path]) -> np.ndarray:
    local = sample_dir / "original_image.png"
    if local.is_file():
        return np.array(Image.open(local).convert("RGB"), dtype=np.uint8)
    if image_path is not None and image_path.is_file():
        return np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    # Fallback 28x28 black canvas.
    return np.zeros((28, 28, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill GT-path mask images into DecoyMNIST saliency sample folders.")
    p.add_argument("--samples-dir", required=True, help="Path to .../decoy_multimodel_saliency_*/samples")
    p.add_argument("--mask-root", required=True, help="Path to attention mask root (e.g., clip_rn50_attention_gradcam)")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples_dir = Path(args.samples_dir).expanduser().resolve()
    mask_root = Path(args.mask_root).expanduser().resolve()
    if not samples_dir.is_dir():
        raise RuntimeError(f"Missing samples dir: {samples_dir}")
    if not mask_root.is_dir():
        raise RuntimeError(f"Missing mask root: {mask_root}")

    sample_dirs = [p for p in sorted(samples_dir.iterdir()) if p.is_dir()]
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found in {samples_dir}")

    print(f"[INFO] samples_dir={samples_dir}")
    print(f"[INFO] mask_root={mask_root}")
    print(f"[INFO] sample_count={len(sample_dirs)}")
    print("[INFO] Building mask index...")
    idx = _build_mask_index(mask_root)
    print(
        f"[INFO] index sizes: triplet={len(idx.by_triplet)} pair={len(idx.by_pair)} stem={len(idx.by_stem)}",
        flush=True,
    )

    rows: List[Dict[str, object]] = []
    done = 0
    missing = 0
    errored = 0

    for i, sd in enumerate(sample_dirs, start=1):
        info_path = sd / "sample_info.json"
        info = {}
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception:
                info = {}

        split, cls, stem, image_path = _extract_sample_hint(sd, info)
        mask_path = _resolve_mask_path(idx=idx, split=split, cls=cls, stem=stem)

        row: Dict[str, object] = {
            "sample_dir": str(sd),
            "split_hint": split or "",
            "class_hint": cls or "",
            "stem_hint": stem or "",
            "image_path": str(image_path) if image_path is not None else "",
            "mask_path": str(mask_path) if mask_path is not None else "",
            "status": "",
            "error": "",
        }

        if mask_path is None:
            row["status"] = "missing_mask"
            missing += 1
            rows.append(row)
            continue

        out_gray = sd / "gt_mask_grayscale_white_black.png"
        if out_gray.is_file() and not args.overwrite:
            row["status"] = "exists_skip"
            done += 1
            rows.append(row)
            continue

        try:
            mask01 = _load_mask_any(mask_path)
            base_rgb = _load_base_rgb(sd, image_path)
            h, w = base_rgb.shape[:2]
            if mask01.shape != (h, w):
                mask01 = cv2.resize(mask01.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                mask01 = _norm01(mask01)

            heat = _heatmap_rgb(mask01)
            ov = _overlay(base_rgb, heat, alpha=0.45)
            cnt = _contours(base_rgb, mask01, thresh=0.75)
            gray = _to_u8(mask01)
            binary = ((mask01 >= 0.75).astype(np.uint8) * 255)

            _save_gray(sd / "gt_mask_grayscale_white_black.png", gray)
            _save_rgb(sd / "gt_mask_heatmap_blue_red.png", heat)
            _save_rgb(sd / "gt_mask_overlay_on_image.png", ov)
            _save_gray(sd / "gt_mask_binary_white_black.png", binary)
            _save_rgb(sd / "gt_mask_contours_on_image.png", cnt)

            row["status"] = "ok"
            done += 1
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            errored += 1

        rows.append(row)
        if i % 20 == 0 or i == len(sample_dirs):
            print(f"[INFO] processed={i}/{len(sample_dirs)} ok={done} missing={missing} error={errored}", flush=True)

    out_csv = samples_dir.parent / "gt_mask_backfill_report.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_dir",
                "split_hint",
                "class_hint",
                "stem_hint",
                "image_path",
                "mask_path",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] ok={done} missing={missing} error={errored}")
    print(f"[DONE] report={out_csv}")


if __name__ == "__main__":
    main()
