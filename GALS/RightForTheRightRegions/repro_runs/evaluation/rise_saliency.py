#!/usr/bin/env python3
"""Generate simple RISE saliency maps for a checkpoint and image manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_loading import build_model, logits_from_output


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"image_path", "label"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return rows


def sample_id(row: Dict[str, str], index: int) -> str:
    raw = row.get("sample_id") or row.get("id") or Path(row["image_path"]).stem
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(raw))
    return f"{index:04d}_{safe}"


def load_image(path: str, image_size: int) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    with Image.open(path) as im:
        return tfm(im.convert("RGB"))


def load_image_rgb(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def generate_masks(
    num_masks: int,
    grid_size: int,
    image_size: int,
    p1: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    cell = math.ceil(image_size / grid_size)
    up_size = (grid_size + 1) * cell
    masks = np.empty((num_masks, image_size, image_size), dtype=np.float32)

    for i in range(num_masks):
        small = (rng.random((grid_size, grid_size)) < p1).astype(np.float32)
        small = torch.from_numpy(small)[None, None]
        up = F.interpolate(small, size=(up_size, up_size), mode="bilinear", align_corners=False)[0, 0]
        x0 = int(rng.integers(0, up_size - image_size + 1))
        y0 = int(rng.integers(0, up_size - image_size + 1))
        masks[i] = up[y0 : y0 + image_size, x0 : x0 + image_size].numpy()

    return torch.from_numpy(masks[:, None]).to(device)


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - float(arr.min())
    denom = float(arr.max())
    if denom <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / denom


def save_gray(path: Path, arr: np.ndarray) -> None:
    arr = (normalize_map(arr) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def colorize(arr: np.ndarray) -> np.ndarray:
    try:
        import matplotlib.cm as cm

        rgba = cm.get_cmap("jet")(normalize_map(arr))
        return (rgba[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        gray = (normalize_map(arr) * 255).astype(np.uint8)
        return np.stack([gray, np.zeros_like(gray), 255 - gray], axis=-1)


def save_overlay(path: Path, image_rgb: np.ndarray, saliency: np.ndarray, alpha: float) -> None:
    heat = colorize(saliency)
    out = (image_rgb.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha)
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(path)


@torch.no_grad()
def rise_for_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    masks: torch.Tensor,
    target_mode: str,
    label: int,
    batch_size: int,
    p1: float,
) -> tuple[np.ndarray, int, float]:
    device = masks.device
    image = image.to(device)
    logits = logits_from_output(model(image[None]))
    pred = int(logits.argmax(dim=1).item())
    target = pred if target_mode == "predicted" else int(label)

    total = torch.zeros(image.shape[-2:], device=device)
    score_total = 0.0
    for start in range(0, masks.shape[0], batch_size):
        batch_masks = masks[start : start + batch_size]
        masked = image[None] * batch_masks
        scores = torch.softmax(logits_from_output(model(masked)), dim=1)[:, target]
        total += (scores[:, None, None] * batch_masks[:, 0]).sum(dim=0)
        score_total += float(scores.sum().item())

    sal = total / max(float(masks.shape[0]) * float(p1), 1e-12)
    return sal.detach().cpu().numpy(), pred, score_total / max(int(masks.shape[0]), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV with image_path,label[,sample_id,mask_path,group].")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arch", default="resnet50_cam")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained", action="store_true", help="Initialize torchvision pretrained weights before checkpoint load.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-masks", type=int, default=2000)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--p1", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-mode", choices=["label", "predicted"], default="label")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(Path(args.manifest))

    device = torch.device(args.device)
    model, model_meta = build_model(
        arch=args.arch,
        num_classes=args.num_classes,
        checkpoint=args.checkpoint,
        pretrained=args.pretrained,
        device=device,
    )
    masks = generate_masks(args.num_masks, args.grid_size, args.image_size, args.p1, device, args.seed)

    manifest_out = out_root / "saliency_manifest.csv"
    fieldnames = list(rows[0].keys()) + ["saliency_path", "overlay_path", "prediction", "target_score_mean"]
    seen = set()
    fieldnames = [x for x in fieldnames if not (x in seen or seen.add(x))]

    with manifest_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            sid = sample_id(row, idx)
            sample_dir = out_root / sid
            sample_dir.mkdir(parents=True, exist_ok=True)
            label = int(row["label"])
            image = load_image(row["image_path"], args.image_size)
            rgb = load_image_rgb(row["image_path"], args.image_size)
            sal, pred, score_mean = rise_for_image(
                model=model,
                image=image,
                masks=masks,
                target_mode=args.target_mode,
                label=label,
                batch_size=args.batch_size,
                p1=args.p1,
            )
            npy_path = sample_dir / "rise.npy"
            png_path = sample_dir / "rise.png"
            overlay_path = sample_dir / "overlay.png"
            np.save(npy_path, sal.astype(np.float32))
            save_gray(png_path, sal)
            save_overlay(overlay_path, rgb, sal, args.overlay_alpha)

            out_row = dict(row)
            out_row.update(
                {
                    "saliency_path": str(npy_path),
                    "overlay_path": str(overlay_path),
                    "prediction": pred,
                    "target_score_mean": f"{score_mean:.8g}",
                }
            )
            writer.writerow(out_row)
            print(f"[RISE] {idx + 1}/{len(rows)} {sid} label={label} pred={pred}", flush=True)

    summary = {
        "model": model_meta,
        "manifest": os.path.abspath(args.manifest),
        "output_dir": str(out_root.resolve()),
        "num_images": len(rows),
        "rise": {
            "num_masks": args.num_masks,
            "grid_size": args.grid_size,
            "p1": args.p1,
            "seed": args.seed,
            "target_mode": args.target_mode,
        },
    }
    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[DONE] wrote {manifest_out}")


if __name__ == "__main__":
    main()

