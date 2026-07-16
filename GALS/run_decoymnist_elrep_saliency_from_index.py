#!/usr/bin/env python3
"""Backfill ElRep Grad-CAM saliency maps for a DecoyMNIST saliency run.

This reads an existing DecoyMNIST multimodel saliency output directory, uses
its sample_index.csv as the source of truth for the exact images, then writes
matching ElRep saliency variants for each sample.

By default this creates a new output directory. Use --write-into-source to also
write elrep_saliency_* files into the existing sample folders.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor

from run_decoymnist_multimodel_saliency import (
    LeNet,
    gradcam_single,
    infer_pred,
    load_checkpoint_flex,
    save_rgb,
    save_saliency_variants,
)


def resolve_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def read_sample_index(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing sample index: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def copy_source_run_summary(source_run: Path) -> Dict[str, object]:
    summary_path = source_run / "run_summary.json"
    if not summary_path.is_file():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ElRep saliency maps for existing DecoyMNIST samples.")
    p.add_argument(
        "--source-run",
        required=True,
        help="Existing DecoyMNIST multimodel saliency output directory containing sample_index.csv and samples/.",
    )
    p.add_argument("--elrep-ckpt", required=True)
    p.add_argument(
        "--png-root",
        default="",
        help="Optional local DecoyMNIST_png root used to recover images if sample_index.csv has stale paths.",
    )
    p.add_argument("--output-dir", default="")
    p.add_argument("--target-class", choices=["label", "pred"], default="label")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--write-into-source",
        action="store_true",
        help="Also write elrep_saliency_* files into the existing source sample folders.",
    )
    return p.parse_args()


def resolve_image_path(row: Dict[str, str], png_root: Path) -> Path:
    raw = Path(str(row["image_path"])).expanduser()
    if raw.is_file():
        return raw
    if not str(png_root):
        return raw

    sample = str(row["sample"])
    label = str(int(row["label"]))
    # sample format: 000_digit0_053257_y0. Recover the original image stem.
    parts = sample.split("_", 2)
    stem = parts[2] if len(parts) >= 3 else Path(str(row["image_path"])).stem
    for split in ("train", "test", "val"):
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = png_root / split / label / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
    for ext in (".png", ".jpg", ".jpeg"):
        matches = list(png_root.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return raw


def main() -> None:
    args = parse_args()
    source_run = Path(args.source_run).expanduser().resolve()
    source_samples = source_run / "samples"
    if not source_samples.is_dir():
        raise FileNotFoundError(f"Missing source samples directory: {source_samples}")

    elrep_ckpt = Path(args.elrep_ckpt).expanduser().resolve()
    if not elrep_ckpt.is_file():
        raise FileNotFoundError(f"Missing ElRep checkpoint: {elrep_ckpt}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = source_run.parent / f"{source_run.name}_elrep_backfill_{ts}"
    output_samples = output_dir / "samples"
    output_samples.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    png_root = Path(args.png_root).expanduser().resolve() if args.png_root else Path("")
    model = LeNet().to(device)
    load_checkpoint_flex(model, elrep_ckpt)
    model.eval()

    rows = read_sample_index(source_run / "sample_index.csv")
    tf = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda t: t * 2.0 - 1.0)])

    out_rows: List[Dict[str, object]] = []
    for row in rows:
        sample = str(row["sample"])
        image_path = resolve_image_path(row, png_root)
        label = int(row["label"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing DecoyMNIST image for sample={sample}: {image_path}")

        pil = Image.open(image_path).convert("L")
        rgb = np.repeat(np.array(pil, dtype=np.uint8)[:, :, None], 3, axis=2)
        x = tf(pil).unsqueeze(0).to(device)

        pred = infer_pred(model, x)
        target_cls = label if args.target_class == "label" else pred
        sal = gradcam_single(model, model.conv2, x, target_class=target_cls)

        dest = output_samples / sample
        dest.mkdir(parents=True, exist_ok=True)
        save_rgb(dest / "original_image.png", rgb)
        save_saliency_variants("elrep", sal, rgb, dest)

        source_dest = source_samples / sample
        if args.write_into_source:
            if not source_dest.is_dir():
                raise FileNotFoundError(f"Missing source sample folder: {source_dest}")
            save_saliency_variants("elrep", sal, rgb, source_dest)
            info_path = source_dest / "sample_info.json"
            if info_path.is_file():
                with info_path.open("r", encoding="utf-8") as f:
                    info = json.load(f)
                info["elrep_pred"] = int(pred)
                info["elrep_target_for_saliency"] = int(target_cls)
                info["elrep_checkpoint"] = str(elrep_ckpt)
                with info_path.open("w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2)

        info = {
            "sample": sample,
            "image_path": str(image_path),
            "label": int(label),
            "elrep_pred": int(pred),
            "elrep_target_for_saliency": int(target_cls),
            "output_sample_dir": str(dest),
            "source_sample_dir": str(source_dest),
        }
        with (dest / "sample_info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        out_rows.append(info)

    index_path = output_dir / "sample_index.csv"
    fieldnames = [
        "sample",
        "image_path",
        "label",
        "elrep_pred",
        "elrep_target_for_saliency",
        "output_sample_dir",
        "source_sample_dir",
    ]
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {
        "source_run": str(source_run),
        "source_summary": copy_source_run_summary(source_run),
        "elrep_checkpoint": str(elrep_ckpt),
        "output_dir": str(output_dir),
        "target_class": args.target_class,
        "write_into_source": bool(args.write_into_source),
        "num_samples": len(out_rows),
        "saliency_method": "Grad-CAM",
        "saliency_layer": "LeNet.conv2",
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] wrote ElRep saliency maps for {len(out_rows)} DecoyMNIST samples")
    print(f"[DONE] output_dir={output_dir}")
    print(f"[DONE] sample_index={index_path}")
    if args.write_into_source:
        print(f"[DONE] also wrote elrep_saliency_* into source samples: {source_samples}")


if __name__ == "__main__":
    main()
