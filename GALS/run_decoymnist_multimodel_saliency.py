#!/usr/bin/env python3
"""Generate DecoyMNIST saliency maps for multiple pretrained checkpoints.

Models:
- guided (LeNet guided checkpoint from run_decoy_param_optuna artifacts)
- vanilla (LeNet)
- gals_rrr (LeNet RRR checkpoint)
- afr (LeNet AFR stage2 checkpoint)
- abn (ABN LeNet)
- upweight (LeNet)

The script samples a deterministic validation subset from train using the same
90/10 split convention (split seed configurable), then picks N samples per
digit and writes saliency visualizations for every model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor


# ----------------------------- Models --------------------------------------


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward_logits(x), dim=1)


class ABNLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.att_conv = nn.Conv2d(50, 1, kernel_size=1, stride=1)
        self.abn_fc = nn.Linear(50, 10)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        feat = F.relu(self.conv2(x))
        att = torch.sigmoid(self.att_conv(feat))
        feat_att = feat * (1.0 + att)
        x_main = F.max_pool2d(feat_att, 2, 2)
        x_main = x_main.view(-1, 4 * 4 * 50)
        x_main = F.relu(self.fc1(x_main))
        logits_main = self.fc2(x_main)
        return logits_main

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        feat = F.relu(self.conv2(x))
        att = torch.sigmoid(self.att_conv(feat))
        feat_att = feat * (1.0 + att)

        x_main = F.max_pool2d(feat_att, 2, 2)
        x_main = x_main.view(-1, 4 * 4 * 50)
        x_main = F.relu(self.fc1(x_main))
        logits_main = self.fc2(x_main)

        feat_aux = (feat * att).mean(dim=(2, 3))
        logits_aux = self.abn_fc(feat_aux)

        return F.log_softmax(logits_main, dim=1), logits_aux


@dataclass
class ModelSpec:
    name: str
    ckpt_path: Path
    model: nn.Module


# -------------------------- IO + Utils -------------------------------------


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise RuntimeError("Could not extract state_dict from checkpoint payload")


def candidate_state_dicts(state: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    cands = [state]

    def strip_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in d.items():
            if k.startswith(prefix):
                out[k[len(prefix) :]] = v
            else:
                out[k] = v
        return out

    def add_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {prefix + k: v for k, v in d.items()}

    for p in ("module.", "model.", "net.", "base."):
        cands.append(strip_prefix(state, p))
    cands.append(add_prefix(state, "base."))
    cands.append(add_prefix(state, "module."))

    uniq = []
    seen = set()
    for d in cands:
        sig = tuple(sorted(d.keys()))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(d)
    return uniq


def load_checkpoint_flex(model: nn.Module, ckpt_path: Path) -> None:
    payload = _torch_load(ckpt_path)
    raw_state = extract_state_dict(payload)
    last_missing = None
    last_unexpected = None

    for cand in candidate_state_dicts(raw_state):
        missing, unexpected = model.load_state_dict(cand, strict=False)
        # Accept a candidate if it loaded at least some keys and has no gross mismatch.
        loaded_keys = len(set(model.state_dict().keys()) & set(cand.keys()))
        if loaded_keys > 0 and len(unexpected) < max(5, int(0.1 * len(cand))):
            return
        last_missing = missing
        last_unexpected = unexpected

    raise RuntimeError(
        f"Failed to align checkpoint keys for {ckpt_path}. "
        f"last_missing={last_missing} last_unexpected={last_unexpected}"
    )


def norm01(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32)
    mn = float(out.min())
    mx = float(out.max())
    if mx > mn:
        out = (out - mn) / (mx - mn)
    else:
        out = np.zeros_like(out, dtype=np.float32)
    return out


def to_u8(arr01: np.ndarray) -> np.ndarray:
    return np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def save_gray(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def heatmap_rgb(arr01: np.ndarray) -> np.ndarray:
    u8 = to_u8(arr01)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 255).astype(np.uint8)


def contours(rgb: np.ndarray, arr01: np.ndarray, thresh: float = 0.75) -> np.ndarray:
    canvas = rgb.copy()
    binary = ((arr01 >= thresh).astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(canvas, cnts, -1, (255, 255, 0), 1)
    return canvas


def save_saliency_variants(prefix: str, sal_28: np.ndarray, rgb_28: np.ndarray, out_dir: Path) -> None:
    sal = norm01(sal_28)
    h, w = rgb_28.shape[:2]
    if sal.shape != (h, w):
        sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = heatmap_rgb(sal)
    ov = overlay(rgb_28, heat, alpha=0.45)
    cnt = contours(rgb_28, sal, thresh=0.75)
    gray = to_u8(sal)
    binary = ((sal >= 0.75).astype(np.uint8) * 255)

    save_rgb(out_dir / f"{prefix}_saliency_overlay_blue_red.png", ov)
    save_rgb(out_dir / f"{prefix}_saliency_heatmap_blue_red.png", heat)
    save_gray(out_dir / f"{prefix}_saliency_grayscale_white_black.png", gray)
    save_gray(out_dir / f"{prefix}_saliency_binary_white_black.png", binary)
    save_rgb(out_dir / f"{prefix}_saliency_contours_on_image.png", cnt)


def gradcam_single(model: nn.Module, layer: nn.Module, x: torch.Tensor, target_class: int) -> np.ndarray:
    feats = []
    grads = []

    def fwd_hook(_m, _i, o):
        feats.append(o)

    def bwd_hook(_m, _gin, gout):
        grads.append(gout[0])

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model.forward_logits(x)
        score = logits[:, int(target_class)].sum()
        score.backward(retain_graph=False)

        fmap = feats[-1]
        grad = grads[-1]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam_raw = (weights * fmap).sum(dim=1, keepdim=True)
        cam = F.relu(cam_raw)
        # Fallback: if ReLU fully wipes the CAM (common for weak/negative evidence),
        # use absolute raw CAM so saliency is still informative instead of all-zero.
        if float(cam.max().detach().item()) <= 0.0:
            cam = torch.abs(cam_raw)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        cam = norm01(cam)
        return cam
    finally:
        h1.remove()
        h2.remove()


# -------------------------- Dataset sampling --------------------------------


def build_val_pool_by_digit(
    png_root: Path,
    val_frac: float,
    split_seed: int,
    sample_seed: int,
) -> Dict[int, List[Tuple[int, Path, int]]]:
    full = ImageFolder(str(png_root / "train"), transform=None)
    n_total = len(full)
    n_val = int(val_frac * n_total)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(int(split_seed))
    _train_subset, val_subset = torch.utils.data.random_split(full, [n_train, n_val], generator=g)

    assert isinstance(val_subset, Subset)
    class_to_items: Dict[int, List[Tuple[int, Path, int]]] = {i: [] for i in range(10)}

    for local_idx, full_idx in enumerate(val_subset.indices):
        path_str, label = full.samples[full_idx]
        if int(label) not in class_to_items:
            continue
        class_to_items[int(label)].append((local_idx, Path(path_str), int(label)))

    rng = np.random.default_rng(sample_seed)
    for digit in range(10):
        items = class_to_items.get(digit, [])
        if len(items) <= 1:
            continue
        order = rng.permutation(len(items)).tolist()
        class_to_items[digit] = [items[i] for i in order]
    return class_to_items


def saliency_is_nonzero(sal: np.ndarray, eps: float) -> bool:
    if sal.size == 0:
        return False
    mx = float(np.max(sal))
    mass = float(np.sum(sal))
    return (mx > eps) and (mass > eps)


# ------------------------------- Main ---------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DecoyMNIST saliency maps for 6 checkpoints.")
    p.add_argument("--png-root", default="/workspace/Waterbird_Runs/MakeMNIST/data/DecoyMNIST_png")
    p.add_argument("--output-dir", default="")
    p.add_argument("--per-digit", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--target-class", choices=["label", "pred"], default="label")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--sample-filter",
        choices=["none", "all_nonzero"],
        default="all_nonzero",
        help="Filter sampled examples by saliency quality across models.",
    )
    p.add_argument(
        "--nonzero-eps",
        type=float,
        default=1e-8,
        help="Minimum saliency peak/sum threshold for non-zero filtering.",
    )
    p.add_argument(
        "--max-candidates-per-digit",
        type=int,
        default=2000,
        help="Upper bound on candidates scanned per digit when filtering.",
    )

    p.add_argument(
        "--guided-ckpt",
        default="/workspace/logsMNIST/decoy_fixed_guided_19ep_one_20260303_065125_artifacts/checkpoints/trial_0_bestVal.ckpt",
    )
    p.add_argument(
        "--vanilla-ckpt",
        default="/workspace/logsMNIST/decoy_vanilla_seed0_ckpts/decoymnist_rrr_seed0_bestval_99.60_epoch18_20260303_063411.pth",
    )
    p.add_argument(
        "--gals-ckpt",
        default="/workspace/Waterbird_Runs/GALS/DecoyMNIST_GALS_Checkpoints/decoymnist_rrr_seed4_bestval_10.75_epoch1_20260303_101952.pth",
    )
    p.add_argument(
        "--afr-ckpt",
        default="/workspace/logsMNIST/ckpts/afr/decoy_afr_seed4_stage2_20260303_225544.pth",
    )
    p.add_argument(
        "--abn-ckpt",
        default="/workspace/logsMNIST/ckpts/abn/decoy_abn_seed0_bestval_99.28_test_85.91_epoch17_20260303_230608.pth",
    )
    p.add_argument(
        "--upweight-ckpt",
        default="/workspace/logsMNIST/ckpts/upweight/decoy_upweight_seed0_bestval_99.58_test_49.98_epoch12_20260303_230554.pth",
    )
    return p.parse_args()


def resolve_device(s: str) -> torch.device:
    if s.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(s)


def ensure_exists(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} checkpoint: {path}")


def infer_pred(model: nn.Module, x: torch.Tensor) -> int:
    logits = model.forward_logits(x)
    return int(torch.argmax(logits, dim=1).item())


def run() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    png_root = Path(args.png_root).expanduser().resolve()
    if not png_root.is_dir():
        raise RuntimeError(f"Missing png-root: {png_root}")

    guided_ckpt = Path(args.guided_ckpt).expanduser().resolve()
    vanilla_ckpt = Path(args.vanilla_ckpt).expanduser().resolve()
    gals_ckpt = Path(args.gals_ckpt).expanduser().resolve()
    afr_ckpt = Path(args.afr_ckpt).expanduser().resolve()
    abn_ckpt = Path(args.abn_ckpt).expanduser().resolve()
    upweight_ckpt = Path(args.upweight_ckpt).expanduser().resolve()

    ensure_exists(guided_ckpt, "guided")
    ensure_exists(vanilla_ckpt, "vanilla")
    ensure_exists(gals_ckpt, "gals")
    ensure_exists(afr_ckpt, "afr")
    ensure_exists(abn_ckpt, "abn")
    ensure_exists(upweight_ckpt, "upweight")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("/workspace/logsMNIST") / f"decoy_multimodel_saliency_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Build models.
    guided = LeNet().to(device)
    vanilla = LeNet().to(device)
    gals = LeNet().to(device)
    afr = LeNet().to(device)
    upweight = LeNet().to(device)
    abn = ABNLeNet().to(device)

    load_checkpoint_flex(guided, guided_ckpt)
    load_checkpoint_flex(vanilla, vanilla_ckpt)
    load_checkpoint_flex(gals, gals_ckpt)
    load_checkpoint_flex(afr, afr_ckpt)
    load_checkpoint_flex(abn, abn_ckpt)
    load_checkpoint_flex(upweight, upweight_ckpt)

    guided.eval()
    vanilla.eval()
    gals.eval()
    afr.eval()
    abn.eval()
    upweight.eval()

    specs: List[ModelSpec] = [
        ModelSpec("guided", guided_ckpt, guided),
        ModelSpec("vanilla", vanilla_ckpt, vanilla),
        ModelSpec("gals", gals_ckpt, gals),
        ModelSpec("afr", afr_ckpt, afr),
        ModelSpec("abn", abn_ckpt, abn),
        ModelSpec("upweight", upweight_ckpt, upweight),
    ]

    by_digit = build_val_pool_by_digit(
        png_root=png_root,
        val_frac=float(args.val_frac),
        split_seed=int(args.split_seed),
        sample_seed=int(args.sample_seed),
    )
    if not any(len(v) > 0 for v in by_digit.values()):
        raise RuntimeError("No validation samples available.")

    tf = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda t: t * 2.0 - 1.0)])

    # Select examples per digit, optionally enforcing saliency non-zero across all models.
    selected: List[Dict[str, object]] = []
    dropped_zero = 0
    scanned_total = 0
    per_digit_shortfall: Dict[int, int] = {}

    for digit in range(10):
        items = by_digit.get(digit, [])
        want = int(args.per_digit)
        if want <= 0:
            continue
        kept_for_digit = 0
        scanned_for_digit = 0
        max_scan = min(len(items), int(args.max_candidates_per_digit))

        for (_local_idx, img_path, label) in items[:max_scan]:
            scanned_total += 1
            scanned_for_digit += 1
            pil = Image.open(img_path).convert("L")
            x = tf(pil).unsqueeze(0).to(device)
            rgb = np.repeat(np.array(pil, dtype=np.uint8)[:, :, None], 3, axis=2)

            preds: Dict[str, int] = {}
            targets: Dict[str, int] = {}
            sal_maps: Dict[str, np.ndarray] = {}
            all_nonzero = True

            for spec in specs:
                model = spec.model
                pred = infer_pred(model, x)
                target_cls = int(label) if args.target_class == "label" else int(pred)
                sal = gradcam_single(model, model.conv2, x, target_class=target_cls)
                preds[spec.name] = int(pred)
                targets[spec.name] = int(target_cls)
                sal_maps[spec.name] = sal
                if args.sample_filter == "all_nonzero" and (not saliency_is_nonzero(sal, eps=float(args.nonzero_eps))):
                    all_nonzero = False

            if args.sample_filter == "all_nonzero" and not all_nonzero:
                dropped_zero += 1
                continue

            selected.append(
                {
                    "img_path": img_path,
                    "label": int(label),
                    "rgb": rgb,
                    "preds": preds,
                    "targets": targets,
                    "sal_maps": sal_maps,
                }
            )
            kept_for_digit += 1
            if kept_for_digit >= want:
                break

        if kept_for_digit < want:
            per_digit_shortfall[digit] = want - kept_for_digit

    if not selected:
        raise RuntimeError(
            f"No samples passed filtering. filter={args.sample_filter} eps={args.nonzero_eps} scanned={scanned_total}"
        )

    rows = []
    for k, rec in enumerate(selected):
        img_path = Path(str(rec["img_path"]))
        label = int(rec["label"])
        rgb = rec["rgb"]
        preds = rec["preds"]
        targets = rec["targets"]
        sal_maps = rec["sal_maps"]

        sample_name = f"{k:03d}_digit{label}_{img_path.stem}"
        sd = samples_dir / sample_name
        sd.mkdir(parents=True, exist_ok=True)
        save_rgb(sd / "original_image.png", rgb)

        info = {
            "sample": sample_name,
            "image_path": str(img_path),
            "label": int(label),
        }

        for spec in specs:
            sal = sal_maps[spec.name]
            save_saliency_variants(spec.name, sal, rgb, sd)
            info[f"{spec.name}_pred"] = int(preds[spec.name])
            info[f"{spec.name}_target_for_saliency"] = int(targets[spec.name])

        with open(sd / "sample_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        rows.append(info)

    csv_path = out_dir / "sample_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "png_root": str(png_root),
        "output_dir": str(out_dir),
        "device": str(device),
        "per_digit": int(args.per_digit),
        "val_frac": float(args.val_frac),
        "split_seed": int(args.split_seed),
        "sample_seed": int(args.sample_seed),
        "target_class_mode": args.target_class,
        "num_samples": len(rows),
        "sample_filter": args.sample_filter,
        "nonzero_eps": float(args.nonzero_eps),
        "scanned_total": int(scanned_total),
        "dropped_zero": int(dropped_zero),
        "per_digit_shortfall": {str(k): int(v) for k, v in per_digit_shortfall.items()},
        "checkpoints": {spec.name: str(spec.ckpt_path) for spec in specs},
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[DONE] samples={len(rows)} out_dir={out_dir} "
        f"filter={args.sample_filter} scanned={scanned_total} dropped_zero={dropped_zero}"
    )
    if per_digit_shortfall:
        print(f"[WARN] per-digit shortfall={per_digit_shortfall}")
    print(f"[DONE] sample_index={csv_path}")


if __name__ == "__main__":
    run()
