#!/usr/bin/env python3
"""
Train/evaluate Waterbirds models with fixed hyperparameters and generate saliency
visualizations for sampled validation images.

Models supported:
- guided (run_guided_waterbird.py)
- vanilla (run_vanilla_waterbird.py)
- gals_vit (main.py + configs/waterbirds_*_gals_vit.yaml)
"""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import run_guided_waterbird as rgw  # noqa: E402
import run_vanilla_waterbird as rvw  # noqa: E402
from models.resnet import resnet50 as gals_resnet50  # noqa: E402


GROUP_NAMES = ["Land_on_Land", "Land_on_Water", "Water_on_Land", "Water_on_Water"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
FLOAT_RE = re.compile(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_map(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32)
    out -= out.min()
    mx = out.max()
    if mx > 1e-8:
        out /= mx
    return out


def map_to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def heatmap_rgb(norm_map: np.ndarray) -> np.ndarray:
    u8 = map_to_u8(norm_map)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay_rgb(base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip((1.0 - alpha) * base_rgb + alpha * heat_rgb, 0, 255).astype(np.uint8)


def contour_overlay(base_rgb: np.ndarray, norm_map: np.ndarray, threshold: float = 0.75) -> np.ndarray:
    canvas = base_rgb.copy()
    binary = (norm_map >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(canvas, contours, -1, (255, 255, 0), 2)
    return canvas


def safe_token(text: str) -> str:
    token = text.replace("\\", "__").replace("/", "__").replace(".", "_")
    token = token.replace(" ", "_").replace(":", "_")
    return token[:180]


def save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def save_gray(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def open_rgb_with_retry(path: Path, retries: int = 5, sleep_s: float = 0.2) -> Image.Image:
    return rgw._open_pil_with_retry(str(path), mode="RGB", retries=retries, sleep_s=sleep_s)


def open_gray_with_retry(path: Path, retries: int = 5, sleep_s: float = 0.2) -> Image.Image:
    return rgw._open_pil_with_retry(str(path), mode="L", retries=retries, sleep_s=sleep_s)


def resize_map(norm_map: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(norm_map, (width, height), interpolation=cv2.INTER_LINEAR)


def select_val_rows(metadata_df: pd.DataFrame, num_samples: int, seed: int, strategy: str) -> pd.DataFrame:
    val_df = metadata_df[metadata_df["split"] == 1].copy()
    if len(val_df) == 0:
        raise RuntimeError("No validation rows found in metadata.csv (split == 1).")

    val_df["y"] = val_df["y"].astype(int)
    val_df["place"] = val_df["place"].astype(int)
    val_df["group"] = val_df["y"] * 2 + val_df["place"]

    n = min(num_samples, len(val_df))
    rng = np.random.default_rng(seed)

    if strategy == "random":
        return val_df.sample(n=n, random_state=seed).reset_index(drop=True)

    groups = sorted(val_df["group"].unique().tolist())
    base = n // len(groups)
    rem = n % len(groups)

    picked_idx: List[int] = []
    for i, g in enumerate(groups):
        g_df = val_df[val_df["group"] == g]
        want = base + (1 if i < rem else 0)
        take = min(want, len(g_df))
        if take > 0:
            chosen = rng.choice(g_df.index.to_numpy(), size=take, replace=False)
            picked_idx.extend(chosen.tolist())

    if len(picked_idx) < n:
        remaining = val_df.drop(index=picked_idx)
        need = min(n - len(picked_idx), len(remaining))
        if need > 0:
            chosen = rng.choice(remaining.index.to_numpy(), size=need, replace=False)
            picked_idx.extend(chosen.tolist())

    out = val_df.loc[picked_idx].copy()
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


class GALSBinaryCAMModel(nn.Module):
    """
    Wrapper around repo ResNet with return_fmaps=True and binary (single-logit) head.
    """

    def __init__(self):
        super().__init__()
        self.net = gals_resnet50(pretrained=False, return_fmaps=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, fmaps = self.net(x)
        return logits, fmaps


class GuidedProbModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return torch.softmax(logits, dim=1)


class VanillaProbModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.softmax(logits, dim=1)


class GALSProbModel(nn.Module):
    def __init__(self, model: GALSBinaryCAMModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        prob_1 = torch.sigmoid(logits[:, 0:1])
        prob_0 = 1.0 - prob_1
        return torch.cat([prob_0, prob_1], dim=1)


class RISEExplainer(nn.Module):
    def __init__(self, prob_model: nn.Module, masks: torch.Tensor, gpu_batch: int, p1: float):
        super().__init__()
        self.prob_model = prob_model
        self.masks = masks
        self.gpu_batch = int(gpu_batch)
        self.p1 = float(p1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_masks = int(self.masks.shape[0])
        _, _, h, w = x.size()
        stack = torch.mul(self.masks, x.data)

        probs: List[torch.Tensor] = []
        for i in range(0, n_masks, self.gpu_batch):
            probs.append(self.prob_model(stack[i : min(i + self.gpu_batch, n_masks)]))
        p = torch.cat(probs, dim=0)

        num_classes = int(p.size(1))
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(n_masks, h * w))
        sal = sal.view((num_classes, h, w))
        sal = sal / float(n_masks) / self.p1
        return sal


def generate_rise_masks_array(
    num_masks: int,
    input_size: Tuple[int, int],
    grid_size: int,
    p1: float,
    seed: int,
) -> np.ndarray:
    h, w = input_size
    cell_h = int(np.ceil(float(h) / float(grid_size)))
    cell_w = int(np.ceil(float(w) / float(grid_size)))
    up_h = (grid_size + 1) * cell_h
    up_w = (grid_size + 1) * cell_w

    rng = np.random.default_rng(seed)
    grid = (rng.random((num_masks, grid_size, grid_size)) < p1).astype(np.float32)
    masks = np.empty((num_masks, 1, h, w), dtype=np.float32)

    for i in range(num_masks):
        x = int(rng.integers(0, cell_h))
        y = int(rng.integers(0, cell_w))
        upsampled = cv2.resize(grid[i], (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        masks[i, 0] = upsampled[x : x + h, y : y + w]

    return masks


def load_or_create_rise_masks(
    mask_path: Path,
    num_masks: int,
    input_size: Tuple[int, int],
    grid_size: int,
    p1: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    expected_shape = (num_masks, 1, input_size[0], input_size[1])
    masks_np: Optional[np.ndarray] = None

    if mask_path.is_file():
        loaded = np.load(mask_path)
        if loaded.shape == expected_shape:
            masks_np = loaded.astype(np.float32)
        else:
            print(
                f"[WARN] Existing RISE mask file has shape {loaded.shape}, expected {expected_shape}. Regenerating.",
                flush=True,
            )

    if masks_np is None:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        masks_np = generate_rise_masks_array(
            num_masks=num_masks,
            input_size=input_size,
            grid_size=grid_size,
            p1=p1,
            seed=seed,
        )
        np.save(mask_path, masks_np)

    return torch.from_numpy(masks_np).float().to(device)


def build_rise_explainer(
    prob_model: nn.Module,
    masks: torch.Tensor,
    input_size: Tuple[int, int],
    num_classes: int,
    gpu_batch: int,
    p1: float,
) -> RISEExplainer:
    _ = input_size
    _ = num_classes
    explainer = RISEExplainer(prob_model=prob_model, masks=masks, gpu_batch=gpu_batch, p1=p1)
    return explainer


def extract_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        # Could already be a state_dict mapping
        if ckpt_obj:
            sample_key = next(iter(ckpt_obj.keys()))
            if isinstance(sample_key, str):
                return ckpt_obj  # type: ignore[return-value]
    raise RuntimeError("Could not extract model state_dict from checkpoint object.")


def align_state_dict_keys(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    model_keys = list(model.state_dict().keys())
    if not model_keys:
        return state_dict
    model_has_module = model_keys[0].startswith("module.")

    ckpt_keys = list(state_dict.keys())
    if not ckpt_keys:
        return state_dict
    ckpt_has_module = ckpt_keys[0].startswith("module.")

    if ckpt_has_module and not model_has_module:
        return {k[7:]: v for k, v in state_dict.items() if k.startswith("module.")}
    if model_has_module and not ckpt_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def find_best_ckpt(run_dir: Path) -> Path:
    patterns = [
        "best_balanced_valacc_*.ckpt",
        "best_valacc_*.ckpt",
        "*.ckpt",
    ]
    for pat in patterns:
        candidates = list(run_dir.glob(pat))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    raise RuntimeError(f"No checkpoint file found in run dir: {run_dir}")


def _parse_metric_line(line: str) -> Optional[Tuple[str, float]]:
    parts = line.strip().split(":")
    if len(parts) < 2:
        return None
    key = parts[0].strip()
    if not key:
        return None
    m = FLOAT_RE.search(line)
    if m is None:
        return None
    return key, float(m.group(1))


def run_command_with_log(cmd: List[str], cwd: Path, log_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as lf:
        lf.write("[CMD] " + " ".join(cmd) + "\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None

        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            print(line, end="", flush=True)
            parsed = _parse_metric_line(line)
            if parsed is not None:
                k, v = parsed
                metrics[k] = v

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with code {rc}. See log: {log_path}")

    return metrics


def scale_if_fraction(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * 100.0 if value <= 1.0 else value


def parse_gals_group_metrics(metrics: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    per_group = scale_if_fraction(metrics.get("balanced_test_acc"))

    group_values: Dict[str, float] = {}
    for gname in GROUP_NAMES:
        key = f"{gname}_test_acc"
        if key in metrics:
            group_values[gname] = scale_if_fraction(metrics[key])  # type: ignore[arg-type]

    worst_group = min(group_values.values()) if group_values else None
    return per_group, worst_group, group_values


def train_guided(args, out_dir: Path) -> Dict[str, object]:
    if args.guided_ckpt and Path(args.guided_ckpt).is_file():
        return {
            "checkpoint": str(Path(args.guided_ckpt).resolve()),
            "best_balanced_val_acc": None,
            "test_acc": None,
            "per_group": None,
            "worst_group": None,
            "from_existing_checkpoint": True,
        }

    rgw.SEED = int(args.guided_seed)
    rgw.base_lr = float(args.guided_base_lr)
    rgw.classifier_lr = float(args.guided_classifier_lr)
    rgw.lr2_mult = float(args.guided_lr2_mult)
    rgw.checkpoint_dir = str(out_dir / "checkpoints" / "guided")

    old_workers = os.environ.get("GUIDED_NUM_WORKERS")
    old_save = os.environ.get("SAVE_CHECKPOINTS")
    os.environ["GUIDED_NUM_WORKERS"] = str(args.guided_num_workers)
    os.environ["SAVE_CHECKPOINTS"] = "1"

    try:
        run_args = SimpleNamespace(data_path=args.data_path, gt_path=args.guided_gt_root)
        best_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(
            run_args,
            int(args.guided_attention_epoch),
            float(args.guided_kl_lambda),
            float(args.guided_kl_incr),
        )
    finally:
        if old_workers is None:
            os.environ.pop("GUIDED_NUM_WORKERS", None)
        else:
            os.environ["GUIDED_NUM_WORKERS"] = old_workers
        if old_save is None:
            os.environ.pop("SAVE_CHECKPOINTS", None)
        else:
            os.environ["SAVE_CHECKPOINTS"] = old_save

    return {
        "checkpoint": ckpt,
        "best_balanced_val_acc": best_val,
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "from_existing_checkpoint": False,
    }


def train_vanilla(args, out_dir: Path) -> Dict[str, object]:
    if args.vanilla_ckpt and Path(args.vanilla_ckpt).is_file():
        return {
            "checkpoint": str(Path(args.vanilla_ckpt).resolve()),
            "best_balanced_val_acc": None,
            "test_acc": None,
            "per_group": None,
            "worst_group": None,
            "from_existing_checkpoint": True,
        }

    run_args = SimpleNamespace(
        data_path=args.data_path,
        seed=int(args.vanilla_seed),
        model="resnet50",
        pretrained=True,
        batch_size=96,
        num_epochs=200,
        lr=float(args.vanilla_base_lr),
        base_lr=float(args.vanilla_base_lr),
        classifier_lr=float(args.vanilla_classifier_lr),
        momentum=float(args.vanilla_momentum),
        weight_decay=float(args.vanilla_weight_decay),
        nesterov=bool(args.vanilla_nesterov),
        num_workers=int(args.vanilla_num_workers),
        checkpoint_dir=str(out_dir / "checkpoints" / "vanilla"),
    )

    old_save = os.environ.get("SAVE_CHECKPOINTS")
    os.environ["SAVE_CHECKPOINTS"] = "1"
    try:
        best_val, test_acc, per_group, worst_group, ckpt = rvw.run_single(run_args)
    finally:
        if old_save is None:
            os.environ.pop("SAVE_CHECKPOINTS", None)
        else:
            os.environ["SAVE_CHECKPOINTS"] = old_save

    return {
        "checkpoint": ckpt,
        "best_balanced_val_acc": best_val,
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "from_existing_checkpoint": False,
    }


def train_gals_vit(args, out_dir: Path) -> Dict[str, object]:
    if not args.run_gals:
        return {
            "enabled": False,
            "checkpoint": None,
            "best_balanced_val_acc": None,
            "test_acc": None,
            "per_group": None,
            "worst_group": None,
            "group_acc": None,
            "from_existing_checkpoint": False,
            "log_path": None,
        }

    if args.gals_ckpt and Path(args.gals_ckpt).is_file():
        return {
            "enabled": True,
            "checkpoint": str(Path(args.gals_ckpt).resolve()),
            "best_balanced_val_acc": None,
            "test_acc": None,
            "per_group": None,
            "worst_group": None,
            "group_acc": None,
            "from_existing_checkpoint": True,
            "log_path": None,
        }

    script_dir = Path(__file__).resolve().parent
    run_name = args.gals_run_name or f"gals_vit_fixed_{safe_token(args.gals_waterbirds_dir)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = script_dir / "trained_weights" / "waterbirds" / run_name
    log_path = out_dir / "logs" / f"{run_name}.log"

    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--config",
        str(args.gals_config),
        "--name",
        run_name,
        f"DATA.ROOT={args.gals_data_root}",
        f"DATA.WATERBIRDS_DIR={args.gals_waterbirds_dir}",
        f"DATA.ATTENTION_DIR={args.gals_attention_dir}",
        f"DATA.NUM_WORKERS={int(args.gals_num_workers)}",
        f"SEED={int(args.gals_seed)}",
        f"EXP.BASE.LR={float(args.gals_base_lr)}",
        f"EXP.CLASSIFIER.LR={float(args.gals_classifier_lr)}",
        f"EXP.WEIGHT_DECAY={float(args.gals_weight_decay)}",
        f"EXP.MOMENTUM={float(args.gals_momentum)}",
        "EXP.LOSSES.GRADIENT_OUTSIDE.COMPUTE=True",
        "EXP.LOSSES.GRADIENT_OUTSIDE.LOG=False",
        f"EXP.LOSSES.GRADIENT_OUTSIDE.WEIGHT={float(args.gals_grad_weight)}",
        f"EXP.LOSSES.GRADIENT_OUTSIDE.CRITERION={args.gals_grad_criterion}",
        "EXP.LOSSES.GRADCAM.COMPUTE=False",
        "EXP.LOSSES.GRADCAM.LOG=False",
        "EXP.AUX_LOSSES_ON_VAL=False",
        "LOGGING.SAVE_BEST=True",
        "LOGGING.SAVE_LAST=False",
        "LOGGING.SAVE_EVERY=0",
        "LOGGING.SAVE_STATS_PATH=NONE",
    ]

    metrics = run_command_with_log(cmd=cmd, cwd=script_dir, log_path=log_path)
    ckpt_path = find_best_ckpt(run_dir)

    per_group, worst_group, group_acc = parse_gals_group_metrics(metrics)
    test_acc = scale_if_fraction(metrics.get("test_acc"))
    best_balanced_val = scale_if_fraction(metrics.get("balanced_val_acc"))

    return {
        "enabled": True,
        "checkpoint": str(ckpt_path.resolve()),
        "best_balanced_val_acc": best_balanced_val,
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "group_acc": group_acc,
        "from_existing_checkpoint": False,
        "log_path": str(log_path),
    }


def save_saliency_variants(prefix: str, saliency_224: np.ndarray, image_rgb: np.ndarray, sample_dir: Path) -> Dict[str, np.ndarray]:
    h, w = image_rgb.shape[:2]
    sal = resize_map(normalize_map(saliency_224), w, h)
    sal_u8 = map_to_u8(sal)
    heat = heatmap_rgb(sal)
    overlay = overlay_rgb(image_rgb, heat, alpha=0.45)
    contour = contour_overlay(image_rgb, sal, threshold=0.75)
    binary = ((sal >= 0.75).astype(np.uint8) * 255)

    save_rgb(sample_dir / f"{prefix}_saliency_overlay_blue_red.png", overlay)
    save_rgb(sample_dir / f"{prefix}_saliency_heatmap_blue_red.png", heat)
    save_gray(sample_dir / f"{prefix}_saliency_grayscale_white_black.png", sal_u8)
    save_gray(sample_dir / f"{prefix}_saliency_binary_white_black.png", binary)
    save_rgb(sample_dir / f"{prefix}_saliency_contours_on_image.png", contour)

    return {
        "overlay": overlay,
        "heatmap": heat,
        "gray": np.repeat(sal_u8[:, :, None], 3, axis=2),
        "contour": contour,
    }


def save_gt_mask_variants(mask_path: Path, image_rgb: np.ndarray, sample_dir: Path) -> bool:
    if not mask_path.is_file():
        return False

    mask = np.array(open_gray_with_retry(mask_path), dtype=np.float32)
    h, w = image_rgb.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = normalize_map(mask)
    mask_u8 = map_to_u8(mask)
    heat = heatmap_rgb(mask)
    overlay = overlay_rgb(image_rgb, heat, alpha=0.45)
    contour = contour_overlay(image_rgb, mask, threshold=0.5)
    binary = ((mask >= 0.5).astype(np.uint8) * 255)

    save_gray(sample_dir / "gt_mask_grayscale_white_black.png", mask_u8)
    save_gray(sample_dir / "gt_mask_binary_white_black.png", binary)
    save_rgb(sample_dir / "gt_mask_overlay_blue_red.png", overlay)
    save_rgb(sample_dir / "gt_mask_heatmap_blue_red.png", heat)
    save_rgb(sample_dir / "gt_mask_contours_on_image.png", contour)
    return True


def load_guided_model(guided_ckpt: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = rgw.make_cam_model(num_classes, model_name="resnet50", pretrained=True).to(device)
    state = torch.load(guided_ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_vanilla_model(vanilla_ckpt: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = rvw.make_model("resnet50", num_classes, pretrained=True).to(device)
    state = torch.load(vanilla_ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_gals_model(gals_ckpt: Path, device: torch.device) -> GALSBinaryCAMModel:
    model = GALSBinaryCAMModel().to(device)
    ckpt = torch.load(gals_ckpt, map_location=device)
    state = extract_state_dict(ckpt)
    state = align_state_dict_keys(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading GALS checkpoint: {missing}", flush=True)
    if unexpected:
        print(f"[WARN] Unexpected keys while loading GALS checkpoint: {unexpected}", flush=True)
    model.eval()
    return model


def write_comparison_panels(sample_dir: Path, vis_by_model: Dict[str, Dict[str, np.ndarray]]) -> None:
    model_names = list(vis_by_model.keys())
    if len(model_names) < 2:
        return

    viz_keys = ["overlay", "heatmap", "gray", "contour"]

    for key in viz_keys:
        # Pairwise
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a = model_names[i]
                b = model_names[j]
                pair = np.concatenate([vis_by_model[a][key], vis_by_model[b][key]], axis=1)
                save_rgb(sample_dir / f"pair_{a}_vs_{b}_{key}.png", pair)

        # All-model strip
        strip = np.concatenate([vis_by_model[m][key] for m in model_names], axis=1)
        save_rgb(sample_dir / f"all_models_{'_'.join(model_names)}_{key}.png", strip)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train fixed Waterbirds guided/vanilla/GALS-ViT models and produce saliency visualizations."
    )
    p.add_argument(
        "--data-path",
        default="/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2",
        help="Waterbirds dataset root containing metadata.csv.",
    )
    p.add_argument(
        "--guided-gt-root",
        default="/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap",
        help="Mask root used for guided training and GT visualization lookup.",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Output directory. If empty, creates timestamped folder under <dataset_parent>/logsWaterbird/.",
    )
    p.add_argument("--num-val-samples", type=int, default=150)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--sample-strategy", choices=["balanced", "random"], default="balanced")
    p.add_argument("--target-class", choices=["label", "pred"], default="label")
    p.add_argument(
        "--saliency-method",
        choices=["rise"],
        default="rise",
        help="Saliency explainer used for map generation. This script currently supports only RISE.",
    )
    p.add_argument("--rise-num-masks", type=int, default=2000)
    p.add_argument("--rise-grid-size", type=int, default=8)
    p.add_argument("--rise-p1", type=float, default=0.1)
    p.add_argument("--rise-gpu-batch", type=int, default=16)
    p.add_argument("--rise-seed", type=int, default=0)
    p.add_argument(
        "--rise-masks-path",
        default="",
        help="Optional .npy mask-bank path for RISE. If unset, uses <output_dir>/rise_masks_*.npy.",
    )

    # Guided fixed params (WB100 defaults)
    p.add_argument("--guided-seed", type=int, default=0)
    p.add_argument("--guided-attention-epoch", type=int, default=73)
    p.add_argument("--guided-kl-lambda", type=float, default=495.60509512105125)
    p.add_argument("--guided-kl-incr", type=float, default=0.0)
    p.add_argument("--guided-base-lr", type=float, default=5.721272343067835e-05)
    p.add_argument("--guided-classifier-lr", type=float, default=0.003571025068268372)
    p.add_argument("--guided-lr2-mult", type=float, default=0.122728778135035)
    p.add_argument("--guided-num-workers", type=int, default=0)
    p.add_argument("--guided-ckpt", default="", help="Optional existing guided checkpoint to skip guided training.")

    # Vanilla fixed params (WB100 defaults from user trial)
    p.add_argument("--vanilla-seed", type=int, default=0)
    p.add_argument("--vanilla-base-lr", type=float, default=0.031210590691245817)
    p.add_argument("--vanilla-classifier-lr", type=float, default=0.0008517287145349147)
    p.add_argument("--vanilla-momentum", type=float, default=0.8914661939990524)
    p.add_argument("--vanilla-weight-decay", type=float, default=1e-5)
    p.add_argument("--vanilla-nesterov", action="store_true", default=False)
    p.add_argument("--vanilla-num-workers", type=int, default=0)
    p.add_argument("--vanilla-ckpt", default="", help="Optional existing vanilla checkpoint to skip vanilla training.")

    # GALS-ViT fixed params (WB100 defaults from user trial)
    p.add_argument("--run-gals", dest="run_gals", action="store_true", default=True)
    p.add_argument("--no-gals", dest="run_gals", action="store_false")
    p.add_argument("--gals-seed", type=int, default=0)
    p.add_argument("--gals-config", default="configs/waterbirds_100_gals_vit.yaml")
    p.add_argument("--gals-data-root", default="/home/ryreu/guided_cnn/waterbirds")
    p.add_argument("--gals-waterbirds-dir", default="waterbird_1.0_forest2water2")
    p.add_argument("--gals-attention-dir", default="clip_vit_attention")
    p.add_argument("--gals-base-lr", type=float, default=0.01906797717764731)
    p.add_argument("--gals-classifier-lr", type=float, default=0.0005160317852697908)
    p.add_argument("--gals-grad-weight", type=float, default=97631.97904483072)
    p.add_argument("--gals-grad-criterion", choices=["L1", "L2"], default="L1")
    p.add_argument("--gals-weight-decay", type=float, default=1e-5)
    p.add_argument("--gals-momentum", type=float, default=0.9)
    p.add_argument("--gals-num-workers", type=int, default=0)
    p.add_argument("--gals-ckpt", default="", help="Optional existing GALS checkpoint to skip GALS training.")
    p.add_argument("--gals-run-name", default="", help="Optional explicit run name for GALS training.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.sample_seed)

    data_path = Path(args.data_path).resolve()
    gt_root = Path(args.guided_gt_root).resolve()

    if not data_path.is_dir():
        raise RuntimeError(f"Missing data path: {data_path}")
    if not gt_root.is_dir():
        raise RuntimeError(f"Missing guided GT root: {gt_root}")

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = data_path.parent / "logsWaterbird" / f"wb_saliency_guided_vanilla_gals_{safe_token(data_path.name)}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    metadata_path = data_path / "metadata.csv"
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing metadata.csv at: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)

    print(f"[INFO] data_path={data_path}", flush=True)
    print(f"[INFO] guided_gt_root={gt_root}", flush=True)
    print(f"[INFO] output_dir={out_dir}", flush=True)

    guided_metrics = train_guided(args, out_dir)
    vanilla_metrics = train_vanilla(args, out_dir)
    gals_metrics = train_gals_vit(args, out_dir)

    guided_ckpt = Path(str(guided_metrics["checkpoint"])).resolve()
    vanilla_ckpt = Path(str(vanilla_metrics["checkpoint"])).resolve()

    if not guided_ckpt.is_file():
        raise RuntimeError(f"Guided checkpoint not found: {guided_ckpt}")
    if not vanilla_ckpt.is_file():
        raise RuntimeError(f"Vanilla checkpoint not found: {vanilla_ckpt}")

    gals_ckpt = None
    if args.run_gals:
        gals_ckpt = Path(str(gals_metrics["checkpoint"])).resolve() if gals_metrics.get("checkpoint") else None
        if gals_ckpt is None or not gals_ckpt.is_file():
            raise RuntimeError(f"GALS checkpoint not found: {gals_ckpt}")

    num_classes = int(metadata_df["y"].nunique())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    guided_model = load_guided_model(guided_ckpt, num_classes, device)
    vanilla_model = load_vanilla_model(vanilla_ckpt, num_classes, device)
    gals_model = load_gals_model(gals_ckpt, device) if gals_ckpt is not None else None

    if args.saliency_method != "rise":
        raise RuntimeError(f"Unsupported saliency method: {args.saliency_method}")

    rise_input_size = (224, 224)
    if args.rise_masks_path:
        rise_masks_path = Path(args.rise_masks_path).expanduser().resolve()
    else:
        p1_token = str(args.rise_p1).replace(".", "p")
        rise_masks_path = out_dir / f"rise_masks_n{args.rise_num_masks}_s{args.rise_grid_size}_p{p1_token}_seed{args.rise_seed}.npy"

    shared_rise_masks = load_or_create_rise_masks(
        mask_path=rise_masks_path,
        num_masks=int(args.rise_num_masks),
        input_size=rise_input_size,
        grid_size=int(args.rise_grid_size),
        p1=float(args.rise_p1),
        seed=int(args.rise_seed),
        device=device,
    )

    guided_rise = build_rise_explainer(
        prob_model=GuidedProbModel(guided_model).to(device).eval(),
        masks=shared_rise_masks,
        input_size=rise_input_size,
        num_classes=num_classes,
        gpu_batch=int(args.rise_gpu_batch),
        p1=float(args.rise_p1),
    )
    vanilla_rise = build_rise_explainer(
        prob_model=VanillaProbModel(vanilla_model).to(device).eval(),
        masks=shared_rise_masks,
        input_size=rise_input_size,
        num_classes=num_classes,
        gpu_batch=int(args.rise_gpu_batch),
        p1=float(args.rise_p1),
    )
    gals_rise = None
    if gals_model is not None:
        gals_rise = build_rise_explainer(
            prob_model=GALSProbModel(gals_model).to(device).eval(),
            masks=shared_rise_masks,
            input_size=rise_input_size,
            num_classes=2,
            gpu_batch=int(args.rise_gpu_batch),
            p1=float(args.rise_p1),
        )

    selected = select_val_rows(
        metadata_df=metadata_df,
        num_samples=args.num_val_samples,
        seed=args.sample_seed,
        strategy=args.sample_strategy,
    )
    print(f"[INFO] Selected {len(selected)} val images for saliency generation.", flush=True)
    print(
        (
            "[INFO] Saliency method=RISE "
            f"(num_masks={int(args.rise_num_masks)} grid_size={int(args.rise_grid_size)} "
            f"p1={float(args.rise_p1)} gpu_batch={int(args.rise_gpu_batch)} seed={int(args.rise_seed)})"
        ),
        flush=True,
    )
    print(f"[INFO] RISE mask bank: {rise_masks_path}", flush=True)

    preprocess = build_preprocess()
    sample_rows: List[Dict[str, object]] = []

    for i, row in selected.iterrows():
        rel_path = str(row["img_filename"])
        img_path = data_path / rel_path
        if not img_path.is_file():
            print(f"[WARN] Missing image, skipping: {img_path}", flush=True)
            continue

        image_pil = open_rgb_with_retry(img_path)
        image_rgb = np.array(image_pil, dtype=np.uint8)
        input_tensor = preprocess(image_pil).unsqueeze(0).to(device)

        label = int(row["y"])
        place = int(row["place"])
        group = int(label * 2 + place)
        use_label_target = args.target_class == "label"

        vis_by_model: Dict[str, Dict[str, np.ndarray]] = {}
        saliency_by_model: Dict[str, np.ndarray] = {}
        info: Dict[str, object] = {
            "index": int(i),
            "img_filename": rel_path,
            "image_path": str(img_path),
            "label": label,
            "place": place,
            "group": group,
            "group_name": GROUP_NAMES[group] if 0 <= group < len(GROUP_NAMES) else str(group),
        }

        with torch.no_grad():
            # Guided
            guided_logits, _ = guided_model(input_tensor)
            guided_pred = int(guided_logits.argmax(dim=1).item())
            guided_target = label if use_label_target else guided_pred
            guided_conf = float(torch.softmax(guided_logits, dim=1)[0, guided_pred].item())
            guided_rise_sal = guided_rise(input_tensor)
            saliency_by_model["guided"] = guided_rise_sal[guided_target].detach().cpu().numpy().astype(np.float32)

            # Vanilla
            vanilla_logits = vanilla_model(input_tensor)
            vanilla_pred = int(vanilla_logits.argmax(dim=1).item())
            vanilla_target = label if use_label_target else vanilla_pred
            vanilla_conf = float(torch.softmax(vanilla_logits, dim=1)[0, vanilla_pred].item())
            vanilla_rise_sal = vanilla_rise(input_tensor)
            saliency_by_model["vanilla"] = vanilla_rise_sal[vanilla_target].detach().cpu().numpy().astype(np.float32)

            # GALS-ViT
            if gals_model is not None and gals_rise is not None:
                gals_logits, _ = gals_model(input_tensor)
                gals_prob_1 = torch.sigmoid(gals_logits[:, 0])
                gals_prob_0 = 1.0 - gals_prob_1
                gals_pred = int((gals_prob_1 >= 0.5).long().item())
                gals_conf = float(torch.maximum(gals_prob_0, gals_prob_1).item())
                gals_target = label if use_label_target else gals_pred
                gals_rise_sal = gals_rise(input_tensor)
                saliency_by_model["gals_vit"] = gals_rise_sal[gals_target].detach().cpu().numpy().astype(np.float32)
            else:
                gals_pred = None
                gals_conf = None
                gals_target = None

        sample_token = safe_token(rel_path)
        sample_dir = out_dir / "samples" / f"{i:03d}_{sample_token}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        save_rgb(sample_dir / "original_image.png", image_rgb)

        mask_name = rgw.mask_name_from_path(str(img_path))
        gt_mask_path = gt_root / mask_name
        has_mask = save_gt_mask_variants(gt_mask_path, image_rgb, sample_dir)
        if not has_mask:
            with open(sample_dir / "gt_mask_missing.txt", "w") as f:
                f.write(f"GT mask not found at: {gt_mask_path}\n")

        # Save visualizations for each model in this sample folder.
        for model_name, sal_map in saliency_by_model.items():
            vis_by_model[model_name] = save_saliency_variants(model_name, sal_map, image_rgb, sample_dir)

        write_comparison_panels(sample_dir, vis_by_model)

        info.update(
            {
                "guided_pred": guided_pred,
                "guided_confidence": guided_conf,
                "guided_saliency_target_class": guided_target,
                "vanilla_pred": vanilla_pred,
                "vanilla_confidence": vanilla_conf,
                "vanilla_saliency_target_class": vanilla_target,
                "gals_vit_pred": gals_pred,
                "gals_vit_confidence": gals_conf,
                "gals_vit_saliency_target_class": gals_target,
                "gt_mask_path": str(gt_mask_path) if has_mask else None,
            }
        )

        with open(sample_dir / "sample_info.json", "w") as f:
            json.dump(info, f, indent=2)
        sample_rows.append(info)

    summary = {
        "data_path": str(data_path),
        "guided_gt_root": str(gt_root),
        "output_dir": str(out_dir),
        "num_val_samples_requested": int(args.num_val_samples),
        "num_val_samples_generated": int(len(sample_rows)),
        "sample_strategy": args.sample_strategy,
        "target_class_mode": args.target_class,
        "saliency_method": args.saliency_method,
        "rise": {
            "num_masks": int(args.rise_num_masks),
            "grid_size": int(args.rise_grid_size),
            "p1": float(args.rise_p1),
            "gpu_batch": int(args.rise_gpu_batch),
            "seed": int(args.rise_seed),
            "masks_path": str(rise_masks_path),
        },
        "guided": guided_metrics,
        "vanilla": vanilla_metrics,
        "gals_vit": gals_metrics,
        "guided_checkpoint": str(guided_ckpt),
        "vanilla_checkpoint": str(vanilla_ckpt),
        "gals_vit_checkpoint": str(gals_ckpt) if gals_ckpt is not None else None,
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "sample_index.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "index",
            "img_filename",
            "image_path",
            "label",
            "place",
            "group",
            "group_name",
            "guided_pred",
            "guided_confidence",
            "guided_saliency_target_class",
            "vanilla_pred",
            "vanilla_confidence",
            "vanilla_saliency_target_class",
            "gals_vit_pred",
            "gals_vit_confidence",
            "gals_vit_saliency_target_class",
            "gt_mask_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    with open(out_dir / "README.txt", "w") as f:
        f.write(
            "Per-sample folders are under samples/.\n"
            "Each folder contains:\n"
            "- original_image.png\n"
            "- guided RISE saliency variants (overlay/heatmap/grayscale/binary/contours)\n"
            "- vanilla RISE saliency variants (same set)\n"
            "- gals_vit RISE saliency variants (same set, if enabled)\n"
            "- pair_* comparison panels for each visualization style\n"
            "- all_models_* strips for each visualization style\n"
            "- GT mask visualization variants when available\n"
            "- sample_info.json\n"
        )

    print(f"[DONE] Generated {len(sample_rows)} sample folders at: {out_dir / 'samples'}", flush=True)
    print(f"[DONE] Summary: {out_dir / 'run_summary.json'}", flush=True)
    print(f"[DONE] Sample index CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
