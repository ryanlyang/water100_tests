#!/usr/bin/env python3
"""
Generate Waterbirds-style saliency outputs for RedMeat using:
- guided checkpoint (loaded from path),
- GALS checkpoint (loaded from path),
- vanilla model trained in-script with fixed hyperparameters.

Outputs include:
- per-sample model saliency variants (overlay/heatmap/grayscale/binary/contours),
- pairwise and all-model comparison strips,
- GT mask visualization variants when available,
- sample_index.csv and run_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


_THIS_DIR = Path(__file__).resolve().parent
_GALS_ROOT = _THIS_DIR.parent
if str(_GALS_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_GALS_ROOT))

from RedMeat_Runs import run_guided_redmeat as rgm  # noqa: E402
from RedMeat_Runs import run_vanilla_redmeat as rvm  # noqa: E402
from models.resnet import resnet50 as gals_resnet50  # noqa: E402


FLOAT_RE = re.compile(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_token(text: str) -> str:
    token = str(text).replace("\\", "__").replace("/", "__").replace(".", "_")
    token = token.replace(" ", "_").replace(":", "_")
    return token[:180]


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


def save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def save_gray(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def resize_map(norm_map: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(norm_map, (width, height), interpolation=cv2.INTER_LINEAR)


def torch_load_compat(path: Path, device: torch.device):
    """
    PyTorch >=2.6 defaults torch.load(..., weights_only=True), which can fail on
    legacy checkpoint payloads. Retry with weights_only=False for trusted files.
    """
    try:
        return torch.load(path, map_location=device)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        msg = str(exc)
        if "Weights only load failed" not in msg:
            raise
        print(
            f"[WARN] weights_only load failed for {path}. Retrying with weights_only=False.",
            flush=True,
        )
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            # Older torch versions may not support the weights_only kwarg.
            return torch.load(path, map_location=device)


def extract_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
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


class GALSResnet50CAM(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = gals_resnet50(pretrained=False, return_fmaps=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GALSProbModel(nn.Module):
    def __init__(self, model: GALSResnet50CAM):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        return torch.softmax(logits, dim=1)


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


def write_comparison_panels(sample_dir: Path, vis_by_model: Dict[str, Dict[str, np.ndarray]]) -> None:
    model_names = list(vis_by_model.keys())
    if len(model_names) < 2:
        return

    viz_keys = ["overlay", "heatmap", "gray", "contour"]
    for key in viz_keys:
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a = model_names[i]
                b = model_names[j]
                pair = np.concatenate([vis_by_model[a][key], vis_by_model[b][key]], axis=1)
                save_rgb(sample_dir / f"pair_{a}_vs_{b}_{key}.png", pair)

        strip = np.concatenate([vis_by_model[m][key] for m in model_names], axis=1)
        save_rgb(sample_dir / f"all_models_{'_'.join(model_names)}_{key}.png", strip)


def train_vanilla_if_needed(args: argparse.Namespace, out_dir: Path) -> Dict[str, object]:
    if args.vanilla_ckpt and Path(args.vanilla_ckpt).is_file():
        ckpt = str(Path(args.vanilla_ckpt).resolve())
        return {
            "checkpoint": ckpt,
            "from_existing_checkpoint": True,
            "best_balanced_val_acc": None,
            "test_acc": None,
            "per_group": None,
            "worst_group": None,
        }

    run_args = SimpleNamespace(
        data_path=args.data_path,
        seed=int(args.vanilla_seed),
        model="resnet50",
        clip_model="RN50",
        tune_mode="full",
        pretrained=True,
        batch_size=int(args.batch_size),
        num_epochs=int(args.vanilla_epochs),
        lr=float(args.vanilla_base_lr),
        base_lr=float(args.vanilla_base_lr),
        classifier_lr=float(args.vanilla_classifier_lr),
        momentum=float(args.vanilla_momentum),
        weight_decay=float(args.vanilla_weight_decay),
        nesterov=bool(args.vanilla_nesterov),
        num_workers=int(args.vanilla_num_workers),
        checkpoint_dir=str(out_dir / "checkpoints" / "vanilla"),
        classes=str(args.classes),
    )

    old_save = os.environ.get("SAVE_CHECKPOINTS")
    os.environ["SAVE_CHECKPOINTS"] = "1"
    try:
        best_val, test_acc, per_group, worst_group, ckpt = rvm.run_single(run_args)
    finally:
        if old_save is None:
            os.environ.pop("SAVE_CHECKPOINTS", None)
        else:
            os.environ["SAVE_CHECKPOINTS"] = old_save

    return {
        "checkpoint": str(ckpt),
        "from_existing_checkpoint": False,
        "best_balanced_val_acc": float(best_val),
        "test_acc": float(test_acc),
        "per_group": float(per_group),
        "worst_group": float(worst_group),
    }


def load_guided_model(guided_ckpt: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = rgm.make_redmeat_cam_model(
        num_classes=num_classes,
        model_name="resnet50",
        pretrained=True,
        clip_model="RN50",
    ).to(device)
    state = torch_load_compat(guided_ckpt, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_vanilla_model(vanilla_ckpt: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = rvm.make_model("resnet50", num_classes, pretrained=True, clip_model="RN50").to(device)
    state = torch_load_compat(vanilla_ckpt, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_gals_model(gals_ckpt: Path, num_classes: int, device: torch.device) -> GALSResnet50CAM:
    model = GALSResnet50CAM(num_classes=num_classes).to(device)
    ckpt = torch_load_compat(gals_ckpt, device)
    state = extract_state_dict(ckpt)
    state = align_state_dict_keys(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading GALS checkpoint: {missing}", flush=True)
    if unexpected:
        print(f"[WARN] Unexpected keys while loading GALS checkpoint: {unexpected}", flush=True)
    model.eval()
    return model


def save_gt_mask_variants(mask_path: Path, image_rgb: np.ndarray, sample_dir: Path) -> bool:
    if not mask_path.is_file():
        return False

    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
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


def _select_val_indices_per_class(labels: np.ndarray, num_classes: int, per_class: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        take = min(int(per_class), int(idx.size))
        selected.extend(idx[:take].tolist())
    selected = list(dict.fromkeys(selected))
    return selected


def _build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RedMeat guided+vanilla+GALS saliency runner (Waterbirds-style outputs).")
    p.add_argument("--data-path", default="/workspace/data/food-101-redmeat")
    p.add_argument("--guided-gt-root", default="/workspace/data/results_redmeat_openclip_dinovit/val/prediction_cmap")
    p.add_argument("--output-dir", default="")
    p.add_argument("--classes", default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon")

    p.add_argument(
        "--guided-ckpt",
        default="/workspace/Waterbird_Runs/GALS/RedMeat_Guided_Checkpoints/resnet50_redmeat_final_kl11_attn2_20260302_073649.pth",
    )
    p.add_argument(
        "--gals-ckpt",
        default="/workspace/SavedChecks/gals_food-101-redmeat_21068187_trial_029/best_valacc_0.64_epoch_133.ckpt",
    )
    p.add_argument("--vanilla-ckpt", default="", help="Optional existing vanilla checkpoint; if set, skip vanilla training.")

    # Vanilla training hyperparams requested by user.
    p.add_argument("--vanilla-seed", type=int, default=0)
    p.add_argument("--vanilla-epochs", type=int, default=150)
    p.add_argument("--vanilla-base-lr", type=float, default=0.0016966944834563556)
    p.add_argument("--vanilla-classifier-lr", type=float, default=0.001036350233934804)
    p.add_argument("--vanilla-momentum", type=float, default=0.9)
    p.add_argument("--vanilla-weight-decay", type=float, default=1e-5)
    p.add_argument("--vanilla-nesterov", action="store_true", default=False)
    p.add_argument("--vanilla-num-workers", type=int, default=8)

    p.add_argument("--batch-size", type=int, default=96, help="Used for vanilla training.")

    p.add_argument("--num-val-per-class", type=int, default=30)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--target-class", choices=["label", "pred"], default="label")

    p.add_argument("--rise-num-masks", type=int, default=2000)
    p.add_argument("--rise-grid-size", type=int, default=8)
    p.add_argument("--rise-p1", type=float, default=0.1)
    p.add_argument("--rise-gpu-batch", type=int, default=16)
    p.add_argument("--rise-seed", type=int, default=0)
    p.add_argument("--rise-masks-path", default="", help="Optional .npy mask-bank path.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.sample_seed))

    data_path = Path(args.data_path).expanduser().resolve()
    gt_root = Path(args.guided_gt_root).expanduser().resolve()
    guided_ckpt = Path(args.guided_ckpt).expanduser().resolve()
    gals_ckpt = Path(args.gals_ckpt).expanduser().resolve()

    if not data_path.is_dir():
        raise RuntimeError(f"Missing data path: {data_path}")
    if not gt_root.is_dir():
        print(f"[WARN] guided-gt-root does not exist: {gt_root}", flush=True)
    if not guided_ckpt.is_file():
        raise RuntimeError(f"Missing guided checkpoint: {guided_ckpt}")
    if not gals_ckpt.is_file():
        raise RuntimeError(f"Missing GALS checkpoint: {gals_ckpt}")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = data_path.parent / "logsRedMeat" / f"redmeat_guided_vanilla_gals_saliency_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_path={data_path}", flush=True)
    print(f"[INFO] guided_gt_root={gt_root}", flush=True)
    print(f"[INFO] output_dir={out_dir}", flush=True)
    print(f"[INFO] guided_ckpt={guided_ckpt}", flush=True)
    print(f"[INFO] gals_ckpt={gals_ckpt}", flush=True)

    vanilla_metrics = train_vanilla_if_needed(args, out_dir)
    vanilla_ckpt = Path(str(vanilla_metrics["checkpoint"])).resolve()
    if not vanilla_ckpt.is_file():
        raise RuntimeError(f"Missing vanilla checkpoint after training: {vanilla_ckpt}")

    val_ds = rgm.RedMeatMetadataDataset(
        data_root=str(data_path),
        split="val",
        image_transform=_build_eval_transform(),
        return_mask=False,
        return_path=True,
        classes=[c.strip() for c in str(args.classes).split(",") if c.strip()],
        split_col="split",
        label_col="label",
        path_col="abs_file_path",
    )
    num_classes = len(val_ds.classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    guided_model = load_guided_model(guided_ckpt, num_classes, device)
    vanilla_model = load_vanilla_model(vanilla_ckpt, num_classes, device)
    gals_model = load_gals_model(gals_ckpt, num_classes, device)

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
    guided_rise = RISEExplainer(
        prob_model=GuidedProbModel(guided_model).to(device).eval(),
        masks=shared_rise_masks,
        gpu_batch=int(args.rise_gpu_batch),
        p1=float(args.rise_p1),
    )
    vanilla_rise = RISEExplainer(
        prob_model=VanillaProbModel(vanilla_model).to(device).eval(),
        masks=shared_rise_masks,
        gpu_batch=int(args.rise_gpu_batch),
        p1=float(args.rise_p1),
    )
    gals_rise = RISEExplainer(
        prob_model=GALSProbModel(gals_model).to(device).eval(),
        masks=shared_rise_masks,
        gpu_batch=int(args.rise_gpu_batch),
        p1=float(args.rise_p1),
    )

    labels = np.array(val_ds.labels, dtype=np.int64)
    selected_indices = _select_val_indices_per_class(
        labels=labels,
        num_classes=num_classes,
        per_class=int(args.num_val_per_class),
        seed=int(args.sample_seed),
    )
    print(
        f"[INFO] Selected {len(selected_indices)} validation images ({args.num_val_per_class} per class requested).",
        flush=True,
    )
    print(
        (
            "[INFO] Saliency method=RISE "
            f"(num_masks={int(args.rise_num_masks)} grid_size={int(args.rise_grid_size)} "
            f"p1={float(args.rise_p1)} gpu_batch={int(args.rise_gpu_batch)} seed={int(args.rise_seed)})"
        ),
        flush=True,
    )
    print(f"[INFO] RISE mask bank: {rise_masks_path}", flush=True)

    sample_rows: List[Dict[str, object]] = []
    use_label_target = args.target_class == "label"

    for i, idx in enumerate(selected_indices):
        image_t, label, img_path_str = val_ds[idx]
        img_path = Path(str(img_path_str))
        label = int(label)

        image_rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        input_tensor = image_t.unsqueeze(0).to(device)

        vis_by_model: Dict[str, Dict[str, np.ndarray]] = {}
        saliency_by_model: Dict[str, np.ndarray] = {}

        with torch.no_grad():
            guided_logits, _ = guided_model(input_tensor)
            guided_pred = int(guided_logits.argmax(dim=1).item())
            guided_target = label if use_label_target else guided_pred
            guided_conf = float(torch.softmax(guided_logits, dim=1)[0, guided_pred].item())
            guided_sal = guided_rise(input_tensor)
            saliency_by_model["guided"] = guided_sal[guided_target].detach().cpu().numpy().astype(np.float32)

            vanilla_logits = vanilla_model(input_tensor)
            vanilla_pred = int(vanilla_logits.argmax(dim=1).item())
            vanilla_target = label if use_label_target else vanilla_pred
            vanilla_conf = float(torch.softmax(vanilla_logits, dim=1)[0, vanilla_pred].item())
            vanilla_sal = vanilla_rise(input_tensor)
            saliency_by_model["vanilla"] = vanilla_sal[vanilla_target].detach().cpu().numpy().astype(np.float32)

            gals_out = gals_model(input_tensor)
            gals_logits = gals_out[0] if isinstance(gals_out, tuple) else gals_out
            gals_pred = int(gals_logits.argmax(dim=1).item())
            gals_target = label if use_label_target else gals_pred
            gals_conf = float(torch.softmax(gals_logits, dim=1)[0, gals_pred].item())
            gals_sal = gals_rise(input_tensor)
            saliency_by_model["gals"] = gals_sal[gals_target].detach().cpu().numpy().astype(np.float32)

        sample_token = safe_token(str(img_path.name))
        class_name = str(val_ds.classes[label])
        sample_dir = out_dir / "samples" / f"{i:03d}_{class_name}_{sample_token}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        save_rgb(sample_dir / "original_image.png", image_rgb)

        gt_mask_path = None
        if gt_root.is_dir():
            for cand in rgm._mask_candidates(str(gt_root), class_name, str(img_path)):
                p = Path(cand)
                if p.is_file():
                    gt_mask_path = p
                    break
            if gt_mask_path is not None:
                save_gt_mask_variants(gt_mask_path, image_rgb, sample_dir)
            else:
                with open(sample_dir / "gt_mask_missing.txt", "w", encoding="utf-8") as f:
                    f.write(f"No GT mask found for image: {img_path}\n")

        for model_name, sal_map in saliency_by_model.items():
            vis_by_model[model_name] = save_saliency_variants(model_name, sal_map, image_rgb, sample_dir)
        write_comparison_panels(sample_dir, vis_by_model)

        info: Dict[str, object] = {
            "index": int(i),
            "val_dataset_index": int(idx),
            "image_path": str(img_path),
            "label": int(label),
            "class_name": class_name,
            "guided_pred": guided_pred,
            "guided_confidence": guided_conf,
            "guided_saliency_target_class": int(guided_target),
            "vanilla_pred": vanilla_pred,
            "vanilla_confidence": vanilla_conf,
            "vanilla_saliency_target_class": int(vanilla_target),
            "gals_pred": gals_pred,
            "gals_confidence": gals_conf,
            "gals_saliency_target_class": int(gals_target),
            "gt_mask_path": str(gt_mask_path) if gt_mask_path is not None else None,
        }

        with open(sample_dir / "sample_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        sample_rows.append(info)

    summary = {
        "data_path": str(data_path),
        "guided_gt_root": str(gt_root),
        "output_dir": str(out_dir),
        "num_val_per_class_requested": int(args.num_val_per_class),
        "num_val_samples_generated": int(len(sample_rows)),
        "target_class_mode": args.target_class,
        "classes": list(val_ds.classes),
        "rise": {
            "num_masks": int(args.rise_num_masks),
            "grid_size": int(args.rise_grid_size),
            "p1": float(args.rise_p1),
            "gpu_batch": int(args.rise_gpu_batch),
            "seed": int(args.rise_seed),
            "masks_path": str(rise_masks_path),
        },
        "guided": {
            "checkpoint": str(guided_ckpt),
        },
        "vanilla": vanilla_metrics,
        "gals": {
            "checkpoint": str(gals_ckpt),
        },
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "sample_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "val_dataset_index",
            "image_path",
            "label",
            "class_name",
            "guided_pred",
            "guided_confidence",
            "guided_saliency_target_class",
            "vanilla_pred",
            "vanilla_confidence",
            "vanilla_saliency_target_class",
            "gals_pred",
            "gals_confidence",
            "gals_saliency_target_class",
            "gt_mask_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "Per-sample folders are under samples/.\n"
            "Each folder contains:\n"
            "- original_image.png\n"
            "- guided RISE saliency variants (overlay/heatmap/grayscale/binary/contours)\n"
            "- vanilla RISE saliency variants (same set)\n"
            "- gals RISE saliency variants (same set)\n"
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
