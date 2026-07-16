#!/usr/bin/env python3
"""
Generate curated RedMeat saliency outputs for:
- upweight checkpoint
- ABN checkpoint
- AFR checkpoint

This script mirrors the Waterbirds/RedMeat saliency artifact style:
- per-model saliency variants (overlay/heatmap/grayscale/binary/contours)
- pairwise and all-model comparison panels
- optional GT-mask visualization variants
- sample_index.csv + run_summary.json

Selection behavior:
- Uses val split only
- Resolves an explicit curated list of filename tokens
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models as tv_models
from torchvision import transforms
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


_THIS_DIR = Path(__file__).resolve().parent
_GALS_ROOT = _THIS_DIR.parent
if str(_GALS_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_GALS_ROOT))

from RedMeat_Runs import run_guided_redmeat as rgm  # noqa: E402
from models.resnet import resnet50 as gals_resnet50  # noqa: E402
from models.resnet_abn import resnet50 as gals_resnet50_abn  # noqa: E402


DEFAULT_CLASSES = "prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon"
DEFAULT_IMAGE_TOKENS = [
    "baby_back_ribs_1941026_jpg",
    "baby_back_ribs_1804724_jpg",
    "baby_back_ribs_1676906_jpg",
    "baby_back_ribs_1343043_jpg",
    "baby_back_ribs_1341092_jpg",
    "filet_mignon_522091_jpg",
    "filet_mignon_1472992_jpg",
    "filet_mignon_2899269_jpg",
    "filet_mignon_512874_jpg",
    "filet_mignon_2703493_jpg",
    "pork_chop_450350_jpg",
    "pork_chop_1437232_jpg",
    "pork_chop_1168924_jpg",
    "pork_chop_1830532_jpg",
    "pork_chop_2350249_jpg",
    "prime_rib_1459859_jpg",
    "prime_rib_1408388_jpg",
    "prime_rib_440936_jpg",
    "prime_rib_300881_jpg",
    "prime_rib_3134213_jpg",
    "steak_3663518_jpg",
    "steak_1362989_jpg",
    "steak_438871_jpg",
    "steak_2032669_jpg",
    "steak_3191589_jpg",
]


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
    try:
        return torch.load(path, map_location=device)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if "Weights only load failed" not in str(exc):
            raise
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)


def extract_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "algorithm"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if ckpt_obj:
            k0 = next(iter(ckpt_obj.keys()))
            if isinstance(k0, str):
                return ckpt_obj  # type: ignore[return-value]
    raise RuntimeError("Could not extract state_dict from checkpoint.")


def _candidate_state_dicts(state_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    def strip_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in d.items():
            if k.startswith(prefix):
                out[k[len(prefix) :]] = v
            else:
                out[k] = v
        return out

    def add_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {prefix + k: v for k, v in d.items()}

    cands = [state_dict]
    for p in ("module.", "model.", "net.", "base."):
        cands.append(strip_prefix(state_dict, p))
    cands.append(add_prefix(state_dict, "module."))
    cands.append(add_prefix(state_dict, "net."))
    cands.append(add_prefix(state_dict, "base."))

    out: List[Dict[str, torch.Tensor]] = []
    seen: Set[Tuple[str, ...]] = set()
    for d in cands:
        sig = tuple(sorted(d.keys()))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(d)
    return out


def _load_state_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, List[str], List[str]]:
    best_loaded = -1
    best_missing: List[str] = []
    best_unexpected: List[str] = []
    best_cand: Optional[Dict[str, torch.Tensor]] = None

    model_keys = set(model.state_dict().keys())
    for cand in _candidate_state_dicts(state_dict):
        loaded = len(model_keys.intersection(cand.keys()))
        if loaded <= 0:
            continue
        if loaded > best_loaded:
            best_loaded = loaded
            best_cand = cand

    if best_cand is None:
        return 0, list(model.state_dict().keys()), list(state_dict.keys())

    missing, unexpected = model.load_state_dict(best_cand, strict=False)
    best_missing = list(missing)
    best_unexpected = list(unexpected)
    return best_loaded, best_missing, best_unexpected


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


def _extract_logits(output, prefer_second: bool = False) -> torch.Tensor:
    if isinstance(output, tuple):
        if prefer_second and len(output) >= 2 and torch.is_tensor(output[1]):
            return output[1]
        for obj in output:
            if torch.is_tensor(obj) and obj.ndim == 2:
                return obj
    if torch.is_tensor(output):
        return output
    raise RuntimeError("Could not extract logits tensor from model output.")


class GenericProbModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = _extract_logits(self.model(x), prefer_second=False)
        return torch.softmax(logits, dim=1)


class ABNProbModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = _extract_logits(self.model(x), prefer_second=True)
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


def _normalize_text_token(text: str) -> str:
    s = str(text).strip().lower()
    s = s.replace("\\", "/")
    s = re.sub(r"\.[a-z0-9]+$", "", s)  # drop file extension
    s = s.replace("/", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    s = re.sub(r"^\d+_", "", s)  # tolerate optional leading numeric prefix
    return s


def _name_variants(name: str) -> Set[str]:
    """
    Produce robust matching keys from either:
    - a curated token (e.g. baby_back_ribs_1941026_jpg)
    - a full file path (e.g. /.../baby_back_ribs/1941026.jpg)
    """
    p = Path(str(name))
    out: Set[str] = set()

    # Raw token/path normalized forms.
    raw = _normalize_text_token(str(name))
    if raw:
        out.add(raw)

    # Basename and stem variants.
    basename = _normalize_text_token(p.name)
    stem = _normalize_text_token(p.stem)
    if basename:
        out.add(basename)
    if stem:
        out.add(stem)

    # Parent + stem is critical when filenames are numeric (e.g., class/1941026.jpg).
    if p.parent and str(p.parent) not in (".", ""):
        parent_name = _normalize_text_token(p.parent.name)
        if parent_name and stem:
            out.add(f"{parent_name}_{stem}")

    # If token ends with _jpg/_jpeg/_png, also allow stripped form.
    for v in list(out):
        for suf in ("_jpg", "_jpeg", "_png"):
            if v.endswith(suf):
                out.add(v[: -len(suf)])
    return {x for x in out if x}


def _resolve_curated_indices(paths: Sequence[str], requested_tokens: Sequence[str]) -> Tuple[List[int], List[str]]:
    req_variants: Dict[str, Set[str]] = {}
    for tok in requested_tokens:
        req_variants[tok] = _name_variants(tok)

    matched_idx: Dict[str, int] = {}
    for i, p in enumerate(paths):
        stem_vars = _name_variants(Path(p).name)
        for tok, variants in req_variants.items():
            if tok in matched_idx:
                continue
            if len(stem_vars.intersection(variants)) > 0:
                matched_idx[tok] = i

    missing = [tok for tok in requested_tokens if tok not in matched_idx]
    ordered_idx = [matched_idx[tok] for tok in requested_tokens if tok in matched_idx]
    return ordered_idx, missing


def _build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _make_generic_resnet50(num_classes: int) -> nn.Module:
    model = gals_resnet50(pretrained=False, return_fmaps=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _make_abn_resnet50(num_classes: int, add_after_attention: bool = True) -> nn.Module:
    return gals_resnet50_abn(
        pretrained=False,
        num_classes=num_classes,
        add_after_attention=bool(add_after_attention),
    )


def _make_torchvision_resnet50(num_classes: int) -> nn.Module:
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _load_best_of_builders(
    ckpt_path: Path,
    builders: Sequence[Tuple[str, callable]],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object]]:
    ckpt = torch_load_compat(ckpt_path, device)
    state = extract_state_dict(ckpt)

    best_model: Optional[nn.Module] = None
    best_meta: Optional[Dict[str, object]] = None
    best_score = -1

    for name, builder in builders:
        model = builder().to(device)
        loaded, missing, unexpected = _load_state_flexible(model, state)
        if loaded > best_score:
            best_score = loaded
            best_model = model
            best_meta = {
                "builder": name,
                "loaded_key_overlap": int(loaded),
                "missing_key_count": int(len(missing)),
                "unexpected_key_count": int(len(unexpected)),
                "missing_keys_preview": missing[:20],
                "unexpected_keys_preview": unexpected[:20],
            }

    if best_model is None or best_meta is None or best_score <= 0:
        raise RuntimeError(f"Could not load checkpoint into any candidate model: {ckpt_path}")

    best_model.eval()
    return best_model, best_meta


def _extract_last_layer_weight_bias(state_dict: Dict[str, torch.Tensor]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    for cand in _candidate_state_dicts(state_dict):
        w = cand.get("weight", None)
        b = cand.get("bias", None)
        if isinstance(w, torch.Tensor) and isinstance(b, torch.Tensor) and w.ndim == 2 and b.ndim == 1:
            return w, b
    # Some checkpoints may use fc.* keys directly.
    for cand in _candidate_state_dicts(state_dict):
        w = cand.get("fc.weight", None)
        b = cand.get("fc.bias", None)
        if isinstance(w, torch.Tensor) and isinstance(b, torch.Tensor) and w.ndim == 2 and b.ndim == 1:
            return w, b
    return None


def _read_json_dict(path: Path) -> Dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_yaml_dict(path: Path) -> Dict[str, object]:
    if yaml is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _infer_afr_stage1_ckpt(afr_ckpt: Path) -> Tuple[Optional[Path], Dict[str, object]]:
    info: Dict[str, object] = {"afr_ckpt_parent": str(afr_ckpt.parent)}
    parent = afr_ckpt.parent

    # 1) Read sidecar metadata produced by AFR train_embeddings.py
    args_json = parent / "args.json"
    meta_yaml = parent / "meta.yaml"
    cfg: Dict[str, object] = {}
    if args_json.is_file():
        cfg = _read_json_dict(args_json)
        info["sidecar"] = str(args_json)
    if not cfg and meta_yaml.is_file():
        cfg = _read_yaml_dict(meta_yaml)
        if cfg:
            info["sidecar"] = str(meta_yaml)

    base_model_dir = str(cfg.get("base_model_dir", "")).strip() if cfg else ""
    if base_model_dir:
        base_dir = Path(base_model_dir).expanduser()
        if not base_dir.is_absolute():
            base_dir = (parent / base_dir).resolve()
        else:
            base_dir = base_dir.resolve()
        info["base_model_dir"] = str(base_dir)
        for name in ("final_checkpoint.pt", "best_checkpoint.pt", "resumable_checkpoint.pt"):
            cand = base_dir / name
            if cand.is_file():
                info["stage1_source"] = "base_model_dir"
                return cand, info

    # 2) Heuristics for copied SavedChecks layouts
    seed = None
    m = re.search(r"seed[_-]?(\d+)", parent.name)
    if m:
        seed = m.group(1)
        info["seed_inferred"] = seed

    roots = [parent.parent, parent.parent.parent]
    candidates: List[Path] = []
    for root in roots:
        if root is None or not root.exists():
            continue
        if seed is not None:
            candidates.extend(
                [
                    root / f"seed_{seed}" / "final_checkpoint.pt",
                    root / f"seed{seed}" / "final_checkpoint.pt",
                    root / "stage1" / f"seed_{seed}" / "final_checkpoint.pt",
                    root / "stage1" / f"seed{seed}" / "final_checkpoint.pt",
                ]
            )
        candidates.extend(
            [
                root / "seed_0" / "final_checkpoint.pt",
                root / "seed0" / "final_checkpoint.pt",
                root / "stage1" / "seed_0" / "final_checkpoint.pt",
                root / "stage1" / "seed0" / "final_checkpoint.pt",
            ]
        )

    seen: Set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            info["stage1_source"] = "heuristic"
            return cand.resolve(), info

    info["stage1_source"] = "not_found"
    return None, info


def _load_afr_model(
    afr_ckpt: Path,
    builders: Sequence[Tuple[str, callable]],
    device: torch.device,
    stage1_ckpt_override: str = "",
) -> Tuple[nn.Module, Dict[str, object]]:
    afr_obj = torch_load_compat(afr_ckpt, device)
    afr_state = extract_state_dict(afr_obj)

    # Try direct full-model load first.
    try:
        model, meta = _load_best_of_builders(afr_ckpt, builders=builders, device=device)
        meta = dict(meta)
        meta["afr_load_mode"] = "direct_full_model"
        return model, meta
    except Exception as direct_exc:
        direct_err = str(direct_exc)

    # If direct load failed, see if this is a stage-2 last-layer checkpoint.
    ll = _extract_last_layer_weight_bias(afr_state)
    if ll is None:
        raise RuntimeError(
            f"Could not load AFR checkpoint as full model and did not detect last-layer-only state: {afr_ckpt}"
        )
    ll_w, ll_b = ll

    # Find stage-1 checkpoint.
    stage1_ckpt: Optional[Path] = None
    stage1_meta: Dict[str, object] = {}
    if stage1_ckpt_override:
        p = Path(stage1_ckpt_override).expanduser().resolve()
        if p.is_file():
            stage1_ckpt = p
            stage1_meta["stage1_source"] = "user_override"
        else:
            raise FileNotFoundError(f"--afr-stage1-ckpt provided but missing: {p}")
    else:
        stage1_ckpt, stage1_meta = _infer_afr_stage1_ckpt(afr_ckpt)
        if stage1_ckpt is None:
            raise RuntimeError(
                "AFR checkpoint appears to be stage-2 last-layer only (weight/bias), "
                "but stage-1 checkpoint could not be inferred. "
                f"Provide --afr-stage1-ckpt. afr_ckpt={afr_ckpt} details={stage1_meta}"
            )

    model, meta = _load_best_of_builders(stage1_ckpt, builders=builders, device=device)
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise RuntimeError("AFR reconstruction expects model.fc to be nn.Linear.")

    fc: nn.Linear = model.fc
    if tuple(fc.weight.shape) != tuple(ll_w.shape) or tuple(fc.bias.shape) != tuple(ll_b.shape):
        raise RuntimeError(
            "AFR stage-2 last-layer shape mismatch with stage-1 model head: "
            f"fc.weight={tuple(fc.weight.shape)} ll_w={tuple(ll_w.shape)} "
            f"fc.bias={tuple(fc.bias.shape)} ll_b={tuple(ll_b.shape)}"
        )

    with torch.no_grad():
        fc.weight.copy_(ll_w.to(fc.weight.device, dtype=fc.weight.dtype))
        fc.bias.copy_(ll_b.to(fc.bias.device, dtype=fc.bias.dtype))

    meta = dict(meta)
    meta.update(
        {
            "afr_load_mode": "stage1_plus_stage2_last_layer",
            "afr_stage2_ckpt": str(afr_ckpt),
            "afr_stage1_ckpt": str(stage1_ckpt),
            "afr_stage2_weight_shape": list(ll_w.shape),
            "afr_stage2_bias_shape": list(ll_b.shape),
            "afr_direct_load_error": direct_err,
        }
    )
    meta.update(stage1_meta)
    model.eval()
    return model, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curated RedMeat saliency for upweight/abn/afr checkpoints.")
    p.add_argument("--data-path", default="/workspace/data/food-101-redmeat")
    p.add_argument("--guided-gt-root", default="/workspace/data/results_redmeat_openclip_dinovit/val/prediction_cmap")
    p.add_argument("--output-dir", default="")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    p.add_argument("--classes", default=DEFAULT_CLASSES)
    p.add_argument("--target-class", choices=["label", "pred"], default="label")
    p.add_argument("--sample-seed", type=int, default=0)

    p.add_argument(
        "--upweight-ckpt",
        default="/workspace/Waterbird_Runs/GALS/trained_weights/food_subset/upweight_redmeat_fixed_trial30_20260303_231413/best_valacc_0.65_epoch_31.ckpt",
    )
    p.add_argument(
        "--abn-ckpt",
        default="/workspace/SavedChecks/abn_redmeat/abn_att_food-101-redmeat_21068185_best_seed0/best_valacc_0.68_epoch_21.ckpt",
    )
    p.add_argument(
        "--afr-ckpt",
        default="/workspace/SavedChecks/afr_redmeat/seed0_g5p5_r0p2/final_checkpoint.pt",
    )
    p.add_argument(
        "--afr-stage1-ckpt",
        default="",
        help="Optional AFR stage-1 checkpoint (needed when --afr-ckpt is stage-2 last-layer only).",
    )

    p.add_argument(
        "--image-names",
        default=",".join(DEFAULT_IMAGE_TOKENS),
        help="Comma-separated curated filename tokens (without leading numeric prefix).",
    )
    p.add_argument("--allow-missing", action="store_true", default=False)

    p.add_argument("--rise-num-masks", type=int, default=2000)
    p.add_argument("--rise-grid-size", type=int, default=8)
    p.add_argument("--rise-p1", type=float, default=0.1)
    p.add_argument("--rise-gpu-batch", type=int, default=16)
    p.add_argument("--rise-seed", type=int, default=0)
    p.add_argument("--rise-masks-path", default="", help="Optional .npy RISE mask bank path.")

    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@dataclass
class EvalModelSpec:
    name: str
    model: nn.Module
    rise: RISEExplainer
    prefers_second_logits: bool
    ckpt_path: Path
    load_meta: Dict[str, object]


@dataclass
class CuratedEntry:
    request_token: str
    image_path: str
    label: int
    class_name: str
    source: str  # metadata | filesystem


def _strip_image_suffix(token_norm: str) -> str:
    for suf in ("_jpg", "_jpeg", "_png"):
        if token_norm.endswith(suf):
            return token_norm[: -len(suf)]
    return token_norm


def _resolve_token_from_filesystem(
    token: str,
    data_path: Path,
    class_list: Sequence[str],
    class_to_idx: Dict[str, int],
) -> Optional[CuratedEntry]:
    token_norm = _strip_image_suffix(_normalize_text_token(token))
    if not token_norm:
        return None

    class_norm_to_name = {_normalize_text_token(c): c for c in class_list}
    # Prefer longest class-name prefix match to avoid partial collisions.
    sorted_class_norm = sorted(class_norm_to_name.keys(), key=len, reverse=True)

    matched_class_norm = None
    image_id = None
    for cn in sorted_class_norm:
        if token_norm.startswith(cn + "_"):
            matched_class_norm = cn
            image_id = token_norm[len(cn) + 1 :]
            break

    ext_list = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    search_roots = [data_path / "images", data_path]

    if matched_class_norm is not None and image_id:
        class_name = class_norm_to_name[matched_class_norm]
        for root in search_roots:
            cls_dir = root / class_name
            if not cls_dir.is_dir():
                continue
            for ext in ext_list:
                cand = cls_dir / f"{image_id}{ext}"
                if cand.is_file():
                    return CuratedEntry(
                        request_token=token,
                        image_path=str(cand.resolve()),
                        label=int(class_to_idx[class_name]),
                        class_name=class_name,
                        source="filesystem",
                    )

    # Generic fallback: search by basename id under images tree.
    # Keep this narrow to avoid expensive broad scans.
    target_ids = []
    if image_id:
        target_ids.append(image_id)
    target_ids.append(token_norm)
    # Deduplicate while preserving order.
    seen_ids: Set[str] = set()
    target_ids = [x for x in target_ids if x and not (x in seen_ids or seen_ids.add(x))]

    for root in search_roots:
        if not root.is_dir():
            continue
        for tid in target_ids:
            for ext in ext_list:
                pattern = f"{tid}{ext}"
                try:
                    for cand in root.rglob(pattern):
                        if not cand.is_file():
                            continue
                        parent_norm = _normalize_text_token(cand.parent.name)
                        if parent_norm in class_norm_to_name:
                            class_name = class_norm_to_name[parent_norm]
                            return CuratedEntry(
                                request_token=token,
                                image_path=str(cand.resolve()),
                                label=int(class_to_idx[class_name]),
                                class_name=class_name,
                                source="filesystem",
                            )
                except Exception:
                    continue
    return None


def main() -> None:
    args = parse_args()
    set_seed(int(args.sample_seed))

    data_path = Path(args.data_path).expanduser().resolve()
    gt_root = Path(args.guided_gt_root).expanduser().resolve()
    upweight_ckpt = Path(args.upweight_ckpt).expanduser().resolve()
    abn_ckpt = Path(args.abn_ckpt).expanduser().resolve()
    afr_ckpt = Path(args.afr_ckpt).expanduser().resolve()

    for tag, p in [("upweight", upweight_ckpt), ("abn", abn_ckpt), ("afr", afr_ckpt)]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing {tag} checkpoint: {p}")
    if not data_path.is_dir():
        raise RuntimeError(f"Missing data path: {data_path}")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = data_path.parent / "logsRedMeat" / f"redmeat_curated_saliency_up_abn_afr_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_path={data_path}", flush=True)
    print(f"[INFO] guided_gt_root={gt_root}", flush=True)
    print(f"[INFO] output_dir={out_dir}", flush=True)

    class_list = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    eval_tf = _build_eval_transform()

    split_list = ["train", "val", "test"] if args.split == "all" else [str(args.split)]
    pool_paths: List[str] = []
    pool_labels: List[int] = []
    pool_label_names: List[str] = []
    for sp in split_list:
        ds = rgm.RedMeatMetadataDataset(
            data_root=str(data_path),
            split=sp,
            image_transform=None,
            return_mask=False,
            return_path=True,
            classes=class_list,
            split_col="split",
            label_col="label",
            path_col="abs_file_path",
        )
        pool_paths.extend([str(x) for x in ds.paths])
        pool_labels.extend([int(x) for x in ds.labels.tolist()])
        pool_label_names.extend([str(x) for x in ds.label_names])

    num_classes = len(class_list)
    print(f"[INFO] split_pool={split_list} pool_size={len(pool_paths)}", flush=True)
    print(f"[INFO] num_classes={num_classes} classes={class_list}", flush=True)

    requested_tokens = [x.strip() for x in str(args.image_names).split(",") if x.strip()]
    class_to_idx = {c: i for i, c in enumerate(class_list)}

    selected_entries: List[CuratedEntry] = []
    missing: List[str] = []

    # Resolve each token in order: metadata first, then direct filesystem fallback.
    for tok in requested_tokens:
        idx_list, _ = _resolve_curated_indices(pool_paths, [tok])
        if idx_list:
            i0 = int(idx_list[0])
            selected_entries.append(
                CuratedEntry(
                    request_token=tok,
                    image_path=str(pool_paths[i0]),
                    label=int(pool_labels[i0]),
                    class_name=str(pool_label_names[i0]),
                    source="metadata",
                )
            )
            continue

        fs_entry = _resolve_token_from_filesystem(tok, data_path=data_path, class_list=class_list, class_to_idx=class_to_idx)
        if fs_entry is not None:
            selected_entries.append(fs_entry)
            continue

        missing.append(tok)

    if missing:
        msg = f"[WARN] Missing {len(missing)} requested images: {missing}"
        if not args.allow_missing:
            raise RuntimeError(msg)
        print(msg, flush=True)
    if not selected_entries:
        raise RuntimeError("No curated images resolved from requested list.")
    n_meta = sum(1 for e in selected_entries if e.source == "metadata")
    n_fs = sum(1 for e in selected_entries if e.source == "filesystem")
    print(
        f"[INFO] Resolved {len(selected_entries)}/{len(requested_tokens)} curated images "
        f"(metadata={n_meta}, filesystem_fallback={n_fs}).",
        flush=True,
    )

    req_device = str(args.device)
    if req_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(req_device)
    print(f"[INFO] Using device: {device}", flush=True)

    if args.rise_masks_path:
        rise_masks_path = Path(args.rise_masks_path).expanduser().resolve()
    else:
        p1_token = str(args.rise_p1).replace(".", "p")
        rise_masks_path = out_dir / f"rise_masks_n{args.rise_num_masks}_s{args.rise_grid_size}_p{p1_token}_seed{args.rise_seed}.npy"

    shared_rise_masks = load_or_create_rise_masks(
        mask_path=rise_masks_path,
        num_masks=int(args.rise_num_masks),
        input_size=(224, 224),
        grid_size=int(args.rise_grid_size),
        p1=float(args.rise_p1),
        seed=int(args.rise_seed),
        device=device,
    )

    up_model, up_meta = _load_best_of_builders(
        upweight_ckpt,
        builders=[
            ("gals_resnet50", lambda: _make_generic_resnet50(num_classes)),
            ("torchvision_resnet50", lambda: _make_torchvision_resnet50(num_classes)),
        ],
        device=device,
    )
    abn_model, abn_meta = _load_best_of_builders(
        abn_ckpt,
        builders=[
            ("gals_resnet50_abn_addTrue", lambda: _make_abn_resnet50(num_classes, add_after_attention=True)),
            ("gals_resnet50_abn_addFalse", lambda: _make_abn_resnet50(num_classes, add_after_attention=False)),
        ],
        device=device,
    )
    afr_model, afr_meta = _load_afr_model(
        afr_ckpt=afr_ckpt,
        builders=[
            ("gals_resnet50", lambda: _make_generic_resnet50(num_classes)),
            ("torchvision_resnet50", lambda: _make_torchvision_resnet50(num_classes)),
        ],
        device=device,
        stage1_ckpt_override=str(args.afr_stage1_ckpt),
    )

    print(f"[INFO] upweight load: {up_meta}", flush=True)
    print(f"[INFO] abn load: {abn_meta}", flush=True)
    print(f"[INFO] afr load: {afr_meta}", flush=True)

    specs: List[EvalModelSpec] = [
        EvalModelSpec(
            name="upweight",
            model=up_model,
            rise=RISEExplainer(
                prob_model=GenericProbModel(up_model).to(device).eval(),
                masks=shared_rise_masks,
                gpu_batch=int(args.rise_gpu_batch),
                p1=float(args.rise_p1),
            ),
            prefers_second_logits=False,
            ckpt_path=upweight_ckpt,
            load_meta=up_meta,
        ),
        EvalModelSpec(
            name="abn",
            model=abn_model,
            rise=RISEExplainer(
                prob_model=ABNProbModel(abn_model).to(device).eval(),
                masks=shared_rise_masks,
                gpu_batch=int(args.rise_gpu_batch),
                p1=float(args.rise_p1),
            ),
            prefers_second_logits=True,
            ckpt_path=abn_ckpt,
            load_meta=abn_meta,
        ),
        EvalModelSpec(
            name="afr",
            model=afr_model,
            rise=RISEExplainer(
                prob_model=GenericProbModel(afr_model).to(device).eval(),
                masks=shared_rise_masks,
                gpu_batch=int(args.rise_gpu_batch),
                p1=float(args.rise_p1),
            ),
            prefers_second_logits=False,
            ckpt_path=afr_ckpt,
            load_meta=afr_meta,
        ),
    ]

    use_label_target = args.target_class == "label"
    sample_rows: List[Dict[str, object]] = []

    for i, entry in enumerate(selected_entries):
        img_path = Path(str(entry.image_path))
        label = int(entry.label)
        class_name = str(entry.class_name)
        image_t = eval_tf(Image.open(img_path).convert("RGB"))
        image_rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        input_tensor = image_t.unsqueeze(0).to(device)

        sample_dir = out_dir / "samples" / f"{i:03d}_{class_name}_{safe_token(img_path.name)}"
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

        vis_by_model: Dict[str, Dict[str, np.ndarray]] = {}
        info: Dict[str, object] = {
            "index": int(i),
            "request_token": str(entry.request_token),
            "resolved_source": str(entry.source),
            "image_path": str(img_path),
            "label": int(label),
            "class_name": class_name,
            "gt_mask_path": (str(gt_mask_path) if gt_mask_path is not None else None),
        }

        with torch.no_grad():
            for spec in specs:
                logits = _extract_logits(spec.model(input_tensor), prefer_second=spec.prefers_second_logits)
                pred = int(logits.argmax(dim=1).item())
                target_cls = label if use_label_target else pred
                conf = float(torch.softmax(logits, dim=1)[0, pred].item())
                sal = spec.rise(input_tensor)[target_cls].detach().cpu().numpy().astype(np.float32)
                vis_by_model[spec.name] = save_saliency_variants(spec.name, sal, image_rgb, sample_dir)

                info[f"{spec.name}_pred"] = pred
                info[f"{spec.name}_confidence"] = conf
                info[f"{spec.name}_saliency_target_class"] = int(target_cls)

        write_comparison_panels(sample_dir, vis_by_model)
        with open(sample_dir / "sample_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        sample_rows.append(info)

    summary = {
        "data_path": str(data_path),
        "guided_gt_root": str(gt_root),
        "output_dir": str(out_dir),
        "target_class_mode": args.target_class,
        "split": str(args.split),
        "split_pool": split_list,
        "classes": list(class_list),
        "requested_image_names": requested_tokens,
        "resolved_count": int(len(selected_entries)),
        "resolved_from_metadata": int(sum(1 for e in selected_entries if e.source == "metadata")),
        "resolved_from_filesystem_fallback": int(sum(1 for e in selected_entries if e.source == "filesystem")),
        "missing_image_names": missing,
        "rise": {
            "num_masks": int(args.rise_num_masks),
            "grid_size": int(args.rise_grid_size),
            "p1": float(args.rise_p1),
            "gpu_batch": int(args.rise_gpu_batch),
            "seed": int(args.rise_seed),
            "masks_path": str(rise_masks_path),
        },
        "models": {
            "upweight": {"checkpoint": str(upweight_ckpt), "load_meta": up_meta},
            "abn": {"checkpoint": str(abn_ckpt), "load_meta": abn_meta},
            "afr": {"checkpoint": str(afr_ckpt), "load_meta": afr_meta},
        },
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "sample_index.csv"
    fieldnames = [
        "index",
        "request_token",
        "resolved_source",
        "image_path",
        "label",
        "class_name",
        "upweight_pred",
        "upweight_confidence",
        "upweight_saliency_target_class",
        "abn_pred",
        "abn_confidence",
        "abn_saliency_target_class",
        "afr_pred",
        "afr_confidence",
        "afr_saliency_target_class",
        "gt_mask_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sample_rows:
            w.writerow({k: row.get(k) for k in fieldnames})

    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "Per-sample folders are under samples/.\n"
            "Each folder contains:\n"
            "- original_image.png\n"
            "- upweight saliency variants (overlay/heatmap/grayscale/binary/contours)\n"
            "- abn saliency variants (overlay/heatmap/grayscale/binary/contours)\n"
            "- afr saliency variants (overlay/heatmap/grayscale/binary/contours)\n"
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
