#!/usr/bin/env python3
"""
Generate curated RISE saliency maps for an ElRep ResNet-50 checkpoint.

Supported datasets:
- wb95
- wb100
- redmeat

The output mirrors the existing saliency artifact style:
- per-sample folder with original image, optional GT mask variants, and ElRep saliency variants
- a compact grid image for quick visual inspection
- sample_index.csv and run_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torchvision import models, transforms


GALS_ROOT = Path(__file__).resolve().parent
if str(GALS_ROOT) not in sys.path:
    sys.path.insert(0, str(GALS_ROOT))


WATERBIRDS95_TOKENS = [
    "106_Horned_Puffin__Horned_Puffin_0024_100620_jpg",
    "144_Common_Tern__Common_Tern_0117_148944_jpg",
    "060_Glaucous_winged_Gull__Glaucous_Winged_Gull_0110_44377_jpg",
    "146_Forsters_Tern__Forsters_Tern_0127_150418_jpg",
    "021_Eastern_Towhee__Eastern_Towhee_0101_22559_jpg",
    "084_Red_legged_Kittiwake__Red_Legged_Kittiwake_0036_73814_jpg",
    "147_Least_Tern__Least_Tern_0082_154396_jpg",
    "101_White_Pelican__White_Pelican_0010_96876_jpg",
    "082_Ringed_Kingfisher__Ringed_Kingfisher_0050_73002_jpg",
    "011_Rusty_Blackbird__Rusty_Blackbird_0113_6664_jpg",
    "038_Great_Crested_Flycatcher__Great_Crested_Flycatcher_0009_29831_jpg",
    "160_Black_throated_Blue_Warbler__Black_Throated_Blue_Warbler_0081_161427_jpg",
    "171_Myrtle_Warbler__Myrtle_Warbler_0037_166690_jpg",
    "069_Rufous_Hummingbird__Rufous_Hummingbird_0095_60360_jpg",
    "019_Gray_Catbird__Gray_Catbird_0094_21303_jpg",
    "198_Rock_Wren__Rock_Wren_0019_188968_jpg",
]

WATERBIRDS100_TOKENS = [
    "060_Glaucous_winged_Gull__Glaucous_Winged_Gull_0012_44264_jpg",
    "084_Red_legged_Kittiwake__Red_Legged_Kittiwake_0068_795430_jpg",
    "100_Brown_Pelican__Brown_Pelican_0077_93464_jpg",
    "005_Crested_Auklet__Crested_Auklet_0071_785255_jpg",
    "087_Mallard__Mallard_0052_76946_jpg",
    "106_Horned_Puffin__Horned_Puffin_0056_101030_jpg",
    "072_Pomarine_Jaeger__Pomarine_Jaeger_0078_795758_jpg",
    "046_Gadwall__Gadwall_0035_30985_jpg",
    "097_Orchard_Oriole__Orchard_Oriole_0006_91724_jpg",
    "057_Rose_breasted_Grosbeak__Rose_Breasted_Grosbeak_0114_39770_jpg",
    "009_Brewer_Blackbird__Brewer_Blackbird_0140_2586_jpg",
    "018_Spotted_Catbird__Spotted_Catbird_0010_19436_jpg",
    "136_Barn_Swallow__Barn_Swallow_0045_130244_jpg",
    "080_Green_Kingfisher__Green_Kingfisher_0004_71076_jpg",
    "165_Chestnut_sided_Warbler__Chestnut_Sided_Warbler_0014_163801_jpg",
    "178_Swainson_Warbler__Swainson_Warbler_0011_174680_jpg",
]

REDMEAT_TOKENS = [
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

REDMEAT_CLASSES = ["prime_rib", "pork_chop", "steak", "baby_back_ribs", "filet_mignon"]
WATERBIRDS_CLASSES = ["Landbird", "Waterbird"]


@dataclass
class CuratedEntry:
    request_token: str
    image_path: Path
    label: int
    class_name: str
    source: str


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


def _normalize_text_token(text: str) -> str:
    s = str(text).strip().lower().replace("\\", "/")
    s = re.sub(r"\.(jpg|jpeg|png)$", "", s)
    s = s.replace(".", "_")
    s = s.replace("/", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    s = re.sub(r"^\d+_", "", s)
    return s


def _strip_image_suffix(token_norm: str) -> str:
    for suffix in ("_jpg", "_jpeg", "_png"):
        if token_norm.endswith(suffix):
            return token_norm[: -len(suffix)]
    return token_norm


def _name_variants(name: str) -> Set[str]:
    p = Path(str(name))
    out = {_normalize_text_token(str(name)), _normalize_text_token(p.name), _normalize_text_token(p.stem)}
    parent = _normalize_text_token(p.parent.name)
    stem = _normalize_text_token(p.stem)
    if parent and parent != "." and stem:
        out.add(f"{parent}_{stem}")
    for value in list(out):
        out.add(_strip_image_suffix(value))
    return {x for x in out if x}


def _resolve_curated_indices(paths: Sequence[Path], requested_tokens: Sequence[str]) -> Tuple[List[int], List[str]]:
    req_variants = {tok: _name_variants(tok) for tok in requested_tokens}
    matched: Dict[str, int] = {}

    for idx, path in enumerate(paths):
        variants = _name_variants(str(path))
        variants.update(_name_variants(path.name))
        if path.parent.name:
            variants.add(f"{_normalize_text_token(path.parent.name)}_{_normalize_text_token(path.stem)}")
        for token, wanted in req_variants.items():
            if token not in matched and variants.intersection(wanted):
                matched[token] = idx

    return [matched[t] for t in requested_tokens if t in matched], [t for t in requested_tokens if t not in matched]


def _resolve_token_from_paths(paths: Sequence[Path], token: str) -> Optional[int]:
    wanted = _name_variants(token)
    for idx, path in enumerate(paths):
        variants = _name_variants(str(path))
        variants.update(_name_variants(path.name))
        if path.parent.name:
            variants.add(f"{_normalize_text_token(path.parent.name)}_{_normalize_text_token(path.stem)}")
        if variants.intersection(wanted):
            return idx
    return None


def normalize_map(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32)
    out -= float(out.min())
    mx = float(out.max())
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


def save_gt_mask_variants(mask_path: Path, image_rgb: np.ndarray, sample_dir: Path) -> Dict[str, np.ndarray]:
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
    return {"overlay": overlay, "heatmap": heat, "contour": contour}


def generate_rise_masks_array(num_masks: int, input_size: Tuple[int, int], grid_size: int, p1: float, seed: int) -> np.ndarray:
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
        masks_np = generate_rise_masks_array(num_masks, input_size, grid_size, p1, seed)
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
        return sal / float(n_masks) / self.p1


def _extract_logits(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple):
        for obj in output:
            if torch.is_tensor(obj) and obj.ndim == 2:
                return obj
    raise RuntimeError("Could not extract logits tensor from model output.")


class GenericProbModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = _extract_logits(self.model(x))
        return torch.softmax(logits, dim=1)


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
        if ckpt_obj and isinstance(next(iter(ckpt_obj.keys())), str):
            return ckpt_obj  # type: ignore[return-value]
    raise RuntimeError("Could not extract state_dict from checkpoint.")


def _candidate_state_dicts(state_dict: Dict[str, torch.Tensor]) -> Iterable[Dict[str, torch.Tensor]]:
    def strip_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in d.items()}

    yield state_dict
    for prefix in ("module.", "model.", "net.", "base."):
        yield strip_prefix(state_dict, prefix)


def load_resnet50_elrep(ckpt_path: Path, num_classes: int, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = extract_state_dict(torch_load_compat(ckpt_path, device))

    model_keys = set(model.state_dict().keys())
    best_state = None
    best_overlap = -1
    for cand in _candidate_state_dicts(state):
        overlap = len(model_keys.intersection(cand.keys()))
        if overlap > best_overlap:
            best_state = cand
            best_overlap = overlap

    if best_state is None or best_overlap <= 0:
        raise RuntimeError(f"Could not match checkpoint keys to torchvision ResNet-50: {ckpt_path}")

    missing, unexpected = model.load_state_dict(best_state, strict=False)
    meta = {
        "checkpoint": str(ckpt_path),
        "loaded_key_overlap": int(best_overlap),
        "missing_key_count": int(len(missing)),
        "unexpected_key_count": int(len(unexpected)),
        "missing_keys_preview": list(missing)[:20],
        "unexpected_keys_preview": list(unexpected)[:20],
    }
    model.to(device).eval()
    return model, meta


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _resolve_img_path(dataset_root: Path, rel_or_abs: str) -> Path:
    p = Path(str(rel_or_abs))
    if p.is_absolute() and p.exists():
        return p
    return dataset_root / str(rel_or_abs).lstrip("/")


def _split_value_for_waterbirds(split: str) -> int:
    return {"train": 0, "val": 1, "test": 2}[split]


def _resolve_waterbird_token_filesystem(
    token: str,
    data_path: Path,
    label_lookup: Optional[Dict[str, int]] = None,
) -> Optional[CuratedEntry]:
    roots = [data_path / "images", data_path]
    wanted = _name_variants(token)
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for root in roots:
        if not root.is_dir():
            continue
        try:
            candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix in exts]
        except Exception:
            continue
        for path in candidates:
            variants = _name_variants(str(path))
            variants.update(_name_variants(path.name))
            if path.parent.name:
                variants.add(f"{_normalize_text_token(path.parent.name)}_{_normalize_text_token(path.stem)}")
            if not variants.intersection(wanted):
                continue

            key = str(path.resolve())
            label = label_lookup.get(key) if label_lookup is not None else None
            if label is None:
                # CUB Waterbirds classes 1-112 are waterbirds in the canonical split;
                # this fallback is only used when metadata lookup fails.
                m = re.match(r"^(\d+)", path.parent.name)
                species_id = int(m.group(1)) if m else 0
                label = 1 if 1 <= species_id <= 112 else 0
            return CuratedEntry(token, path.resolve(), int(label), WATERBIRDS_CLASSES[int(label)], "filesystem")
    return None


def resolve_waterbirds_entries(data_path: Path, split: str, requested_tokens: Sequence[str]) -> Tuple[List[CuratedEntry], List[str]]:
    meta_path = data_path / "metadata.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing Waterbirds metadata.csv: {meta_path}")
    df_all = pd.read_csv(meta_path)
    df = df_all[df_all["split"].astype(int) == _split_value_for_waterbirds(split)].copy()

    paths = [_resolve_img_path(data_path, p) for p in df["img_filename"].astype(str).tolist()]
    labels = df["y"].astype(int).tolist()
    all_paths = [_resolve_img_path(data_path, p) for p in df_all["img_filename"].astype(str).tolist()]
    all_labels = df_all["y"].astype(int).tolist()
    label_lookup = {str(p.resolve()): int(y) for p, y in zip(all_paths, all_labels)}

    entries: List[CuratedEntry] = []
    missing: List[str] = []
    for token in requested_tokens:
        idx = _resolve_token_from_paths(paths, token)
        if idx is not None:
            label = int(labels[idx])
            entries.append(
                CuratedEntry(
                    request_token=token,
                    image_path=paths[idx],
                    label=label,
                    class_name=WATERBIRDS_CLASSES[label],
                    source="metadata",
                )
            )
            continue

        fs_entry = _resolve_waterbird_token_filesystem(token, data_path, label_lookup=label_lookup)
        if fs_entry is not None:
            entries.append(fs_entry)
        else:
            missing.append(token)
    return entries, missing


def _resolve_redmeat_token_filesystem(
    token: str,
    data_path: Path,
    class_list: Sequence[str],
    class_to_idx: Dict[str, int],
) -> Optional[CuratedEntry]:
    token_norm = _strip_image_suffix(_normalize_text_token(token))
    class_norm_to_name = {_normalize_text_token(c): c for c in class_list}
    sorted_class_norm = sorted(class_norm_to_name.keys(), key=len, reverse=True)

    matched_class_norm = None
    image_id = None
    for class_norm in sorted_class_norm:
        if token_norm.startswith(class_norm + "_"):
            matched_class_norm = class_norm
            image_id = token_norm[len(class_norm) + 1 :]
            break
    if matched_class_norm is None or not image_id:
        return None

    class_name = class_norm_to_name[matched_class_norm]
    for root in (data_path / "images", data_path):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate = class_dir / f"{image_id}{ext}"
            if candidate.is_file():
                return CuratedEntry(token, candidate.resolve(), class_to_idx[class_name], class_name, "filesystem")
    return None


def resolve_redmeat_entries(
    data_path: Path,
    split: str,
    requested_tokens: Sequence[str],
    class_list: Sequence[str],
) -> Tuple[List[CuratedEntry], List[str]]:
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    meta_path = data_path / "all_images.csv"
    paths: List[Path] = []
    labels: List[int] = []
    label_names: List[str] = []

    if meta_path.is_file():
        df = pd.read_csv(meta_path)
        split_list = ["train", "val", "test"] if split == "all" else [split]
        df = df[df["split"].astype(str).isin(split_list)].copy()
        paths = [_resolve_img_path(data_path, p) for p in df["abs_file_path"].astype(str).tolist()]
        label_names = df["label"].astype(str).tolist()
        labels = [class_to_idx[x] for x in label_names]

    entries: List[CuratedEntry] = []
    missing: List[str] = []
    for token in requested_tokens:
        idxs, _ = _resolve_curated_indices(paths, [token]) if paths else ([], [token])
        if idxs:
            idx = idxs[0]
            entries.append(CuratedEntry(token, paths[idx], labels[idx], label_names[idx], "metadata"))
            continue
        fs_entry = _resolve_redmeat_token_filesystem(token, data_path, class_list, class_to_idx)
        if fs_entry is not None:
            entries.append(fs_entry)
        else:
            missing.append(token)
    return entries, missing


def candidate_waterbirds_mask_paths(mask_root: Path, image_path: Path, data_path: Path) -> List[Path]:
    try:
        rel = image_path.relative_to(data_path)
    except Exception:
        rel = Path(image_path.name)
    rel_no_ext = rel.with_suffix("")
    parent = rel.parent.name
    parent_underscored = parent.replace(".", "_")
    base = rel.stem
    return [
        mask_root / f"{parent_underscored}_{base}.png",
        mask_root / parent_underscored / f"{base}.png",
        mask_root / parent / f"{base}.png",
        mask_root / rel_no_ext.with_suffix(".png"),
        mask_root / rel_no_ext.with_suffix(".jpg"),
        mask_root / rel_no_ext.with_suffix(".jpeg"),
    ]


def candidate_redmeat_mask_paths(mask_root: Path, class_name: str, image_path: Path) -> List[Path]:
    stem = image_path.stem
    parent = image_path.parent.name
    return [
        mask_root / class_name / f"{stem}.png",
        mask_root / parent / f"{stem}.png",
        mask_root / f"{class_name}_{stem}.png",
        mask_root / f"{stem}.png",
        mask_root / f"{class_name}_{stem}_jpg.png",
        mask_root / f"{stem}_jpg.png",
    ]


def resolve_mask_path(dataset: str, mask_root: Path, data_path: Path, class_name: str, image_path: Path) -> Optional[Path]:
    if not mask_root.is_dir():
        return None
    candidates = (
        candidate_redmeat_mask_paths(mask_root, class_name, image_path)
        if dataset == "redmeat"
        else candidate_waterbirds_mask_paths(mask_root, image_path, data_path)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def make_grid(rows: Sequence[Dict[str, object]], output_path: Path, thumb: int = 140) -> None:
    columns = [
        ("image_path", "Image"),
        ("gt_overlay_path", "GT"),
        ("elrep_overlay_path", "ElRep overlay"),
        ("elrep_heatmap_path", "ElRep heatmap"),
        ("elrep_contour_path", "ElRep contour"),
    ]
    pad = 8
    header_h = 24
    cell_w = thumb
    cell_h = thumb
    width = len(columns) * (cell_w + pad) + pad
    height = header_h + len(rows) * (cell_h + pad) + pad
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col, (_, title) in enumerate(columns):
        x = pad + col * (cell_w + pad)
        draw.text((x, 6), title, fill=(0, 0, 0), font=font)

    for row_idx, row in enumerate(rows):
        y = header_h + pad + row_idx * (cell_h + pad)
        for col, (key, _) in enumerate(columns):
            x = pad + col * (cell_w + pad)
            path_value = row.get(key)
            if not path_value:
                tile = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
            else:
                tile = Image.open(str(path_value)).convert("RGB")
                tile.thumbnail((cell_w, cell_h), Image.BILINEAR)
                framed = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
                framed.paste(tile, ((cell_w - tile.width) // 2, (cell_h - tile.height) // 2))
                tile = framed
            canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curated RISE saliency maps for ElRep checkpoints.")
    p.add_argument("--dataset", choices=["wb95", "wb100", "redmeat"], required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--mask-root", default="")
    p.add_argument("--elrep-ckpt", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--image-names", default="", help="Comma-separated curated tokens. Defaults to dataset list.")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--target-class", choices=["label", "pred"], default="label")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--rise-num-masks", type=int, default=2000)
    p.add_argument("--rise-grid-size", type=int, default=8)
    p.add_argument("--rise-p1", type=float, default=0.1)
    p.add_argument("--rise-gpu-batch", type=int, default=16)
    p.add_argument("--rise-seed", type=int, default=0)
    p.add_argument("--rise-masks-path", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.sample_seed)

    dataset = str(args.dataset)
    data_path = Path(args.data_path).expanduser().resolve()
    mask_root = Path(args.mask_root).expanduser().resolve() if args.mask_root else Path("")
    ckpt_path = Path(args.elrep_ckpt).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data path: {data_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing ElRep checkpoint: {ckpt_path}")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.cwd() / f"elrep_{dataset}_curated_rise_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    default_tokens = {
        "wb95": WATERBIRDS95_TOKENS,
        "wb100": WATERBIRDS100_TOKENS,
        "redmeat": REDMEAT_TOKENS,
    }[dataset]
    requested_tokens = [x.strip() for x in (args.image_names or ",".join(default_tokens)).split(",") if x.strip()]

    if dataset in {"wb95", "wb100"}:
        entries, missing = resolve_waterbirds_entries(data_path, args.split, requested_tokens)
        class_names = WATERBIRDS_CLASSES
    else:
        entries, missing = resolve_redmeat_entries(data_path, args.split, requested_tokens, REDMEAT_CLASSES)
        class_names = REDMEAT_CLASSES

    if missing:
        msg = f"[WARN] Missing {len(missing)} requested images: {missing}"
        if not args.allow_missing:
            raise RuntimeError(msg)
        print(msg, flush=True)
    if not entries:
        raise RuntimeError("No curated images resolved.")

    req_device = str(args.device)
    if req_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(req_device)

    print(f"[INFO] dataset={dataset}", flush=True)
    print(f"[INFO] data_path={data_path}", flush=True)
    print(f"[INFO] mask_root={mask_root if str(mask_root) else 'NONE'}", flush=True)
    print(f"[INFO] checkpoint={ckpt_path}", flush=True)
    print(f"[INFO] output_dir={out_dir}", flush=True)
    print(f"[INFO] resolved={len(entries)}/{len(requested_tokens)} split={args.split}", flush=True)
    print(f"[INFO] device={device}", flush=True)

    model, load_meta = load_resnet50_elrep(ckpt_path, num_classes=len(class_names), device=device)
    print(f"[INFO] elrep load: {load_meta}", flush=True)

    if args.rise_masks_path:
        rise_masks_path = Path(args.rise_masks_path).expanduser().resolve()
    else:
        p1_token = str(args.rise_p1).replace(".", "p")
        rise_masks_path = out_dir / f"rise_masks_n{args.rise_num_masks}_s{args.rise_grid_size}_p{p1_token}_seed{args.rise_seed}.npy"
    rise_masks = load_or_create_rise_masks(
        rise_masks_path,
        num_masks=args.rise_num_masks,
        input_size=(224, 224),
        grid_size=args.rise_grid_size,
        p1=args.rise_p1,
        seed=args.rise_seed,
        device=device,
    )
    rise = RISEExplainer(GenericProbModel(model).to(device).eval(), rise_masks, args.rise_gpu_batch, args.rise_p1)
    eval_tf = build_eval_transform()

    rows: List[Dict[str, object]] = []
    grid_rows: List[Dict[str, object]] = []
    use_label_target = args.target_class == "label"

    for idx, entry in enumerate(entries):
        img = Image.open(entry.image_path).convert("RGB")
        image_rgb = np.array(img, dtype=np.uint8)
        x = eval_tf(img).unsqueeze(0).to(device)

        sample_dir = out_dir / "samples" / f"{idx:03d}_{entry.class_name}_{safe_token(entry.image_path.name)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        original_path = sample_dir / "original_image.png"
        save_rgb(original_path, image_rgb)

        mask_path = resolve_mask_path(dataset, mask_root, data_path, entry.class_name, entry.image_path) if args.mask_root else None
        gt_overlay_path = ""
        if mask_path is not None:
            gt_vis = save_gt_mask_variants(mask_path, image_rgb, sample_dir)
            gt_overlay_path = str(sample_dir / "gt_mask_overlay_blue_red.png")

        with torch.no_grad():
            logits = _extract_logits(model(x))
            pred = int(logits.argmax(dim=1).item())
            conf = float(torch.softmax(logits, dim=1)[0, pred].item())
            target_cls = int(entry.label if use_label_target else pred)
            sal = rise(x)[target_cls].detach().cpu().numpy().astype(np.float32)

        vis = save_saliency_variants("elrep", sal, image_rgb, sample_dir)
        overlay_path = sample_dir / "elrep_saliency_overlay_blue_red.png"
        heatmap_path = sample_dir / "elrep_saliency_heatmap_blue_red.png"
        contour_path = sample_dir / "elrep_saliency_contours_on_image.png"

        info = {
            "index": idx,
            "request_token": entry.request_token,
            "resolved_source": entry.source,
            "image_path": str(entry.image_path),
            "label": entry.label,
            "class_name": entry.class_name,
            "pred": pred,
            "confidence": conf,
            "saliency_target_class": target_cls,
            "mask_path": str(mask_path) if mask_path is not None else "",
            "sample_dir": str(sample_dir),
        }
        with open(sample_dir / "sample_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        rows.append(info)
        grid_rows.append(
            {
                "image_path": str(original_path),
                "gt_overlay_path": gt_overlay_path,
                "elrep_overlay_path": str(overlay_path),
                "elrep_heatmap_path": str(heatmap_path),
                "elrep_contour_path": str(contour_path),
            }
        )

    csv_path = out_dir / "sample_index.csv"
    fieldnames = [
        "index",
        "request_token",
        "resolved_source",
        "image_path",
        "label",
        "class_name",
        "pred",
        "confidence",
        "saliency_target_class",
        "mask_path",
        "sample_dir",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    grid_path = out_dir / "grid" / f"elrep_{dataset}_curated_rise_grid.png"
    make_grid(grid_rows, grid_path)

    summary = {
        "dataset": dataset,
        "data_path": str(data_path),
        "mask_root": str(mask_root) if args.mask_root else "",
        "checkpoint": str(ckpt_path),
        "output_dir": str(out_dir),
        "target_class": args.target_class,
        "split": args.split,
        "classes": class_names,
        "requested_image_names": requested_tokens,
        "resolved_count": len(entries),
        "missing_image_names": missing,
        "rise": {
            "num_masks": args.rise_num_masks,
            "grid_size": args.rise_grid_size,
            "p1": args.rise_p1,
            "gpu_batch": args.rise_gpu_batch,
            "seed": args.rise_seed,
            "masks_path": str(rise_masks_path),
        },
        "load_meta": load_meta,
        "grid_path": str(grid_path),
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "ElRep curated RISE saliency artifacts.\n"
            "Per-sample folders are under samples/.\n"
            "Each sample includes original_image.png, ElRep saliency variants, optional GT mask variants, and sample_info.json.\n"
            f"Grid: {grid_path}\n"
        )

    print(f"[DONE] Generated {len(rows)} ElRep saliency sample folders.", flush=True)
    print(f"[DONE] Grid: {grid_path}", flush=True)
    print(f"[DONE] Summary: {out_dir / 'run_summary.json'}", flush=True)
    print(f"[DONE] Sample index: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
