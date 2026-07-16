#!/usr/bin/env python3
"""Utilities for RedMeat Optuna objectives based on
log_optim_num = log(val_acc) - beta * ig_fwd_kl.

This module is intentionally standalone so new "optimnum" sweeps can run
without altering existing sweep scripts.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# Import GALS internals (this file lives in GALS/RedMeat_Runs).
GALS_ROOT = Path(__file__).resolve().parent.parent
if str(GALS_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(GALS_ROOT))

import datasets  # noqa: E402
from datasets import normalizations  # noqa: E402
import utils.general_utils as gu  # noqa: E402
import utils.loss_utils as lu  # noqa: E402


def _load_cfg(config_path: str, overrides: Optional[List[str]] = None):
    """Build the merged OmegaConf object exactly like main.py."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    base_cfg = OmegaConf.load(str(GALS_ROOT / "configs" / "base.yaml"))
    cli = OmegaConf.from_dotlist(list(overrides or []))
    args = OmegaConf.merge(base_cfg, cfg, cli)
    return args


def _dataset_cls(dataset_name: str):
    if dataset_name == "food_subset":
        from datasets.food import FoodSubset as Dataset

        return Dataset
    if dataset_name == "waterbirds":
        from datasets.waterbirds import Waterbirds as Dataset

        return Dataset
    if dataset_name == "waterbirds_background":
        from datasets.waterbirds_background_task import WaterbirdsBackgroundTask as Dataset

        return Dataset
    if dataset_name == "coco_gender":
        from datasets.coco import COCOGender as Dataset

        return Dataset
    raise NotImplementedError(f"Unsupported dataset for optimnum eval: {dataset_name}")


def _approach_cls(approach_name: str):
    if approach_name == "generic":
        from approaches.generic_cnn import GenericCNN as Approach

        return Approach
    if approach_name == "abn":
        from approaches.abn import ABN as Approach

        return Approach
    if approach_name == "coco_gender":
        from approaches.coco_gender import COCOGenderCNN as Approach

        return Approach
    if approach_name == "coco_abn":
        from approaches.coco_abn import COCOABN as Approach

        return Approach
    raise NotImplementedError(f"Unsupported approach for optimnum eval: {approach_name}")


def _resolve_checkpoint_path(checkpoint_path: str, *, cwd: Optional[str] = None) -> str:
    if os.path.isabs(checkpoint_path):
        return checkpoint_path
    if cwd is None:
        cwd = os.getcwd()
    return os.path.abspath(os.path.join(cwd, checkpoint_path))


def _forward_kl_from_maps(pred_maps: torch.Tensor, gt_maps: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute mean forward KL = KL(gt || pred) over a batch of saliency maps.

    pred_maps: [B, 1, H, W]
    gt_maps:   [B, 1, H, W]
    """
    if gt_maps.shape[-2:] != pred_maps.shape[-2:]:
        gt_maps = F.interpolate(gt_maps, size=pred_maps.shape[-2:], mode="nearest")

    p = pred_maps.reshape(pred_maps.shape[0], -1)
    q = gt_maps.reshape(gt_maps.shape[0], -1)

    p = torch.clamp(p, min=0.0) + eps
    q = torch.clamp(q, min=0.0) + eps

    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)

    fwd_kl = (q * (torch.log(q) - torch.log(p))).sum(dim=1)
    return float(fwd_kl.mean().detach().cpu().item())


def _integrated_gradients_saliency(
    inputs: torch.Tensor,
    cls_loss_fn,
    *,
    ig_steps: int = 16,
) -> torch.Tensor:
    """Compute RRR-style saliency via integrated gradients.

    Uses a zero baseline and returns channel-collapsed |IG| saliency [B, 1, H, W].
    """
    if ig_steps < 1:
        raise ValueError(f"ig_steps must be >= 1, got {ig_steps}")

    baseline = torch.zeros_like(inputs)
    delta = inputs - baseline
    grad_sum = torch.zeros_like(inputs)

    # Trapezoid/riemann-style integral approximation over alpha in (0, 1].
    alphas = torch.linspace(
        1.0 / float(ig_steps),
        1.0,
        ig_steps,
        device=inputs.device,
        dtype=inputs.dtype,
    )

    for alpha in alphas:
        scaled_inputs = (baseline + alpha * delta).detach()
        scaled_inputs.requires_grad_(True)
        cls_loss = cls_loss_fn(scaled_inputs)
        grads = torch.autograd.grad(
            cls_loss,
            scaled_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        grad_sum += grads

    avg_grads = grad_sum / float(ig_steps)
    ig = delta * avg_grads
    return ig.abs().sum(dim=1, keepdim=True)


def _loss_settings_for_method(cfg, method: str) -> Optional[Dict]:
    losses = cfg.EXP.LOSSES

    if method in ("gals", "rrr"):
        src = losses.GRADIENT_OUTSIDE
    elif method == "gradcam":
        src = losses.GRADCAM
    elif method in ("abn_att", "abn_vit"):
        src = losses.ABN_SUPERVISION
    else:
        return None

    out = {"GT": str(src.GT)}
    if "COMBINE_ATT_MODE" in src:
        out["COMBINE_ATT_MODE"] = str(src.COMBINE_ATT_MODE)
    if "INVERT" in src:
        out["INVERT"] = bool(src.INVERT)
    return out


def compute_main_checkpoint_optimnum(
    *,
    config_path: str,
    overrides: Optional[List[str]],
    checkpoint_path: str,
    method: str,
    beta: float,
    data_root: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    ig_steps: int = 16,
) -> Dict[str, float]:
    """Evaluate val_acc and IG forward-KL for a trained main.py checkpoint.

    Returns dict with:
      val_acc (fraction in [0,1]),
      val_ig_fwd_kl,
      log_optim_num,
      optim_value (alias for log_optim_num for sweep compatibility).
    """
    args = _load_cfg(config_path, overrides=overrides)

    # Force data location from caller for robustness. In some SLURM runs,
    # relying on OmegaConf dotlist overrides alone can still leave DATA.ROOT
    # at defaults (e.g., "./data"), which breaks post-trial metric eval.
    if data_root is not None:
        args.DATA.ROOT = str(data_root)
    if dataset_dir is not None:
        args.DATA.FOOD_SUBSET_DIR = str(dataset_dir)
        args.DATA.SUBDIR = str(dataset_dir)

    loss_settings = _loss_settings_for_method(args, method)

    # For segmentation-guided configs (e.g., gals_ourmasks), some configs set
    # SEG_TRAIN_ONLY=True, which disables val masks. We force val segmentation on
    # here because optimnum needs IG forward-KL on the validation split.
    if loss_settings is not None and str(loss_settings.get("GT", "")).lower() == "segmentation":
        args.DATA.SEG_TRAIN_ONLY = False
        args.DATA.RETURN_SEG = True

    mean = normalizations.normalizations[args.DATA.NORMALIZATION]["mean"]
    std = normalizations.normalizations[args.DATA.NORMALIZATION]["std"]
    transform = transforms.Compose(
        [
            transforms.Resize((args.DATA.SIZE, args.DATA.SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    Dataset = _dataset_cls(str(args.DATA.DATASET))
    if str(args.DATA.DATASET) == "food_subset":
        resolved_root = os.path.join(str(args.DATA.ROOT), str(args.DATA.FOOD_SUBSET_DIR))
        resolved_train = os.path.join(resolved_root, "train")
        if not os.path.isdir(resolved_train):
            raise FileNotFoundError(
                "Resolved food_subset train split not found for optimnum eval: "
                f"{resolved_train} (DATA.ROOT={args.DATA.ROOT}, "
                f"DATA.FOOD_SUBSET_DIR={args.DATA.FOOD_SUBSET_DIR})"
            )
    val_dataset = Dataset(root=args.DATA.ROOT, cfg=args, transform=transform, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.DATA.BATCH_SIZE,
        num_workers=args.DATA.NUM_WORKERS,
        shuffle=False,
    )

    Approach = _approach_cls(str(args.EXP.APPROACH))
    approach = Approach(args, [None, None])

    ckpt_file = _resolve_checkpoint_path(checkpoint_path)
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint for optimnum eval not found: {ckpt_file}")
    approach.load_checkpoint(ckpt_file)
    approach.net.eval()

    total = 0
    total_correct = 0
    fwdkl_weighted_sum = 0.0
    fwdkl_count = 0

    for batch in val_loader:
        inputs = batch["image"].to(approach.device)
        labels = batch["label"].to(approach.device)

        with torch.no_grad():
            if str(args.EXP.APPROACH) == "abn":
                provided_att = approach.get_provided_att(batch)
                if str(args.EXP.MODEL) == "resnet50_abn":
                    _att_logits, logits, _ = approach.net(inputs, provided_att=provided_att)
                else:
                    logits = approach.net(inputs, provided_att=provided_att)
            else:
                net_out = approach.net(inputs)
                if isinstance(net_out, (tuple, list)):
                    logits = net_out[0]
                else:
                    logits = net_out

        _, _, preds = gu.calc_preds(
            logits,
            approach.activation,
            approach.classifier_classes,
            enforce_binary=approach.enforce_binary_eval,
        )

        total += int(labels.shape[0])
        total_correct += int((preds == labels).sum().item())

        if loss_settings is not None:
            if str(args.EXP.APPROACH) == "abn":
                provided_att = approach.get_provided_att(batch)
            else:
                provided_att = None

            def _cls_loss_fn(scaled_inputs: torch.Tensor) -> torch.Tensor:
                if str(args.EXP.APPROACH) == "abn":
                    if str(args.EXP.MODEL) == "resnet50_abn":
                        _att_logits, scaled_logits, _ = approach.net(scaled_inputs, provided_att=provided_att)
                    else:
                        scaled_logits = approach.net(scaled_inputs, provided_att=provided_att)
                else:
                    scaled_out = approach.net(scaled_inputs)
                    if isinstance(scaled_out, (tuple, list)):
                        scaled_logits = scaled_out[0]
                    else:
                        scaled_logits = scaled_out

                return lu.calc_classification_loss(
                    approach.criterion,
                    approach.activation,
                    approach.classifier_classes,
                    scaled_logits,
                    labels,
                )

            saliency = _integrated_gradients_saliency(inputs, _cls_loss_fn, ig_steps=ig_steps)

            gt_maps, pred_maps = lu.get_gt_pred_attentions(
                loss_settings,
                "OPTIMNUM_IG_FWDKL",
                batch,
                inputs,
                approach.device,
                pred_attentions=saliency,
            )
            if gt_maps is not None and pred_maps is not None and pred_maps.shape[0] > 0:
                b_fwd = _forward_kl_from_maps(pred_maps, gt_maps)
                n = int(pred_maps.shape[0])
                fwdkl_weighted_sum += b_fwd * n
                fwdkl_count += n

    if total == 0:
        raise RuntimeError("Validation loader had zero samples while computing optimnum.")

    val_acc = float(total_correct / total)
    val_ig_fwd_kl = float(fwdkl_weighted_sum / fwdkl_count) if fwdkl_count > 0 else float("nan")

    if not np.isfinite(val_ig_fwd_kl):
        # If no valid guidance masks appear in val, fall back to accuracy-only objective.
        val_ig_fwd_kl = 0.0

    log_optim_num = float(math.log(max(val_acc, 1e-12)) - float(beta) * val_ig_fwd_kl)

    return {
        "val_acc": val_acc,
        "val_ig_fwd_kl": val_ig_fwd_kl,
        "log_optim_num": log_optim_num,
        # Compatibility alias used by sweep drivers (now in log-space).
        "optim_value": log_optim_num,
        # Legacy key retained for backward compatibility.
        "val_rev_kl": val_ig_fwd_kl,
    }


def compute_guided_checkpoint_optimnum_png(
    *,
    checkpoint_path: str,
    data_path: str,
    gt_path: str,
    classes: Optional[List[str]],
    split_col: str,
    label_col: str,
    path_col: str,
    beta: float,
    batch_size: int,
    num_workers: int = 4,
    ig_steps: int = 16,
) -> Dict[str, float]:
    """Compute optimnum for guided PNG-mask runner checkpoints."""
    from RedMeat_Runs import run_guided_redmeat as rg_png
    import run_guided_waterbird as base

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            base.Brighten(8.0),
        ]
    )

    val_ds = rg_png.RedMeatMetadataDataset(
        data_root=data_path,
        split="val",
        image_transform=img_tf,
        mask_root=gt_path,
        mask_transform=mask_tf,
        return_mask=True,
        return_path=True,
        classes=classes,
        split_col=split_col,
        label_col=label_col,
        path_col=path_col,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(val_ds.classes)
    model = base.make_cam_model(num_classes, model_name="resnet50", pretrained=False).to(device)
    ckpt = _resolve_checkpoint_path(checkpoint_path)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    total = 0
    total_correct = 0
    fwdkl_weighted_sum = 0.0
    fwdkl_count = 0

    for images, labels, masks, _paths in loader:
        images = images.to(device)
        labels = labels.to(device).long()
        masks = masks.to(device)

        with torch.no_grad():
            logits, _feats = model(images)
            preds = logits.argmax(dim=1)
        total += int(labels.shape[0])
        total_correct += int((preds == labels).sum().item())

        def _cls_loss_fn(scaled_images: torch.Tensor) -> torch.Tensor:
            scaled_logits, _scaled_feats = model(scaled_images)
            return F.cross_entropy(scaled_logits, labels)

        saliency = _integrated_gradients_saliency(images, _cls_loss_fn, ig_steps=ig_steps)

        b_fwd = _forward_kl_from_maps(saliency, masks)
        n = int(images.shape[0])
        fwdkl_weighted_sum += b_fwd * n
        fwdkl_count += n

    if total == 0:
        raise RuntimeError("Validation loader had zero samples while computing guided optimnum.")

    val_acc = float(total_correct / total)
    val_ig_fwd_kl = float(fwdkl_weighted_sum / fwdkl_count) if fwdkl_count > 0 else 0.0
    log_optim_num = float(math.log(max(val_acc, 1e-12)) - float(beta) * val_ig_fwd_kl)

    return {
        "val_acc": val_acc,
        "val_ig_fwd_kl": val_ig_fwd_kl,
        "log_optim_num": log_optim_num,
        "optim_value": log_optim_num,
        "val_rev_kl": val_ig_fwd_kl,
    }


def compute_guided_checkpoint_optimnum_pth(
    *,
    checkpoint_path: str,
    data_path: str,
    att_path: str,
    classes: Optional[List[str]],
    split_col: str,
    label_col: str,
    path_col: str,
    att_key: str,
    att_combine: str,
    att_norm01: bool,
    att_brighten: float,
    beta: float,
    batch_size: int,
    num_workers: int = 4,
    ig_steps: int = 16,
) -> Dict[str, float]:
    """Compute optimnum for guided .pth-attention runner checkpoints."""
    from RedMeat_Runs import run_guided_redmeat_gals_vitatt as rg_vit
    import run_guided_waterbird as base

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_ds = rg_vit.RedMeatMetadataDatasetPthAttention(
        data_root=data_path,
        split="val",
        image_transform=img_tf,
        attention_root=att_path,
        attention_key=att_key,
        attention_combine=att_combine,
        attention_size=224,
        attention_normalize_01=att_norm01,
        attention_brighten=att_brighten,
        return_attention=True,
        return_path=True,
        classes=classes,
        split_col=split_col,
        label_col=label_col,
        path_col=path_col,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(val_ds.classes)
    model = base.make_cam_model(num_classes, model_name="resnet50", pretrained=False).to(device)
    ckpt = _resolve_checkpoint_path(checkpoint_path)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    total = 0
    total_correct = 0
    fwdkl_weighted_sum = 0.0
    fwdkl_count = 0

    for images, labels, att_maps, _paths in loader:
        images = images.to(device)
        labels = labels.to(device).long()
        att_maps = att_maps.to(device)

        with torch.no_grad():
            logits, _feats = model(images)
            preds = logits.argmax(dim=1)

        total += int(labels.shape[0])
        total_correct += int((preds == labels).sum().item())

        def _cls_loss_fn(scaled_images: torch.Tensor) -> torch.Tensor:
            scaled_logits, _scaled_feats = model(scaled_images)
            return F.cross_entropy(scaled_logits, labels)

        saliency = _integrated_gradients_saliency(images, _cls_loss_fn, ig_steps=ig_steps)

        b_fwd = _forward_kl_from_maps(saliency, att_maps)
        n = int(images.shape[0])
        fwdkl_weighted_sum += b_fwd * n
        fwdkl_count += n

    if total == 0:
        raise RuntimeError("Validation loader had zero samples while computing guided optimnum.")

    val_acc = float(total_correct / total)
    val_ig_fwd_kl = float(fwdkl_weighted_sum / fwdkl_count) if fwdkl_count > 0 else 0.0
    log_optim_num = float(math.log(max(val_acc, 1e-12)) - float(beta) * val_ig_fwd_kl)

    return {
        "val_acc": val_acc,
        "val_ig_fwd_kl": val_ig_fwd_kl,
        "log_optim_num": log_optim_num,
        "optim_value": log_optim_num,
        "val_rev_kl": val_ig_fwd_kl,
    }
