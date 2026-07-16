#!/usr/bin/env python3
"""
Grid-search alternative optim_value definitions for guided RedMeat.

Workflow:
1) Train N guided models with random hyperparameters.
2) For each checkpoint, compute val saliency alignment loss for a grid of:
   - saliency methods (default: 5)
   - loss functions (default: 4)
3) Define optim_value = log(val_acc) - beta * val_saliency_loss for each combo.
4) For each combo, pick the best checkpoint by optim_value and run test eval.
5) Report correlation between test accuracy and optim_value across all trials
   for each combo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


# Ensure local package imports work no matter the cwd.
GALS_ROOT = Path(__file__).resolve().parent.parent
if str(GALS_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(GALS_ROOT))

_RGM = None


def _get_rgm():
    global _RGM
    if _RGM is None:
        from RedMeat_Runs import run_guided_redmeat as _rgm  # noqa: E402

        _RGM = _rgm
    return _RGM


DEFAULT_GT_NEWCLIP = (
    "/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/"
    "results_redmeat_openclip_dinovit/val/prediction_cmap/"
)

DEFAULT_METHODS = [
    "ig",
    "grad_abs",
    "grad_x_input",
    "smoothgrad",
    "smoothgrad_x_input",
]
DEFAULT_LOSSES = [
    "fwd_kl",
    "rev_kl",
    "jsd",
    "bce_dice",
]


def loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def write_row(csv_path: str, row: Dict, header: List[str]) -> None:
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _parse_classes(classes_csv: str) -> List[str]:
    return [c.strip() for c in str(classes_csv).split(",") if c.strip()]


def _parse_csv_list(csv_text: str) -> List[str]:
    return [x.strip() for x in str(csv_text).split(",") if x.strip()]


def _pct_or_nan(v):
    return float(v) if v is not None else float("nan")


def _build_val_loader_with_masks(
    *,
    data_path: str,
    gt_path: str,
    classes: List[str],
    split_col: str,
    label_col: str,
    path_col: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, int]:
    rgm = _get_rgm()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_tf = transforms.Compose(
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
            rgm.base.Brighten(8.0),
        ]
    )

    val_ds = rgm.RedMeatMetadataDataset(
        data_root=data_path,
        split="val",
        image_transform=image_tf,
        mask_root=gt_path,
        mask_transform=mask_tf,
        return_mask=True,
        return_path=True,
        classes=classes,
        split_col=split_col,
        label_col=label_col,
        path_col=path_col,
    )
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader, len(val_ds.classes)


def _build_test_loader(
    *,
    data_path: str,
    classes: List[str],
    split_col: str,
    label_col: str,
    path_col: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, int]:
    rgm = _get_rgm()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_ds = rgm.RedMeatMetadataDataset(
        data_root=data_path,
        split="test",
        image_transform=image_tf,
        return_mask=False,
        return_path=True,
        classes=classes,
        split_col=split_col,
        label_col=label_col,
        path_col=path_col,
    )
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader, len(test_ds.classes)


def _load_guided_resnet50(checkpoint_path: str, num_classes: int, device: torch.device):
    rgm = _get_rgm()
    model = rgm.base.make_cam_model(num_classes, model_name="resnet50", pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _map_to_prob(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: [B, 1, H, W]
    flat = x.reshape(x.shape[0], -1)
    flat = torch.clamp(flat, min=0.0) + eps
    flat = flat / flat.sum(dim=1, keepdim=True)
    return flat


def _map_to_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    out = (flat - lo) / (hi - lo + eps)
    return out.reshape_as(x)


def _loss_from_maps(loss_name: str, saliency: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if gt.shape[-2:] != saliency.shape[-2:]:
        gt = F.interpolate(gt, size=saliency.shape[-2:], mode="nearest")

    if loss_name == "fwd_kl":
        p = _map_to_prob(saliency, eps=eps)
        q = _map_to_prob(gt, eps=eps)
        return (q * (torch.log(q) - torch.log(p))).sum(dim=1).mean()

    if loss_name == "rev_kl":
        p = _map_to_prob(saliency, eps=eps)
        q = _map_to_prob(gt, eps=eps)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=1).mean()

    if loss_name == "jsd":
        p = _map_to_prob(saliency, eps=eps)
        q = _map_to_prob(gt, eps=eps)
        m = 0.5 * (p + q)
        kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=1)
        kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=1)
        return 0.5 * (kl_pm + kl_qm).mean()

    if loss_name == "bce_dice":
        s01 = _map_to_01(saliency, eps=eps)
        g01 = torch.clamp(gt, 0.0, 1.0)
        s01 = torch.clamp(s01, eps, 1.0 - eps)
        g01 = torch.clamp(g01, eps, 1.0 - eps)

        bce = F.binary_cross_entropy(s01, g01)

        s_flat = s01.reshape(s01.shape[0], -1)
        g_flat = g01.reshape(g01.shape[0], -1)
        inter = (s_flat * g_flat).sum(dim=1)
        denom = s_flat.sum(dim=1) + g_flat.sum(dim=1)
        dice_loss = 1.0 - (2.0 * inter + eps) / (denom + eps)
        return 0.5 * bce + 0.5 * dice_loss.mean()

    raise ValueError(f"Unknown loss_name: {loss_name}")


def _saliency_grad(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    multiply_input: bool,
) -> torch.Tensor:
    x = images.detach().clone().requires_grad_(True)
    logits, _ = model(x)
    cls_loss = F.cross_entropy(logits, labels)
    grads = torch.autograd.grad(cls_loss, x, retain_graph=False, create_graph=False, allow_unused=False)[0]
    if multiply_input:
        sal = (grads * x).abs().sum(dim=1, keepdim=True)
    else:
        sal = grads.abs().sum(dim=1, keepdim=True)
    return sal.detach()


def _saliency_ig(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    ig_steps: int,
) -> torch.Tensor:
    if ig_steps < 1:
        raise ValueError(f"ig_steps must be >= 1, got {ig_steps}")

    baseline = torch.zeros_like(images)
    delta = images - baseline
    grad_sum = torch.zeros_like(images)

    alphas = torch.linspace(
        1.0 / float(ig_steps),
        1.0,
        ig_steps,
        device=images.device,
        dtype=images.dtype,
    )

    for alpha in alphas:
        x = (baseline + alpha * delta).detach().requires_grad_(True)
        logits, _ = model(x)
        cls_loss = F.cross_entropy(logits, labels)
        grads = torch.autograd.grad(cls_loss, x, retain_graph=False, create_graph=False, allow_unused=False)[0]
        grad_sum += grads

    avg_grads = grad_sum / float(ig_steps)
    ig = delta * avg_grads
    return ig.abs().sum(dim=1, keepdim=True).detach()


def _saliency_smoothgrad(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_samples: int,
    noise_std: float,
    multiply_input: bool,
) -> torch.Tensor:
    if n_samples < 1:
        raise ValueError(f"smoothgrad_samples must be >=1, got {n_samples}")
    sal_sum = None
    for _ in range(int(n_samples)):
        noise = torch.randn_like(images) * float(noise_std)
        x = (images + noise).detach().requires_grad_(True)
        logits, _ = model(x)
        cls_loss = F.cross_entropy(logits, labels)
        grads = torch.autograd.grad(cls_loss, x, retain_graph=False, create_graph=False, allow_unused=False)[0]
        if multiply_input:
            sal = (grads * x).abs().sum(dim=1, keepdim=True)
        else:
            sal = grads.abs().sum(dim=1, keepdim=True)
        if sal_sum is None:
            sal_sum = sal
        else:
            sal_sum = sal_sum + sal
    return (sal_sum / float(n_samples)).detach()


def _compute_saliency(
    method: str,
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    ig_steps: int,
    smoothgrad_samples: int,
    smoothgrad_noise_std: float,
) -> torch.Tensor:
    if method == "ig":
        return _saliency_ig(model, images, labels, ig_steps=ig_steps)
    if method == "grad_abs":
        return _saliency_grad(model, images, labels, multiply_input=False)
    if method == "grad_x_input":
        return _saliency_grad(model, images, labels, multiply_input=True)
    if method == "smoothgrad":
        return _saliency_smoothgrad(
            model,
            images,
            labels,
            n_samples=smoothgrad_samples,
            noise_std=smoothgrad_noise_std,
            multiply_input=False,
        )
    if method == "smoothgrad_x_input":
        return _saliency_smoothgrad(
            model,
            images,
            labels,
            n_samples=smoothgrad_samples,
            noise_std=smoothgrad_noise_std,
            multiply_input=True,
        )
    raise ValueError(f"Unknown saliency method: {method}")


@torch.no_grad()
def _val_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    total = 0
    correct = 0
    for images, labels, _masks, _paths in loader:
        images = images.to(device)
        labels = labels.to(device).long()
        logits, _ = model(images)
        preds = logits.argmax(dim=1)
        total += int(labels.shape[0])
        correct += int((preds == labels).sum().item())
    if total == 0:
        raise RuntimeError("Validation loader is empty while computing val accuracy.")
    return float(correct / total)


def score_checkpoint_grid(
    *,
    checkpoint_path: str,
    val_loader: DataLoader,
    num_classes: int,
    methods: List[str],
    losses: List[str],
    beta: float,
    ig_steps: int,
    smoothgrad_samples: int,
    smoothgrad_noise_std: float,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model = _load_guided_resnet50(checkpoint_path, num_classes, device)
    val_acc = _val_accuracy(model, val_loader, device)

    loss_sums = {(m, l): 0.0 for m in methods for l in losses}
    n_total = 0

    model.eval()
    for images, labels, masks, _paths in val_loader:
        images = images.to(device)
        labels = labels.to(device).long()
        masks = masks.to(device)
        bs = int(images.shape[0])
        n_total += bs

        saliency_by_method = {}
        for m in methods:
            saliency_by_method[m] = _compute_saliency(
                m,
                model,
                images,
                labels,
                ig_steps=ig_steps,
                smoothgrad_samples=smoothgrad_samples,
                smoothgrad_noise_std=smoothgrad_noise_std,
            )

        for m in methods:
            saliency = saliency_by_method[m]
            for l in losses:
                loss_val = _loss_from_maps(l, saliency, masks)
                loss_sums[(m, l)] += float(loss_val.detach().cpu().item()) * bs

    if n_total == 0:
        raise RuntimeError("Validation loader is empty while scoring optim_value grid.")

    out: Dict[str, Dict[str, float]] = {}
    for m in methods:
        for l in losses:
            avg_loss = float(loss_sums[(m, l)] / float(n_total))
            optim_value = float(math.log(max(val_acc, 1e-12)) - float(beta) * avg_loss)
            key = f"{m}__{l}"
            out[key] = {
                "saliency_method": m,
                "loss_name": l,
                "val_acc": float(val_acc),
                "val_saliency_loss": avg_loss,
                "optim_value": optim_value,
            }
    return out


def run_test_from_checkpoint(
    *,
    checkpoint_path: str,
    data_path: str,
    classes: List[str],
    split_col: str,
    label_col: str,
    path_col: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, float]:
    rgm = _get_rgm()
    test_loader, num_classes = _build_test_loader(
        data_path=data_path,
        classes=classes,
        split_col=split_col,
        label_col=label_col,
        path_col=path_col,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = _load_guided_resnet50(checkpoint_path, num_classes, device)
    test_loss, test_acc, _cls_acc, per_group, worst_group = rgm.evaluate_test(model, test_loader, num_classes)
    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "per_group": float(per_group),
        "worst_group": float(worst_group),
    }


def _corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    s1 = pd.Series(x)
    s2 = pd.Series(y)
    try:
        return float(s1.corr(s2, method="spearman"))
    except Exception:
        return float("nan")


def _sample_hparams(args, rng: np.random.Generator):
    return {
        "attention_epoch": int(rng.integers(args.attn_min, args.attn_max + 1)),
        "kl_lambda": loguniform(rng, args.kl_min, args.kl_max),
        "kl_incr": 0.0,
        "base_lr": loguniform(rng, args.base_lr_min, args.base_lr_max),
        "classifier_lr": loguniform(rng, args.cls_lr_min, args.cls_lr_max),
        "lr2_mult": loguniform(rng, args.lr2_mult_min, args.lr2_mult_max),
    }


def main():
    p = argparse.ArgumentParser(
        description=(
            "Run random guided RedMeat trials, evaluate multiple saliency/loss "
            "optim_value definitions, and report test correlation."
        )
    )
    p.add_argument(
        "--data-path",
        default="/home/ryreu/guided_cnn/Food101/data/food-101-redmeat",
        help="RedMeat dataset root containing all_images.csv",
    )
    p.add_argument(
        "--gt-path",
        default=DEFAULT_GT_NEWCLIP,
        help="GT mask root (defaults to GT_NEWCLIP OpenCLIP-DINOvIT masks).",
    )
    p.add_argument("--split-col", default="split")
    p.add_argument("--label-col", default="label")
    p.add_argument("--path-col", default="abs_file_path")
    p.add_argument(
        "--classes",
        default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon",
        help="Comma-separated class list",
    )

    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument(
        "--train-seed-mode",
        choices=["fixed", "by_trial"],
        default="fixed",
        help="Use one training seed for all trials, or seed+trial_id.",
    )
    p.add_argument("--num-epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--attn-min", type=int, default=0)
    p.add_argument("--attn-max", type=int, default=149)
    p.add_argument("--kl-min", type=float, default=1.0)
    p.add_argument("--kl-max", type=float, default=500.0)
    p.add_argument("--base-lr-min", type=float, default=1e-5)
    p.add_argument("--base-lr-max", type=float, default=5e-2)
    p.add_argument("--cls-lr-min", type=float, default=1e-5)
    p.add_argument("--cls-lr-max", type=float, default=5e-2)
    p.add_argument("--lr2-mult-min", type=float, default=1e-1)
    p.add_argument("--lr2-mult-max", type=float, default=3.0)

    p.add_argument(
        "--saliency-methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated saliency methods",
    )
    p.add_argument(
        "--losses",
        default=",".join(DEFAULT_LOSSES),
        help="Comma-separated loss names",
    )
    p.add_argument("--optim-beta", type=float, default=1.0)
    p.add_argument("--ig-steps", type=int, default=16)
    p.add_argument("--smoothgrad-samples", type=int, default=8)
    p.add_argument("--smoothgrad-noise-std", type=float, default=0.10)

    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: /home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_optim_grid_<timestamp>",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining trials if a trial fails.",
    )

    args = p.parse_args()

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"Missing data path: {args.data_path}")
    if not os.path.isdir(args.gt_path):
        raise FileNotFoundError(f"Missing GT mask path: {args.gt_path}")

    methods = _parse_csv_list(args.saliency_methods)
    losses = _parse_csv_list(args.losses)
    if not methods:
        raise ValueError("No saliency methods provided.")
    if not losses:
        raise ValueError("No losses provided.")

    classes = _parse_classes(args.classes)
    if not classes:
        raise ValueError("Class list is empty.")

    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be >= 1")
    args.attn_max = min(int(args.attn_max), max(0, int(args.num_epochs) - 1))
    if args.attn_min > args.attn_max:
        args.attn_min = args.attn_max

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_optim_grid_{ts}"
    output_dir = os.path.abspath(output_dir)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    trials_csv = os.path.join(output_dir, "trials.csv")
    combo_csv = os.path.join(output_dir, "combo_scores.csv")
    corr_csv = os.path.join(output_dir, "correlations.csv")
    best_combo_csv = os.path.join(output_dir, "best_by_combo.csv")
    failures_csv = os.path.join(output_dir, "failed_trials.csv")
    summary_json = os.path.join(output_dir, "summary.json")

    print(f"[INFO] Output dir: {output_dir}", flush=True)
    print(f"[INFO] GT path (NEWCLIP): {args.gt_path}", flush=True)
    print(f"[INFO] Methods ({len(methods)}): {methods}", flush=True)
    print(f"[INFO] Losses  ({len(losses)}): {losses}", flush=True)
    print(f"[INFO] Total combos: {len(methods) * len(losses)}", flush=True)

    # Force checkpoint retention for this experiment.
    os.environ["SAVE_CHECKPOINTS"] = "1"

    rgm = _get_rgm()

    # Configure guided runner globals.
    rgm.num_epochs = int(args.num_epochs)
    rgm.batch_size = int(args.batch_size)
    rgm.checkpoint_dir = checkpoints_dir

    # Build val loader once for all checkpoint scoring.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_loader, val_num_classes = _build_val_loader_with_masks(
        data_path=args.data_path,
        gt_path=args.gt_path,
        classes=classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trial_header = [
        "trial",
        "train_seed",
        "attention_epoch",
        "kl_lambda",
        "kl_incr",
        "base_lr",
        "classifier_lr",
        "lr2_mult",
        "best_balanced_val_acc",
        "test_acc",
        "per_group",
        "worst_group",
        "checkpoint",
        "seconds",
    ]
    combo_header = [
        "trial",
        "checkpoint",
        "train_seed",
        "saliency_method",
        "loss_name",
        "combo_name",
        "val_acc_for_optim",
        "val_saliency_loss",
        "optim_value",
        "test_acc",
        "per_group",
        "worst_group",
    ]
    fail_header = ["trial", "train_seed", "error"]

    rng = np.random.default_rng(args.seed)
    all_trial_rows: List[Dict] = []
    all_combo_rows: List[Dict] = []
    failed_rows: List[Dict] = []

    run_args = argparse.Namespace(
        data_path=args.data_path,
        gt_path=args.gt_path,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
        classes=classes,
    )

    for trial_id in range(int(args.n_trials)):
        if args.train_seed_mode == "by_trial":
            train_seed = int(args.train_seed + trial_id)
        else:
            train_seed = int(args.train_seed)

        hp = _sample_hparams(args, rng)
        t0 = time.time()
        print(
            f"[TRIAL {trial_id:03d}] seed={train_seed} "
            f"attn={hp['attention_epoch']} kl={hp['kl_lambda']:.4f} "
            f"base_lr={hp['base_lr']:.3e} cls_lr={hp['classifier_lr']:.3e} "
            f"lr2_mult={hp['lr2_mult']:.4f}",
            flush=True,
        )

        try:
            rgm.SEED = train_seed
            rgm.base_lr = float(hp["base_lr"])
            rgm.classifier_lr = float(hp["classifier_lr"])
            rgm.lr2_mult = float(hp["lr2_mult"])

            best_balanced_val_acc, test_acc, per_group, worst_group, checkpoint_path = rgm.run_single(
                run_args,
                int(hp["attention_epoch"]),
                float(hp["kl_lambda"]),
                float(hp["kl_incr"]),
            )
            elapsed = int(time.time() - t0)

            if not checkpoint_path or str(checkpoint_path).upper() == "NONE":
                raise RuntimeError(
                    "Checkpoint was not saved. This experiment requires checkpoints. "
                    "Set SAVE_CHECKPOINTS=1."
                )
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            trial_row = {
                "trial": trial_id,
                "train_seed": train_seed,
                "attention_epoch": int(hp["attention_epoch"]),
                "kl_lambda": float(hp["kl_lambda"]),
                "kl_incr": float(hp["kl_incr"]),
                "base_lr": float(hp["base_lr"]),
                "classifier_lr": float(hp["classifier_lr"]),
                "lr2_mult": float(hp["lr2_mult"]),
                "best_balanced_val_acc": float(best_balanced_val_acc),
                "test_acc": float(test_acc),
                "per_group": float(per_group),
                "worst_group": float(worst_group),
                "checkpoint": checkpoint_path,
                "seconds": elapsed,
            }
            write_row(trials_csv, trial_row, trial_header)
            all_trial_rows.append(trial_row)

            grid_scores = score_checkpoint_grid(
                checkpoint_path=checkpoint_path,
                val_loader=val_loader,
                num_classes=val_num_classes,
                methods=methods,
                losses=losses,
                beta=float(args.optim_beta),
                ig_steps=int(args.ig_steps),
                smoothgrad_samples=int(args.smoothgrad_samples),
                smoothgrad_noise_std=float(args.smoothgrad_noise_std),
                device=device,
            )

            for combo_name, metrics in grid_scores.items():
                row = {
                    "trial": trial_id,
                    "checkpoint": checkpoint_path,
                    "train_seed": train_seed,
                    "saliency_method": metrics["saliency_method"],
                    "loss_name": metrics["loss_name"],
                    "combo_name": combo_name,
                    "val_acc_for_optim": float(metrics["val_acc"]),
                    "val_saliency_loss": float(metrics["val_saliency_loss"]),
                    "optim_value": float(metrics["optim_value"]),
                    "test_acc": float(test_acc),
                    "per_group": float(per_group),
                    "worst_group": float(worst_group),
                }
                write_row(combo_csv, row, combo_header)
                all_combo_rows.append(row)

            print(
                f"[TRIAL {trial_id:03d}] done in {elapsed}s | "
                f"test_acc={float(test_acc):.2f}% | checkpoint={checkpoint_path}",
                flush=True,
            )

        except Exception as exc:
            err_text = f"{type(exc).__name__}: {exc}"
            print(f"[TRIAL {trial_id:03d}] FAILED: {err_text}", flush=True)
            print(traceback.format_exc(), flush=True)
            fail_row = {"trial": trial_id, "train_seed": train_seed, "error": err_text}
            write_row(failures_csv, fail_row, fail_header)
            failed_rows.append(fail_row)
            if not args.continue_on_error:
                raise

    if not all_combo_rows:
        raise RuntimeError("No successful combo scores were recorded.")

    combo_df = pd.DataFrame(all_combo_rows)
    combo_df["optim_value"] = combo_df["optim_value"].astype(float)
    combo_df["test_acc"] = combo_df["test_acc"].astype(float)
    combo_df["per_group"] = combo_df["per_group"].astype(float)
    combo_df["worst_group"] = combo_df["worst_group"].astype(float)

    corr_header = [
        "combo_name",
        "saliency_method",
        "loss_name",
        "n",
        "pearson_test_acc_vs_optim",
        "spearman_test_acc_vs_optim",
        "pearson_per_group_vs_optim",
        "spearman_per_group_vs_optim",
        "pearson_worst_group_vs_optim",
        "spearman_worst_group_vs_optim",
    ]
    best_header = [
        "combo_name",
        "saliency_method",
        "loss_name",
        "best_trial",
        "best_checkpoint",
        "best_optim_value",
        "trial_test_acc_cached",
        "trial_per_group_cached",
        "trial_worst_group_cached",
        "test_acc_reeval",
        "per_group_reeval",
        "worst_group_reeval",
    ]

    # Cache test re-evaluation so repeated checkpoints across combos are evaluated once.
    reeval_cache: Dict[str, Dict[str, float]] = {}

    # Keep deterministic order.
    combo_names = sorted(combo_df["combo_name"].unique().tolist())
    corr_rows = []
    best_rows = []

    for combo_name in combo_names:
        sdf = combo_df[combo_df["combo_name"] == combo_name].copy()
        if sdf.empty:
            continue
        sdf = sdf.sort_values("optim_value", ascending=False)
        top = sdf.iloc[0]

        x = sdf["optim_value"].to_numpy(dtype=float)
        y_test = sdf["test_acc"].to_numpy(dtype=float)
        y_pg = sdf["per_group"].to_numpy(dtype=float)
        y_wg = sdf["worst_group"].to_numpy(dtype=float)

        corr_row = {
            "combo_name": combo_name,
            "saliency_method": str(top["saliency_method"]),
            "loss_name": str(top["loss_name"]),
            "n": int(sdf.shape[0]),
            "pearson_test_acc_vs_optim": _corr_pearson(x, y_test),
            "spearman_test_acc_vs_optim": _corr_spearman(x, y_test),
            "pearson_per_group_vs_optim": _corr_pearson(x, y_pg),
            "spearman_per_group_vs_optim": _corr_spearman(x, y_pg),
            "pearson_worst_group_vs_optim": _corr_pearson(x, y_wg),
            "spearman_worst_group_vs_optim": _corr_spearman(x, y_wg),
        }
        write_row(corr_csv, corr_row, corr_header)
        corr_rows.append(corr_row)

        ckpt = str(top["checkpoint"])
        if ckpt not in reeval_cache:
            reeval_cache[ckpt] = run_test_from_checkpoint(
                checkpoint_path=ckpt,
                data_path=args.data_path,
                classes=classes,
                split_col=args.split_col,
                label_col=args.label_col,
                path_col=args.path_col,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )

        reeval = reeval_cache[ckpt]
        best_row = {
            "combo_name": combo_name,
            "saliency_method": str(top["saliency_method"]),
            "loss_name": str(top["loss_name"]),
            "best_trial": int(top["trial"]),
            "best_checkpoint": ckpt,
            "best_optim_value": float(top["optim_value"]),
            "trial_test_acc_cached": float(top["test_acc"]),
            "trial_per_group_cached": float(top["per_group"]),
            "trial_worst_group_cached": float(top["worst_group"]),
            "test_acc_reeval": float(reeval["test_acc"]),
            "per_group_reeval": float(reeval["per_group"]),
            "worst_group_reeval": float(reeval["worst_group"]),
        }
        write_row(best_combo_csv, best_row, best_header)
        best_rows.append(best_row)

    # Print compact ranking by test-accuracy Spearman correlation.
    corr_sorted = sorted(
        corr_rows,
        key=lambda r: (
            float("-inf")
            if (r["spearman_test_acc_vs_optim"] is None or np.isnan(r["spearman_test_acc_vs_optim"]))
            else float(r["spearman_test_acc_vs_optim"])
        ),
        reverse=True,
    )
    print("\n=== Correlation Ranking (spearman test_acc vs optim_value) ===", flush=True)
    for r in corr_sorted:
        rho = r["spearman_test_acc_vs_optim"]
        pear = r["pearson_test_acc_vs_optim"]
        print(
            f"{r['combo_name']:<32} n={r['n']:>2} "
            f"spearman={rho:.4f} pearson={pear:.4f}",
            flush=True,
        )

    summary = {
        "data_path": args.data_path,
        "gt_path": args.gt_path,
        "output_dir": output_dir,
        "n_trials_requested": int(args.n_trials),
        "n_trials_succeeded": int(len(all_trial_rows)),
        "n_trials_failed": int(len(failed_rows)),
        "saliency_methods": methods,
        "losses": losses,
        "n_combos": int(len(methods) * len(losses)),
        "optim_beta": float(args.optim_beta),
        "ig_steps": int(args.ig_steps),
        "smoothgrad_samples": int(args.smoothgrad_samples),
        "smoothgrad_noise_std": float(args.smoothgrad_noise_std),
        "files": {
            "trials_csv": trials_csv,
            "combo_scores_csv": combo_csv,
            "correlations_csv": corr_csv,
            "best_by_combo_csv": best_combo_csv,
            "failed_trials_csv": failures_csv if failed_rows else "",
        },
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE] Summary files:", flush=True)
    for k, v in summary["files"].items():
        if v:
            print(f"  {k}: {v}", flush=True)
    print(f"  summary_json: {summary_json}", flush=True)


if __name__ == "__main__":
    main()
