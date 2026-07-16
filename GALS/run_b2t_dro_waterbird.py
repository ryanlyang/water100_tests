#!/usr/bin/env python3
"""B2T pseudo-bias GroupDRO trainer for Waterbirds.

This keeps the student/training setup close to the vanilla Waterbirds runner:
ImageNet-pretrained torchvision ResNet50, separate base/classifier LRs, SGD,
200 epochs by default, and Waterbirds metadata splits. The B2T-specific piece
is fixed CLIP-RN50 pseudo-background inference using the paper's Waterbirds
keyword groups, then GroupDRO over class x pseudo-background groups.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

GROUP_NAMES = ["Land_on_Land", "Land_on_Water", "Water_on_Land", "Water_on_Water"]

B2T_GENERAL_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of my {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a bad photo of a {}.",
    "a bad photo of the {}.",
    "a black and white photo of a {}.",
    "a black and white photo of the {}.",
    "a blurry photo of a {}.",
    "a blurry photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a cropped photo of a {}.",
    "a cropped photo of the {}.",
    "a low resolution photo of a {}.",
    "a low resolution photo of the {}.",
    "a good photo of a {}.",
    "a good photo of the {}.",
    "a pixelated photo of a {}.",
    "a pixelated photo of the {}.",
    "a drawing of a {}.",
    "a drawing of the {}.",
    "itap of a {}.",
    "itap of my {}.",
    "itap of the {}.",
    "a photo of many {}.",
    "a tattoo of a {}.",
    "a tattoo of the {}.",
    "a embroidered {}.",
    "the embroidered {}.",
    "a painting of a {}.",
    "a painting of the {}.",
    "a sculpture of a {}.",
    "a sculpture of the {}.",
    "a rendition of a {}.",
    "a rendition of the {}.",
    "a rendering of a {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a sketch of the {}.",
    "a doodle of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "the origami {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "art of a {}.",
    "art of the {}.",
    "graffiti of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "the toy {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a plushie {}.",
    "the plushie {}.",
]

B2T_WATERBIRD_CLASS_TEMPLATES = [
    "{}",
    "bird on {}",
    "bird on a {}",
    "bird and a {}",
    "fowl on {}",
    "fowl on a {}",
    "fowl and a {}",
]

B2T_WATERBIRD_KEYWORDS = [
    ["forest", "woods", "tree", "branch"],
    ["ocean", "beach", "surfer", "boat", "dock", "water", "lake"],
]


def _try_import_openai_clip():
    try:
        import clip  # type: ignore

        return clip
    except Exception:
        root = str(Path(__file__).resolve().parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from CLIP.clip import clip  # type: ignore

        return clip


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WaterbirdsB2TDataset(Dataset):
    SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        data_root: str,
        split: str,
        image_transform=None,
        pseudo_bias: Optional[Sequence[int]] = None,
    ) -> None:
        self.data_root = data_root
        self.image_transform = image_transform

        metadata_path = os.path.join(self.data_root, "metadata.csv")
        df = pd.read_csv(metadata_path)
        df = df[df["split"] == self.SPLIT_MAP[split]].reset_index(drop=True)

        self.paths = [os.path.join(self.data_root, p) for p in df["img_filename"].values]
        self.labels = df["y"].astype(int).values
        self.places = df["place"].astype(int).values

        if pseudo_bias is None:
            self.train_bias = self.places.copy()
        else:
            pseudo = np.asarray(pseudo_bias, dtype=np.int64)
            if pseudo.shape[0] != len(self.paths):
                raise ValueError(
                    f"pseudo_bias length {pseudo.shape[0]} does not match {split} length {len(self.paths)}"
                )
            self.train_bias = pseudo

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        label = int(self.labels[idx])
        place = int(self.places[idx])
        train_bias = int(self.train_bias[idx])
        return image, label, place, train_bias, self.paths[idx]


class WaterbirdsPathDataset(Dataset):
    SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

    def __init__(self, data_root: str, split: str, image_transform=None) -> None:
        self.data_root = data_root
        self.image_transform = image_transform
        df = pd.read_csv(os.path.join(self.data_root, "metadata.csv"))
        df = df[df["split"] == self.SPLIT_MAP[split]].reset_index(drop=True)
        self.paths = [os.path.join(self.data_root, p) for p in df["img_filename"].values]
        self.labels = df["y"].astype(int).values
        self.places = df["place"].astype(int).values

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image, int(self.labels[idx]), int(self.places[idx]), self.paths[idx]


def make_student(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name != "resnet50":
        raise ValueError(f"B2T-DRO currently uses the paper/ours ResNet50 student, got {model_name}")
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_b2t_prompts(keywords: Sequence[str]) -> List[str]:
    prompts: List[str] = []
    for general in B2T_GENERAL_TEMPLATES:
        for class_template in B2T_WATERBIRD_CLASS_TEMPLATES:
            for keyword in keywords:
                prompts.append(general.format(class_template.format(keyword)))
    return prompts


def _build_b2t_text_weights(clip_model, clip_tokenize, device: torch.device) -> torch.Tensor:
    weights = []
    with torch.no_grad():
        for keywords in B2T_WATERBIRD_KEYWORDS:
            tokens = clip_tokenize(_build_b2t_prompts(keywords)).to(device)
            text_features = clip_model.encode_text(tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_feature = text_features.mean(dim=0)
            text_feature = text_feature / text_feature.norm()
            weights.append(text_feature)
    return torch.stack(weights, dim=1).to(device)


def ensure_b2t_pseudo_bias(
    data_path: str,
    pseudo_bias_path: str,
    clip_model_name: str = "RN50",
    batch_size: int = 256,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    overwrite: bool = False,
) -> str:
    if os.path.exists(pseudo_bias_path) and not overwrite:
        print(f"[B2T] Using cached pseudo-bias labels: {pseudo_bias_path}", flush=True)
        return pseudo_bias_path

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(os.path.abspath(pseudo_bias_path)) or ".", exist_ok=True)

    clip_mod = _try_import_openai_clip()
    clip_model, clip_preprocess = clip_mod.load(clip_model_name, device=device, jit=False)
    clip_model.eval()
    crop = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    transform = transforms.Compose([crop, clip_preprocess])
    dataset = WaterbirdsPathDataset(data_path, "train", image_transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    text_weights = _build_b2t_text_weights(clip_model, clip_mod.tokenize, device)

    preds = []
    labels = []
    places = []
    with torch.no_grad():
        for images, y, place, _paths in loader:
            images = images.to(device, non_blocking=True)
            image_features = clip_model.encode_image(images).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_weights
            pred = logits.argmax(dim=1).detach().cpu().long()
            preds.append(pred)
            labels.append(y.long())
            places.append(place.long())

    pred_tensor = torch.cat(preds, dim=0)
    y_tensor = torch.cat(labels, dim=0)
    place_tensor = torch.cat(places, dim=0)
    pseudo_acc = float((pred_tensor == place_tensor).float().mean().item() * 100.0)
    pseudo_group = y_tensor * 2 + pred_tensor
    counts = torch.bincount(pseudo_group, minlength=4).cpu().numpy().astype(int).tolist()
    torch.save(pred_tensor, pseudo_bias_path)
    print(
        f"[B2T] Saved pseudo-bias labels to {pseudo_bias_path} "
        f"(n={len(pred_tensor)} pseudo_place_acc={pseudo_acc:.2f}% pseudo_group_counts={counts})",
        flush=True,
    )
    return pseudo_bias_path


class GroupEMA:
    def __init__(self, size: int, step_size: float, device: torch.device) -> None:
        self.step_size = float(step_size)
        self.group_weights = torch.ones(size, device=device) / float(size)

    def update(self, group_loss: torch.Tensor) -> torch.Tensor:
        self.group_weights = self.group_weights * torch.exp(self.step_size * group_loss.detach())
        self.group_weights = self.group_weights / self.group_weights.sum()
        return group_loss @ self.group_weights


def _group_metrics(
    losses: np.ndarray,
    correct: np.ndarray,
    labels: np.ndarray,
    places: np.ndarray,
) -> Dict[str, object]:
    groups = labels * 2 + places
    group_acc = np.zeros(4, dtype=np.float64)
    group_loss = np.zeros(4, dtype=np.float64)
    group_total = np.zeros(4, dtype=np.int64)
    for group_id in range(4):
        idx = np.where(groups == group_id)[0]
        group_total[group_id] = idx.size
        if idx.size == 0:
            group_acc[group_id] = np.nan
            group_loss[group_id] = np.nan
        else:
            group_acc[group_id] = float(correct[idx].mean() * 100.0)
            group_loss[group_id] = float(losses[idx].mean())

    class_acc = np.zeros(2, dtype=np.float64)
    for cls in range(2):
        idx = np.where(labels == cls)[0]
        class_acc[cls] = float(correct[idx].mean() * 100.0) if idx.size else np.nan

    return {
        "loss": float(np.mean(losses)),
        "acc": float(np.mean(correct) * 100.0),
        "balanced_class": float(np.nanmean(class_acc)),
        "balanced_group": float(np.nanmean(group_acc)),
        "worst_group": float(np.nanmin(group_acc)),
        "group_acc": group_acc,
        "group_loss": group_loss,
        "group_total": group_total,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses, corrects, labels_all, places_all = [], [], [], []
    with torch.no_grad():
        for images, labels, places, _train_bias, _paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            logits = model(images)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
            losses.append(loss.detach().cpu().numpy())
            corrects.append((pred == labels).detach().cpu().numpy().astype(np.float64))
            labels_all.append(labels.detach().cpu().numpy().astype(np.int64))
            places_all.append(places.numpy().astype(np.int64))
    return _group_metrics(
        losses=np.concatenate(losses),
        correct=np.concatenate(corrects),
        labels=np.concatenate(labels_all),
        places=np.concatenate(places_all),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: optim.Optimizer,
    dro_step_size: float,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object], int]:
    criterion = nn.CrossEntropyLoss(reduction="none")
    group_ema = GroupEMA(size=4, step_size=dro_step_size, device=device)
    best_wts = copy.deepcopy(model.state_dict())
    best_metrics: Optional[Dict[str, object]] = None
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for images, labels, _places, train_bias, _paths in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            train_bias = train_bias.to(device, non_blocking=True).long()
            logits = model(images)
            sample_loss = criterion(logits, labels)
            group_idx = labels * 2 + train_bias
            group_map = (group_idx.unsqueeze(0) == torch.arange(4, device=device).unsqueeze(1)).float()
            group_count = group_map.sum(dim=1)
            group_denom = group_count + (group_count == 0).float()
            group_loss = (group_map @ sample_loss.view(-1)) / group_denom
            loss = group_ema.update(group_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(sample_loss.detach().mean().cpu().item()))

        val_metrics = evaluate(model, val_loader, device)
        if best_metrics is None or float(val_metrics["balanced_group"]) > float(best_metrics["balanced_group"]):
            best_metrics = val_metrics
            best_epoch = epoch
            best_wts = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch + 1}/{num_epochs} "
            f"train_loss={np.mean(train_losses):.4f} "
            f"val_acc={float(val_metrics['acc']):.2f} "
            f"val_bal_group={float(val_metrics['balanced_group']):.2f} "
            f"val_worst_group={float(val_metrics['worst_group']):.2f}",
            flush=True,
        )

    if best_metrics is None:
        raise RuntimeError("No validation metrics were produced")
    model.load_state_dict(best_wts)
    return model, best_metrics, best_epoch


def build_dataloaders(args, pseudo_bias: np.ndarray, generator: torch.Generator):
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = WaterbirdsB2TDataset(args.data_path, "train", tfm, pseudo_bias=pseudo_bias)
    val_ds = WaterbirdsB2TDataset(args.data_path, "val", tfm, pseudo_bias=None)
    test_ds = WaterbirdsB2TDataset(args.data_path, "test", tfm, pseudo_bias=None)

    if args.balanced_sampler:
        group = train_ds.labels * 2 + train_ds.train_bias
        group_count = np.bincount(group, minlength=4).astype(np.float64)
        weights = np.zeros(4, dtype=np.float64)
        nonzero = group_count > 0
        weights[nonzero] = 1.0 / group_count[nonzero]
        sample_weights = torch.from_numpy(weights[group]).double()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, val_loader, test_loader


def run_single(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    pseudo_bias_path = ensure_b2t_pseudo_bias(
        data_path=args.data_path,
        pseudo_bias_path=args.pseudo_bias_path,
        clip_model_name=args.b2t_clip_model,
        batch_size=args.pseudo_batch_size,
        num_workers=args.num_workers,
        device=device,
        overwrite=args.overwrite_pseudo_bias,
    )
    pseudo_bias = torch.load(pseudo_bias_path, map_location="cpu").long().numpy()
    train_loader, val_loader, test_loader = build_dataloaders(args, pseudo_bias, generator)

    model = make_student(args.model, num_classes=2, pretrained=args.pretrained).to(device)
    base_params = []
    fc_params = []
    for name, param in model.named_parameters():
        if "fc" in name:
            fc_params.append(param)
        else:
            base_params.append(param)
    optimizer = optim.SGD(
        [
            {"params": base_params, "lr": float(args.base_lr)},
            {"params": fc_params, "lr": float(args.classifier_lr)},
        ],
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        nesterov=bool(args.nesterov),
    )

    print(
        f"\n=== B2T-DRO RUN: epochs={args.num_epochs} base_lr={args.base_lr} "
        f"classifier_lr={args.classifier_lr} momentum={args.momentum} "
        f"wd={args.weight_decay} dro_step_size={args.dro_step_size} seed={args.seed} "
        f"balanced_sampler={args.balanced_sampler} ===",
        flush=True,
    )

    best_model, best_val_metrics, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        optimizer=optimizer,
        dro_step_size=args.dro_step_size,
        device=device,
    )
    test_metrics = evaluate(best_model, test_loader, device)

    group_acc = test_metrics["group_acc"]
    assert isinstance(group_acc, np.ndarray)
    print(
        f"[BEST VAL] epoch={best_epoch + 1} "
        f"val_bal_group={float(best_val_metrics['balanced_group']):.2f} "
        f"val_worst_group={float(best_val_metrics['worst_group']):.2f}",
        flush=True,
    )
    print(
        f"[TEST] acc={float(test_metrics['acc']):.2f} "
        f"bal_group={float(test_metrics['balanced_group']):.2f} "
        f"worst_group={float(test_metrics['worst_group']):.2f}",
        flush=True,
    )
    for name, acc in zip(GROUP_NAMES, group_acc):
        print(f"[TEST] {name}: {float(acc):.2f}%", flush=True)

    save_checkpoints = os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "n")
    if save_checkpoints:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = (
            f"b2t_dro_{args.model}_seed{args.seed}_"
            f"valbg{float(best_val_metrics['balanced_group']):.2f}_epoch{best_epoch + 1}_{ts}.pth"
        )
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        torch.save(best_model.state_dict(), ckpt_path)
    else:
        ckpt_path = "NONE"

    return {
        "best_epoch": best_epoch + 1,
        "best_val_acc": float(best_val_metrics["acc"]),
        "best_val_balanced_group": float(best_val_metrics["balanced_group"]),
        "best_val_worst_group": float(best_val_metrics["worst_group"]),
        "test_acc": float(test_metrics["acc"]),
        "test_balanced_group": float(test_metrics["balanced_group"]),
        "test_worst_group": float(test_metrics["worst_group"]),
        "test_group_acc": group_acc,
        "checkpoint": ckpt_path,
        "pseudo_bias_path": pseudo_bias_path,
    }


def parse_args():
    p = argparse.ArgumentParser(description="B2T-DRO Waterbirds trainer.")
    p.add_argument("data_path", help="Waterbirds root containing metadata.csv")
    p.add_argument("--pseudo-bias-path", required=True)
    p.add_argument("--overwrite-pseudo-bias", action="store_true", default=False)
    p.add_argument("--b2t-clip-model", default="RN50")
    p.add_argument("--pseudo-batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=["resnet50"], default="resnet50")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--base-lr", type=float, required=True)
    p.add_argument("--classifier-lr", type=float, required=True)
    p.add_argument("--momentum", type=float, required=True)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dro-step-size", type=float, default=0.01)
    p.add_argument("--balanced-sampler", action="store_true", default=True)
    p.add_argument("--no-balanced-sampler", action="store_false", dest="balanced_sampler")
    p.add_argument("--nesterov", action="store_true", default=False)
    p.add_argument("--no-nesterov", action="store_false", dest="nesterov")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-dir", default="B2T_DRO_Checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    run_single(parse_args())
