#!/usr/bin/env python3
"""NSF post-hoc debiasing for Waterbirds.

This keeps the same ResNet-50 student/training setup used by the other
Waterbirds baselines, then applies Neutralizing Spurious Features (NSF) to the
trained representation. NSF is fit on validation features and evaluated on test
features, following the public NSF implementation.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

GROUP_NAMES = ["Land_on_Land", "Land_on_Water", "Water_on_Land", "Water_on_Water"]


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


class WaterbirdsNSFDataset(Dataset):
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
        label = int(self.labels[idx])
        place = int(self.places[idx])
        return image, label, place, self.paths[idx]


def make_student(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def forward_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


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
        for images, labels, places, _paths in loader:
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


def build_dataloaders(args, generator: torch.Generator):
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = WaterbirdsNSFDataset(args.data_path, "train", tfm)
    val_ds = WaterbirdsNSFDataset(args.data_path, "val", tfm)
    test_ds = WaterbirdsNSFDataset(args.data_path, "test", tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def train_erm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object], int]:
    criterion = nn.CrossEntropyLoss()
    best_wts = copy.deepcopy(model.state_dict())
    best_metrics: Optional[Dict[str, object]] = None
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for images, labels, _places, _paths in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

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


def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    features, labels, places = [], [], []
    with torch.no_grad():
        for images, y, p, _paths in loader:
            images = images.to(device, non_blocking=True)
            feat = forward_features(model, images)
            features.append(feat.detach())
            labels.append(y.to(device, non_blocking=True).long())
            places.append(p.to(device, non_blocking=True).long())
    return {
        "f": torch.cat(features, dim=0),
        "y": torch.cat(labels, dim=0),
        "p": torch.cat(places, dim=0),
    }


def _class_prototypes(
    labels: torch.Tensor,
    features: torch.Tensor,
    n_classes: int,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    centers = []
    counts = []
    for cls in range(n_classes):
        cls_mask = (labels == cls) & mask
        count = cls_mask.sum()
        counts.append(count.float())
        if int(count.item()) == 0:
            centers.append(torch.zeros(features.shape[1], device=features.device, dtype=features.dtype))
        else:
            centers.append(features[cls_mask].mean(dim=0))
    return torch.stack(centers, dim=0), torch.stack(counts, dim=0)


def nsf_get_centers(data: Dict[str, torch.Tensor], n_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = data["y"].long()
    features = data["f"]
    global_centers, _ = _class_prototypes(labels, features, n_classes)
    nearest = torch.cdist(features, global_centers).argmin(dim=1)
    outliers = nearest != labels

    in_centers, _in_counts = _class_prototypes(labels, features, n_classes, mask=~outliers)
    out_centers, out_counts = _class_prototypes(labels, features, n_classes, mask=outliers)
    debiased = in_centers.clone()
    has_outliers = out_counts > 1
    debiased[has_outliers] = 0.5 * (in_centers[has_outliers] + out_centers[has_outliers])
    return debiased, global_centers, outliers


class NSFTransformation(nn.Module):
    def __init__(self, w: torch.Tensor, b: torch.Tensor) -> None:
        super().__init__()
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.b) * self.w - self.b


def nsf_transform_features(
    centers: torch.Tensor,
    data: Dict[str, torch.Tensor],
    steps: int,
    lr: float,
    beta_reg_weight: float,
) -> NSFTransformation:
    features = data["f"].detach()
    labels = data["y"].long().detach()
    w = torch.ones((1, features.shape[1]), device=features.device, dtype=features.dtype)
    b = torch.zeros((1, features.shape[1]), device=features.device, dtype=features.dtype)
    transform = NSFTransformation(w=w, b=b).to(features.device)
    optimizer = torch.optim.AdamW(transform.parameters(), lr=float(lr), weight_decay=0.0)

    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        transformed = transform(features)
        target = centers[labels]
        loss = F.pairwise_distance(transformed, target).mean() + float(beta_reg_weight) * transform.w.mean()
        loss.backward()
        optimizer.step()
    return transform


def _nearest_center_pred(features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return torch.cdist(features, centers).argmin(dim=1)


def nsf_adjust_classifier(
    old_fc: nn.Linear,
    transformed_data: Dict[str, torch.Tensor],
    original_data: Dict[str, torch.Tensor],
    global_centers: torch.Tensor,
    debiased_centers: torch.Tensor,
    steps: int,
    lr: float,
) -> nn.Linear:
    features = transformed_data["f"].detach()
    labels = transformed_data["y"].long().detach()
    pred_original = _nearest_center_pred(original_data["f"].detach(), global_centers)
    pred_new = _nearest_center_pred(features, debiased_centers)
    mask1 = (pred_original != labels) & (pred_new == labels)
    mask2 = pred_original == labels

    fc = nn.Linear(old_fc.in_features, old_fc.out_features).to(features.device)
    optimizer = torch.optim.AdamW(fc.parameters(), lr=float(lr), weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    if int(mask1.sum().item()) == 0 or int(mask2.sum().item()) == 0:
        train_features = features
        train_labels = labels
        batch_size = min(64, len(train_labels))
        for _step in range(int(steps)):
            idx = torch.randint(0, len(train_labels), (batch_size,), device=features.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(fc(train_features[idx]), train_labels[idx])
            loss.backward()
            optimizer.step()
        return fc

    feat1, y1 = features[mask1], labels[mask1]
    feat2, y2 = features[mask2], labels[mask2]
    batch_size = min(len(y1), len(y2))
    for _step in range(int(steps)):
        idx1 = torch.randint(0, len(y1), (batch_size,), device=features.device)
        idx2 = torch.randint(0, len(y2), (batch_size,), device=features.device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(fc(feat1[idx1]), y1[idx1]) + criterion(fc(feat2[idx2]), y2[idx2])
        loss.backward()
        optimizer.step()
    return fc


def evaluate_feature_classifier(
    fc: nn.Module,
    data: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    fc.eval()
    with torch.no_grad():
        logits = fc(data["f"])
        labels = data["y"].long()
        losses = F.cross_entropy(logits, labels, reduction="none")
        preds = logits.argmax(dim=1)
        correct = preds == labels
    return _group_metrics(
        losses=losses.detach().cpu().numpy(),
        correct=correct.detach().cpu().numpy().astype(np.float64),
        labels=labels.detach().cpu().numpy().astype(np.int64),
        places=data["p"].detach().cpu().numpy().astype(np.int64),
    )


def run_nsf_posthoc(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    transform_steps: int,
    classifier_steps: int,
    transform_lr: float,
    nsf_classifier_lr: float,
    beta_reg_weight: float,
) -> Tuple[Dict[str, object], Dict[str, object], NSFTransformation, nn.Linear, Dict[str, object]]:
    val_data = extract_features(model, val_loader, device)
    test_data = extract_features(model, test_loader, device)
    n_classes = int(val_data["y"].max().item()) + 1

    debiased_centers, global_centers, outliers = nsf_get_centers(val_data, n_classes)
    transform = nsf_transform_features(
        centers=debiased_centers,
        data=val_data,
        steps=transform_steps,
        lr=transform_lr,
        beta_reg_weight=beta_reg_weight,
    )
    val_transformed = {k: v for k, v in val_data.items()}
    test_transformed = {k: v for k, v in test_data.items()}
    val_transformed["f"] = transform(val_data["f"]).detach()
    test_transformed["f"] = transform(test_data["f"]).detach()

    fc = nsf_adjust_classifier(
        old_fc=model.fc,
        transformed_data=val_transformed,
        original_data=val_data,
        global_centers=global_centers,
        debiased_centers=debiased_centers,
        steps=classifier_steps,
        lr=nsf_classifier_lr,
    )
    val_metrics = evaluate_feature_classifier(fc, val_transformed)
    test_metrics = evaluate_feature_classifier(fc, test_transformed)
    stats = {
        "outlier_count": int(outliers.sum().item()),
        "outlier_frac": float(outliers.float().mean().item()),
    }
    return val_metrics, test_metrics, transform, fc, stats


def run_single(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader, val_loader, test_loader = build_dataloaders(args, generator)
    model = make_student(num_classes=2, pretrained=args.pretrained).to(device)

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
        f"\n=== NSF RUN: epochs={args.num_epochs} base_lr={args.base_lr} "
        f"classifier_lr={args.classifier_lr} momentum={args.momentum} wd={args.weight_decay} "
        f"transform_steps={args.transform_steps} classifier_steps={args.classifier_steps} "
        f"transform_lr={args.transform_lr} nsf_classifier_lr={args.nsf_classifier_lr} "
        f"beta_reg={args.beta_reg_weight} seed={args.seed} ===",
        flush=True,
    )

    t0 = time.time()
    best_model, erm_val_metrics, erm_best_epoch = train_erm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        optimizer=optimizer,
        device=device,
    )
    nsf_val_metrics, nsf_test_metrics, transform, nsf_fc, nsf_stats = run_nsf_posthoc(
        model=best_model,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        transform_steps=args.transform_steps,
        classifier_steps=args.classifier_steps,
        transform_lr=args.transform_lr,
        nsf_classifier_lr=args.nsf_classifier_lr,
        beta_reg_weight=args.beta_reg_weight,
    )

    group_acc = nsf_test_metrics["group_acc"]
    assert isinstance(group_acc, np.ndarray)
    print(
        f"[ERM BEST VAL] epoch={erm_best_epoch + 1} "
        f"val_bal_group={float(erm_val_metrics['balanced_group']):.2f} "
        f"val_worst_group={float(erm_val_metrics['worst_group']):.2f}",
        flush=True,
    )
    print(
        f"[NSF VAL] bal_group={float(nsf_val_metrics['balanced_group']):.2f} "
        f"worst_group={float(nsf_val_metrics['worst_group']):.2f} "
        f"outliers={nsf_stats['outlier_count']} ({100.0 * nsf_stats['outlier_frac']:.2f}%)",
        flush=True,
    )
    print(
        f"[NSF TEST] acc={float(nsf_test_metrics['acc']):.2f} "
        f"bal_group={float(nsf_test_metrics['balanced_group']):.2f} "
        f"worst_group={float(nsf_test_metrics['worst_group']):.2f}",
        flush=True,
    )
    for name, acc in zip(GROUP_NAMES, group_acc):
        print(f"[NSF TEST] {name}: {float(acc):.2f}%", flush=True)

    save_checkpoints = os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "n")
    if save_checkpoints:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = (
            f"nsf_resnet50_seed{args.seed}_"
            f"valbg{float(nsf_val_metrics['balanced_group']):.2f}_"
            f"testwg{float(nsf_test_metrics['worst_group']):.2f}_{ts}.pth"
        )
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        torch.save(
            {
                "model": best_model.state_dict(),
                "nsf_transform": transform.state_dict(),
                "nsf_fc": nsf_fc.state_dict(),
                "args": vars(args),
                "erm_val_metrics": _metrics_for_save(erm_val_metrics),
                "nsf_val_metrics": _metrics_for_save(nsf_val_metrics),
                "nsf_test_metrics": _metrics_for_save(nsf_test_metrics),
                "nsf_stats": nsf_stats,
            },
            ckpt_path,
        )
    else:
        ckpt_path = "NONE"

    return {
        "erm_best_epoch": erm_best_epoch + 1,
        "erm_val_acc": float(erm_val_metrics["acc"]),
        "erm_val_balanced_group": float(erm_val_metrics["balanced_group"]),
        "erm_val_worst_group": float(erm_val_metrics["worst_group"]),
        "nsf_val_acc": float(nsf_val_metrics["acc"]),
        "nsf_val_balanced_group": float(nsf_val_metrics["balanced_group"]),
        "nsf_val_worst_group": float(nsf_val_metrics["worst_group"]),
        "test_acc": float(nsf_test_metrics["acc"]),
        "test_balanced_group": float(nsf_test_metrics["balanced_group"]),
        "test_worst_group": float(nsf_test_metrics["worst_group"]),
        "test_group_acc": group_acc,
        "outlier_count": int(nsf_stats["outlier_count"]),
        "outlier_frac": float(nsf_stats["outlier_frac"]),
        "checkpoint": ckpt_path,
        "seconds": int(time.time() - t0),
    }


def _metrics_for_save(metrics: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Waterbirds NSF trainer.")
    p.add_argument("data_path", help="Waterbirds root containing metadata.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--base-lr", type=float, required=True)
    p.add_argument("--classifier-lr", type=float, required=True)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--nesterov", action="store_true", default=False)
    p.add_argument("--no-nesterov", action="store_false", dest="nesterov")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--transform-steps", type=int, default=10)
    p.add_argument("--classifier-steps", type=int, default=500)
    p.add_argument("--transform-lr", type=float, default=1e-3)
    p.add_argument("--nsf-classifier-lr", type=float, default=1e-3)
    p.add_argument("--beta-reg-weight", type=float, default=10.0)
    p.add_argument("--checkpoint-dir", default="NSF_Checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    run_single(parse_args())
