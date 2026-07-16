#!/usr/bin/env python3
"""NSF post-hoc debiasing for RedMeat.

This mirrors the RedMeat vanilla ResNet-50 setup, then applies Neutralizing
Spurious Features (NSF) on validation features. Since RedMeat does not have
explicit spurious-group labels in the current setup, the reported "balanced
group" and "worst group" quantities are class-balanced and worst-class metrics.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

DEFAULT_CLASSES = ["prime_rib", "pork_chop", "steak", "baby_back_ribs", "filet_mignon"]


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


def _resolve_img_path(dataset_root: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    rel_path = str(rel_or_abs).lstrip("/")
    return os.path.join(dataset_root, rel_path)


def _parse_classes(classes: Optional[str], metadata_path: str) -> List[str]:
    if classes is not None and str(classes).strip():
        return [item.strip() for item in str(classes).split(",") if item.strip()]
    metadata_df = pd.read_csv(metadata_path)
    return sorted(metadata_df["label"].astype(str).unique().tolist())


class RedMeatNSFDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        image_transform=None,
        classes: Optional[List[str]] = None,
        split_col: str = "split",
        label_col: str = "label",
        path_col: str = "abs_file_path",
    ) -> None:
        self.data_root = data_root
        self.image_transform = image_transform

        metadata_path = os.path.join(self.data_root, "all_images.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        metadata_df = pd.read_csv(metadata_path)
        for col_name in (split_col, label_col, path_col):
            if col_name not in metadata_df.columns:
                raise KeyError(f"Missing column '{col_name}' in {metadata_path}. Found: {list(metadata_df.columns)}")

        if classes is None:
            classes = sorted(metadata_df[label_col].astype(str).unique().tolist())
        self.classes = list(classes)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        split_df = metadata_df[metadata_df[split_col].astype(str) == str(split)].copy()
        if len(split_df) == 0:
            raise ValueError(f"Split '{split}' has 0 rows in {metadata_path}")

        labels_raw = split_df[label_col].astype(str).tolist()
        missing = sorted(set(labels_raw) - set(self.class_to_idx))
        if missing:
            raise ValueError(f"Split '{split}' contains labels not in class list: {missing}")

        self.paths = [_resolve_img_path(self.data_root, value) for value in split_df[path_col].astype(str).tolist()]
        self.labels = np.array([self.class_to_idx[label] for label in labels_raw], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = int(self.labels[idx])
        image = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image, label, path


def make_student(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def forward_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    features = model.conv1(images)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)
    features = model.avgpool(features)
    return torch.flatten(features, 1)


def _class_metrics(
    losses: np.ndarray,
    correct: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    class_acc = np.zeros(num_classes, dtype=np.float64)
    class_loss = np.zeros(num_classes, dtype=np.float64)
    class_total = np.zeros(num_classes, dtype=np.int64)
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        class_total[class_idx] = int(class_mask.sum())
        if class_total[class_idx] == 0:
            class_acc[class_idx] = np.nan
            class_loss[class_idx] = np.nan
        else:
            class_acc[class_idx] = float(correct[class_mask].mean() * 100.0)
            class_loss[class_idx] = float(losses[class_mask].mean())

    return {
        "loss": float(np.mean(losses)),
        "acc": float(np.mean(correct) * 100.0),
        "balanced_group": float(np.nanmean(class_acc)),
        "worst_group": float(np.nanmin(class_acc)),
        "class_acc": class_acc,
        "class_loss": class_loss,
        "class_total": class_total,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, object]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses_all, correct_all, labels_all = [], [], []
    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            logits = model(images)
            losses = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            losses_all.append(losses.detach().cpu().numpy())
            correct_all.append((preds == labels).detach().cpu().numpy().astype(np.float64))
            labels_all.append(labels.detach().cpu().numpy().astype(np.int64))
    return _class_metrics(
        losses=np.concatenate(losses_all),
        correct=np.concatenate(correct_all),
        labels=np.concatenate(labels_all),
        num_classes=num_classes,
    )


def build_dataloaders(args, generator: torch.Generator):
    data_path = getattr(args, "data_path", None) or getattr(args, "data_root", None)
    if not data_path:
        raise AttributeError("Expected args.data_path or args.data_root for RedMeat dataset root")
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    metadata_path = os.path.join(data_path, "all_images.csv")
    classes = _parse_classes(args.classes, metadata_path)
    train_dataset = RedMeatNSFDataset(data_path, "train", image_transform, classes=classes)
    val_dataset = RedMeatNSFDataset(data_path, "val", image_transform, classes=train_dataset.classes)
    test_dataset = RedMeatNSFDataset(data_path, "test", image_transform, classes=train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_dataset.classes


def train_erm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_classes: int,
) -> Tuple[nn.Module, Dict[str, object], int]:
    criterion = nn.CrossEntropyLoss()
    best_wts = copy.deepcopy(model.state_dict())
    best_metrics: Optional[Dict[str, object]] = None
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for images, labels, _paths in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        val_metrics = evaluate(model, val_loader, device, num_classes)
        if best_metrics is None or float(val_metrics["balanced_group"]) > float(best_metrics["balanced_group"]):
            best_metrics = val_metrics
            best_epoch = epoch
            best_wts = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch + 1}/{num_epochs} "
            f"train_loss={np.mean(train_losses):.4f} "
            f"val_acc={float(val_metrics['acc']):.2f} "
            f"val_balanced={float(val_metrics['balanced_group']):.2f} "
            f"val_worst_class={float(val_metrics['worst_group']):.2f}",
            flush=True,
        )

    if best_metrics is None:
        raise RuntimeError("No validation metrics were produced")
    model.load_state_dict(best_wts)
    return model, best_metrics, best_epoch


def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    features_all, labels_all = [], []
    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device, non_blocking=True)
            features_all.append(forward_features(model, images).detach())
            labels_all.append(labels.to(device, non_blocking=True).long())
    return {
        "f": torch.cat(features_all, dim=0),
        "y": torch.cat(labels_all, dim=0),
    }


def _class_prototypes(
    labels: torch.Tensor,
    features: torch.Tensor,
    num_classes: int,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    centers = []
    counts = []
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx) & mask
        count = class_mask.sum()
        counts.append(count.float())
        if int(count.item()) == 0:
            centers.append(torch.zeros(features.shape[1], device=features.device, dtype=features.dtype))
        else:
            centers.append(features[class_mask].mean(dim=0))
    return torch.stack(centers, dim=0), torch.stack(counts, dim=0)


def nsf_get_centers(data: Dict[str, torch.Tensor], num_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = data["y"].long()
    features = data["f"]
    global_centers, _ = _class_prototypes(labels, features, num_classes)
    nearest = torch.cdist(features, global_centers).argmin(dim=1)
    outliers = nearest != labels
    in_centers, _in_counts = _class_prototypes(labels, features, num_classes, mask=~outliers)
    out_centers, out_counts = _class_prototypes(labels, features, num_classes, mask=outliers)
    debiased_centers = in_centers.clone()
    has_outliers = out_counts > 1
    debiased_centers[has_outliers] = 0.5 * (in_centers[has_outliers] + out_centers[has_outliers])
    return debiased_centers, global_centers, outliers


class NSFTransformation(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features + self.bias) * self.weight - self.bias


def nsf_transform_features(
    centers: torch.Tensor,
    data: Dict[str, torch.Tensor],
    steps: int,
    lr: float,
    beta_reg_weight: float,
) -> NSFTransformation:
    features = data["f"].detach()
    labels = data["y"].long().detach()
    weight = torch.ones((1, features.shape[1]), device=features.device, dtype=features.dtype)
    bias = torch.zeros((1, features.shape[1]), device=features.device, dtype=features.dtype)
    transform = NSFTransformation(weight=weight, bias=bias).to(features.device)
    optimizer = torch.optim.AdamW(transform.parameters(), lr=float(lr), weight_decay=0.0)
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        transformed = transform(features)
        target = centers[labels]
        loss = F.pairwise_distance(transformed, target).mean() + float(beta_reg_weight) * transform.weight.mean()
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

    classifier = nn.Linear(old_fc.in_features, old_fc.out_features).to(features.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=float(lr), weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    if int(mask1.sum().item()) == 0 or int(mask2.sum().item()) == 0:
        batch_size = min(64, len(labels))
        for _step in range(int(steps)):
            batch_idx = torch.randint(0, len(labels), (batch_size,), device=features.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(classifier(features[batch_idx]), labels[batch_idx])
            loss.backward()
            optimizer.step()
        return classifier

    feat1, labels1 = features[mask1], labels[mask1]
    feat2, labels2 = features[mask2], labels[mask2]
    batch_size = min(len(labels1), len(labels2))
    for _step in range(int(steps)):
        idx1 = torch.randint(0, len(labels1), (batch_size,), device=features.device)
        idx2 = torch.randint(0, len(labels2), (batch_size,), device=features.device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(classifier(feat1[idx1]), labels1[idx1]) + criterion(classifier(feat2[idx2]), labels2[idx2])
        loss.backward()
        optimizer.step()
    return classifier


def evaluate_feature_classifier(
    classifier: nn.Module,
    data: Dict[str, torch.Tensor],
    num_classes: int,
) -> Dict[str, object]:
    classifier.eval()
    with torch.no_grad():
        logits = classifier(data["f"])
        labels = data["y"].long()
        losses = F.cross_entropy(logits, labels, reduction="none")
        preds = logits.argmax(dim=1)
        correct = preds == labels
    return _class_metrics(
        losses=losses.detach().cpu().numpy(),
        correct=correct.detach().cpu().numpy().astype(np.float64),
        labels=labels.detach().cpu().numpy().astype(np.int64),
        num_classes=num_classes,
    )


def run_nsf_posthoc(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    transform_steps: int,
    classifier_steps: int,
    transform_lr: float,
    nsf_classifier_lr: float,
    beta_reg_weight: float,
) -> Tuple[Dict[str, object], Dict[str, object], NSFTransformation, nn.Linear, Dict[str, object]]:
    val_data = extract_features(model, val_loader, device)
    test_data = extract_features(model, test_loader, device)
    debiased_centers, global_centers, outliers = nsf_get_centers(val_data, num_classes)
    transform = nsf_transform_features(
        centers=debiased_centers,
        data=val_data,
        steps=transform_steps,
        lr=transform_lr,
        beta_reg_weight=beta_reg_weight,
    )
    val_transformed = {key: value for key, value in val_data.items()}
    test_transformed = {key: value for key, value in test_data.items()}
    val_transformed["f"] = transform(val_data["f"]).detach()
    test_transformed["f"] = transform(test_data["f"]).detach()

    classifier = nsf_adjust_classifier(
        old_fc=model.fc,
        transformed_data=val_transformed,
        original_data=val_data,
        global_centers=global_centers,
        debiased_centers=debiased_centers,
        steps=classifier_steps,
        lr=nsf_classifier_lr,
    )
    val_metrics = evaluate_feature_classifier(classifier, val_transformed, num_classes)
    test_metrics = evaluate_feature_classifier(classifier, test_transformed, num_classes)
    stats = {
        "outlier_count": int(outliers.sum().item()),
        "outlier_frac": float(outliers.float().mean().item()),
    }
    return val_metrics, test_metrics, transform, classifier, stats


def _metrics_for_save(metrics: Dict[str, object]) -> Dict[str, object]:
    serializable = {}
    for key, value in metrics.items():
        serializable[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return serializable


def run_single(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader, val_loader, test_loader, classes = build_dataloaders(args, generator)
    num_classes = len(classes)
    model = make_student(num_classes=num_classes, pretrained=args.pretrained).to(device)
    base_params, classifier_params = [], []
    for param_name, param in model.named_parameters():
        if "fc" in param_name:
            classifier_params.append(param)
        else:
            base_params.append(param)
    optimizer = optim.SGD(
        [
            {"params": base_params, "lr": float(args.base_lr)},
            {"params": classifier_params, "lr": float(args.classifier_lr)},
        ],
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        nesterov=bool(args.nesterov),
    )

    print(
        f"\n=== REDMEAT NSF RUN: epochs={args.num_epochs} base_lr={args.base_lr} "
        f"classifier_lr={args.classifier_lr} momentum={args.momentum} wd={args.weight_decay} "
        f"transform_steps={args.transform_steps} classifier_steps={args.classifier_steps} "
        f"transform_lr={args.transform_lr} nsf_classifier_lr={args.nsf_classifier_lr} "
        f"beta_reg={args.beta_reg_weight} seed={args.seed} classes={classes} ===",
        flush=True,
    )

    start_time = time.time()
    best_model, erm_val_metrics, erm_best_epoch = train_erm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        optimizer=optimizer,
        device=device,
        num_classes=num_classes,
    )
    nsf_val_metrics, nsf_test_metrics, transform, nsf_classifier, nsf_stats = run_nsf_posthoc(
        model=best_model,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        transform_steps=args.transform_steps,
        classifier_steps=args.classifier_steps,
        transform_lr=args.transform_lr,
        nsf_classifier_lr=args.nsf_classifier_lr,
        beta_reg_weight=args.beta_reg_weight,
    )

    class_acc = nsf_test_metrics["class_acc"]
    assert isinstance(class_acc, np.ndarray)
    print(
        f"[ERM BEST VAL] epoch={erm_best_epoch + 1} "
        f"val_balanced={float(erm_val_metrics['balanced_group']):.2f} "
        f"val_worst_class={float(erm_val_metrics['worst_group']):.2f}",
        flush=True,
    )
    print(
        f"[NSF VAL] balanced={float(nsf_val_metrics['balanced_group']):.2f} "
        f"worst_class={float(nsf_val_metrics['worst_group']):.2f} "
        f"outliers={nsf_stats['outlier_count']} ({100.0 * nsf_stats['outlier_frac']:.2f}%)",
        flush=True,
    )
    print(
        f"[NSF TEST] acc={float(nsf_test_metrics['acc']):.2f} "
        f"balanced={float(nsf_test_metrics['balanced_group']):.2f} "
        f"worst_class={float(nsf_test_metrics['worst_group']):.2f}",
        flush=True,
    )
    for class_name, acc in zip(classes, class_acc):
        print(f"[NSF TEST] {class_name}: {float(acc):.2f}%", flush=True)

    save_checkpoints = os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "n")
    if save_checkpoints:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = (
            f"nsf_redmeat_resnet50_seed{args.seed}_"
            f"valbg{float(nsf_val_metrics['balanced_group']):.2f}_"
            f"testwg{float(nsf_test_metrics['worst_group']):.2f}_{timestamp}.pth"
        )
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        torch.save(
            {
                "model": best_model.state_dict(),
                "nsf_transform": transform.state_dict(),
                "nsf_classifier": nsf_classifier.state_dict(),
                "classes": classes,
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
        "test_group_acc": class_acc,
        "outlier_count": int(nsf_stats["outlier_count"]),
        "outlier_frac": float(nsf_stats["outlier_frac"]),
        "checkpoint": ckpt_path,
        "classes": ",".join(classes),
        "seconds": int(time.time() - start_time),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="RedMeat NSF trainer.")
    parser.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--base-lr", type=float, required=True)
    parser.add_argument("--classifier-lr", type=float, required=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--nesterov", action="store_true", default=False)
    parser.add_argument("--no-nesterov", action="store_false", dest="nesterov")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--transform-steps", type=int, default=10)
    parser.add_argument("--classifier-steps", type=int, default=500)
    parser.add_argument("--transform-lr", type=float, default=1e-3)
    parser.add_argument("--nsf-classifier-lr", type=float, default=1e-3)
    parser.add_argument("--beta-reg-weight", type=float, default=10.0)
    parser.add_argument("--checkpoint-dir", default="NSF_RedMeat_Checkpoints")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    return parser.parse_args()


if __name__ == "__main__":
    run_single(parse_args())
