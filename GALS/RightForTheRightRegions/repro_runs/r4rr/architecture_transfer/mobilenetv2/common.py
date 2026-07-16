#!/usr/bin/env python3
"""Shared MobileNetV2 helpers for R4RR architecture-transfer runs.

The implementation intentionally mirrors the working MobileNetV2 setup used in
the main experiment code:

- ImageNet-pretrained torchvision MobileNetV2.
- Dropout-free GAP + Linear classifier head.
- ``forward`` returns ``(logits, feature_maps)`` so the existing CAM/KL R4RR
  training loop can compute class-specific CAM evidence maps exactly like the
  ResNet-50 runner.
"""

from __future__ import annotations

import copy
import importlib
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


REPRO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = REPRO_ROOT / "r4rr" / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))


class _LazyModule:
    """Defer canonical trainer imports until a run actually needs them.

    This keeps CLI help usable in lightweight environments that may not have
    optional training dependencies such as OpenCV installed.
    """

    def __init__(self, module_name: str):
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_module", None)

    def _load(self):
        module = object.__getattribute__(self, "_module")
        if module is None:
            module_name = object.__getattribute__(self, "_module_name")
            module = importlib.import_module(module_name)
            object.__setattr__(self, "_module", module)
        return module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __setattr__(self, name: str, value):
        if name in {"_module_name", "_module"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._load(), name, value)


wb_base = _LazyModule("r4rr_waterbirds")
rm_base = _LazyModule("r4rr_redmeat")


MOBILENETV2_NAME = "mobilenet_v2"
DEFAULT_REMEAT_CLASSES = "prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon"


@dataclass
class RunResult:
    best_balanced_val_acc: float
    test_acc: float
    per_group: float
    worst_group: float
    checkpoint: str
    best_epoch: int
    seconds: int


def _mobilenet_v2(pretrained: bool) -> nn.Module:
    """Build MobileNetV2 across old and new torchvision APIs."""
    if hasattr(models, "MobileNet_V2_Weights"):
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v2(weights=weights)
    return models.mobilenet_v2(pretrained=pretrained)


class MobileNetV2CAM(nn.Module):
    """Torchvision MobileNetV2 with an exact CAM-compatible GAP + Linear head."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.base = _mobilenet_v2(pretrained=pretrained)
        if not hasattr(self.base, "features"):
            raise TypeError("Expected torchvision MobileNetV2 to expose `.features`.")
        if not isinstance(self.base.classifier, nn.Sequential):
            raise TypeError("Expected torchvision MobileNetV2 classifier to be nn.Sequential.")
        if not isinstance(self.base.classifier[-1], nn.Linear):
            raise TypeError("Expected final MobileNetV2 classifier module to be nn.Linear.")

        in_features = int(self.base.classifier[-1].in_features)
        self.classifier = nn.Linear(in_features, num_classes)
        # Avoid dropout so logits and CAM maps use the same deterministic
        # feature channels: features -> GAP -> Linear.
        self.base.classifier = self.classifier
        self.features = None

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.base.features(images)
        self.features = feature_maps
        pooled = nn.functional.adaptive_avg_pool2d(feature_maps, 1).flatten(1)
        logits = self.classifier(pooled)
        return logits, feature_maps


def make_model(num_classes: int, pretrained: bool = True) -> MobileNetV2CAM:
    return MobileNetV2CAM(num_classes=num_classes, pretrained=pretrained)


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _set_training_globals(device: torch.device, momentum: float, weight_decay: float) -> None:
    """Keep canonical R4RR trainer globals synchronized for reused routines."""
    wb_base.device = device
    wb_base.momentum = float(momentum)
    wb_base.weight_decay = float(weight_decay)
    rm_base.device = device
    rm_base.momentum = float(momentum)
    rm_base.weight_decay = float(weight_decay)
    rm_base.base.device = device
    rm_base.base.momentum = float(momentum)
    rm_base.base.weight_decay = float(weight_decay)


def _seed_everything(seed: int) -> torch.Generator:
    wb_base.seed_everything(int(seed))
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _imagenet_transforms(img_size: int, include_mask: bool = False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_train = transforms.Compose(
        [
            transforms.Resize((int(img_size), int(img_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    image_eval = transforms.Compose(
        [
            transforms.Resize((int(img_size), int(img_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    if not include_mask:
        return image_train, image_eval, None
    mask_train = transforms.Compose(
        [
            transforms.Resize((int(img_size), int(img_size))),
            transforms.ToTensor(),
            wb_base.Brighten(8.0),
        ]
    )
    return image_train, image_eval, mask_train


def _parse_classes(classes_arg: Optional[str]) -> Optional[List[str]]:
    if not classes_arg:
        return None
    classes = [c.strip() for c in str(classes_arg).split(",") if c.strip()]
    return classes or None


def _checkpoint_enabled() -> bool:
    return os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "n")


def _save_checkpoint(model: nn.Module, checkpoint_dir: str, prefix: str, seed: int) -> str:
    if not _checkpoint_enabled():
        print("[RUN DONE] Checkpoint saving disabled via SAVE_CHECKPOINTS=0", flush=True)
        return "NONE"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(checkpoint_dir, f"{prefix}_seed{seed}_{ts}.pth")
    torch.save(model.state_dict(), save_path)
    return save_path


def build_waterbirds_loaders(
    data_path: str,
    teacher_map_path: Optional[str],
    batch_size: int,
    img_size: int,
    use_attention: bool,
    generator: torch.Generator,
    num_workers: Optional[int] = None,
):
    image_train, image_eval, mask_train = _imagenet_transforms(img_size, include_mask=use_attention)
    if num_workers is None:
        num_workers = wb_base.get_num_workers(default=4)

    metadata_path = os.path.join(data_path, "metadata.csv")
    if os.path.exists(metadata_path):
        train_dataset = wb_base.WaterbirdsMetadataDataset(
            data_root=data_path,
            split="train",
            image_transform=image_train,
            mask_root=teacher_map_path,
            mask_transform=mask_train,
            return_mask=use_attention,
            return_path=True,
            return_group=False,
        )
        val_dataset = wb_base.WaterbirdsMetadataDataset(
            data_root=data_path,
            split="val",
            image_transform=image_eval,
            return_mask=False,
            return_path=True,
            return_group=False,
        )
        test_dataset = wb_base.WaterbirdsMetadataDataset(
            data_root=data_path,
            split="test",
            image_transform=image_eval,
            return_mask=False,
            return_path=True,
            return_group=True,
        )
        num_classes = len(np.unique(train_dataset.labels))
    else:
        if use_attention and teacher_map_path is None:
            raise ValueError("teacher_map_path is required for guided folder-layout Waterbirds runs.")
        full_train = (
            wb_base.GuidedImageFolder(
                image_root=os.path.join(data_path, "train"),
                mask_root=teacher_map_path,
                image_transform=image_train,
                mask_transform=mask_train,
            )
            if use_attention
            else wb_base.ImageFolderWithPaths(
                root=os.path.join(data_path, "train"),
                transform=image_train,
            )
        )
        n_total = len(full_train)
        n_val = max(1, int(0.16 * n_total))
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(full_train, [n_train, n_val], generator=generator)
        test_dataset = wb_base.ImageFolderWithPaths(
            root=os.path.join(data_path, "test"),
            transform=image_eval,
        )
        num_classes = len(full_train.images.classes) if use_attention else len(full_train.classes)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=wb_base.seed_worker,
            generator=generator,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=wb_base.seed_worker,
            generator=generator,
        ),
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=wb_base.seed_worker,
        generator=generator,
    )
    return dataloaders, dataset_sizes, test_loader, int(num_classes)


def build_redmeat_loaders(
    data_path: str,
    teacher_map_path: Optional[str],
    batch_size: int,
    img_size: int,
    use_attention: bool,
    generator: torch.Generator,
    classes_arg: Optional[str] = DEFAULT_REMEAT_CLASSES,
    split_col: str = "split",
    label_col: str = "label",
    path_col: str = "abs_file_path",
    num_workers: Optional[int] = None,
):
    image_train, image_eval, mask_train = _imagenet_transforms(img_size, include_mask=use_attention)
    if num_workers is None:
        num_workers = wb_base.get_num_workers(default=4)
    classes = _parse_classes(classes_arg)

    train_dataset = rm_base.RedMeatMetadataDataset(
        data_root=data_path,
        split="train",
        image_transform=image_train,
        mask_root=teacher_map_path,
        mask_transform=mask_train,
        return_mask=use_attention,
        return_path=True,
        classes=classes,
        split_col=split_col,
        label_col=label_col,
        path_col=path_col,
    )
    val_dataset = rm_base.RedMeatMetadataDataset(
        data_root=data_path,
        split="val",
        image_transform=image_eval,
        return_mask=False,
        return_path=True,
        classes=train_dataset.classes,
        split_col=split_col,
        label_col=label_col,
        path_col=path_col,
    )
    test_dataset = rm_base.RedMeatMetadataDataset(
        data_root=data_path,
        split="test",
        image_transform=image_eval,
        return_mask=False,
        return_path=True,
        classes=train_dataset.classes,
        split_col=split_col,
        label_col=label_col,
        path_col=path_col,
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=wb_base.seed_worker,
            generator=generator,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=wb_base.seed_worker,
            generator=generator,
        ),
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=wb_base.seed_worker,
        generator=generator,
    )
    return dataloaders, dataset_sizes, test_loader, int(len(train_dataset.classes)), train_dataset.classes


def run_guided_waterbirds(args, attention_epoch: int, kl_lambda: float, kl_increment=None) -> RunResult:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _set_training_globals(device, args.momentum, args.weight_decay)
    g = _seed_everything(args.seed)

    use_attention = int(attention_epoch) < int(args.num_epochs) and float(kl_lambda) > 0.0
    dataloaders, dataset_sizes, test_loader, num_classes = build_waterbirds_loaders(
        data_path=args.data_path,
        teacher_map_path=args.teacher_map_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        use_attention=use_attention,
        generator=g,
        num_workers=getattr(args, "num_workers", None),
    )

    model = make_model(num_classes=num_classes, pretrained=args.pretrained).to(device)
    print(
        f"[MODEL] model=mobilenet_v2 pretrained={args.pretrained} "
        f"trainable_params={count_trainable_params(model):,}",
        flush=True,
    )
    print(
        f"\n=== RUN: MobileNetV2 R4RR | kl_lambda={kl_lambda} attention_epoch={attention_epoch} "
        f"base_lr={args.base_lr} classifier_lr={args.classifier_lr} lr2_mult={args.lr2_mult} "
        f"momentum={args.momentum} weight_decay={args.weight_decay} batch={args.batch_size} "
        f"epochs={args.num_epochs} img_size={args.img_size} seed={args.seed} ===",
        flush=True,
    )

    if kl_increment is None:
        kl_increment = float(kl_lambda) / 10.0

    t0 = time.time()
    best_model, best_score, best_epoch = wb_base.train_model(
        model,
        dataloaders,
        dataset_sizes,
        int(attention_epoch),
        float(kl_lambda),
        int(args.num_epochs),
        base_lr=float(args.base_lr),
        classifier_lr=float(args.classifier_lr),
        lr2_mult=float(args.lr2_mult),
        kl_incr=float(kl_increment),
        use_attention=use_attention,
        num_classes=int(num_classes),
    )
    test_loss, test_acc, group_acc, per_group, worst_group = wb_base.evaluate_test(best_model, test_loader)

    print(f"\n[VAL] Best Balanced Acc: {best_score:.4f} at epoch {best_epoch}", flush=True)
    print(f"[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%", flush=True)
    if group_acc is not None:
        for name, acc in zip(wb_base.GROUP_NAMES, group_acc):
            print(f"[TEST] {name}: {acc:.2f}%", flush=True)
        print(f"[TEST] Per Group: {per_group:.2f}%  Worst Group: {worst_group:.2f}%", flush=True)

    ckpt = _save_checkpoint(
        best_model,
        args.checkpoint_dir,
        f"mobilenet_v2_r4rr_kl{float(kl_lambda):.5g}_attn{int(attention_epoch)}",
        int(args.seed),
    )
    print(
        f"[RUN DONE] best_balanced_val_acc={best_score:.4f} | test_acc={test_acc:.2f}% "
        f"| per_group={per_group:.2f}% | worst_group={worst_group:.2f}% | saved: {ckpt}",
        flush=True,
    )
    return RunResult(float(best_score), float(test_acc), float(per_group), float(worst_group), ckpt, int(best_epoch), int(time.time() - t0))


def run_guided_redmeat(args, attention_epoch: int, kl_lambda: float, kl_increment=None) -> RunResult:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _set_training_globals(device, args.momentum, args.weight_decay)
    g = _seed_everything(args.seed)

    use_attention = int(attention_epoch) < int(args.num_epochs) and float(kl_lambda) > 0.0
    dataloaders, dataset_sizes, test_loader, num_classes, classes = build_redmeat_loaders(
        data_path=args.data_path,
        teacher_map_path=args.teacher_map_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        use_attention=use_attention,
        generator=g,
        classes_arg=args.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
        num_workers=getattr(args, "num_workers", None),
    )

    model = make_model(num_classes=num_classes, pretrained=args.pretrained).to(device)
    print(
        f"[MODEL] model=mobilenet_v2 pretrained={args.pretrained} "
        f"trainable_params={count_trainable_params(model):,}",
        flush=True,
    )
    print(
        f"\n=== RUN: RedMeat MobileNetV2 R4RR | kl_lambda={kl_lambda} attention_epoch={attention_epoch} "
        f"base_lr={args.base_lr} classifier_lr={args.classifier_lr} lr2_mult={args.lr2_mult} "
        f"momentum={args.momentum} weight_decay={args.weight_decay} batch={args.batch_size} "
        f"epochs={args.num_epochs} img_size={args.img_size} seed={args.seed} ===",
        flush=True,
    )

    if kl_increment is None:
        kl_increment = float(kl_lambda) / 10.0

    t0 = time.time()
    best_model, best_score, best_epoch = wb_base.train_model(
        model,
        dataloaders,
        dataset_sizes,
        int(attention_epoch),
        float(kl_lambda),
        int(args.num_epochs),
        base_lr=float(args.base_lr),
        classifier_lr=float(args.classifier_lr),
        lr2_mult=float(args.lr2_mult),
        kl_incr=float(kl_increment),
        use_attention=use_attention,
        num_classes=int(num_classes),
    )
    test_loss, test_acc, class_acc, per_group, worst_group = rm_base.evaluate_test(best_model, test_loader, num_classes)

    print(f"\n[VAL] Best Balanced Acc: {best_score:.4f} at epoch {best_epoch}", flush=True)
    print(f"[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%", flush=True)
    for cls_name, acc in zip(classes, class_acc):
        print(f"[TEST] {cls_name}: {acc:.2f}%", flush=True)
    print(f"[TEST] Per-class mean: {per_group:.2f}%  Worst-class: {worst_group:.2f}%", flush=True)

    ckpt = _save_checkpoint(
        best_model,
        args.checkpoint_dir,
        f"mobilenet_v2_redmeat_r4rr_kl{float(kl_lambda):.5g}_attn{int(attention_epoch)}",
        int(args.seed),
    )
    print(
        f"[RUN DONE] best_balanced_val_acc={best_score:.4f} | test_acc={test_acc:.2f}% "
        f"| per_group={per_group:.2f}% | worst_group={worst_group:.2f}% | saved: {ckpt}",
        flush=True,
    )
    return RunResult(float(best_score), float(test_acc), float(per_group), float(worst_group), ckpt, int(best_epoch), int(time.time() - t0))


def _get_param_groups(model: nn.Module, base_lr: float, classifier_lr: float):
    base_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".classifier" in name or name.startswith("classifier."):
            classifier_params.append(param)
        else:
            base_params.append(param)
    if not classifier_params:
        classifier_params = [p for p in model.parameters() if p.requires_grad]
        base_params = []
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": float(base_lr)})
    groups.append({"params": classifier_params, "lr": float(classifier_lr)})
    return groups


def train_ce_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    num_classes: int,
    num_epochs: int,
    base_lr: float,
    classifier_lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    device: torch.device,
):
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(
        _get_param_groups(model, base_lr, classifier_lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
        nesterov=bool(nesterov),
    )
    best_wts = copy.deepcopy(model.state_dict())
    best_balanced = -1.0
    best_epoch = -1
    t0 = time.time()

    for epoch in range(int(num_epochs)):
        print(f"Epoch {epoch + 1}/{num_epochs}", flush=True)
        for phase in ("train", "val"):
            is_train = phase == "train"
            model.train() if is_train else model.eval()
            running_loss = 0.0
            running_corrects = 0
            class_correct = np.zeros(num_classes, dtype=np.int64)
            class_total = np.zeros(num_classes, dtype=np.int64)

            for batch in dataloaders[phase]:
                inputs, labels = batch[0], batch[1]
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                if is_train:
                    opt.zero_grad()
                with torch.set_grad_enabled(is_train):
                    outputs, _features = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    if is_train:
                        loss.backward()
                        opt.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                labels_cpu = labels.detach().cpu().numpy()
                preds_cpu = preds.detach().cpu().numpy()
                for cls in range(num_classes):
                    mask = labels_cpu == cls
                    if np.any(mask):
                        class_correct[cls] += np.sum(preds_cpu[mask] == labels_cpu[mask])
                        class_total[cls] += np.sum(mask)

            epoch_loss = running_loss / max(dataset_sizes[phase], 1)
            epoch_acc = running_corrects / max(dataset_sizes[phase], 1)
            balanced_acc = float((class_correct / np.maximum(class_total, 1)).mean())
            print(
                f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Balanced Acc: {balanced_acc:.4f}",
                flush=True,
            )
            if phase == "val" and balanced_acc > best_balanced:
                best_balanced = balanced_acc
                best_epoch = epoch
                best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    return model, float(best_balanced), int(best_epoch), int(time.time() - t0)


def run_baseline_waterbirds(args) -> RunResult:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _set_training_globals(device, args.momentum, args.weight_decay)
    g = _seed_everything(args.seed)
    dataloaders, dataset_sizes, test_loader, num_classes = build_waterbirds_loaders(
        data_path=args.data_path,
        teacher_map_path=None,
        batch_size=args.batch_size,
        img_size=args.img_size,
        use_attention=False,
        generator=g,
        num_workers=getattr(args, "num_workers", None),
    )
    model = make_model(num_classes=num_classes, pretrained=args.pretrained).to(device)
    print(
        f"\n=== RUN: MobileNetV2 CE baseline | base_lr={args.base_lr} "
        f"classifier_lr={args.classifier_lr} momentum={args.momentum} wd={args.weight_decay} "
        f"nesterov={args.nesterov} batch={args.batch_size} epochs={args.num_epochs} "
        f"img_size={args.img_size} seed={args.seed} ===",
        flush=True,
    )
    best_model, best_score, best_epoch, seconds = train_ce_model(
        model,
        dataloaders,
        dataset_sizes,
        num_classes,
        args.num_epochs,
        args.base_lr,
        args.classifier_lr,
        args.momentum,
        args.weight_decay,
        args.nesterov,
        device,
    )
    test_loss, test_acc, group_acc, per_group, worst_group = wb_base.evaluate_test(best_model, test_loader)
    print(f"\n[VAL] Best Balanced Acc: {best_score:.4f} at epoch {best_epoch}", flush=True)
    print(f"[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%", flush=True)
    if group_acc is not None:
        for name, acc in zip(wb_base.GROUP_NAMES, group_acc):
            print(f"[TEST] {name}: {acc:.2f}%", flush=True)
        print(f"[TEST] Per Group: {per_group:.2f}%  Worst Group: {worst_group:.2f}%", flush=True)
    ckpt = _save_checkpoint(best_model, args.checkpoint_dir, "mobilenet_v2_ce", int(args.seed))
    return RunResult(float(best_score), float(test_acc), float(per_group), float(worst_group), ckpt, int(best_epoch), int(seconds))


def run_baseline_redmeat(args) -> RunResult:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _set_training_globals(device, args.momentum, args.weight_decay)
    g = _seed_everything(args.seed)
    dataloaders, dataset_sizes, test_loader, num_classes, classes = build_redmeat_loaders(
        data_path=args.data_path,
        teacher_map_path=None,
        batch_size=args.batch_size,
        img_size=args.img_size,
        use_attention=False,
        generator=g,
        classes_arg=args.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
        num_workers=getattr(args, "num_workers", None),
    )
    model = make_model(num_classes=num_classes, pretrained=args.pretrained).to(device)
    print(
        f"\n=== RUN: RedMeat MobileNetV2 CE baseline | base_lr={args.base_lr} "
        f"classifier_lr={args.classifier_lr} momentum={args.momentum} wd={args.weight_decay} "
        f"nesterov={args.nesterov} batch={args.batch_size} epochs={args.num_epochs} "
        f"img_size={args.img_size} seed={args.seed} ===",
        flush=True,
    )
    best_model, best_score, best_epoch, seconds = train_ce_model(
        model,
        dataloaders,
        dataset_sizes,
        num_classes,
        args.num_epochs,
        args.base_lr,
        args.classifier_lr,
        args.momentum,
        args.weight_decay,
        args.nesterov,
        device,
    )
    test_loss, test_acc, class_acc, per_group, worst_group = rm_base.evaluate_test(best_model, test_loader, num_classes)
    print(f"\n[VAL] Best Balanced Acc: {best_score:.4f} at epoch {best_epoch}", flush=True)
    print(f"[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%", flush=True)
    for cls_name, acc in zip(classes, class_acc):
        print(f"[TEST] {cls_name}: {acc:.2f}%", flush=True)
    print(f"[TEST] Per-class mean: {per_group:.2f}%  Worst-class: {worst_group:.2f}%", flush=True)
    ckpt = _save_checkpoint(best_model, args.checkpoint_dir, "mobilenet_v2_redmeat_ce", int(args.seed))
    return RunResult(float(best_score), float(test_acc), float(per_group), float(worst_group), ckpt, int(best_epoch), int(seconds))


def write_csv_row(csv_path: str, row: Dict[str, object], header: Sequence[str]) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(header))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_rows(rows: Sequence[Dict[str, object]], keys: Sequence[str]) -> None:
    print("\n[MULTI-SEED DONE]", flush=True)
    for key in keys:
        arr = np.array([float(r[key]) for r in rows], dtype=float)
        suffix = "%" if key != "best_balanced_val_acc" else ""
        print(f"{key}: mean={arr.mean():.4f}{suffix}, std={arr.std(ddof=0):.4f}{suffix}", flush=True)
