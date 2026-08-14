#!/usr/bin/env python3
"""Train one ImageNet-9 baseline on Original train/validation data."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from imagenet9_data import (
    CLASS_NAMES,
    NUM_CLASSES,
    TUNING_OBJECTIVE,
    ImageNet9Dataset,
    build_eval_transform,
    build_train_transform,
    class_counts,
    load_original_samples,
)


METHODS = (
    "erm",
    "upweight",
    "abn",
    "elrep",
    "gals",
    "gals_gradcam",
    "gals_abn",
)


class ImageNet9GALSDataset(Dataset):
    """Original ImageNet-9 images joined to manifest-keyed CLIP ViT maps."""

    def __init__(self, samples, transform, map_root: Path, image_size: int) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.map_root = map_root
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        from PIL import Image

        sample = self.samples[index]
        with Image.open(sample.path) as image_file:
            image = image_file.convert("RGB")
        image = self.transform(image)

        map_path = self.map_root / "maps" / sample.class_name / f"{sample.sample_id}.pth"
        if not map_path.is_file():
            raise FileNotFoundError(f"Missing GALS map: {map_path}")
        payload = torch.load(str(map_path), map_location="cpu")
        attention = payload.get("unnormalized_attentions")
        if not torch.is_tensor(attention) or tuple(attention.shape) != (2, 1, 7, 7):
            shape = tuple(attention.shape) if torch.is_tensor(attention) else None
            raise RuntimeError(f"Invalid GALS map {map_path}: shape={shape}")
        if not bool(torch.isfinite(attention).all()):
            raise RuntimeError(f"Non-finite GALS map: {map_path}")
        attention = F.interpolate(
            attention.float(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return {
            "image": image,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "attention": attention,
            "sample_id": sample.sample_id,
        }


def combine_gals_attention(attention: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match GALS average_nonzero prompt aggregation and per-map normalization."""
    if attention.ndim != 5:
        raise RuntimeError(f"Expected BxPx1xHxW attention, got {tuple(attention.shape)}")
    prompt_nonzero = attention.flatten(2).ne(0).any(dim=2)
    valid_samples = prompt_nonzero.any(dim=1)
    weights = prompt_nonzero.to(attention.dtype).view(attention.shape[0], attention.shape[1], 1, 1, 1)
    combined = (attention * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    combined = combined / denominator
    flat = combined.flatten(1)
    minimum = flat.min(dim=1, keepdim=True).values
    maximum = flat.max(dim=1, keepdim=True).values
    normalized = (flat - minimum) / maximum.clamp_min(1e-12)
    return normalized.view_as(combined), valid_samples


def ground_truth_gradcam(
    feature_maps: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Grad-CAM used by the existing GALS Grad-CAM baseline."""
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, targets.view(-1, 1), 1.0)
    gradients = torch.autograd.grad(
        logits,
        feature_maps,
        grad_outputs=one_hot,
        retain_graph=True,
        create_graph=True,
    )[0]
    weights = F.adaptive_avg_pool2d(gradients, 1)
    gradcam = F.relu((feature_maps * weights).sum(dim=1, keepdim=True))
    flat = gradcam.flatten(1)
    maximum = flat.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    minimum = flat.min(dim=1, keepdim=True).values
    return ((flat - minimum) / maximum).view_as(gradcam)


@dataclass
class TrainResult:
    method: str
    seed: int
    best_epoch: int
    best_val_accuracy: float
    best_val_macro_class_accuracy: float
    best_val_per_class_accuracy: List[float]
    train_seconds: float
    checkpoint: str
    train_samples: int
    val_samples: int
    class_weights: List[float]


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _torchvision_resnet50(pretrained: bool) -> nn.Module:
    if hasattr(models, "ResNet50_Weights"):
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights)
    return models.resnet50(pretrained=pretrained)


def _load_abn_pretrained(model: nn.Module, checkpoint_path: Path) -> Dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"ABN requires the pretrained checkpoint: {checkpoint_path}. "
            "Expected the 349 MB resnet50_abn_imagenet.pth.tar file."
        )
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise RuntimeError(f"ABN checkpoint has no state_dict: {checkpoint_path}")
    raw_state = payload["state_dict"]
    state = {}
    removed = []
    for raw_key, value in raw_state.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        # Match the existing ABN baseline: replace ImageNet-specific attention
        # classifiers and final classifier for the nine-class task.
        if "att_conv" in key or "bn_att2" in key or "fc" in key:
            removed.append(key)
            continue
        state[key] = value
    incompatible = model.load_state_dict(state, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded": len(state),
        "removed": len(removed),
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }


def build_model(
    method: str,
    pretrained: bool,
    abn_checkpoint: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    if method in {"abn", "gals_abn"}:
        from models.resnet_abn import resnet50 as resnet50_abn

        model = resnet50_abn(
            pretrained=False,
            num_classes=NUM_CLASSES,
            add_after_attention=True,
        )
        if not pretrained:
            return model, {"pretrained": False}
        if abn_checkpoint is None:
            raise ValueError("--abn-checkpoint is required for pretrained ABN")
        return model, _load_abn_pretrained(model, abn_checkpoint)

    model = _torchvision_resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model, {"pretrained": pretrained, "source": "torchvision_resnet50"}


def split_parameter_groups(
    model: nn.Module,
    base_lr: float,
    classifier_lr: float,
) -> List[Dict[str, object]]:
    base_parameters = []
    classifier_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "fc" in name:
            classifier_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    if not base_parameters or not classifier_parameters:
        raise RuntimeError(
            f"Expected non-empty backbone and fc parameter groups; "
            f"base={len(base_parameters)} classifier={len(classifier_parameters)}"
        )
    return [
        {"params": base_parameters, "lr": base_lr},
        {"params": classifier_parameters, "lr": classifier_lr},
    ]


def forward_resnet_features(
    model: nn.Module,
    images: torch.Tensor,
    return_feature_map: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = model.conv1(images)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)
    pooled = torch.flatten(model.avgpool(features), 1)
    return model.fc(pooled), features if return_feature_map else pooled


def elrep_penalty(features: torch.Tensor, theta1: float, theta2: float) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(features.float())
    batch_size = max(int(features.shape[0]), 1)
    nuclear = theta1 * singular_values.abs().sum() / batch_size
    frobenius = theta2 * singular_values.square().sum() / batch_size
    return nuclear + frobenius


def _forward(
    method: str,
    model: nn.Module,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if method in {"abn", "gals_abn"}:
        attention_logits, logits, auxiliary = model(images)
        attention = auxiliary[0] if method == "gals_abn" else None
        return logits, attention_logits, attention
    if method == "elrep":
        logits, features = forward_resnet_features(model, images)
        return logits, None, features
    if method == "gals_gradcam":
        logits, feature_maps = forward_resnet_features(
            model, images, return_feature_map=True
        )
        return logits, None, feature_maps
    return model(images), None, None


def inverse_frequency_class_weights(samples: Sequence[object]) -> torch.Tensor:
    counts = Counter(int(sample.label) for sample in samples)
    if any(counts[label] == 0 for label in range(NUM_CLASSES)):
        raise RuntimeError(f"Training data is missing an ImageNet-9 class: {dict(counts)}")
    inverse = torch.tensor([1.0 / counts[label] for label in range(NUM_CLASSES)])
    return inverse / inverse.max()


def build_loaders(
    manifest_path: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    image_size: int = 224,
    verify_files: bool = True,
    method: str = "erm",
    gals_map_root: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, Sequence[object], Sequence[object]]:
    train_samples = load_original_samples(manifest_path, "train", verify_files)
    val_samples = load_original_samples(manifest_path, "val", verify_files)
    expected_train = {name: 5045 for name in CLASS_NAMES}
    expected_val = {name: 450 for name in CLASS_NAMES}
    if class_counts(train_samples) != expected_train:
        raise RuntimeError(f"Unexpected IN-9 train counts: {class_counts(train_samples)}")
    if class_counts(val_samples) != expected_val:
        raise RuntimeError(f"Unexpected IN-9 validation counts: {class_counts(val_samples)}")

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if method in {"gals", "gals_gradcam", "gals_abn"}:
        if gals_map_root is None:
            raise ValueError("GALS requires --gals-map-root")
        train_dataset = ImageNet9GALSDataset(
            train_samples,
            build_train_transform(image_size),
            gals_map_root,
            image_size,
        )
    else:
        train_dataset = ImageNet9Dataset(train_samples, build_train_transform(image_size))
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        ImageNet9Dataset(val_samples, build_eval_transform(image_size)),
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, train_samples, val_samples


def _metric_arrays() -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros(NUM_CLASSES, dtype=np.int64),
        np.zeros(NUM_CLASSES, dtype=np.int64),
    )


def _update_metrics(
    correct: np.ndarray,
    total: np.ndarray,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    for label in range(NUM_CLASSES):
        mask = targets_np == label
        total[label] += int(mask.sum())
        correct[label] += int((predictions_np[mask] == targets_np[mask]).sum())


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    result: TrainResult,
    model_metadata: Mapping[str, object],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": args.method,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "hparams": {
                "base_lr": args.base_lr,
                "classifier_lr": args.classifier_lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "abn_cls_weight": args.abn_cls_weight,
                "theta1": args.theta1,
                "theta2": args.theta2,
                "grad_weight": args.grad_weight,
                "grad_criterion": args.grad_criterion,
                "cam_weight": args.cam_weight,
                "abn_att_weight": args.abn_att_weight,
                "gals_map_root": str(args.gals_map_root) if args.gals_map_root else "",
                "epochs": args.epochs,
                "seed": args.seed,
            },
            "result": asdict(result),
            "model_metadata": dict(model_metadata),
        },
        str(temporary),
    )
    temporary.replace(checkpoint_path)


def train(args: argparse.Namespace) -> TrainResult:
    if args.method not in METHODS:
        raise ValueError(f"Unsupported method: {args.method}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train_loader, val_loader, train_samples, val_samples = build_loaders(
        args.manifest,
        args.batch_size,
        args.num_workers,
        args.seed,
        args.image_size,
        verify_files=not args.skip_file_checks,
        method=args.method,
        gals_map_root=args.gals_map_root,
    )
    class_weights = inverse_frequency_class_weights(train_samples)
    if args.method != "upweight":
        class_weights = torch.ones(NUM_CLASSES)

    model, model_metadata = build_model(
        args.method,
        args.pretrained,
        args.abn_checkpoint,
    )
    model.to(device)
    optimizer = SGD(
        split_parameter_groups(model, args.base_lr, args.classifier_lr),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    train_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    val_criterion = nn.CrossEntropyLoss()

    best_macro = -1.0
    best_accuracy = -1.0
    best_epoch = -1
    best_per_class: List[float] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    start = time.time()

    print(
        f"[RUN] method={args.method} seed={args.seed} epochs={args.epochs} "
        f"train={len(train_samples)} val={len(val_samples)} device={device}",
        flush=True,
    )
    if args.method == "gals":
        print(
            f"[GALS] map_root={args.gals_map_root} grad_weight={args.grad_weight:.9g} "
            f"grad_criterion={args.grad_criterion} combine=average_nonzero "
            "mode=suppress_outside",
            flush=True,
        )
    elif args.method == "gals_gradcam":
        print(
            f"[GALS-GRADCAM] map_root={args.gals_map_root} "
            f"cam_weight={args.cam_weight:.9g} criterion=L1 mode=match "
            "combine=average_nonzero target_layer=layer4",
            flush=True,
        )
    elif args.method == "gals_abn":
        print(
            f"[GALS-ABN] map_root={args.gals_map_root} "
            f"abn_cls_weight={args.abn_cls_weight:.9g} "
            f"abn_att_weight={args.abn_att_weight:.9g} criterion=L1 "
            "mode=suppress_outside combine=average_nonzero attention_shape=14x14",
            flush=True,
        )
    print(
        f"[HPARAMS] base_lr={args.base_lr:.9g} classifier_lr={args.classifier_lr:.9g} "
        f"momentum={args.momentum:.6g} weight_decay={args.weight_decay:.9g} "
        f"abn_cls_weight={args.abn_cls_weight:.9g} theta1={args.theta1:.9g} "
        f"theta2={args.theta2:.9g}",
        flush=True,
    )
    print(f"[MODEL] {json.dumps(model_metadata, default=str, sort_keys=True)}", flush=True)
    print(f"[CLASS WEIGHTS] {class_weights.tolist()}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            if args.method == "gals":
                images.requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            logits, attention_logits, features = _forward(args.method, model, images)
            loss = train_criterion(logits, targets)
            if args.method in {"abn", "gals_abn"}:
                if attention_logits is None:
                    raise RuntimeError("ABN did not return attention-branch logits")
                loss = loss + args.abn_cls_weight * train_criterion(attention_logits, targets)
                if args.method == "gals_abn":
                    if features is None:
                        raise RuntimeError("GALS ABN did not return its spatial attention map")
                    teacher, valid = combine_gals_attention(
                        batch["attention"].to(device, non_blocking=True)
                    )
                    if not bool(valid.all()):
                        invalid = int((~valid).sum().item())
                        raise RuntimeError(
                            f"GALS ABN batch contains {invalid} all-zero teacher maps"
                        )
                    teacher = F.interpolate(
                        teacher,
                        size=features.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    attention_loss = F.l1_loss(features, features * teacher)
                    loss = loss + args.abn_att_weight * attention_loss
            elif args.method == "elrep":
                if features is None:
                    raise RuntimeError("ElRep did not return penultimate features")
                loss = loss + elrep_penalty(features, args.theta1, args.theta2)
            elif args.method == "gals":
                teacher, valid = combine_gals_attention(
                    batch["attention"].to(device, non_blocking=True)
                )
                if not bool(valid.all()):
                    invalid = int((~valid).sum().item())
                    raise RuntimeError(f"GALS batch contains {invalid} all-zero teacher maps")
                input_gradient = torch.autograd.grad(
                    loss,
                    images,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                outside_gradient = input_gradient * (1.0 - teacher)
                if args.grad_criterion == "L1":
                    gradient_loss = outside_gradient.abs().mean()
                elif args.grad_criterion == "L2":
                    gradient_loss = outside_gradient.square().mean()
                else:
                    raise ValueError(f"Unknown gradient criterion: {args.grad_criterion}")
                loss = loss + args.grad_weight * gradient_loss
            elif args.method == "gals_gradcam":
                if features is None:
                    raise RuntimeError("GALS Grad-CAM did not return layer4 feature maps")
                teacher, valid = combine_gals_attention(
                    batch["attention"].to(device, non_blocking=True)
                )
                if not bool(valid.all()):
                    invalid = int((~valid).sum().item())
                    raise RuntimeError(
                        f"GALS Grad-CAM batch contains {invalid} all-zero teacher maps"
                    )
                gradcam = ground_truth_gradcam(features, logits, targets)
                teacher = F.interpolate(
                    teacher,
                    size=gradcam.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                cam_loss = F.l1_loss(gradcam, teacher)
                loss = loss + args.cam_weight * cam_loss
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().item()) * images.shape[0]
            train_count += images.shape[0]

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        correct, total = _metric_arrays()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["label"].to(device, non_blocking=True)
                logits, _attention_logits, _features = _forward(args.method, model, images)
                loss = val_criterion(logits, targets)
                predictions = logits.argmax(dim=1)
                val_loss_sum += float(loss.item()) * images.shape[0]
                val_count += images.shape[0]
                _update_metrics(correct, total, predictions, targets)

        if np.any(total == 0):
            raise RuntimeError(f"Validation epoch is missing classes: support={total.tolist()}")
        per_class = correct / total
        macro = float(per_class.mean())
        accuracy = float(correct.sum() / total.sum())
        if macro > best_macro:
            best_macro = macro
            best_accuracy = accuracy
            best_epoch = epoch + 1
            best_per_class = per_class.tolist()
            if args.checkpoint:
                best_state = copy.deepcopy(model.state_dict())
        print(
            f"[EPOCH] {epoch + 1:02d}/{args.epochs} "
            f"train_loss={train_loss_sum / max(train_count, 1):.6f} "
            f"val_loss={val_loss_sum / max(val_count, 1):.6f} "
            f"val_acc={accuracy:.6f} val_macro={macro:.6f} best_macro={best_macro:.6f}",
            flush=True,
        )

    checkpoint = ""
    result = TrainResult(
        method=args.method,
        seed=args.seed,
        best_epoch=best_epoch,
        best_val_accuracy=best_accuracy,
        best_val_macro_class_accuracy=best_macro,
        best_val_per_class_accuracy=best_per_class,
        train_seconds=time.time() - start,
        checkpoint=checkpoint,
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        class_weights=[float(value) for value in class_weights.tolist()],
    )
    if args.checkpoint:
        if best_state is None:
            raise RuntimeError("Checkpoint requested but no best model state was captured")
        model.load_state_dict(best_state)
        checkpoint_path = Path(args.checkpoint)
        result.checkpoint = str(checkpoint_path.resolve())
        _save_checkpoint(checkpoint_path, model, args, result, model_metadata)

    print(f"[RESULT] {json.dumps(asdict(result), sort_keys=True)}", flush=True)
    print(
        f"[OBJECTIVE] name={TUNING_OBJECTIVE} value={result.best_val_macro_class_accuracy:.9f}",
        flush=True,
    )
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--base-lr", type=float, required=True)
    parser.add_argument("--classifier-lr", type=float, required=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--abn-cls-weight", type=float, default=1.0)
    parser.add_argument("--theta1", type=float, default=1e-4)
    parser.add_argument("--theta2", type=float, default=1e-5)
    parser.add_argument("--gals-map-root", type=Path)
    parser.add_argument("--grad-weight", type=float, default=1e4)
    parser.add_argument("--grad-criterion", choices=("L1", "L2"), default="L1")
    parser.add_argument("--cam-weight", type=float, default=1.0)
    parser.add_argument("--abn-att-weight", type=float, default=1.0)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--abn-checkpoint", type=Path)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-file-checks", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
