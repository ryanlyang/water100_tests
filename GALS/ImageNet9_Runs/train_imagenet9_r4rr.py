#!/usr/bin/env python3
"""Train one R4RR ResNet-50 trial on ImageNet-9 Original data."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF

from audit_imagenet9_r4rr_weclip_maps import voc_colormap
from imagenet9_data import (
    CLASS_NAMES,
    NUM_CLASSES,
    TUNING_OBJECTIVE,
    ImageNet9Dataset,
    build_eval_transform,
    class_counts,
    load_original_samples,
)
from train_imagenet9_baseline import (
    _metric_arrays,
    _torchvision_resnet50,
    _update_metrics,
    seed_everything,
    seed_worker,
    split_parameter_groups,
)


EXPECTED_TRAIN = 45405
EXPECTED_VAL = 4050
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class R4RRTrainResult:
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
    attention_epoch: int
    kl_lambda: float
    kl_increment: float
    base_lr: float
    classifier_lr: float
    lr2_mult: float
    invalid_teacher_samples_seen: int
    aligned_teacher_samples_seen: int


def decode_target_mask(encoded: np.ndarray, target_label: int) -> np.ndarray:
    """Decode one target class from a VOC-color semantic prediction."""
    if encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(f"Expected an RGB teacher map, got {encoded.shape}")
    if not 0 <= target_label < 9:
        raise ValueError(f"ImageNet-9 label is outside 0..8: {target_label}")
    target_color = voc_colormap(10)[target_label + 1]
    return np.all(encoded == target_color, axis=2).astype(np.uint8)


def joint_train_transform(
    image: Image.Image,
    target_mask: Image.Image,
    image_size: int,
    crop_params: Optional[Tuple[int, int, int, int]] = None,
    flip: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the baseline crop/flip to the image and teacher mask together."""
    if image.size != target_mask.size:
        raise RuntimeError(f"Image/map size mismatch: {image.size} != {target_mask.size}")
    if crop_params is None:
        crop_params = RandomResizedCrop.get_params(
            image,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        )
    top, left, height, width = crop_params
    image = TF.resized_crop(
        image,
        top,
        left,
        height,
        width,
        (image_size, image_size),
        interpolation=InterpolationMode.BILINEAR,
    )
    target_mask = TF.resized_crop(
        target_mask,
        top,
        left,
        height,
        width,
        (image_size, image_size),
        interpolation=InterpolationMode.NEAREST,
    )
    if flip is None:
        flip = random.random() < 0.5
    if flip:
        image = TF.hflip(image)
        target_mask = TF.hflip(target_mask)
    image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
    mask_tensor = torch.from_numpy(
        np.array(target_mask, dtype=np.uint8, copy=True)
    ).float()
    return image_tensor, mask_tensor


class ImageNet9R4RRDataset(Dataset):
    """Original training images paired with target-class WeCLIP+ masks."""

    def __init__(self, samples: Sequence[object], teacher_map_root: Path, image_size: int) -> None:
        self.samples = list(samples)
        self.teacher_map_root = teacher_map_root
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        sample = self.samples[index]
        map_path = self.teacher_map_root / f"{sample.sample_id}.png"
        if not map_path.is_file():
            raise FileNotFoundError(f"Missing R4RR teacher map: {map_path}")
        with Image.open(sample.path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(map_path) as map_file:
            if map_file.mode != "RGB":
                raise RuntimeError(f"Teacher map is not RGB: {map_path} mode={map_file.mode}")
            if map_file.size != image.size:
                raise RuntimeError(
                    f"Teacher/source size mismatch for {sample.sample_id}: "
                    f"{map_file.size} != {image.size}"
                )
            encoded = np.array(map_file, dtype=np.uint8, copy=True)
        target_mask = Image.fromarray(
            decode_target_mask(encoded, sample.label), mode="L"
        )
        image_tensor, mask_tensor = joint_train_transform(
            image, target_mask, self.image_size
        )
        return {
            "image": image_tensor,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "teacher_mask": mask_tensor,
            "sample_id": sample.sample_id,
        }


def forward_resnet_cam(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return logits and normalized ground-truth-class CAMs from layer4."""
    features = model.conv1(images)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)
    pooled = torch.flatten(model.avgpool(features), 1)
    logits = model.fc(pooled)
    weights = model.fc.weight[targets]
    cams = F.relu(torch.einsum("bc,bchw->bhw", weights, features))
    flat = cams.flatten(1)
    minimum = flat.min(dim=1, keepdim=True).values
    maximum = flat.max(dim=1, keepdim=True).values
    normalized = (flat - minimum) / (maximum - minimum + 1e-8)
    return logits, normalized.view_as(cams)


def r4rr_alignment_loss(
    cams: torch.Tensor,
    teacher_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward KL over valid target masks; invalid maps remain CE-only."""
    if cams.ndim != 3 or teacher_masks.ndim != 3:
        raise ValueError(
            f"Expected BxHxW CAM/mask tensors, got {cams.shape}, {teacher_masks.shape}"
        )
    teacher_small = F.interpolate(
        teacher_masks.unsqueeze(1),
        size=cams.shape[-2:],
        mode="area",
    ).squeeze(1)
    teacher_flat = teacher_small.flatten(1)
    valid = teacher_flat.sum(dim=1) > 0
    if not bool(valid.any()):
        return cams.sum() * 0.0, valid
    cam_flat = cams[valid].flatten(1)
    teacher_flat = teacher_flat[valid]
    teacher_prob = teacher_flat / teacher_flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
    loss = F.kl_div(
        F.log_softmax(cam_flat, dim=1),
        teacher_prob,
        reduction="batchmean",
    )
    return loss, valid


def build_loaders(args: argparse.Namespace):
    verify = not args.skip_file_checks
    train_samples = load_original_samples(args.manifest, "train", verify)
    val_samples = load_original_samples(args.manifest, "val", verify)
    expected_train = {name: 5045 for name in CLASS_NAMES}
    expected_val = {name: 450 for name in CLASS_NAMES}
    if len(train_samples) != EXPECTED_TRAIN or class_counts(train_samples) != expected_train:
        raise RuntimeError(f"Unexpected ImageNet-9 train split: {class_counts(train_samples)}")
    if len(val_samples) != EXPECTED_VAL or class_counts(val_samples) != expected_val:
        raise RuntimeError(f"Unexpected ImageNet-9 validation split: {class_counts(val_samples)}")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "worker_init_fn": seed_worker,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    train_loader = DataLoader(
        ImageNet9R4RRDataset(train_samples, args.teacher_map_root, args.image_size),
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        ImageNet9Dataset(val_samples, build_eval_transform(args.image_size)),
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, train_samples, val_samples


def make_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
    learning_rate_multiplier: float,
) -> SGD:
    return SGD(
        split_parameter_groups(
            model,
            args.base_lr * learning_rate_multiplier,
            args.classifier_lr * learning_rate_multiplier,
        ),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    result: R4RRTrainResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": "r4rr",
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "hparams": {
                "attention_epoch": args.attention_epoch,
                "kl_lambda": args.kl_lambda,
                "kl_increment": args.kl_increment,
                "base_lr": args.base_lr,
                "classifier_lr": args.classifier_lr,
                "lr2_mult": args.lr2_mult,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "seed": args.seed,
                "teacher_map_root": str(args.teacher_map_root),
            },
            "result": asdict(result),
        },
        str(temporary),
    )
    temporary.replace(path)


def train(args: argparse.Namespace) -> R4RRTrainResult:
    if args.method != "r4rr":
        raise ValueError(f"This trainer only supports r4rr, got {args.method}")
    if not 0 <= args.attention_epoch < args.epochs:
        raise ValueError(
            f"attention_epoch must be in [0, {args.epochs - 1}], got {args.attention_epoch}"
        )
    if args.kl_lambda <= 0 or args.lr2_mult <= 0:
        raise ValueError("kl_lambda and lr2_mult must be positive")
    if args.kl_increment is None:
        args.kl_increment = args.kl_lambda / 10.0

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train_loader, val_loader, train_samples, val_samples = build_loaders(args)

    model = _torchvision_resnet50(args.pretrained)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)
    optimizer = make_optimizer(model, args, 1.0)
    criterion = nn.CrossEntropyLoss()

    best_macro = -1.0
    best_accuracy = -1.0
    best_epoch = -1
    best_per_class: List[float] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    invalid_seen = 0
    aligned_seen = 0
    start = time.time()

    print(
        f"[RUN] method=r4rr seed={args.seed} epochs={args.epochs} "
        f"train={len(train_samples)} val={len(val_samples)} device={device}",
        flush=True,
    )
    print(
        f"[R4RR] attention_epoch={args.attention_epoch} kl_lambda={args.kl_lambda:.9g} "
        f"kl_increment={args.kl_increment:.9g} lr2_mult={args.lr2_mult:.9g} "
        f"teacher_map_root={args.teacher_map_root}",
        flush=True,
    )
    print(
        f"[HPARAMS] base_lr={args.base_lr:.9g} classifier_lr={args.classifier_lr:.9g} "
        f"momentum={args.momentum:.6g} weight_decay={args.weight_decay:.9g}",
        flush=True,
    )
    print(
        "[PROTOCOL] CAM=ResNet50 layer4 ground-truth class; alignment=forward_kl; "
        "invalid target maps=classification_only; selection=Original val macro class accuracy",
        flush=True,
    )

    for epoch in range(args.epochs):
        if epoch == args.attention_epoch:
            optimizer = make_optimizer(model, args, args.lr2_mult)
            best_macro = -1.0
            best_accuracy = -1.0
            best_epoch = -1
            best_per_class = []
            best_state = None
            print(
                f"[PHASE] epoch={epoch} Classify->Align optimizer restart "
                f"base_lr={args.base_lr * args.lr2_mult:.9g} "
                f"classifier_lr={args.classifier_lr * args.lr2_mult:.9g}",
                flush=True,
            )

        align_active = epoch >= args.attention_epoch
        current_kl = args.kl_lambda + max(0, epoch - args.attention_epoch) * args.kl_increment
        model.train()
        train_loss_sum = 0.0
        ce_loss_sum = 0.0
        alignment_loss_sum = 0.0
        train_count = 0
        epoch_invalid = 0
        epoch_aligned = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            teacher_masks = batch["teacher_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, cams = forward_resnet_cam(model, images, targets)
            ce_loss = criterion(logits, targets)
            alignment_loss, valid = r4rr_alignment_loss(cams, teacher_masks)
            loss = ce_loss + current_kl * alignment_loss if align_active else ce_loss
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            valid_count = int(valid.sum().item())
            epoch_aligned += valid_count
            epoch_invalid += batch_size - valid_count
            train_loss_sum += float(loss.detach().item()) * batch_size
            ce_loss_sum += float(ce_loss.detach().item()) * batch_size
            alignment_loss_sum += float(alignment_loss.detach().item()) * valid_count
            train_count += batch_size

        invalid_seen += epoch_invalid
        aligned_seen += epoch_aligned
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        correct, total = _metric_arrays()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["label"].to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, targets)
                predictions = logits.argmax(dim=1)
                val_loss_sum += float(loss.item()) * images.shape[0]
                val_count += images.shape[0]
                _update_metrics(correct, total, predictions, targets)

        if np.any(total == 0):
            raise RuntimeError(f"Validation epoch is missing classes: support={total.tolist()}")
        per_class = correct / total
        macro = float(per_class.mean())
        accuracy = float(correct.sum() / total.sum())
        if align_active and macro > best_macro:
            best_macro = macro
            best_accuracy = accuracy
            best_epoch = epoch + 1
            best_per_class = per_class.tolist()
            if args.checkpoint:
                best_state = copy.deepcopy(model.state_dict())
        print(
            f"[EPOCH] {epoch + 1:02d}/{args.epochs} phase={'align' if align_active else 'classify'} "
            f"kl={current_kl:.9g} train_loss={train_loss_sum / max(train_count, 1):.6f} "
            f"ce={ce_loss_sum / max(train_count, 1):.6f} "
            f"alignment={alignment_loss_sum / max(epoch_aligned, 1):.6f} "
            f"aligned={epoch_aligned} invalid={epoch_invalid} "
            f"val_loss={val_loss_sum / max(val_count, 1):.6f} "
            f"val_acc={accuracy:.6f} val_macro={macro:.6f} best_macro={best_macro:.6f}",
            flush=True,
        )

    if best_epoch < 0:
        raise RuntimeError("No validation checkpoint was eligible after attention activation")
    result = R4RRTrainResult(
        method="r4rr",
        seed=args.seed,
        best_epoch=best_epoch,
        best_val_accuracy=best_accuracy,
        best_val_macro_class_accuracy=best_macro,
        best_val_per_class_accuracy=best_per_class,
        train_seconds=time.time() - start,
        checkpoint="",
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        attention_epoch=args.attention_epoch,
        kl_lambda=args.kl_lambda,
        kl_increment=args.kl_increment,
        base_lr=args.base_lr,
        classifier_lr=args.classifier_lr,
        lr2_mult=args.lr2_mult,
        invalid_teacher_samples_seen=invalid_seen,
        aligned_teacher_samples_seen=aligned_seen,
    )
    if args.checkpoint:
        if best_state is None:
            raise RuntimeError("Checkpoint requested but no best state was captured")
        model.load_state_dict(best_state)
        checkpoint_path = Path(args.checkpoint)
        result.checkpoint = str(checkpoint_path.resolve())
        _save_checkpoint(checkpoint_path, model, args, result)

    print(f"[RESULT] {json.dumps(asdict(result), sort_keys=True)}", flush=True)
    print(
        f"[OBJECTIVE] name={TUNING_OBJECTIVE} value={result.best_val_macro_class_accuracy:.9f}",
        flush=True,
    )
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("r4rr",), default="r4rr")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-map-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--attention-epoch", type=int, required=True)
    parser.add_argument("--kl-lambda", type=float, required=True)
    parser.add_argument("--kl-increment", type=float)
    parser.add_argument("--base-lr", type=float, required=True)
    parser.add_argument("--classifier-lr", type=float, required=True)
    parser.add_argument("--lr2-mult", type=float, required=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-file-checks", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
