"""Step-5 candidate training primitives for the full DecoyMNIST FCV study.

This module intentionally owns no persistence code.  A model state becomes an
online candidate at the end of each epoch and is consumed immediately by the
later selectors; checkpoints and optimizer states are never written here.
"""

from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from decoy_full_config import CampaignRun
from decoy_manifest_provenance import (
    ManifestProvenanceError,
    validate_manifest_bundle,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CandidateTrainingError(ValueError):
    """Raised when candidate training differs from the locked protocol."""


@dataclass(frozen=True)
class ClassificationMetrics:
    count: int
    loss: float
    accuracy: float
    balanced_class_accuracy: float
    worst_class_accuracy: float
    per_class_accuracy: tuple[float, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "balanced_class_accuracy": self.balanced_class_accuracy,
            "worst_class_accuracy": self.worst_class_accuracy,
            "per_class_accuracy": list(self.per_class_accuracy),
        }


def training_transform_spec(
    config: Mapping[str, Any], run: CampaignRun
) -> Dict[str, Any]:
    """Return the auditable train geometry without importing Torchvision."""

    allowed = [float(value) for value in config["training"]["crop_scale_mins"]]
    if float(run.crop_scale_min) not in allowed:
        raise CandidateTrainingError(
            f"crop_scale_min={run.crop_scale_min} is outside the frozen grid {allowed}."
        )
    augmentation = config["training"]["augmentation"]
    crop = augmentation["train_random_resized_crop"]
    return {
        "operation": "RandomResizedCrop",
        "size": [int(config["model"]["image_size"])] * 2,
        "scale": [float(run.crop_scale_min), float(crop["maximum_scale"])],
        "ratio": [float(value) for value in crop["ratio"]],
        "interpolation": str(crop["interpolation"]),
        "horizontal_flip_probability": float(
            augmentation["horizontal_flip_probability"]
        ),
        "grayscale_to_rgb": "identical_channels",
        "normalization": "imagenet",
    }


def evaluation_transform_spec(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the one evaluation geometry shared by every augmentation regime."""

    preprocessing = config["data"]["preprocessing"]
    return {
        "operation": "Resize",
        "size": [int(value) for value in preprocessing["resize"]["size"]],
        "interpolation": str(preprocessing["resize"]["interpolation"]),
        "crop": None,
        "horizontal_flip_probability": 0.0,
        "grayscale_to_rgb": "identical_channels",
        "normalization": "imagenet",
    }


def _torchvision_transforms():
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except ImportError as exc:  # pragma: no cover - exercised on Tigris
        raise RuntimeError("Step 5 requires the locked Torchvision environment.") from exc
    return transforms, InterpolationMode


def build_training_transform(config: Mapping[str, Any], run: CampaignRun):
    transforms, interpolation_mode = _torchvision_transforms()
    spec = training_transform_spec(config, run)
    if spec["interpolation"] != "bicubic":
        raise CandidateTrainingError("Candidate crops must use bicubic interpolation.")
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                spec["size"][0],
                scale=tuple(spec["scale"]),
                ratio=tuple(spec["ratio"]),
                interpolation=interpolation_mode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_evaluation_transform(config: Mapping[str, Any]):
    transforms, interpolation_mode = _torchvision_transforms()
    spec = evaluation_transform_spec(config)
    if spec["interpolation"] != "bicubic":
        raise CandidateTrainingError("Evaluation resize must use bicubic interpolation.")
    return transforms.Compose(
        [
            transforms.Resize(
                tuple(spec["size"]),
                interpolation=interpolation_mode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ManifestImageDataset:
    """Read an authenticated public manifest without exposing hidden group data."""

    def __init__(
        self,
        config: Mapping[str, Any],
        manifest_path: str | Path,
        role: str,
        transform: Any,
    ) -> None:
        try:
            self.binding = validate_manifest_bundle(config, manifest_path, role)
        except ManifestProvenanceError as exc:
            raise CandidateTrainingError(str(exc)) from exc
        self.frame = pd.read_csv(self.binding.manifest_path)
        self.data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
        self.transform = transform
        self.paths = []
        for relative in self.frame["image_rel_path"].astype(str):
            path = (self.data_root / relative).resolve()
            if not path.is_relative_to(self.data_root):
                raise CandidateTrainingError(f"Manifest path escapes data root: {relative}")
            self.paths.append(path)
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise CandidateTrainingError(f"Manifest images are missing: {missing[:3]}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[int(index)]
        path = self.paths[int(index)]
        try:
            with Image.open(path) as source:
                source.load()
                image = source.convert("L").convert("RGB")
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Could not read {path}: {exc}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), str(row["sample_id"])


def seed_everything(config: Mapping[str, Any], seed: int) -> None:
    """Apply the campaign's deterministic seed policy."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on Tigris
        raise RuntimeError("Step 5 requires PyTorch.") from exc
    deterministic = bool(config["reproducibility"]["deterministic_algorithms"])
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    if deterministic:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            str(config["reproducibility"]["cublas_workspace_config"]),
        )
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = bool(
            config["reproducibility"]["cudnn_benchmark"]
        )
        torch.backends.cudnn.deterministic = deterministic


def seed_worker(_worker_id: int) -> None:
    import torch

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_candidate_dataloaders(
    config: Mapping[str, Any],
    run: CampaignRun,
    train_manifest: str | Path,
    validation_manifest: str | Path,
):
    """Build one seeded train loader and one augmentation-invariant val loader."""

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - exercised on Tigris
        raise RuntimeError("Step 5 requires PyTorch.") from exc
    train = ManifestImageDataset(
        config,
        train_manifest,
        "candidate_train",
        build_training_transform(config, run),
    )
    validation = ManifestImageDataset(
        config,
        validation_manifest,
        "biased_validation",
        build_evaluation_transform(config),
    )
    if train.binding.bundle_sha256 != validation.binding.bundle_sha256:
        raise CandidateTrainingError("Train and validation use different split bundles.")
    generator = torch.Generator()
    generator.manual_seed(int(run.seed))
    validation_generator = torch.Generator()
    validation_generator.manual_seed(10_000 + int(run.seed))
    common = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
        "drop_last": False,
    }
    loaders = {
        "train": DataLoader(train, shuffle=True, generator=generator, **common),
        "biased_validation": DataLoader(
            validation,
            shuffle=False,
            generator=validation_generator,
            **common,
        ),
    }
    return loaders, {"train": train, "biased_validation": validation}


def build_model(config: Mapping[str, Any], *, pretrained: bool = True):
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - exercised on Tigris
        raise RuntimeError("Step 5 requires timm==1.0.28.") from exc
    model_cfg = config["model"]
    model = timm.create_model(
        str(model_cfg["name"]),
        pretrained=bool(pretrained),
        num_classes=int(model_cfg["num_classes"]),
    )
    patch_size = model.patch_embed.patch_size
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
    expected_size = int(model_cfg["patch_size"])
    expected_patches = int(model_cfg["patch_grid_size"]) ** 2
    if tuple(int(value) for value in patch_size) != (expected_size, expected_size):
        raise CandidateTrainingError("Loaded ViT has the wrong patch size.")
    if int(model.patch_embed.num_patches) != expected_patches:
        raise CandidateTrainingError("Loaded ViT has the wrong patch count.")
    return model


def warmup_cosine_factor(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise CandidateTrainingError("Invalid warmup/cosine schedule length.")
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    cosine_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / cosine_steps, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer_and_scheduler(model: Any, config: Mapping[str, Any], run: CampaignRun, steps_per_epoch: int):
    import torch

    if steps_per_epoch <= 0:
        raise CandidateTrainingError("Training loader must contain at least one batch.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(run.learning_rate), weight_decay=float(run.weight_decay)
    )
    total_steps = int(config["training"]["epochs"]) * int(steps_per_epoch)
    warmup_steps = int(config["training"]["scheduler"]["warmup_epochs"]) * int(steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_factor(
            step, total_steps=total_steps, warmup_steps=warmup_steps
        ),
    )
    return optimizer, scheduler


def autocast_context(device: Any, precision: str):
    import torch

    if precision == "amp_bfloat16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "amp_bfloat16" and device.type != "cuda":
        return nullcontext()
    raise CandidateTrainingError(f"Unsupported precision: {precision}")


def train_one_epoch(model: Any, loader: Any, optimizer: Any, scheduler: Any, device: Any, precision: str) -> Dict[str, float]:
    import torch

    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels, _sample_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        count = int(labels.numel())
        total_loss += float(loss.detach().item()) * count
        total_correct += int(logits.argmax(dim=1).eq(labels).sum().item())
        total += count
    if not total:
        raise CandidateTrainingError("Training loader was empty.")
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def evaluate_classifier(model: Any, loader: Any, device: Any, precision: str, num_classes: int) -> ClassificationMetrics:
    import torch

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    class_correct = np.zeros(int(num_classes), dtype=np.int64)
    class_total = np.zeros(int(num_classes), dtype=np.int64)
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for images, labels, _sample_ids in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(device, precision):
                logits = model(images)
                loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            labels_np = labels.cpu().numpy()
            matches_np = predictions.eq(labels).cpu().numpy()
            class_total += np.bincount(labels_np, minlength=int(num_classes))
            class_correct += np.bincount(
                labels_np[matches_np], minlength=int(num_classes)
            )
            total_loss += float(loss.item())
    if np.any(class_total == 0):
        raise CandidateTrainingError("Evaluation split is missing at least one class.")
    per_class = class_correct.astype(np.float64) / class_total
    count = int(class_total.sum())
    return ClassificationMetrics(
        count=count,
        loss=total_loss / count,
        accuracy=float(class_correct.sum() / count),
        balanced_class_accuracy=float(per_class.mean()),
        worst_class_accuracy=float(per_class.min()),
        per_class_accuracy=tuple(float(value) for value in per_class),
    )
