#!/usr/bin/env python3
"""Manifest-backed datasets and validation metrics for ImageNet-9."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES: Tuple[str, ...] = (
    "dog",
    "bird",
    "vehicle",
    "reptile",
    "carnivore",
    "insect",
    "instrument",
    "primate",
    "fish",
)
NUM_CLASSES = len(CLASS_NAMES)
TUNING_OBJECTIVE = "val_macro_class_accuracy"
FINAL_EVALUATION_VARIANTS: Tuple[str, ...] = (
    "original",
    "mixed_same",
    "mixed_rand",
    "mixed_next",
)
FORBIDDEN_SELECTION_VARIANTS: Tuple[str, ...] = (
    "original",
    "mixed_same",
    "mixed_rand",
    "mixed_next",
    "only_fg",
    "only_bg_b",
    "only_bg_t",
    "no_fg",
)


@dataclass(frozen=True)
class ImageNet9Sample:
    sample_id: str
    path: Path
    label: int
    class_name: str
    split: str
    imagenet_index: int = -1
    synset: str = ""


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_label(label: int, class_name: str, source: Path) -> None:
    if not 0 <= label < NUM_CLASSES:
        raise RuntimeError(f"Invalid ImageNet-9 label {label} in {source}")
    expected = CLASS_NAMES[label]
    if class_name.lower() != expected:
        raise RuntimeError(
            f"Label/name mismatch in {source}: label={label} expects {expected}, got {class_name}"
        )


def load_original_samples(
    manifest_path: Path,
    split: str,
    verify_files: bool = True,
) -> List[ImageNet9Sample]:
    if split not in {"train", "val"}:
        raise ValueError(f"Original manifest split must be train or val, got {split}")
    rows = _read_csv(manifest_path)
    samples: List[ImageNet9Sample] = []
    for row in rows:
        if row["split"] != split:
            continue
        label = int(row["label"])
        class_name = row["class_name"].lower()
        _validate_label(label, class_name, manifest_path)
        path = Path(row["source_path"])
        if verify_files and not path.is_file():
            raise FileNotFoundError(path)
        samples.append(
            ImageNet9Sample(
                sample_id=row["sample_id"],
                path=path,
                label=label,
                class_name=class_name,
                split=split,
                imagenet_index=int(row["imagenet_index"]),
                synset=row["synset"],
            )
        )
    if not samples:
        raise RuntimeError(f"No {split} samples found in {manifest_path}")
    _validate_unique_ids(samples, manifest_path)
    return samples


def load_official_variant_samples(
    official_manifest_path: Path,
    official_test_root: Path,
    variant: str,
    verify_files: bool = True,
) -> List[ImageNet9Sample]:
    if variant not in FORBIDDEN_SELECTION_VARIANTS:
        raise ValueError(f"Unknown official Backgrounds Challenge variant: {variant}")
    rows = _read_csv(official_manifest_path)
    samples: List[ImageNet9Sample] = []
    for row in rows:
        if row["variant"] != variant:
            continue
        label = int(row["label"])
        class_name = row["class_name"].lower()
        _validate_label(label, class_name, official_manifest_path)
        relative_path = Path(row["relative_path"])
        path = official_test_root / relative_path
        if verify_files and not path.is_file():
            raise FileNotFoundError(path)
        samples.append(
            ImageNet9Sample(
                sample_id=f"{variant}:{relative_path.as_posix()}",
                path=path,
                label=label,
                class_name=class_name,
                split=f"official_test:{variant}",
            )
        )
    if not samples:
        raise RuntimeError(f"No {variant} rows found in {official_manifest_path}")
    _validate_unique_ids(samples, official_manifest_path)
    return samples


def _validate_unique_ids(samples: Sequence[ImageNet9Sample], source: Path) -> None:
    counts = Counter(sample.sample_id for sample in samples)
    duplicates = [sample_id for sample_id, count in counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate sample IDs in {source}: {duplicates[:10]}")


def class_counts(samples: Iterable[ImageNet9Sample]) -> Dict[str, int]:
    counts = Counter(sample.label for sample in samples)
    return {CLASS_NAMES[label]: counts[label] for label in range(NUM_CLASSES)}


class ImageNet9Dataset(Dataset):
    """Dataset with stable metadata keys for training and teacher-map joins."""

    def __init__(
        self,
        samples: Sequence[ImageNet9Sample],
        transform: Optional[Callable] = None,
    ) -> None:
        if not samples:
            raise ValueError("ImageNet9Dataset requires at least one sample")
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        sample = self.samples[index]
        with Image.open(sample.path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_id,
            "image_path": str(sample.path),
            "class_name": sample.class_name,
            "split": sample.split,
            "index": index,
        }


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_eval_transform(image_size: int = 224, resize_size: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_original_datasets(
    manifest_path: Path,
    image_size: int = 224,
    verify_files: bool = True,
) -> Tuple[ImageNet9Dataset, ImageNet9Dataset]:
    train_samples = load_original_samples(manifest_path, "train", verify_files)
    val_samples = load_original_samples(manifest_path, "val", verify_files)
    return (
        ImageNet9Dataset(train_samples, build_train_transform(image_size)),
        ImageNet9Dataset(val_samples, build_eval_transform(image_size)),
    )


def build_original_dataloaders(
    manifest_path: Path,
    batch_size: int = 96,
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = build_original_datasets(manifest_path, image_size)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def build_official_variant_dataset(
    official_manifest_path: Path,
    official_test_root: Path,
    variant: str,
    image_size: int = 224,
    verify_files: bool = True,
) -> ImageNet9Dataset:
    samples = load_official_variant_samples(
        official_manifest_path,
        official_test_root,
        variant,
        verify_files,
    )
    return ImageNet9Dataset(samples, build_eval_transform(image_size))


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    require_all_classes: bool = True,
) -> Dict[str, object]:
    predictions = torch.as_tensor(predictions).detach().cpu().long().reshape(-1)
    targets = torch.as_tensor(targets).detach().cpu().long().reshape(-1)
    if predictions.numel() != targets.numel() or targets.numel() == 0:
        raise ValueError("Predictions and targets must be non-empty and have equal length")
    if torch.any((targets < 0) | (targets >= num_classes)):
        raise ValueError("Targets contain labels outside the configured class range")

    per_class: List[float] = []
    support: List[int] = []
    for label in range(num_classes):
        mask = targets == label
        count = int(mask.sum().item())
        support.append(count)
        if count == 0:
            if require_all_classes:
                raise RuntimeError(f"Metric input is missing class {label}")
            per_class.append(float("nan"))
            continue
        per_class.append(float((predictions[mask] == targets[mask]).float().mean().item()))

    valid = [value for value in per_class if value == value]
    return {
        "accuracy": float((predictions == targets).float().mean().item()),
        "macro_class_accuracy": sum(valid) / len(valid),
        "per_class_accuracy": per_class,
        "class_support": support,
    }


class ClassificationMeter:
    """Accumulates logits/predictions and computes the fixed validation objective."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes
        self._predictions: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        outputs = torch.as_tensor(outputs).detach()
        if outputs.ndim == 2:
            predictions = outputs.argmax(dim=1)
        elif outputs.ndim == 1:
            predictions = outputs
        else:
            raise ValueError(f"Expected logits [N,C] or predictions [N], got {outputs.shape}")
        self._predictions.append(predictions.cpu().long())
        self._targets.append(torch.as_tensor(targets).detach().cpu().long().reshape(-1))

    def compute(self) -> Dict[str, object]:
        if not self._predictions:
            raise RuntimeError("No batches were added to ClassificationMeter")
        return classification_metrics(
            torch.cat(self._predictions),
            torch.cat(self._targets),
            num_classes=self.num_classes,
        )

    def objective(self) -> float:
        return float(self.compute()["macro_class_accuracy"])

