"""Utilities for the unmodified-DecoyMNIST ViT susceptibility pilot.

The original DecoyMNIST construction writes a class-coded 5x5 grayscale block
in one of four corners.  Training blocks have intensity ``255 - 25*y`` and
test blocks have intensity ``25*y``.  This module detects that block from the
published construction itself; it never changes the source PNGs on disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


MODEL_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"
LEARNING_RATES = (1.0e-5, 3.0e-5, 1.0e-4)
WEIGHT_DECAY = 0.05
SEEDS = (0, 1, 2)
PATCH_SIZE = 5
IMAGE_SIZE = 224
NUM_CLASSES = 10


@dataclass(frozen=True)
class PilotRun:
    run_index: int
    learning_rate: float
    seed: int

    @property
    def run_id(self) -> str:
        lr = f"{self.learning_rate:.8g}".replace("-", "m").replace(".", "p")
        return f"run_{self.run_index:02d}_lr_{lr}_wd_0p05_seed_{self.seed}"


def enumerate_runs() -> List[PilotRun]:
    runs: List[PilotRun] = []
    for learning_rate in LEARNING_RATES:
        for seed in SEEDS:
            runs.append(PilotRun(len(runs), learning_rate, seed))
    return runs


def get_run(run_index: int) -> PilotRun:
    runs = enumerate_runs()
    if run_index < 0 or run_index >= len(runs):
        raise IndexError(f"run_index must be in [0, {len(runs) - 1}]")
    return runs[run_index]


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def expected_patch_intensity(label: int, split: str) -> int:
    if not 0 <= int(label) < NUM_CLASSES:
        raise ValueError(f"Invalid DecoyMNIST label: {label}")
    if split == "train":
        return 255 - 25 * int(label)
    if split == "test":
        return 25 * int(label)
    raise ValueError(f"split must be 'train' or 'test', found {split!r}")


def _corner_slices(height: int, width: int) -> List[Tuple[slice, slice]]:
    p = PATCH_SIZE
    if height < p or width < p:
        raise ValueError(f"Image is too small for a {p}x{p} corner patch")
    return [
        (slice(0, p), slice(0, p)),
        (slice(0, p), slice(width - p, width)),
        (slice(height - p, height), slice(0, p)),
        (slice(height - p, height), slice(width - p, width)),
    ]


def locate_decoy_patch(
    grayscale: np.ndarray, label: int, split: str, *, tolerance: float = 1.0
) -> Tuple[slice, slice]:
    """Locate and validate the class-coded corner block.

    Test-class zero has a zero-valued patch indistinguishable from the black
    background.  Any tied zero corner is equivalent for erasure/patch-only
    diagnostics, so deterministic first-tie selection is intentional.
    """

    if grayscale.ndim != 2:
        raise ValueError(f"Expected a grayscale HxW image, got {grayscale.shape}")
    expected = float(expected_patch_intensity(label, split))
    corners = _corner_slices(*grayscale.shape)
    errors = [
        float(np.abs(grayscale[rows, cols].astype(np.float32) - expected).mean())
        for rows, cols in corners
    ]
    selected = int(np.argmin(np.asarray(errors)))
    rows, cols = corners[selected]
    max_error = float(
        np.abs(grayscale[rows, cols].astype(np.float32) - expected).max()
    )
    if max_error > tolerance:
        raise ValueError(
            "Image does not match the unmodified DecoyMNIST encoding: "
            f"label={label} split={split} expected={expected:.1f} "
            f"best_mean_error={errors[selected]:.3f} max_error={max_error:.3f}"
        )
    return rows, cols


def diagnostic_views(
    grayscale: np.ndarray, label: int, split: str
) -> Dict[str, np.ndarray]:
    """Return original, digit-only (patch erased), and patch-only views."""

    rows, cols = locate_decoy_patch(grayscale, label, split)
    original = np.asarray(grayscale, dtype=np.uint8).copy()
    digit_only = original.copy()
    digit_only[rows, cols] = 0
    patch_only = np.zeros_like(original)
    patch_only[rows, cols] = original[rows, cols]
    return {
        "original": original,
        "digit_only": digit_only,
        "patch_only": patch_only,
    }


def discover_samples(root: str | Path, split: str) -> Dict[int, List[Path]]:
    split_root = Path(root).expanduser().resolve() / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing DecoyMNIST split: {split_root}")
    by_label: Dict[int, List[Path]] = {}
    for label in range(NUM_CLASSES):
        class_root = split_root / str(label)
        paths = sorted(class_root.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No PNGs found under {class_root}")
        by_label[label] = paths
    return by_label


def stratified_train_holdout(
    by_label: Mapping[int, Sequence[Path]],
    *,
    validation_fraction: float = 0.10,
    split_seed: int = 0,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    # MNIST's ten classes are not exactly equal-sized.  Rounding each class
    # independently yields 5,999 rather than 6,000 holdout examples.  Use
    # deterministic largest-remainder apportionment so the split is both
    # class-stratified and exactly the requested global fraction.
    class_sizes = {label: len(by_label[label]) for label in range(NUM_CLASSES)}
    total_size = sum(class_sizes.values())
    target_validation_size = int(round(total_size * validation_fraction))
    exact_quotas = {
        label: class_sizes[label] * validation_fraction
        for label in range(NUM_CLASSES)
    }
    validation_counts = {
        label: int(math.floor(exact_quotas[label])) for label in range(NUM_CLASSES)
    }
    remaining = target_validation_size - sum(validation_counts.values())
    remainder_order = sorted(
        range(NUM_CLASSES),
        key=lambda label: (-(exact_quotas[label] - validation_counts[label]), label),
    )
    for label in remainder_order[:remaining]:
        validation_counts[label] += 1

    rng = np.random.default_rng(split_seed)
    train: List[Tuple[Path, int]] = []
    validation: List[Tuple[Path, int]] = []
    for label in range(NUM_CLASSES):
        paths = list(by_label[label])
        order = rng.permutation(len(paths))
        n_val = validation_counts[label]
        val_indices = set(int(index) for index in order[:n_val])
        for index, path in enumerate(paths):
            target = validation if index in val_indices else train
            target.append((path, label))
    train.sort(key=lambda item: (item[1], item[0].as_posix()))
    validation.sort(key=lambda item: (item[1], item[0].as_posix()))
    return train, validation


def flatten_samples(by_label: Mapping[int, Sequence[Path]]) -> List[Tuple[Path, int]]:
    return [
        (path, label)
        for label in range(NUM_CLASSES)
        for path in by_label[label]
    ]


def split_fingerprint(
    train: Sequence[Tuple[Path, int]], validation: Sequence[Tuple[Path, int]]
) -> str:
    payload = {
        "train": [(path.as_posix(), int(label)) for path, label in train],
        "validation": [
            (path.as_posix(), int(label)) for path, label in validation
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_transform() -> transforms.Compose:
    # No crop or flip: this pilot asks whether the unmodified benchmark shortcut
    # is learnable by ViT, so augmentation must not intermittently remove it.
    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def _rgb_pil(grayscale: np.ndarray) -> Image.Image:
    return Image.fromarray(grayscale, mode="L").convert("RGB")


class OriginalDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]], transform) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        return self.transform(_rgb_pil(grayscale)), int(label), path.name


class DiagnosticDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        split: str,
        transform,
    ) -> None:
        self.samples = list(samples)
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        views = diagnostic_views(grayscale, int(label), self.split)
        return (
            self.transform(_rgb_pil(views["original"])),
            self.transform(_rgb_pil(views["digit_only"])),
            self.transform(_rgb_pil(views["patch_only"])),
            int(label),
            path.name,
        )


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def verify_encoding(
    by_label: Mapping[int, Sequence[Path]], split: str, *, per_class: int = 100
) -> None:
    for label in range(NUM_CLASSES):
        paths = list(by_label[label])
        if len(paths) < per_class:
            raise ValueError(
                f"Need {per_class} {split} examples for class {label}, found {len(paths)}"
            )
        for path in paths[:per_class]:
            with Image.open(path) as image:
                grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
            locate_decoy_patch(grayscale, label, split)


def build_model(*, pretrained: bool = True) -> nn.Module:
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - exercised on Tigris
        raise RuntimeError("Install timm==1.0.28 in fcv_gh200") from exc
    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=NUM_CLASSES)
    patch_size = tuple(int(value) for value in model.patch_embed.patch_size)
    if patch_size != (16, 16) or int(model.patch_embed.num_patches) != 196:
        raise RuntimeError(
            f"Unexpected ViT geometry: patch_size={patch_size} "
            f"num_patches={model.patch_embed.num_patches}"
        )
    return model


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def warmup_cosine_factor(
    step: int, *, total_steps: int, warmup_steps: int
) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    cosine_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / cosine_steps, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels, _names in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        n = int(labels.numel())
        total_loss += float(loss.detach()) * n
        total_correct += int((logits.argmax(1) == labels).sum().item())
        total += n
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def evaluate_diagnostics(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, object]:
    names = ("original", "digit_only", "patch_only")
    correct = {name: 0 for name in names}
    class_correct = {name: np.zeros(NUM_CLASSES, dtype=np.int64) for name in names}
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    total = 0
    model.eval()
    with torch.inference_mode():
        for original, digit_only, patch_only, labels, _sample_names in loader:
            labels = labels.to(device, non_blocking=True)
            views = torch.cat((original, digit_only, patch_only), dim=0).to(
                device, non_blocking=True
            )
            with autocast_context(device):
                logits = model(views)
            chunks = logits.chunk(3, dim=0)
            labels_cpu = labels.cpu().numpy()
            class_total += np.bincount(labels_cpu, minlength=NUM_CLASSES)
            for name, chunk in zip(names, chunks):
                predictions = chunk.argmax(1)
                matches = predictions.eq(labels)
                correct[name] += int(matches.sum().item())
                matched_labels = labels[matches].cpu().numpy()
                class_correct[name] += np.bincount(
                    matched_labels, minlength=NUM_CLASSES
                )
            total += int(labels.numel())
    result: Dict[str, object] = {"count": total}
    for name in names:
        class_accuracy = np.divide(
            class_correct[name],
            class_total,
            out=np.zeros(NUM_CLASSES, dtype=np.float64),
            where=class_total > 0,
        )
        result[f"{name}_accuracy"] = correct[name] / total
        result[f"{name}_balanced_class_accuracy"] = float(class_accuracy.mean())
        result[f"{name}_worst_class_accuracy"] = float(class_accuracy.min())
        result[f"{name}_class_accuracies"] = class_accuracy.tolist()
    return result


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
