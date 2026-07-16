#!/usr/bin/env python3
import argparse
import os
from os.path import join as oj

from typing import Optional, Tuple

import numpy as np
from PIL import Image


def _to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    if xmin < 0.0:
        # Assume [-1, 1] range.
        x = (x + 1.0) / 2.0
        x = np.clip(x, 0.0, 1.0) * 255.0
    elif xmax <= 1.0:
        # Assume [0, 1] range.
        x = np.clip(x, 0.0, 1.0) * 255.0
    else:
        # Assume [0, 255] (or similar) range.
        x = np.clip(x, 0.0, 255.0)
    return x.round().astype(np.uint8)


def save_images(
    x: np.ndarray,
    out_dir: str,
    labels: Optional[np.ndarray],
    start: int,
    step: int,
    max_images: int,
):
    os.makedirs(out_dir, exist_ok=True)

    if x.ndim == 2:
        x = x[None, None, :, :]
    elif x.ndim == 3:
        x = x[:, None, :, :]
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported array shape: {x.shape}")

    n, c, h, w = x.shape
    if c not in (1, 3):
        raise ValueError(f"Unsupported channel count: {c}")

    idxs = list(range(start, n, step))[:max_images]
    for i in idxs:
        img = _to_uint8(x[i])
        if c == 1:
            img = img[0]
            pil = Image.fromarray(img, mode="L")
        else:
            img = np.transpose(img, (1, 2, 0))
            pil = Image.fromarray(img, mode="RGB")

        if labels is not None:
            name = f"{i:06d}_y{int(labels[i])}.png"
        else:
            name = f"{i:06d}.png"
        pil.save(oj(out_dir, name))


def save_per_class(
    x: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
    per_class: int,
    start: int,
    step: int,
):
    if labels is None:
        raise ValueError("labels are required for per-class export")
    labels = labels.astype(int)
    num_classes = int(labels.max()) + 1
    for cls in range(num_classes):
        idxs = np.where(labels == cls)[0]
        idxs = idxs[start::step]
        idxs = idxs[:per_class]
        cls_dir = oj(out_dir, str(cls))
        os.makedirs(cls_dir, exist_ok=True)
        for i in idxs:
            img = _to_uint8(x[i])
            if img.ndim == 3 and img.shape[0] in (1, 3):
                if img.shape[0] == 1:
                    pil = Image.fromarray(img[0], mode="L")
                else:
                    pil = Image.fromarray(np.transpose(img, (1, 2, 0)), mode="RGB")
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
            name = f"{i:06d}_y{int(labels[i])}.png"
            pil.save(oj(cls_dir, name))


def _load_npy_pair(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y


def _find_decoy_test(base_dir: str) -> str:
    # Prefer the repo-root data path; fall back to the older nested path.
    candidate = oj(base_dir, "test_x_decoy.npy")
    if os.path.exists(candidate):
        return candidate
    fallback = oj("mnist", "data", "ColorMNIST", "test_x_decoy.npy")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError("Could not find test_x_decoy.npy in data/ColorMNIST or mnist/data/ColorMNIST.")

def _load_mnist_labels(train: bool, root: str) -> np.ndarray:
    try:
        from torchvision import datasets
    except Exception as exc:
        raise RuntimeError("torchvision is required to load MNIST labels for decoy export") from exc
    ds = datasets.MNIST(root=root, train=train, download=False, transform=None)
    # torchvision uses targets
    return np.asarray(ds.targets)

def _safe_labels_for_decoy(x: np.ndarray, y_path: str, mnist_root: str, train: bool) -> np.ndarray:
    if os.path.exists(y_path):
        y = np.load(y_path)
        if len(y) == len(x):
            return y
    # Fallback: load original MNIST labels to match full 60k/10k arrays.
    return _load_mnist_labels(train=train, root=mnist_root)

def main():
    parser = argparse.ArgumentParser(description="Convert MNIST .npy arrays to PNG images.")
    parser.add_argument("--npy", help="Path to .npy file (shape N,C,H,W).")
    parser.add_argument("--out", required=True, help="Output directory for PNGs.")
    parser.add_argument("--labels", default=None, help="Optional labels .npy for filename suffix.")
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--step", type=int, default=1, help="Stride between saved images.")
    parser.add_argument("--max-images", type=int, default=100, help="Maximum number of images to save.")
    parser.add_argument("--per-class", type=int, default=0, help="If set, save this many images per class into subfolders.")
    parser.add_argument(
        "--make-dataset",
        choices=["colormnist", "decoymnist"],
        default=None,
        help="Generate per-class previews for train/test splits.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/ColorMNIST",
        help="Base directory for ColorMNIST .npy files (default: data/ColorMNIST).",
    )
    parser.add_argument(
        "--mnist-root",
        default="data",
        help="Root directory that contains the MNIST dataset (default: data).",
    )
    args = parser.parse_args()

    if args.make_dataset:
        os.makedirs(args.out, exist_ok=True)
        if args.make_dataset == "colormnist":
            train_x, train_y = _load_npy_pair(oj(args.data_dir, "train_x.npy"), oj(args.data_dir, "train_y.npy"))
            test_x, test_y = _load_npy_pair(oj(args.data_dir, "test_x.npy"), oj(args.data_dir, "test_y.npy"))
        else:
            train_x = np.load(oj(args.data_dir, "train_x_decoy.npy"))
            test_x = np.load(_find_decoy_test(args.data_dir))
            train_y = _safe_labels_for_decoy(train_x, oj(args.data_dir, "train_y.npy"), args.mnist_root, train=True)
            test_y = _safe_labels_for_decoy(test_x, oj(args.data_dir, "test_y.npy"), args.mnist_root, train=False)

        if args.per_class <= 0:
            raise ValueError("--per-class must be > 0 when using --make-dataset")

        save_per_class(train_x, train_y, oj(args.out, "train"), args.per_class, args.start, args.step)
        save_per_class(test_x, test_y, oj(args.out, "test"), args.per_class, args.start, args.step)
        return

    if not args.npy:
        raise ValueError("--npy is required unless --make-dataset is provided")

    x = np.load(args.npy)
    labels = np.load(args.labels) if args.labels else None
    if args.per_class > 0:
        if labels is None:
            raise ValueError("--per-class requires --labels")
        save_per_class(x, labels, args.out, args.per_class, args.start, args.step)
    else:
        save_images(x, args.out, labels, args.start, args.step, args.max_images)


if __name__ == "__main__":
    main()
