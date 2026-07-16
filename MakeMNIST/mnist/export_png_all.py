#!/usr/bin/env python3
import argparse
import os
from os.path import join as oj

import numpy as np
from PIL import Image
from tqdm import tqdm


def _to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    if xmin < 0.0:
        x = (x + 1.0) / 2.0
        x = np.clip(x, 0.0, 1.0) * 255.0
    elif xmax <= 1.0:
        x = np.clip(x, 0.0, 1.0) * 255.0
    else:
        x = np.clip(x, 0.0, 255.0)
    return x.round().astype(np.uint8)


def _save_split(x: np.ndarray, y: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError(f"Unsupported shape: {x.shape}")
    n, c, _, _ = x.shape
    for i in tqdm(range(n), desc=f"Saving {out_dir}"):
        cls = int(y[i])
        cls_dir = oj(out_dir, str(cls))
        os.makedirs(cls_dir, exist_ok=True)
        img = _to_uint8(x[i])
        if c == 1:
            pil = Image.fromarray(img[0], mode="L")
        elif c == 3:
            pil = Image.fromarray(np.transpose(img, (1, 2, 0)), mode="RGB")
        else:
            raise ValueError(f"Unsupported channel count: {c}")
        pil.save(oj(cls_dir, f"{i:06d}_y{cls}.png"))


def main():
    parser = argparse.ArgumentParser(description="Export full ColorMNIST/DecoyMNIST splits to PNGs.")
    parser.add_argument(
        "--dataset",
        choices=["colormnist", "decoymnist", "both"],
        default="both",
        help="Which dataset to export.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/ColorMNIST",
        help="Directory containing .npy files (default: data/ColorMNIST).",
    )
    parser.add_argument(
        "--out-root",
        default="data",
        help="Root output directory (default: data).",
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    data_dir = os.path.join(repo_root, args.data_dir)
    out_root = os.path.join(repo_root, args.out_root)

    if args.dataset in ("colormnist", "both"):
        out = oj(out_root, "ColorMNIST_png")
        train_x = np.load(oj(data_dir, "train_x.npy"))
        train_y = np.load(oj(data_dir, "train_y.npy"))
        val_x = np.load(oj(data_dir, "val_x.npy"))
        val_y = np.load(oj(data_dir, "val_y.npy"))
        test_x = np.load(oj(data_dir, "test_x.npy"))
        test_y = np.load(oj(data_dir, "test_y.npy"))
        _save_split(train_x, train_y, oj(out, "train"))
        _save_split(val_x, val_y, oj(out, "val"))
        _save_split(test_x, test_y, oj(out, "test"))

    if args.dataset in ("decoymnist", "both"):
        out = oj(out_root, "DecoyMNIST_png")
        train_x = np.load(oj(data_dir, "train_x_decoy.npy"))
        # Use full MNIST labels (60k/10k) if available, else fall back to ColorMNIST labels.
        train_y_path = oj(data_dir, "train_y.npy")
        test_y_path = oj(data_dir, "test_y.npy")
        if os.path.exists(train_y_path) and len(np.load(train_y_path)) == len(train_x):
            train_y = np.load(train_y_path)
        else:
            from torchvision import datasets
            mnist = datasets.MNIST(root=oj(repo_root, "data"), train=True, download=False, transform=None)
            train_y = np.asarray(mnist.targets)
        test_x = np.load(oj(data_dir, "test_x_decoy.npy"))
        if os.path.exists(test_y_path) and len(np.load(test_y_path)) == len(test_x):
            test_y = np.load(test_y_path)
        else:
            from torchvision import datasets
            mnist = datasets.MNIST(root=oj(repo_root, "data"), train=False, download=False, transform=None)
            test_y = np.asarray(mnist.targets)
        _save_split(train_x, train_y, oj(out, "train"))
        _save_split(test_x, test_y, oj(out, "test"))


if __name__ == "__main__":
    main()
