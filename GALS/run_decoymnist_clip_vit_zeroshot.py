#!/usr/bin/env python3
"""CLIP ViT zero-shot evaluation on DecoyMNIST."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _add_repo_to_syspath() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _try_import_clip():
    try:
        import clip  # type: ignore

        return clip
    except Exception:
        _add_repo_to_syspath()
        from CLIP.clip import clip  # type: ignore

        return clip


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _class_acc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.zeros((num_classes,), dtype=np.float64)
    for c in range(num_classes):
        idx = np.where(y_true == c)[0]
        if idx.size == 0:
            acc[c] = float("nan")
        else:
            acc[c] = float(np.mean((y_pred[idx] == y_true[idx]).astype(np.float64)) * 100.0)
    return acc


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def _nanmin(x: np.ndarray) -> float:
    return float(np.nanmin(x))


def _fmt_arr(arr: np.ndarray) -> str:
    return np.array2string(arr, precision=2, separator=",")


def _write_rows(csv_path: str, rows: Iterable[Dict], header: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class _ClipSubset(Dataset):
    def __init__(self, subset: Subset, preprocess):
        self.subset = subset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.preprocess(img), int(label)


class _ClipFolder(Dataset):
    def __init__(self, folder: ImageFolder, preprocess):
        self.folder = folder
        self.preprocess = preprocess

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        img, label = self.folder[idx]
        return self.preprocess(img), int(label)


def _extract_features(ds: Dataset, model, device: str, batch_size: int, num_workers: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
        worker_init_fn=_seed_worker,
        generator=g,
    )
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    X = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(labels, axis=0).astype(np.int64, copy=False)
    return _l2_normalize(X), y


def _build_text_features(clip_module, model, device: str, class_names: Sequence[str], templates: Sequence[str]) -> np.ndarray:
    text_features: List[torch.Tensor] = []
    with torch.no_grad():
        for cls in class_names:
            prompts = [t.format(cls) for t in templates]
            toks = clip_module.tokenize(prompts).to(device)
            feats = model.encode_text(toks).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            cls_feat = feats.mean(dim=0)
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            text_features.append(cls_feat)
    W = torch.stack(text_features, dim=0).cpu().numpy().astype(np.float32)
    return _l2_normalize(W)


def _evaluate(X: np.ndarray, y: np.ndarray, text_features: np.ndarray, num_classes: int) -> Dict[str, object]:
    logits = X @ text_features.T
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean((pred == y).astype(np.float64)) * 100.0)
    class_acc = _class_acc(y, pred, num_classes)
    return {
        "acc": acc,
        "balanced_class_acc": _nanmean(class_acc),
        "worst_class_acc": _nanmin(class_acc),
        "class_accs": _fmt_arr(class_acc),
    }


def _digit_name(c: str) -> str:
    words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    return words.get(c, c)


def _prepare_splits(png_root: str, val_frac: float, split_seed: int):
    train_dir = os.path.join(png_root, "train")
    test_dir = os.path.join(png_root, "test")
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Missing train dir: {train_dir}")
    if not os.path.isdir(test_dir):
        raise RuntimeError(f"Missing test dir: {test_dir}")

    full_train = ImageFolder(train_dir)
    test_ds = ImageFolder(test_dir)
    if full_train.classes != test_ds.classes:
        raise RuntimeError(f"Class mismatch train={full_train.classes} test={test_ds.classes}")

    n_total = len(full_train)
    n_val = int(val_frac * n_total)
    n_train = n_total - n_val
    split_g = torch.Generator().manual_seed(split_seed)
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=split_g)
    return full_train.classes, train_subset, val_subset, test_ds


def _default_templates() -> List[str]:
    return [
        "a handwritten digit {}.",
        "a photo of the digit {}.",
        "an image of the number {}.",
        "a grayscale image of {}.",
        "a centered handwritten {}.",
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DecoyMNIST CLIP ViT zero-shot eval.")
    p.add_argument("--png-root", default="/workspace/Waterbird_Runs/MakeMNIST/data/DecoyMNIST_png")
    p.add_argument("--clip-model", default="ViT-B/32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--splits", default="val,test")
    p.add_argument("--output-csv", default="decoy_clip_vit_zeroshot.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clip = _try_import_clip()

    classes, train_subset, val_subset, test_ds = _prepare_splits(
        png_root=args.png_root,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
    )
    num_classes = len(classes)

    try:
        model, preprocess = clip.load(args.clip_model, device=args.device, jit=False)
    except TypeError:
        model, preprocess = clip.load(args.clip_model, device=args.device)

    prompts = [_digit_name(c) for c in classes]
    text_templates = _default_templates()
    text_features = _build_text_features(clip, model, args.device, prompts, text_templates)

    val_clip = _ClipSubset(val_subset, preprocess)
    test_clip = _ClipFolder(test_ds, preprocess)

    X_val, y_val = _extract_features(val_clip, model, args.device, args.batch_size, args.num_workers, args.split_seed)
    X_test, y_test = _extract_features(test_clip, model, args.device, args.batch_size, args.num_workers, args.split_seed)

    split_map = {
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("Running DecoyMNIST CLIP ViT zero-shot")
    print(f"device={args.device}")
    print(f"png_root={args.png_root}")
    print(f"clip_model={args.clip_model}")
    print(
        f"train={len(train_subset)} val={len(val_subset)} test={len(test_ds)} "
        f"split={1.0 - args.val_frac:.2f}/{args.val_frac:.2f} split_seed={args.split_seed}"
    )
    print(f"splits={splits} seeds={seeds}")

    rows: List[Dict] = []
    for seed in seeds:
        _seed_everything(seed)
        row = {
            "seed": seed,
            "clip_model": args.clip_model,
        }
        for split in splits:
            if split not in split_map:
                raise ValueError(f"Unsupported split '{split}'. Choose from val,test.")
            X, y = split_map[split]
            m = _evaluate(X, y, text_features, num_classes)
            row[f"{split}_acc"] = m["acc"]
            row[f"{split}_balanced_class_acc"] = m["balanced_class_acc"]
            row[f"{split}_worst_class_acc"] = m["worst_class_acc"]
            row[f"{split}_class_accs"] = m["class_accs"]
        rows.append(row)
        msg = [f"seed={seed}"]
        for split in splits:
            msg.append(
                f"{split}_acc={row[f'{split}_acc']:.2f}% {split}_worst={row[f'{split}_worst_class_acc']:.2f}%"
            )
        print(" ".join(msg))

    header = ["seed", "clip_model"]
    for split in splits:
        header.extend(
            [
                f"{split}_acc",
                f"{split}_balanced_class_acc",
                f"{split}_worst_class_acc",
                f"{split}_class_accs",
            ]
        )
    _write_rows(args.output_csv, rows, header)

    print("\nSummary over seeds")
    for split in splits:
        vals = np.array([float(r[f"{split}_acc"]) for r in rows], dtype=float)
        worst = np.array([float(r[f"{split}_worst_class_acc"]) for r in rows], dtype=float)
        print(f"{split}_acc mean={np.mean(vals):.2f}% std={np.std(vals):.2f}%")
        print(f"{split}_worst_class_acc mean={np.mean(worst):.2f}% std={np.std(worst):.2f}%")
    print(f"wrote {args.output_csv}")
    print("[ZERO-SHOT] With fixed CLIP model+prompts, per-seed rows are expected to match.")


if __name__ == "__main__":
    main()
