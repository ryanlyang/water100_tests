#!/usr/bin/env python3
"""Zero-shot CLIP evaluation on DecoyMNIST test split.

This script runs CLIP zero-shot classification on DecoyMNIST PNG folders
(e.g., data/DecoyMNIST_png/test) and reports:
- test accuracy
- balanced class accuracy
- worst class accuracy
- per-class accuracies

It can emit one row per seed for consistency with other experiment tables.
Because this is pure inference, rows are usually identical across seeds.
"""

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
from torchvision.datasets import ImageFolder


DIGIT_WORDS = {
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


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for s in _parse_csv_list(text):
        out.append(int(s))
    return out


def _default_templates() -> List[str]:
    return [
        "a handwritten digit {}.",
        "a photo of the handwritten digit {}.",
        "the number {}.",
    ]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _fmt_arr(arr: np.ndarray) -> str:
    return np.array2string(arr, precision=4, separator=",")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _try_import_clip(clip_repo: str = ""):
    try:
        import clip  # type: ignore

        return clip
    except Exception:
        pass

    candidates: List[Path] = []
    if clip_repo:
        candidates.append(Path(clip_repo).expanduser().resolve())

    repro_root = _repo_root()
    candidates.extend(
        [
            repro_root / "third_party" / "GALS",
            repro_root.parent / "GALS",
            Path.cwd() / "GALS",
            Path.cwd().parent / "GALS",
        ]
    )

    seen = set()
    for c in candidates:
        if not c:
            continue
        c = c.resolve()
        if str(c) in seen:
            continue
        seen.add(str(c))
        if (c / "CLIP" / "clip" / "clip.py").exists():
            sys.path.insert(0, str(c))
            from CLIP.clip import clip  # type: ignore

            return clip

    raise ImportError(
        "Could not import CLIP. Install `clip` in your env, or pass --clip-repo "
        "pointing to a repo root that contains CLIP/clip/clip.py"
    )


def _label_variants(label: str) -> List[str]:
    label = str(label)
    variants = [label.replace("_", " ")]
    if label in DIGIT_WORDS:
        variants.append(DIGIT_WORDS[label])
    return list(dict.fromkeys(variants))


def _build_text_features(
    clip_module,
    model,
    device: str,
    class_names: Sequence[str],
    templates: Sequence[str],
) -> np.ndarray:
    text_features: List[torch.Tensor] = []
    with torch.no_grad():
        for cls in class_names:
            prompts: List[str] = []
            for v in _label_variants(cls):
                for t in templates:
                    prompts.append(t.format(v))
            tokens = clip_module.tokenize(prompts).to(device)
            feats = model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            cls_feat = feats.mean(dim=0)
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            text_features.append(cls_feat)

    W = torch.stack(text_features, dim=0).cpu().numpy().astype(np.float32)
    return _l2_normalize(W)


def _extract_image_features(
    dataset: ImageFolder,
    model,
    device: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
    )

    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            f = model.encode_image(images).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())

    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    return X, y


def _class_acc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.zeros((num_classes,), dtype=np.float64)
    for c in range(num_classes):
        idx = np.where(y_true == c)[0]
        if idx.size == 0:
            acc[c] = float("nan")
            continue
        acc[c] = float(np.mean((y_pred[idx] == y_true[idx]).astype(np.float64)) * 100.0)
    return acc


def _evaluate(X: np.ndarray, y: np.ndarray, text_features: np.ndarray, num_classes: int) -> Dict[str, object]:
    logits = X @ text_features.T
    y_pred = np.argmax(logits, axis=1)

    class_accs = _class_acc(y, y_pred, num_classes=num_classes)
    return {
        "n": int(y.shape[0]),
        "test_acc": float(np.mean((y_pred == y).astype(np.float64)) * 100.0),
        "test_balanced_class_acc": float(np.nanmean(class_accs)),
        "test_worst_class_acc": float(np.nanmin(class_accs)),
        "test_class_accs": _fmt_arr(class_accs),
    }


def _write_rows(csv_path: str, rows: Iterable[Dict[str, object]], header: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-shot CLIP on DecoyMNIST test split")
    p.add_argument("--png-root", type=str, default=None, help="Path to DecoyMNIST_png")
    p.add_argument("--split", type=str, default="test", help="Split folder under --png-root")
    p.add_argument("--clip-model", type=str, default="RN50", help="CLIP model name, e.g. RN50, ViT-B/32")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--templates", type=str, default="", help="Comma-separated prompt templates")
    p.add_argument("--clip-repo", type=str, default="", help="Optional repo root containing CLIP/clip/clip.py")
    p.add_argument("--output-csv", type=str, default="")
    args = p.parse_args()

    repo_root = _repo_root()
    default_png_root = repo_root / "third_party" / "CDEP" / "data" / "DecoyMNIST_png"
    png_root = Path(args.png_root).expanduser().resolve() if args.png_root else default_png_root.resolve()
    split_dir = png_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds is empty")

    templates = _parse_csv_list(args.templates) if args.templates else _default_templates()

    clip_module = _try_import_clip(args.clip_repo)

    print("[INFO] Zero-shot CLIP DecoyMNIST")
    print(f"[INFO] png_root={png_root}")
    print(f"[INFO] split={args.split}")
    print(f"[INFO] clip_model={args.clip_model}")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] seeds={seeds}")

    model, preprocess = clip_module.load(args.clip_model, device=args.device, jit=False)

    dataset = ImageFolder(str(split_dir), transform=preprocess)
    class_names = list(dataset.classes)
    num_classes = len(class_names)

    print(f"[INFO] n_test={len(dataset)} classes={class_names}")

    # Deterministic inference; extract once, then report per seed for table consistency.
    _seed_everything(seeds[0])
    X, y = _extract_image_features(
        dataset=dataset,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    text_features = _build_text_features(
        clip_module=clip_module,
        model=model,
        device=args.device,
        class_names=class_names,
        templates=templates,
    )

    rows: List[Dict[str, object]] = []
    for s in seeds:
        _seed_everything(s)
        metrics = _evaluate(X, y, text_features, num_classes=num_classes)
        row = {
            "seed": s,
            "clip_model": args.clip_model,
            "split": args.split,
            "n": metrics["n"],
            "test_acc": metrics["test_acc"],
            "test_balanced_class_acc": metrics["test_balanced_class_acc"],
            "test_worst_class_acc": metrics["test_worst_class_acc"],
            "test_class_accs": metrics["test_class_accs"],
        }
        rows.append(row)
        print(
            f"[SEED {s}] test_acc={row['test_acc']:.4f} "
            f"balanced={row['test_balanced_class_acc']:.4f} worst={row['test_worst_class_acc']:.4f}"
        )

    accs = np.asarray([float(r["test_acc"]) for r in rows], dtype=np.float64)
    baccs = np.asarray([float(r["test_balanced_class_acc"]) for r in rows], dtype=np.float64)
    waccs = np.asarray([float(r["test_worst_class_acc"]) for r in rows], dtype=np.float64)

    print("\n[SUMMARY] mean +/- std across seeds")
    print(f"test_acc:                {accs.mean():.4f} +/- {accs.std():.4f}")
    print(f"test_balanced_class_acc: {baccs.mean():.4f} +/- {baccs.std():.4f}")
    print(f"test_worst_class_acc:    {waccs.mean():.4f} +/- {waccs.std():.4f}")

    if args.output_csv:
        _write_rows(
            args.output_csv,
            rows,
            header=[
                "seed",
                "clip_model",
                "split",
                "n",
                "test_acc",
                "test_balanced_class_acc",
                "test_worst_class_acc",
                "test_class_accs",
            ],
        )
        print(f"[INFO] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
