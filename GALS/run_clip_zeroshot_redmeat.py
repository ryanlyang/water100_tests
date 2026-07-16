#!/usr/bin/env python3
"""Zero-shot CLIP evaluation on RedMeat (Food101 subset) with per-seed reporting.

No training is performed. This script evaluates CLIP zero-shot prompts on selected
splits and writes one row per seed for easy consistency with other result tables.
"""

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image


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
            continue
        acc[c] = float(np.mean((y_pred[idx] == y_true[idx]).astype(np.float64)) * 100.0)
    return acc


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def _nanmin(x: np.ndarray) -> float:
    return float(np.nanmin(x))


def _fmt_arr(arr: np.ndarray) -> str:
    return np.array2string(arr, precision=4, separator=",")


def _resolve_image_path(dataset_root: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    rel = str(rel_or_abs).lstrip("/")
    return os.path.join(dataset_root, rel)


@dataclass(frozen=True)
class _Sample:
    path: str
    label: int


class _ImageDataset:
    def __init__(self, samples: List[_Sample], preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.preprocess(img)
        return x, s.label


def _build_samples(
    dataset_path: str,
    split_col: str,
    label_col: str,
    path_col: str,
    split_value: str,
    class_to_idx: Dict[str, int],
) -> List[_Sample]:
    meta = os.path.join(dataset_path, "all_images.csv")
    if not os.path.exists(meta):
        raise FileNotFoundError(f"Missing all_images.csv at: {meta}")

    df = pd.read_csv(meta)
    for col in (split_col, label_col, path_col):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in {meta}. Columns={list(df.columns)}")

    d = df[df[split_col].astype(str) == str(split_value)]
    out: List[_Sample] = []
    for _, r in d.iterrows():
        label_name = str(r[label_col])
        if label_name not in class_to_idx:
            continue
        p = _resolve_image_path(dataset_path, str(r[path_col]))
        out.append(_Sample(path=p, label=class_to_idx[label_name]))
    return out


def _extract_features(
    samples: List[_Sample],
    model,
    preprocess,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader

    ds = _ImageDataset(samples, preprocess)
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

    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    return X, y


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
            prompts = [tmpl.format(cls.replace("_", " ")) for tmpl in templates]
            tokens = clip_module.tokenize(prompts).to(device)
            feats = model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            cls_feat = feats.mean(dim=0)
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            text_features.append(cls_feat)

    W = torch.stack(text_features, dim=0).cpu().numpy().astype(np.float32)
    return _l2_normalize(W)


def _evaluate(X: np.ndarray, y: np.ndarray, text_features: np.ndarray, num_classes: int) -> Dict[str, object]:
    logits = X @ text_features.T
    y_pred = np.argmax(logits, axis=1)

    acc = float(np.mean((y_pred == y).astype(np.float64)) * 100.0)
    class_accs = _class_acc(y, y_pred, num_classes=num_classes)
    return {
        "n": int(y.shape[0]),
        "acc": acc,
        "balanced_class_acc": _nanmean(class_accs),
        "worst_class_acc": _nanmin(class_accs),
        "class_accs": _fmt_arr(class_accs),
    }


def _write_rows(csv_path: str, rows: Iterable[Dict], header: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary(rows: Sequence[Dict], splits: Sequence[str]) -> None:
    print("\n[SUMMARY] Mean +/- std across seeds")
    for split in splits:
        for metric in ("acc", "balanced_class_acc", "worst_class_acc"):
            key = f"{split}_{metric}"
            vals = np.array([float(r[key]) for r in rows], dtype=float)
            print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _default_templates() -> List[str]:
    return [
        "a photo of {}.",
        "a blurry photo of {}.",
        "a bright photo of {}.",
        "a close-up photo of {}.",
        "a cropped photo of {}.",
        "a low resolution photo of {}.",
        "a good photo of {}.",
        "a photo of the dish {}.",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-shot CLIP evaluation on RedMeat with per-seed rows.")
    p.add_argument("data_path", help="Path to RedMeat dataset root containing all_images.csv")
    p.add_argument("--clip-model", default="RN50", help='CLIP model name (e.g. "RN50", "ViT-B/32")')
    p.add_argument("--device", default="cuda", help='Torch device ("cuda" or "cpu")')
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds to report")
    p.add_argument("--splits", default="val,test", help="Comma-separated splits from {train,val,test}")
    p.add_argument("--split-col", default="split")
    p.add_argument("--label-col", default="label")
    p.add_argument("--path-col", default="abs_file_path")
    p.add_argument(
        "--class-names",
        default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon",
        help="Comma-separated class names in label-id order",
    )
    p.add_argument(
        "--template",
        action="append",
        default=[],
        help="Prompt template with '{}' placeholder (can be passed multiple times).",
    )
    p.add_argument("--output-csv", default="clip_zeroshot_redmeat_5seeds.csv")
    args = p.parse_args()

    seeds = [int(s) for s in _parse_csv_list(args.seeds)]
    if not seeds:
        raise ValueError("No seeds provided.")

    splits = _parse_csv_list(args.splits)
    valid_splits = {"train", "val", "test"}
    if not splits or any(s not in valid_splits for s in splits):
        raise ValueError(f"--splits must be a comma-list from {sorted(valid_splits)}")

    class_names = _parse_csv_list(args.class_names)
    if len(class_names) < 2:
        raise ValueError("--class-names must contain at least 2 classes.")
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    templates = args.template if args.template else _default_templates()
    for t in templates:
        if "{}" not in t:
            raise ValueError(f"Template missing '{{}}' placeholder: {t}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to cpu")
        args.device = "cpu"

    t0 = time.time()
    _seed_everything(seeds[0])

    clip = _try_import_clip()
    try:
        model, preprocess = clip.load(args.clip_model, device=args.device, jit=False)
    except TypeError:
        model, preprocess = clip.load(args.clip_model, device=args.device)

    text_features = _build_text_features(
        clip_module=clip,
        model=model,
        device=args.device,
        class_names=class_names,
        templates=templates,
    )

    split_metrics: Dict[str, Dict[str, object]] = {}
    for split in splits:
        print(f"[ZERO-SHOT] Extracting features for split={split}...")
        samples = _build_samples(
            dataset_path=args.data_path,
            split_col=args.split_col,
            label_col=args.label_col,
            path_col=args.path_col,
            split_value=split,
            class_to_idx=class_to_idx,
        )
        if not samples:
            raise RuntimeError(f"No samples found for split={split}")
        X, y = _extract_features(
            samples=samples,
            model=model,
            preprocess=preprocess,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=seeds[0],
        )
        X = _l2_normalize(X).astype(np.float32, copy=False)
        split_metrics[split] = _evaluate(X, y, text_features, num_classes=len(class_names))

    if "cuda" in args.device:
        del model
        torch.cuda.empty_cache()

    rows: List[Dict[str, object]] = []
    for s in seeds:
        row: Dict[str, object] = {
            "seed": s,
            "clip_model": args.clip_model,
            "class_names": "|".join(class_names),
            "num_templates": len(templates),
            "seconds": int(time.time() - t0),
        }
        for split in splits:
            m = split_metrics[split]
            row[f"{split}_n"] = m["n"]
            row[f"{split}_acc"] = m["acc"]
            row[f"{split}_balanced_class_acc"] = m["balanced_class_acc"]
            row[f"{split}_worst_class_acc"] = m["worst_class_acc"]
            row[f"{split}_class_accs"] = m["class_accs"]
        rows.append(row)

    header: List[str] = ["seed", "clip_model", "class_names", "num_templates"]
    for split in splits:
        header.extend(
            [
                f"{split}_n",
                f"{split}_acc",
                f"{split}_balanced_class_acc",
                f"{split}_worst_class_acc",
                f"{split}_class_accs",
            ]
        )
    header.append("seconds")

    _write_rows(args.output_csv, rows, header)

    print("\n[ZERO-SHOT] Per-seed results")
    for row in rows:
        split_bits = []
        for split in splits:
            split_bits.append(
                f"{split}: acc={float(row[f'{split}_acc']):.4f}, "
                f"bal_cls={float(row[f'{split}_balanced_class_acc']):.4f}, "
                f"worst_cls={float(row[f'{split}_worst_class_acc']):.4f}"
            )
        print(f"  seed={row['seed']} | " + " | ".join(split_bits))

    _print_summary(rows, splits)
    print(f"\n[ZERO-SHOT] Wrote: {args.output_csv}")
    print("[ZERO-SHOT] Note: For fixed CLIP model+prompts, inference is deterministic, so per-seed rows are expected to match.")


if __name__ == "__main__":
    main()
