#!/usr/bin/env python3
"""
Zero-shot CLIP evaluation on Waterbirds with per-seed reporting.

This script does not train anything. It computes CLIP image features and
zero-shot text features, evaluates accuracy metrics, and writes one row per seed.

Because no training occurs, results are typically identical across seeds for a
fixed model + prompts. We still emit per-seed rows for consistency with your
other experiment tables.
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
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _add_repo_to_syspath() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _try_import_clip():
    # Prefer pip package "clip". Fall back to local CLIP copy.
    try:
        import clip  # type: ignore

        return clip
    except Exception:
        _add_repo_to_syspath()
        from CLIP.clip import clip  # type: ignore

        return clip


def _split_root_and_dir(dataset_path: str) -> Tuple[str, str]:
    p = Path(dataset_path).expanduser().resolve()
    if (p / "metadata.csv").exists():
        return str(p.parent), p.name
    raise FileNotFoundError(f"Expected Waterbirds dataset dir containing metadata.csv, got: {p}")


@dataclass(frozen=True)
class _CfgData:
    WATERBIRDS_DIR: str
    SIZE: int = 224
    REMOVE_BACKGROUND: bool = False
    ATTENTION_DIR: str = "NONE"


@dataclass(frozen=True)
class _Cfg:
    DATA: _CfgData


def _load_waterbirds(dataset_path: str):
    _add_repo_to_syspath()
    from datasets.waterbirds import Waterbirds  # type: ignore

    root, waterbirds_dir = _split_root_and_dir(dataset_path)
    cfg = _Cfg(DATA=_CfgData(WATERBIRDS_DIR=waterbirds_dir))
    return root, cfg, Waterbirds


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


def _group_acc(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, num_groups: int = 4) -> np.ndarray:
    accs = np.zeros((num_groups,), dtype=np.float64)
    for g in range(num_groups):
        idx = np.where(groups == g)[0]
        if idx.size == 0:
            accs[g] = float("nan")
            continue
        accs[g] = float(np.mean((y_true[idx] == y_pred[idx]).astype(np.float64)) * 100.0)
    return accs


def _class_acc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    accs = np.zeros((num_classes,), dtype=np.float64)
    for c in range(num_classes):
        idx = np.where(y_true == c)[0]
        if idx.size == 0:
            accs[c] = float("nan")
            continue
        accs[c] = float(np.mean((y_true[idx] == y_pred[idx]).astype(np.float64)) * 100.0)
    return accs


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def _nanmin(x: np.ndarray) -> float:
    return float(np.nanmin(x))


def _fmt_arr(arr: np.ndarray) -> str:
    return np.array2string(arr, precision=4, separator=",")


def _extract_features(
    dataset,
    model,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
        worker_init_fn=_seed_worker,
        generator=g,
    )

    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    groups: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            out = model.encode_image(images).float()
            out = out / out.norm(dim=-1, keepdim=True)
            feats.append(out.cpu().numpy())

            y = batch["label"].cpu().numpy()
            labels.append(y)

            grp = batch["group"].cpu().numpy()
            if grp.ndim == 2 and grp.shape[1] == 1:
                grp = grp[:, 0]
            groups.append(grp)

    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    grp = np.concatenate(groups, axis=0).astype(np.int64)
    return X, y, grp


def _build_text_features(
    clip_module,
    model,
    device: str,
    class_names: Sequence[str],
    templates: Sequence[str],
) -> np.ndarray:
    # Build one normalized embedding per class by averaging normalized prompt embeddings.
    text_features: List[torch.Tensor] = []
    with torch.no_grad():
        for cls in class_names:
            prompts = [tmpl.format(cls) for tmpl in templates]
            tokens = clip_module.tokenize(prompts).to(device)
            feats = model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            cls_feat = feats.mean(dim=0)
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            text_features.append(cls_feat)

    W = torch.stack(text_features, dim=0).cpu().numpy().astype(np.float32)
    return _l2_normalize(W)


def _evaluate_from_features(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    text_features: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    # Argmax over cosine similarity (features are already normalized).
    logits = X @ text_features.T
    y_pred = np.argmax(logits, axis=1)

    acc = float(np.mean((y_pred == y).astype(np.float64)) * 100.0)
    g_acc = _group_acc(y, y_pred, groups, num_groups=4)
    c_acc = _class_acc(y, y_pred, num_classes=num_classes)

    return {
        "n": int(y.shape[0]),
        "acc": acc,
        "balanced_group_acc": _nanmean(g_acc),
        "worst_group_acc": _nanmin(g_acc),
        "group_accs": _fmt_arr(g_acc),
        "class_accs": _fmt_arr(c_acc),
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
        for metric in ("acc", "balanced_group_acc", "worst_group_acc"):
            key = f"{split}_{metric}"
            vals = np.array([float(r[key]) for r in rows], dtype=float)
            print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _default_templates() -> List[str]:
    return [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a bright photo of a {}.",
        "a close-up photo of a {}.",
        "a cropped photo of a {}.",
        "a low resolution photo of a {}.",
        "a good photo of a {}.",
        "a photo of the {}.",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-shot CLIP evaluation on Waterbirds with per-seed reporting.")
    p.add_argument("data_path", help="Path to Waterbirds dataset directory containing metadata.csv")
    p.add_argument("--clip-model", default="RN50", help='CLIP model name (e.g. "RN50", "ViT-B/32")')
    p.add_argument("--device", default="cuda", help='Torch device ("cuda" or "cpu")')
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds to report")
    p.add_argument("--splits", default="val,test", help="Comma-separated splits from {train,val,test}")
    p.add_argument(
        "--class-names",
        default="landbird,waterbird",
        help="Comma-separated class names matching label ids 0,1",
    )
    p.add_argument(
        "--template",
        action="append",
        default=[],
        help="Prompt template with '{}' placeholder (can be passed multiple times).",
    )
    p.add_argument("--output-csv", default="clip_zeroshot_waterbirds_5seeds.csv")
    args = p.parse_args()

    seeds = [int(s) for s in _parse_csv_list(args.seeds)]
    if not seeds:
        raise ValueError("No seeds provided.")

    splits = _parse_csv_list(args.splits)
    valid_splits = {"train", "val", "test"}
    if not splits or any(s not in valid_splits for s in splits):
        raise ValueError(f"--splits must be a comma-list from {sorted(valid_splits)}")

    class_names = _parse_csv_list(args.class_names)
    if len(class_names) != 2:
        raise ValueError("--class-names must contain exactly 2 class names for Waterbirds.")

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

    root, cfg, Waterbirds = _load_waterbirds(args.data_path)

    # Extract image features once per split (seed-independent for shuffle=False inference).
    split_metrics: Dict[str, Dict[str, object]] = {}
    for split in splits:
        print(f"[ZERO-SHOT] Extracting features for split={split}...")
        ds = Waterbirds(root=root, cfg=cfg, split=split, transform=preprocess)
        X, y, g = _extract_features(
            dataset=ds,
            model=model,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=seeds[0],
        )
        X = _l2_normalize(X).astype(np.float32, copy=False)
        split_metrics[split] = _evaluate_from_features(
            X=X,
            y=y,
            groups=g,
            text_features=text_features,
            num_classes=len(class_names),
        )

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
            row[f"{split}_balanced_group_acc"] = m["balanced_group_acc"]
            row[f"{split}_worst_group_acc"] = m["worst_group_acc"]
            row[f"{split}_group_accs"] = m["group_accs"]
            row[f"{split}_class_accs"] = m["class_accs"]
        rows.append(row)

    header: List[str] = ["seed", "clip_model", "class_names", "num_templates"]
    for split in splits:
        header.extend(
            [
                f"{split}_n",
                f"{split}_acc",
                f"{split}_balanced_group_acc",
                f"{split}_worst_group_acc",
                f"{split}_group_accs",
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
                f"bal_grp={float(row[f'{split}_balanced_group_acc']):.4f}, "
                f"worst_grp={float(row[f'{split}_worst_group_acc']):.4f}"
            )
        print(f"  seed={row['seed']} | " + " | ".join(split_bits))

    _print_summary(rows, splits)
    print(f"\n[ZERO-SHOT] Wrote: {args.output_csv}")
    print("[ZERO-SHOT] Note: For fixed CLIP model+prompts, zero-shot inference is deterministic, so per-seed rows are expected to match.")


if __name__ == "__main__":
    main()
