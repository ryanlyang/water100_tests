#!/usr/bin/env python3
"""Run zero-shot OpenAI CLIP ViT + OpenCLIP LAION + SigLIP2 on DecoyMNIST, Waterbirds-95/100, and RedMeat test splits."""

from __future__ import annotations

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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _add_repo_to_syspath() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _try_import_open_clip():
    try:
        import open_clip  # type: ignore

        return open_clip
    except Exception as exc:
        raise RuntimeError(
            "Failed to import open_clip. Install with: pip install open_clip_torch"
        ) from exc


def _try_import_openai_clip():
    # Prefer pip package "clip". Fall back to local CLIP copy in this repo.
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


def _fmt_arr(arr: np.ndarray, precision: int = 4) -> str:
    return np.array2string(arr, precision=precision, separator=",")


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def _nanmin(x: np.ndarray) -> float:
    return float(np.nanmin(x))


def _class_acc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    accs = np.zeros((num_classes,), dtype=np.float64)
    for c in range(num_classes):
        idx = np.where(y_true == c)[0]
        if idx.size == 0:
            accs[c] = float("nan")
        else:
            accs[c] = float(np.mean((y_pred[idx] == y_true[idx]).astype(np.float64)) * 100.0)
    return accs


def _group_acc(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, num_groups: int = 4) -> np.ndarray:
    accs = np.zeros((num_groups,), dtype=np.float64)
    for g in range(num_groups):
        idx = np.where(groups == g)[0]
        if idx.size == 0:
            accs[g] = float("nan")
        else:
            accs[g] = float(np.mean((y_true[idx] == y_pred[idx]).astype(np.float64)) * 100.0)
    return accs


def _write_rows(csv_path: str, rows: Iterable[Dict], header: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _tokenize_to_device(tokenizer, prompts: Sequence[str], device: str):
    toks = tokenizer(list(prompts))
    if isinstance(toks, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in toks.items()}
    if torch.is_tensor(toks):
        return toks.to(device)
    raise TypeError(f"Unsupported tokenizer output type: {type(toks)}")


def _load_open_clip_model(model_name: str, pretrained: str, device: str):
    open_clip = _try_import_open_clip()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def _load_openai_clip_model(model_name: str, device: str):
    clip_mod = _try_import_openai_clip()
    model, preprocess = clip_mod.load(model_name, device=device, jit=False)
    model.eval()
    tokenizer = clip_mod.tokenize
    return model, preprocess, tokenizer


def _build_text_features(model, tokenizer, device: str, class_names: Sequence[str], templates: Sequence[str]) -> np.ndarray:
    text_features: List[torch.Tensor] = []
    with torch.no_grad():
        for cls in class_names:
            prompts = [t.format(cls) for t in templates]
            toks = _tokenize_to_device(tokenizer, prompts, device)
            feats = model.encode_text(toks).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            cls_feat = feats.mean(dim=0)
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            text_features.append(cls_feat)
    W = torch.stack(text_features, dim=0).cpu().numpy().astype(np.float32)
    return _l2_normalize(W)


@dataclass(frozen=True)
class _CfgData:
    WATERBIRDS_DIR: str
    SIZE: int = 224
    REMOVE_BACKGROUND: bool = False
    ATTENTION_DIR: str = "NONE"


@dataclass(frozen=True)
class _Cfg:
    DATA: _CfgData


def _split_root_and_dir(dataset_path: str) -> Tuple[str, str]:
    p = Path(dataset_path).expanduser().resolve()
    if (p / "metadata.csv").exists():
        return str(p.parent), p.name
    raise FileNotFoundError(f"Expected metadata.csv under: {p}")


class _ImageLabelDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), int(label)


class _ClipFolder(Dataset):
    def __init__(self, folder: ImageFolder, preprocess):
        self.folder = folder
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.folder)

    def __getitem__(self, idx: int):
        img, label = self.folder[idx]
        return self.preprocess(img), int(label)


def _extract_features_simple(ds: Dataset, model, device: str, batch_size: int, num_workers: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
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


def _extract_features_waterbirds(dataset, model, device: str, batch_size: int, num_workers: int, seed: int):
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
    ys: List[np.ndarray] = []
    gs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            f = model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            ys.append(batch["label"].cpu().numpy())
            grp = batch["group"].cpu().numpy()
            if grp.ndim == 2 and grp.shape[1] == 1:
                grp = grp[:, 0]
            gs.append(grp)

    X = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    g_arr = np.concatenate(gs, axis=0).astype(np.int64, copy=False)
    return _l2_normalize(X), y, g_arr


def _build_redmeat_test_items(dataset_path: str, class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
    meta_csv = os.path.join(dataset_path, "all_images.csv")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Missing all_images.csv: {meta_csv}")

    items: List[Tuple[str, int]] = []
    with open(meta_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("split", "")) != "test":
                continue
            label_name = str(row.get("label", ""))
            if label_name not in class_to_idx:
                continue
            rel_or_abs = str(row.get("abs_file_path", "")).lstrip("/")
            path = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(dataset_path, rel_or_abs)
            if not os.path.exists(path):
                continue
            items.append((path, class_to_idx[label_name]))
    if not items:
        raise RuntimeError(f"No RedMeat test samples resolved from {meta_csv}")
    return items


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


def _eval_classification(X: np.ndarray, y: np.ndarray, text_features: np.ndarray, num_classes: int) -> Dict[str, object]:
    logits = X @ text_features.T
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean((pred == y).astype(np.float64)) * 100.0)
    c_acc = _class_acc(y, pred, num_classes)
    return {
        "n": int(y.shape[0]),
        "acc": acc,
        "balanced_class_acc": _nanmean(c_acc),
        "worst_class_acc": _nanmin(c_acc),
        "class_accs": _fmt_arr(c_acc, precision=4),
    }


def _eval_waterbirds(X: np.ndarray, y: np.ndarray, groups: np.ndarray, text_features: np.ndarray, num_classes: int) -> Dict[str, object]:
    logits = X @ text_features.T
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean((pred == y).astype(np.float64)) * 100.0)
    g_acc = _group_acc(y, pred, groups, num_groups=4)
    c_acc = _class_acc(y, pred, num_classes)
    return {
        "n": int(y.shape[0]),
        "acc": acc,
        "balanced_group_acc": _nanmean(g_acc),
        "worst_group_acc": _nanmin(g_acc),
        "group_accs": _fmt_arr(g_acc, precision=4),
        "balanced_class_acc": _nanmean(c_acc),
        "worst_class_acc": _nanmin(c_acc),
        "class_accs": _fmt_arr(c_acc, precision=4),
    }


def _waterbirds_templates() -> List[str]:
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


def _redmeat_templates() -> List[str]:
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


def _decoy_templates() -> List[str]:
    return [
        "a handwritten digit {}.",
        "a photo of the digit {}.",
        "an image of the number {}.",
        "a grayscale image of {}.",
        "a centered handwritten {}.",
    ]


def _evaluate_waterbirds_dataset(
    dataset_tag: str,
    dataset_path: str,
    model,
    preprocess,
    tokenizer,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, object]:
    _add_repo_to_syspath()
    from datasets.waterbirds import Waterbirds  # type: ignore

    class_names = ["landbird", "waterbird"]
    text_features = _build_text_features(model, tokenizer, device, class_names, _waterbirds_templates())

    root, wb_dir = _split_root_and_dir(dataset_path)
    cfg = _Cfg(DATA=_CfgData(WATERBIRDS_DIR=wb_dir))
    ds = Waterbirds(root=root, cfg=cfg, split="test", transform=preprocess)
    X, y, groups = _extract_features_waterbirds(ds, model, device, batch_size, num_workers, seed)
    metrics = _eval_waterbirds(X, y, groups, text_features, num_classes=2)
    metrics["dataset"] = dataset_tag
    metrics["split"] = "test"
    metrics["class_names"] = "|".join(class_names)
    return metrics


def _evaluate_redmeat_dataset(
    dataset_path: str,
    model,
    preprocess,
    tokenizer,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    class_names: Sequence[str],
) -> Dict[str, object]:
    text_class_names = [x.replace("_", " ") for x in class_names]
    text_features = _build_text_features(model, tokenizer, device, text_class_names, _redmeat_templates())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    items = _build_redmeat_test_items(dataset_path, class_to_idx)
    ds = _ImageLabelDataset(items, preprocess)
    X, y = _extract_features_simple(ds, model, device, batch_size, num_workers, seed)
    metrics = _eval_classification(X, y, text_features, num_classes=len(class_names))
    metrics["dataset"] = "redmeat"
    metrics["split"] = "test"
    metrics["class_names"] = "|".join(class_names)
    return metrics


def _evaluate_decoy_dataset(
    png_root: str,
    model,
    preprocess,
    tokenizer,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, object]:
    test_dir = os.path.join(png_root, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Missing DecoyMNIST test dir: {test_dir}")
    folder = ImageFolder(test_dir)
    class_names = [_digit_name(c) for c in folder.classes]
    text_features = _build_text_features(model, tokenizer, device, class_names, _decoy_templates())
    ds = _ClipFolder(folder, preprocess)
    X, y = _extract_features_simple(ds, model, device, batch_size, num_workers, seed)
    metrics = _eval_classification(X, y, text_features, num_classes=len(folder.classes))
    metrics["dataset"] = "decoymnist"
    metrics["split"] = "test"
    metrics["class_names"] = "|".join(folder.classes)
    return metrics


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot OpenAI CLIP ViT + OpenCLIP LAION + SigLIP2 on DecoyMNIST, Waterbirds95/100, and RedMeat test splits."
    )
    p.add_argument(
        "--wb95-path",
        default="/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2",
    )
    p.add_argument(
        "--wb100-path",
        default="/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2",
    )
    p.add_argument(
        "--redmeat-path",
        default="/home/ryreu/guided_cnn/Food101/data/food-101-redmeat",
    )
    p.add_argument(
        "--decoy-png-root",
        default="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png",
    )
    p.add_argument(
        "--redmeat-class-names",
        default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon",
    )
    p.add_argument("--openai-model", default="ViT-B/32")
    p.add_argument("--laion-model", default="ViT-B-32")
    p.add_argument("--laion-pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--siglip2-model", default="ViT-B-16-SigLIP2-256")
    p.add_argument("--siglip2-pretrained", default="webli")
    p.add_argument("--variants", default="openai_vit,openclip_laion,siglip2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seeds", default="0")
    p.add_argument("--output-csv", default="zeroshot_openai_openclip_siglip2_all_test.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to cpu.")
        args.device = "cpu"

    seeds = [int(s) for s in _parse_csv_list(args.seeds)]
    if not seeds:
        raise ValueError("No seeds provided.")

    variants = _parse_csv_list(args.variants)
    if not variants:
        raise ValueError("No variants provided.")

    redmeat_classes = _parse_csv_list(args.redmeat_class_names)
    if len(redmeat_classes) < 2:
        raise ValueError("Expected at least 2 redmeat classes.")

    variant_cfg = {
        "openai_vit": {
            "family": "openai_clip",
            "model_name": args.openai_model,
            "pretrained": "openai",
        },
        "openclip_laion": {
            "family": "open_clip",
            "model_name": args.laion_model,
            "pretrained": args.laion_pretrained,
        },
        "siglip2": {
            "family": "open_clip",
            "model_name": args.siglip2_model,
            "pretrained": args.siglip2_pretrained,
        },
    }

    rows: List[Dict[str, object]] = []
    t0 = time.time()

    for seed in seeds:
        _seed_everything(seed)
        for variant in variants:
            if variant not in variant_cfg:
                raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(variant_cfg)}")
            vcfg = variant_cfg[variant]
            model_name = str(vcfg["model_name"])
            pretrained = str(vcfg["pretrained"])
            family = str(vcfg["family"])
            print(
                f"\n[MODEL] variant={variant} seed={seed} model={model_name} pretrained={pretrained} device={args.device}"
            )
            if family == "openai_clip":
                model, preprocess, tokenizer = _load_openai_clip_model(model_name, args.device)
            else:
                model, preprocess, tokenizer = _load_open_clip_model(model_name, pretrained, args.device)

            # DecoyMNIST
            decoy = _evaluate_decoy_dataset(
                args.decoy_png_root, model, preprocess, tokenizer, args.device, args.batch_size, args.num_workers, seed
            )
            rows.append(
                {
                    "dataset": decoy["dataset"],
                    "variant": variant,
                    "seed": seed,
                    "model_name": model_name,
                    "pretrained": pretrained,
                    "split": decoy["split"],
                    "n": decoy["n"],
                    "acc": decoy["acc"],
                    "balanced_group_acc": "",
                    "worst_group_acc": "",
                    "group_accs": "",
                    "balanced_class_acc": decoy["balanced_class_acc"],
                    "worst_class_acc": decoy["worst_class_acc"],
                    "class_accs": decoy["class_accs"],
                    "class_names": decoy["class_names"],
                }
            )
            print(
                f"[RESULT] {decoy['dataset']} acc={float(decoy['acc']):.4f} "
                f"worst_class={float(decoy['worst_class_acc']):.4f}"
            )

            # Waterbirds 95
            wb95 = _evaluate_waterbirds_dataset(
                "waterbirds95",
                args.wb95_path,
                model,
                preprocess,
                tokenizer,
                args.device,
                args.batch_size,
                args.num_workers,
                seed,
            )
            rows.append(
                {
                    "dataset": wb95["dataset"],
                    "variant": variant,
                    "seed": seed,
                    "model_name": model_name,
                    "pretrained": pretrained,
                    "split": wb95["split"],
                    "n": wb95["n"],
                    "acc": wb95["acc"],
                    "balanced_group_acc": wb95["balanced_group_acc"],
                    "worst_group_acc": wb95["worst_group_acc"],
                    "group_accs": wb95["group_accs"],
                    "balanced_class_acc": wb95["balanced_class_acc"],
                    "worst_class_acc": wb95["worst_class_acc"],
                    "class_accs": wb95["class_accs"],
                    "class_names": wb95["class_names"],
                }
            )
            print(
                f"[RESULT] {wb95['dataset']} acc={float(wb95['acc']):.4f} "
                f"bal_group={float(wb95['balanced_group_acc']):.4f} "
                f"worst_group={float(wb95['worst_group_acc']):.4f}"
            )

            # Waterbirds 100
            wb100 = _evaluate_waterbirds_dataset(
                "waterbirds100",
                args.wb100_path,
                model,
                preprocess,
                tokenizer,
                args.device,
                args.batch_size,
                args.num_workers,
                seed,
            )
            rows.append(
                {
                    "dataset": wb100["dataset"],
                    "variant": variant,
                    "seed": seed,
                    "model_name": model_name,
                    "pretrained": pretrained,
                    "split": wb100["split"],
                    "n": wb100["n"],
                    "acc": wb100["acc"],
                    "balanced_group_acc": wb100["balanced_group_acc"],
                    "worst_group_acc": wb100["worst_group_acc"],
                    "group_accs": wb100["group_accs"],
                    "balanced_class_acc": wb100["balanced_class_acc"],
                    "worst_class_acc": wb100["worst_class_acc"],
                    "class_accs": wb100["class_accs"],
                    "class_names": wb100["class_names"],
                }
            )
            print(
                f"[RESULT] {wb100['dataset']} acc={float(wb100['acc']):.4f} "
                f"bal_group={float(wb100['balanced_group_acc']):.4f} "
                f"worst_group={float(wb100['worst_group_acc']):.4f}"
            )

            # RedMeat
            red = _evaluate_redmeat_dataset(
                args.redmeat_path,
                model,
                preprocess,
                tokenizer,
                args.device,
                args.batch_size,
                args.num_workers,
                seed,
                redmeat_classes,
            )
            rows.append(
                {
                    "dataset": red["dataset"],
                    "variant": variant,
                    "seed": seed,
                    "model_name": model_name,
                    "pretrained": pretrained,
                    "split": red["split"],
                    "n": red["n"],
                    "acc": red["acc"],
                    "balanced_group_acc": "",
                    "worst_group_acc": "",
                    "group_accs": "",
                    "balanced_class_acc": red["balanced_class_acc"],
                    "worst_class_acc": red["worst_class_acc"],
                    "class_accs": red["class_accs"],
                    "class_names": red["class_names"],
                }
            )
            print(
                f"[RESULT] {red['dataset']} acc={float(red['acc']):.4f} "
                f"worst_class={float(red['worst_class_acc']):.4f}"
            )

            if "cuda" in args.device:
                del model
                torch.cuda.empty_cache()

    header = [
        "dataset",
        "variant",
        "seed",
        "model_name",
        "pretrained",
        "split",
        "n",
        "acc",
        "balanced_group_acc",
        "worst_group_acc",
        "group_accs",
        "balanced_class_acc",
        "worst_class_acc",
        "class_accs",
        "class_names",
    ]
    _write_rows(args.output_csv, rows, header)
    print(f"\n[DONE] wrote {args.output_csv}")

    print("\n[SUMMARY] by dataset + variant")
    by_key_acc: Dict[Tuple[str, str], List[float]] = {}
    by_key_worst_class: Dict[Tuple[str, str], List[float]] = {}
    by_key_worst_group: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        key = (str(r["dataset"]), str(r["variant"]))
        by_key_acc.setdefault(key, []).append(float(r["acc"]))

        worst_class_raw = r.get("worst_class_acc", "")
        if worst_class_raw not in ("", None):
            by_key_worst_class.setdefault(key, []).append(float(worst_class_raw))

        worst_group_raw = r.get("worst_group_acc", "")
        if worst_group_raw not in ("", None):
            by_key_worst_group.setdefault(key, []).append(float(worst_group_raw))

    for key in sorted(by_key_acc):
        acc_vals = np.array(by_key_acc[key], dtype=float)
        msg = f"  {key[0]} | {key[1]} acc={np.mean(acc_vals):.4f} +/- {np.std(acc_vals):.4f}"
        if key in by_key_worst_class and len(by_key_worst_class[key]) > 0:
            wc = np.array(by_key_worst_class[key], dtype=float)
            msg += f" | worst_class={np.mean(wc):.4f} +/- {np.std(wc):.4f}"
        if key in by_key_worst_group and len(by_key_worst_group[key]) > 0:
            wg = np.array(by_key_worst_group[key], dtype=float)
            msg += f" | worst_group={np.mean(wg):.4f} +/- {np.std(wg):.4f}"
        print(msg)
    print(f"[TIME] seconds={int(time.time() - t0)}")


if __name__ == "__main__":
    main()
