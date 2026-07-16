#!/usr/bin/env python3
import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Import shared guided-CNN training/model utilities from the project root.
GALS_ROOT = Path(__file__).resolve().parent.parent
if str(GALS_ROOT) not in sys.path:
    sys.path.insert(0, str(GALS_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import run_guided_waterbird as base  # noqa: E402

# Reuse RedMeat metadata dataset + test eval from the PNG-mask guided runner.
from RedMeat_Runs import run_guided_redmeat as red_png  # noqa: E402


batch_size = red_png.batch_size
num_epochs = red_png.num_epochs
base_lr = red_png.base_lr
classifier_lr = red_png.classifier_lr
lr2_mult = red_png.lr2_mult
momentum = red_png.momentum
weight_decay = red_png.weight_decay

checkpoint_dir = "RedMeat_Guided_GALSViT_Checkpoints"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0


def _resolve_img_path(data_root: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    rel = str(rel_or_abs).lstrip("/")
    return os.path.join(data_root, rel)


def _to_rel_under_root(path_or_rel: str, root: str) -> str:
    s = str(path_or_rel)
    if os.path.isabs(s):
        try:
            rel = os.path.relpath(s, root)
            if not rel.startswith(".."):
                return rel
        except Exception:
            pass
        return os.path.basename(s)
    return s.lstrip("/")


def _combine_prompt_attentions(att, combine: str):
    # att can be [P,1,H,W], [1,H,W], [H,W]
    if isinstance(att, np.ndarray):
        att = torch.from_numpy(att)
    att = att.float()

    if att.dim() == 2:
        att = att.unsqueeze(0).unsqueeze(0)
    elif att.dim() == 3:
        att = att.unsqueeze(0)

    if att.shape[0] == 1:
        out = att
    elif combine == "mean":
        out = att.mean(dim=0, keepdim=True)
    elif combine == "max":
        out = att.max(dim=0, keepdim=True).values
    else:
        raise ValueError(f"Unsupported combine='{combine}' (expected mean|max)")

    return out  # [1,1,H,W]


def load_gals_vit_attention(
    pth_path: str,
    *,
    key: str = "unnormalized_attentions",
    combine: str = "mean",
    out_size: int = 224,
    normalize_01: bool = True,
    brighten: float = 1.0,
):
    d = torch.load(pth_path, map_location="cpu")
    if key not in d:
        raise KeyError(f"Missing key '{key}' in {pth_path}. Keys: {list(d.keys())}")

    att = _combine_prompt_attentions(d[key], combine=combine)
    att = torch.clamp(att, min=0.0)

    if normalize_01:
        mn = float(att.min())
        mx = float(att.max())
        if mx - mn > 1e-12:
            att = (att - mn) / (mx - mn)
        else:
            att = torch.zeros_like(att)

    if out_size is not None:
        att = F.interpolate(att, size=(out_size, out_size), mode="bilinear", align_corners=False)

    if brighten != 1.0:
        att = torch.clamp(att * float(brighten), 0.0, 1.0 if normalize_01 else float("inf"))

    return att[0]  # [1,H,W]


def _attention_candidates(att_root: str, rel_path: str, img_path: str):
    rel_no_ext = os.path.splitext(rel_path)[0]
    yield os.path.join(att_root, rel_no_ext + ".pth")

    # Fallbacks for layout differences.
    img_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
    yield os.path.join(att_root, img_name_no_ext + ".pth")

    if rel_path.startswith("images/"):
        yield os.path.join(att_root, os.path.splitext(rel_path[len("images/"):])[0] + ".pth")


def _first_existing(paths):
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p
    return None


class RedMeatMetadataDatasetPthAttention(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        image_transform=None,
        attention_root: str = None,
        attention_subdir: str = "clip_vit_attention",
        attention_key: str = "unnormalized_attentions",
        attention_combine: str = "mean",
        attention_size: int = 224,
        attention_normalize_01: bool = True,
        attention_brighten: float = 1.0,
        return_attention: bool = True,
        return_path: bool = True,
        classes=None,
        split_col: str = "split",
        label_col: str = "label",
        path_col: str = "abs_file_path",
    ):
        self.data_root = data_root
        self.image_transform = image_transform
        self.return_attention = return_attention
        self.return_path = return_path

        self.attention_key = attention_key
        self.attention_combine = attention_combine
        self.attention_size = attention_size
        self.attention_normalize_01 = attention_normalize_01
        self.attention_brighten = attention_brighten

        meta_path = os.path.join(self.data_root, "all_images.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing all_images.csv at: {meta_path}")

        df = pd.read_csv(meta_path)
        for c in (split_col, label_col, path_col):
            if c not in df.columns:
                raise KeyError(f"Missing column '{c}' in {meta_path}. Columns={list(df.columns)}")

        if classes is None:
            classes = sorted(df[label_col].astype(str).unique().tolist())
        self.classes = [str(c) for c in classes]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        sdf = df[df[split_col].astype(str) == str(split)]
        if len(sdf) == 0:
            raise ValueError(f"Split '{split}' is empty in {meta_path}")

        label_names = sdf[label_col].astype(str).tolist()
        unknown = sorted(set(label_names) - set(self.class_to_idx.keys()))
        if unknown:
            raise ValueError(f"Labels not in class list for split '{split}': {unknown}")

        self.labels = np.array([self.class_to_idx[x] for x in label_names], dtype=np.int64)

        rel_paths = [_to_rel_under_root(p, self.data_root) for p in sdf[path_col].astype(str).tolist()]
        self.paths = [_resolve_img_path(self.data_root, p) for p in rel_paths]

        if attention_root is None:
            attention_root = os.path.join(self.data_root, attention_subdir)
        self.attention_root = attention_root

        self.att_paths = [
            _first_existing(_attention_candidates(self.attention_root, rel_path, img_path))
            for rel_path, img_path in zip(rel_paths, self.paths)
        ]

        if self.return_attention:
            missing = [self.paths[i] for i, ap in enumerate(self.att_paths) if ap is None]
            if missing:
                preview = "\n  - ".join(missing[:5])
                raise FileNotFoundError(
                    f"Missing attention .pth for {len(missing)} samples under {self.attention_root}. "
                    f"Example images with no .pth:\n  - {preview}"
                )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = int(self.labels[idx])

        img = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)

        out = [img, label]

        if self.return_attention:
            att_path = self.att_paths[idx]
            if att_path is None:
                raise FileNotFoundError(f"Missing attention path for sample: {path}")
            att = load_gals_vit_attention(
                att_path,
                key=self.attention_key,
                combine=self.attention_combine,
                out_size=self.attention_size,
                normalize_01=self.attention_normalize_01,
                brighten=self.attention_brighten,
            )
            out.append(att)

        if self.return_path:
            out.append(path)

        return tuple(out)


def run_single(args, attn_epoch, kl_value, kl_increment=None):
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_attention = attn_epoch < num_epochs and kl_value > 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

    base.seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    train_dataset = RedMeatMetadataDatasetPthAttention(
        data_root=args.data_path,
        split="train",
        image_transform=data_transforms["train"],
        attention_root=args.att_path,
        attention_subdir=getattr(args, "att_subdir", "clip_vit_attention"),
        attention_key=getattr(args, "att_key", "unnormalized_attentions"),
        attention_combine=getattr(args, "att_combine", "mean"),
        attention_size=224,
        attention_normalize_01=bool(getattr(args, "att_norm01", True)),
        attention_brighten=float(getattr(args, "att_brighten", 1.0)),
        return_attention=use_attention,
        return_path=True,
        classes=args.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )

    val_dataset = red_png.RedMeatMetadataDataset(
        data_root=args.data_path,
        split="val",
        image_transform=data_transforms["eval"],
        return_mask=False,
        return_path=True,
        classes=train_dataset.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )

    test_dataset = red_png.RedMeatMetadataDataset(
        data_root=args.data_path,
        split="test",
        image_transform=data_transforms["eval"],
        return_mask=False,
        return_path=True,
        classes=train_dataset.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )

    num_classes = len(train_dataset.classes)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=base.seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=base.seed_worker,
            generator=g,
        ),
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
    }

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=base.seed_worker,
        generator=g,
    )

    model = base.make_cam_model(num_classes, model_name="resnet50", pretrained=True).to(device)

    save_checkpoints = os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "n")
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n=== RUN (RedMeat GALS ViT GT): kl_lambda={kl_value}, attention_epoch={attn_epoch} ===", flush=True)
    if kl_increment is None:
        kl_increment = kl_value / 10.0

    best_model, best_score, best_epoch = base.train_model(
        model,
        dataloaders,
        dataset_sizes,
        attn_epoch,
        kl_value,
        num_epochs,
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        lr2_mult=lr2_mult,
        kl_incr=kl_increment,
        use_attention=use_attention,
        num_classes=num_classes,
    )
    print(f"\n[VAL] Best Balanced Acc: {best_score:.4f} at epoch {best_epoch}")

    test_loss, test_acc, class_acc, per_group, worst_group = red_png.evaluate_test(best_model, test_loader, num_classes)
    print(f"\n[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")
    for cls_name, acc in zip(train_dataset.classes, class_acc):
        print(f"[TEST] {cls_name}: {acc:.2f}%")
    print(f"[TEST] Per-class mean: {per_group:.2f}%  Worst-class: {worst_group:.2f}%")

    if save_checkpoints:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"resnet50_redmeat_galsvit_final_kl{int(kl_value)}_attn{attn_epoch}_{ts}.pth"
        save_path = os.path.join(checkpoint_dir, save_name)
        torch.save(best_model.state_dict(), save_path)
    else:
        save_path = "NONE"
        print("[RUN DONE] Checkpoint saving disabled via SAVE_CHECKPOINTS=0", flush=True)

    print(
        f"[RUN DONE] kl={kl_value} attn={attn_epoch} lr2_mult={lr2_mult} kl_incr={kl_increment} "
        f"| best_balanced_val_acc={best_score:.4f} | test_acc={test_acc:.2f}% | saved: {save_path}",
        flush=True,
    )

    return float(best_score), float(test_acc), float(per_group), float(worst_group), save_path


def main():
    global SEED, base_lr, classifier_lr, lr2_mult, num_epochs, checkpoint_dir

    p = argparse.ArgumentParser(
        description="Guided RedMeat runner using GALS ViT .pth attentions as guidance masks."
    )
    p.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")
    p.add_argument("att_path", help="Root folder containing GALS ViT attention .pth files")

    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--attention-epoch", type=int, default=num_epochs)
    p.add_argument("--kl-lambda", type=float, default=0.0)
    p.add_argument("--kl-increment", type=float, default=None)
    p.add_argument("--base_lr", type=float, default=base_lr)
    p.add_argument("--classifier_lr", type=float, default=classifier_lr)
    p.add_argument("--lr2-mult", type=float, default=lr2_mult)
    p.add_argument("--num-epochs", type=int, default=num_epochs)
    p.add_argument("--checkpoint-dir", default=checkpoint_dir)

    p.add_argument("--att-key", default="unnormalized_attentions", choices=["unnormalized_attentions", "attentions"])
    p.add_argument("--att-combine", default="mean", choices=["mean", "max"])
    p.add_argument("--att-norm01", action="store_true", default=True)
    p.add_argument("--att-brighten", type=float, default=1.0)

    p.add_argument("--split-col", default="split")
    p.add_argument("--label-col", default="label")
    p.add_argument("--path-col", default="abs_file_path")
    p.add_argument(
        "--classes",
        default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon",
        help="Comma-separated class list. Empty string = infer from metadata.",
    )

    args = p.parse_args()

    SEED = int(args.seed)
    base_lr = float(args.base_lr)
    classifier_lr = float(args.classifier_lr)
    lr2_mult = float(args.lr2_mult)
    num_epochs = int(args.num_epochs)
    checkpoint_dir = str(args.checkpoint_dir)

    if num_epochs < 1:
        raise ValueError("--num-epochs must be >= 1")

    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()] if args.classes else None

    run_args = argparse.Namespace(
        data_path=args.data_path,
        att_path=args.att_path,
        att_key=args.att_key,
        att_combine=args.att_combine,
        att_norm01=args.att_norm01,
        att_brighten=args.att_brighten,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
        classes=classes,
    )

    run_single(run_args, int(args.attention_epoch), float(args.kl_lambda), args.kl_increment)


if __name__ == "__main__":
    main()
