import os
import time
import copy
import argparse
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd

import run_guided_waterbird as base


# Defaults mirror run_guided_waterbird.py so sweeps are comparable.
batch_size = base.batch_size
num_epochs = base.num_epochs
base_lr = base.base_lr
classifier_lr = base.classifier_lr
lr2_mult = base.lr2_mult
momentum = base.momentum
weight_decay = base.weight_decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0


def seed_everything(seed: int):
    # Required by CUDA for deterministic GEMM kernels when deterministic algs are enabled.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _combine_prompt_attentions(att, combine: str):
    # att: [P, 1, H, W] or [1, H, W] or [H, W]
    if isinstance(att, np.ndarray):
        att = torch.from_numpy(att)
    att = att.float()
    if att.dim() == 2:
        att = att.unsqueeze(0).unsqueeze(0)
    elif att.dim() == 3:
        att = att.unsqueeze(0)

    if att.shape[0] == 1:
        out = att
    else:
        if combine == "mean":
            out = att.mean(dim=0, keepdim=True)
        elif combine == "max":
            out = att.max(dim=0, keepdim=True).values
        else:
            raise ValueError(f"Unsupported combine={combine} (expected mean|max)")
    # out: [1, 1, H, W]
    return out


def load_gals_vit_attention(
    pth_path: str,
    *,
    key: str = "unnormalized_attentions",
    combine: str = "mean",
    out_size: int = 224,
    normalize_01: bool = True,
    brighten: float = 1.0,
):
    """
    Loads GALS ViT attention .pth produced by extract_attention.py (SAVE_FOLDER=clip_vit_attention).

    Returns mask tensor shaped [1, out_size, out_size] with non-negative values.
    """
    d = torch.load(pth_path, map_location="cpu")
    if key not in d:
        raise KeyError(f"Missing key '{key}' in {pth_path}. Keys: {list(d.keys())}")

    att = d[key]
    att = _combine_prompt_attentions(att, combine=combine)  # [1,1,h,w]
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


class WaterbirdsMetadataDatasetPthAttention(Dataset):
    SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

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
        return_group: bool = False,
    ):
        self.data_root = data_root
        self.image_transform = image_transform
        self.return_attention = return_attention
        self.return_path = return_path
        self.return_group = return_group

        self.attention_key = attention_key
        self.attention_combine = attention_combine
        self.attention_size = attention_size
        self.attention_normalize_01 = attention_normalize_01
        self.attention_brighten = attention_brighten

        metadata_path = os.path.join(self.data_root, "metadata.csv")
        df = pd.read_csv(metadata_path)
        split_id = self.SPLIT_MAP[split]
        df = df[df["split"] == split_id]

        self.img_rel = df["img_filename"].values.tolist()
        self.paths = [os.path.join(self.data_root, p) for p in self.img_rel]
        self.labels = df["y"].astype(int).values
        self.places = df["place"].astype(int).values

        if attention_root is None:
            attention_root = os.path.join(self.data_root, attention_subdir)
        self.attention_root = attention_root

        self.att_paths = [
            os.path.join(self.attention_root, p.replace(".jpg", ".pth")) for p in self.img_rel
        ]

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _open_rgb_with_retry(path: str, retries: int = 5, sleep_s: float = 0.2) -> Image.Image:
        last_exc = None
        for attempt in range(retries):
            try:
                with Image.open(path) as im:
                    im.load()
                    return im.convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                last_exc = exc
                if attempt + 1 < retries:
                    time.sleep(sleep_s)
                    continue
                break
        raise last_exc

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = int(self.labels[idx])
        img = self._open_rgb_with_retry(path)
        if self.image_transform is not None:
            img = self.image_transform(img)

        output = [img, label]

        if self.return_attention:
            att_path = self.att_paths[idx]
            if not os.path.exists(att_path):
                raise FileNotFoundError(f"Missing attention file: {att_path}")
            mask = load_gals_vit_attention(
                att_path,
                key=self.attention_key,
                combine=self.attention_combine,
                out_size=self.attention_size,
                normalize_01=self.attention_normalize_01,
                brighten=self.attention_brighten,
            )
            output.append(mask)

        if self.return_path:
            output.append(path)

        if self.return_group:
            group = int(label * 2 + int(self.places[idx]))
            output.append(group)

        return tuple(output)


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

    seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    num_workers = base.get_num_workers(default=4)

    train_dataset = WaterbirdsMetadataDatasetPthAttention(
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
        return_group=False,
    )

    val_dataset = base.WaterbirdsMetadataDataset(
        data_root=args.data_path,
        split="val",
        image_transform=data_transforms["eval"],
        return_mask=False,
        return_path=True,
        return_group=False,
    )

    test_dataset = base.WaterbirdsMetadataDataset(
        data_root=args.data_path,
        split="test",
        image_transform=data_transforms["eval"],
        return_mask=False,
        return_path=True,
        return_group=True,
    )

    num_classes = int(len(np.unique(train_dataset.labels)))
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        ),
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = base.make_cam_model(num_classes, model_name="resnet50", pretrained=True).to(device)

    print(f"\n=== RUN (GALS ViT GT): kl_lambda={kl_value}, attention_epoch={attn_epoch} ===", flush=True)
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

    test_loss, test_acc, group_acc, per_group, worst_group = base.evaluate_test(best_model, test_loader)
    print(f"\n[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")
    print(f"[TEST] Per-group Acc: {per_group:.2f}%  Worst-group: {worst_group:.2f}%")

    ckpt = None
    return float(best_score), float(test_acc), float(per_group), float(worst_group), ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Waterbirds dataset root with metadata.csv")
    parser.add_argument(
        "att_path",
        help="Root folder containing GALS ViT attention .pth files (e.g. <dataset>/clip_vit_attention)",
    )
    parser.add_argument("--att-key", default="unnormalized_attentions", choices=["unnormalized_attentions", "attentions"])
    parser.add_argument("--att-combine", default="mean", choices=["mean", "max"])
    parser.add_argument("--att-norm01", action="store_true", default=True)
    parser.add_argument("--att-brighten", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-epoch", type=int, default=0)
    parser.add_argument("--kl-lambda", type=float, default=1.0)
    parser.add_argument("--lr2-mult", type=float, default=lr2_mult)
    args = parser.parse_args()

    globals()["SEED"] = args.seed
    globals()["lr2_mult"] = args.lr2_mult

    run_args = SimpleNamespace(data_path=args.data_path, att_path=args.att_path, att_key=args.att_key,
                               att_combine=args.att_combine, att_norm01=args.att_norm01, att_brighten=args.att_brighten)
    run_single(run_args, args.attention_epoch, args.kl_lambda)


if __name__ == "__main__":
    main()
