#!/usr/bin/env python3
"""Fixed-hyperparameter DecoyMNIST LeNet with GALS-style RRR loss.

This is a decoy-style LeNet run that keeps the standard Decoy optimizer setup
(Adam, lr=0.001, weight_decay=1e-4) and adds an input-gradient suppression
term outside an external attention mask.
"""

from __future__ import annotations

import argparse
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor
from tqdm.auto import tqdm


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor, return_fmaps: bool = False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        fmaps = F.relu(self.conv2(x))
        x = F.max_pool2d(fmaps, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        if return_fmaps:
            return logits, fmaps
        return logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _load_attention_map(path: Path) -> torch.Tensor:
    # PyTorch >=2.6 defaults torch.load(..., weights_only=True), which
    # rejects non-tensor payloads used by these saved attention maps.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        payload = torch.load(path, map_location="cpu")
    arr = None
    if isinstance(payload, dict):
        for key in ("attentions", "unnormalized_attentions", "attention", "cam", "saliency"):
            if key in payload:
                arr = payload[key]
                break
    else:
        arr = payload
    if arr is None:
        raise ValueError(f"Could not parse attention payload at: {path}")

    att = torch.as_tensor(arr, dtype=torch.float32)
    # Collapse any leading dims (e.g. num_prompts x 1 x H x W) into a single
    # stack and reduce to one 2D map. This avoids shape-dependent infinite
    # loops when the first dim is >1.
    if att.ndim > 2:
        att = att.reshape(-1, att.shape[-2], att.shape[-1]).max(dim=0).values
    if att.ndim != 2:
        raise ValueError(f"Expected 2D attention after reduction, got shape {tuple(att.shape)} at {path}")

    mn = float(att.min())
    mx = float(att.max())
    if mx > mn:
        att = (att - mn) / (mx - mn)
    else:
        att = torch.zeros_like(att)
    return att.unsqueeze(0)  # 1 x H x W


class GuidedImageFolder(utils.Dataset):
    def __init__(self, png_root: str, mask_root: str, split: str, image_transform=None) -> None:
        self.split = split
        self.split_img_root = Path(png_root) / split
        self.split_mask_root = Path(mask_root) / split
        self.images = ImageFolder(str(self.split_img_root), transform=image_transform)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img, label = self.images[idx]
        img_path, _ = self.images.samples[idx]
        rel = Path(img_path).resolve().relative_to(self.split_img_root.resolve())
        mask_path = (self.split_mask_root / rel).with_suffix(".pth")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {img_path}: {mask_path}")
        mask = _load_attention_map(mask_path)
        return img, label, mask


@torch.no_grad()
def evaluate(model: nn.Module, loader: utils.DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        if len(batch) == 3:
            data, target, _ = batch
        else:
            data, target = batch
        data = data.to(device)
        target = target.to(device)
        logits = model(data)
        loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
        correct += logits.argmax(dim=1).eq(target).sum().item()
        total += data.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def train_one_seed(args, seed: int, full_train: GuidedImageFolder, test_dataset: ImageFolder, device, loader_kwargs):
    set_seed(seed)

    # Keep split fixed across seeds (matches cdepstyle behavior used in Decoy runners).
    split_g = torch.Generator().manual_seed(0)
    n_total = len(full_train)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val
    train_subset, val_subset = utils.random_split(full_train, [n_train, n_val], generator=split_g)

    train_loader = utils.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = utils.DataLoader(val_subset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)
    test_loader = utils.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.grad_criterion == "L1":
        grad_criterion = nn.L1Loss()
    else:
        grad_criterion = nn.MSELoss()
    if args.cam_criterion == "L1":
        cam_criterion = nn.L1Loss()
    else:
        cam_criterion = nn.MSELoss()

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_iter = train_loader
        if not args.no_progress_bar:
            epoch_iter = tqdm(
                train_loader,
                desc=f"seed={seed} epoch={epoch}/{args.epochs}",
                leave=False,
                dynamic_ncols=True,
            )

        for data, target, gt_mask in epoch_iter:
            data = data.to(device)
            target = target.to(device)
            gt_mask = gt_mask.to(device)
            gt_mask = F.interpolate(gt_mask, size=data.shape[-2:], mode="nearest")

            data.requires_grad_(True)
            optimizer.zero_grad()

            logits, fmaps = model(data, return_fmaps=True)
            cls_loss = F.cross_entropy(logits, target)
            loss = cls_loss

            if args.loss_mode in ("rrr", "both"):
                dy_dx = torch.autograd.grad(cls_loss, data, create_graph=True, retain_graph=True)[0]
                rrr_loss = grad_criterion(dy_dx, dy_dx * gt_mask)
                loss = loss + args.grad_weight * rrr_loss
            if args.loss_mode in ("gradcam", "both"):
                one_hot = torch.zeros_like(logits)
                one_hot.scatter_(1, target.unsqueeze(1), 1.0)
                grads = torch.autograd.grad(
                    logits,
                    fmaps,
                    grad_outputs=one_hot,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                weights = F.adaptive_avg_pool2d(grads, 1)
                gcam = (fmaps * weights).sum(dim=1, keepdim=True)
                gcam = F.relu(gcam)
                gcam = F.interpolate(gcam, size=data.shape[-2:], mode="bilinear", align_corners=False)
                gflat = gcam.view(gcam.size(0), -1)
                gmax = gflat.max(dim=1, keepdim=True)[0]
                gmax = torch.where(gmax == 0, torch.ones_like(gmax), gmax)
                gmin = gflat.min(dim=1, keepdim=True)[0]
                gcam = ((gflat - gmin) / gmax).view_as(gcam)

                if args.cam_mode == "suppress_outside":
                    cam_loss = cam_criterion(gcam, gcam * gt_mask)
                else:
                    cam_loss = cam_criterion(gcam, gt_mask)
                loss = loss + args.cam_weight * cam_loss

            loss.backward()
            optimizer.step()

            if not args.no_progress_bar:
                epoch_iter.set_postfix(loss=f"{float(loss.item()):.4f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())

        if args.print_every > 0 and (epoch % args.print_every == 0 or epoch == args.epochs):
            print(
                f"seed={seed} epoch={epoch}/{args.epochs} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
            )

    model.load_state_dict(best_weights)
    _, test_acc = evaluate(model, test_loader, device)
    return best_val_acc, test_acc, best_epoch, deepcopy(best_weights)


def main() -> None:
    parser = argparse.ArgumentParser(description="DecoyMNIST LeNet with fixed GALS-style RRR loss")
    parser.add_argument(
        "--png-root",
        type=str,
        default="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png",
    )
    parser.add_argument(
        "--mask-root",
        type=str,
        default="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png/clip_rn50_attention_gradcam",
    )
    parser.add_argument("--epochs", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-weight", type=float, default=72503.48035960984)
    parser.add_argument("--grad-criterion", choices=["L1", "L2"], default="L1")
    parser.add_argument("--loss-mode", choices=["rrr", "gradcam", "both"], default="rrr")
    parser.add_argument("--cam-weight", type=float, default=1.0)
    parser.add_argument("--cam-criterion", choices=["L1", "L2"], default="L1")
    parser.add_argument("--cam-mode", choices=["match", "suppress_outside"], default="match")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--no-progress-bar", action="store_true", default=False)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="DecoyMNIST_GALS_Checkpoints",
        help="Directory to save best-val checkpoint per seed.",
    )
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}

    transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
    full_train = GuidedImageFolder(
        png_root=args.png_root,
        mask_root=args.mask_root,
        split="train",
        image_transform=transform,
    )
    test_dataset = ImageFolder(os.path.join(args.png_root, "test"), transform=transform)

    print("Running DecoyMNIST fixed GALS-RRR")
    print(f"device={device}")
    print(f"png_root={args.png_root}")
    print(f"mask_root={args.mask_root}")
    print(f"train={len(full_train)} test={len(test_dataset)} split={1.0 - args.val_frac:.2f}/{args.val_frac:.2f}")
    print(
        f"optimizer=Adam lr={args.lr} weight_decay={args.weight_decay} "
        f"loss_mode={args.loss_mode} grad_weight={args.grad_weight} grad_criterion={args.grad_criterion} "
        f"cam_weight={args.cam_weight} cam_criterion={args.cam_criterion} cam_mode={args.cam_mode}"
    )
    print(f"checkpoint_dir={args.checkpoint_dir}")

    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    for i in range(args.n_seeds):
        seed = args.seed_start + i
        best_val_acc, test_acc, best_epoch, best_weights = train_one_seed(
            args=args,
            seed=seed,
            full_train=full_train,
            test_dataset=test_dataset,
            device=device,
            loader_kwargs=loader_kwargs,
        )

        ckpt_name = (
            f"decoymnist_{args.loss_mode}_seed{seed}_"
            f"bestval_{best_val_acc:.2f}_epoch{best_epoch}_{run_ts}.pth"
        )
        ckpt_path = ckpt_dir / ckpt_name
        torch.save(
            {
                "model_state_dict": best_weights,
                "seed": int(seed),
                "best_val_acc": float(best_val_acc),
                "best_epoch": int(best_epoch),
                "test_acc": float(test_acc),
                "args": vars(args),
            },
            ckpt_path,
        )

        rows.append((seed, best_val_acc, test_acc, best_epoch))
        print(
            f"seed={seed} best_val_acc={best_val_acc:.2f}% "
            f"best_epoch={best_epoch} test_acc={test_acc:.2f}% "
            f"checkpoint={ckpt_path}"
        )

    vals = np.asarray([r[1] for r in rows], dtype=np.float64)
    tests = np.asarray([r[2] for r in rows], dtype=np.float64)
    print("\nSummary over seeds")
    print(f"val_acc  mean={vals.mean():.2f}% std={vals.std():.2f}%")
    print(f"test_acc mean={tests.mean():.2f}% std={tests.std():.2f}%")


if __name__ == "__main__":
    main()
