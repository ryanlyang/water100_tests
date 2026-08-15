#!/usr/bin/env python3
"""DecoyMNIST guided run with CDEP-style vanilla backbone/training setup.

Key behavior:
- Same core setup as CDEP-style vanilla runner:
  - LeNet conv/fc architecture
  - grayscale input in [-1, 1]
  - Adam, single LR
  - fixed 90/10 train/val split
  - select by best val accuracy
- Adds guided training stage:
  - epoch < attention_epoch: CE only
  - epoch >= attention_epoch: CE + alignment_weight * alignment_loss(GradCAM, Mask)
- Optionally saves every epoch for checkpoint-selection diagnostics.
"""

from __future__ import print_function

import argparse
import os
import random
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from PIL import Image
from torchvision.datasets import ImageFolder

from alignment_losses import (
    ALIGNMENT_LOSSES,
    normalize_alignment_loss_name,
    spatial_alignment_loss,
)
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        return F.log_softmax(self.logits(x), dim=1)


class GradCAMWrap(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.features = None
        self.gradients = None
        self.base.conv2.register_forward_hook(self._fwd_hook)
        self.base.conv2.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.features = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def grad_cam(self):
        weights = self.gradients.mean(dim=(2, 3))
        cams = torch.einsum("bc,bchw->bhw", weights, self.features)
        cams = torch.relu(cams)
        flat = cams.view(cams.size(0), -1)
        mn, _ = flat.min(dim=1, keepdim=True)
        mx, _ = flat.max(dim=1, keepdim=True)
        return ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)

    def forward(self, x):
        return self.base(x)


class ExpandWhite:
    def __init__(self, thr=10, radius=3):
        self.thr = thr
        self.radius = radius

    def __call__(self, mask):
        arr = np.array(mask)
        white = (arr > self.thr).astype(np.uint8)
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * self.radius + 1, 2 * self.radius + 1)
        )
        dil = cv2.dilate(white, k, iterations=1)
        return Image.fromarray((dil * 255).astype(np.uint8))


class EdgeExtract:
    def __init__(self, thr=10, edge_width=1):
        self.thr = thr
        self.edge_width = edge_width

    def __call__(self, mask):
        arr = np.array(mask)
        white = (arr > self.thr).astype(np.uint8)
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * self.edge_width + 1, 2 * self.edge_width + 1)
        )
        edge = cv2.morphologyEx(white, cv2.MORPH_GRADIENT, k)
        return Image.fromarray((edge * 255).astype(np.uint8))


class Brighten:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, mask):
        return torch.clamp(mask * self.factor, 0.0, 1.0)


class GuidedImageFolder(utils.Dataset):
    def __init__(self, image_root, mask_root, image_transform=None, mask_transform=None):
        self.images = ImageFolder(image_root, transform=image_transform)
        self.mask_root = mask_root
        self.mask_transform = mask_transform
        self._mask_exts = (".png", ".jpg", ".jpeg")

    def _resolve_mask_path(self, base, class_name):
        candidates = [f"{class_name}_{base}", base]
        if "_lbl" in base:
            candidates.append(base.split("_lbl")[0])
            candidates.append(re.sub(r"_lbl\d+$", "", base))
            candidates.append(re.sub(r"_lbl\d+", "", base))

        for stem in candidates:
            for ext in self._mask_exts:
                p = os.path.join(self.mask_root, stem + ext)
                if os.path.exists(p):
                    return p
        tried = [os.path.join(self.mask_root, stem + ext) for stem in candidates for ext in self._mask_exts]
        raise FileNotFoundError(
            f"Mask not found for base='{base}', class='{class_name}'. Tried: {tried}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx]
        path, _ = self.images.samples[idx]
        base = os.path.splitext(os.path.basename(path))[0]
        class_name = os.path.basename(os.path.dirname(path))
        mask_path = self._resolve_mask_path(base, class_name)
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, label, mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_attn_loss(cams, gt_masks, alignment_loss="forward_kl"):
    return spatial_alignment_loss(cams, gt_masks, loss_name=alignment_loss)


@torch.no_grad()
def evaluate_classification(model, loader, device):
    avg_loss, acc, _, _ = evaluate_with_class_stats(model, loader, device, num_classes=10)
    return avg_loss, acc


@torch.no_grad()
def evaluate_with_class_stats(model, loader, device, num_classes=10):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    per_class_total = np.zeros((num_classes,), dtype=np.int64)
    per_class_correct = np.zeros((num_classes,), dtype=np.int64)
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        loss_sum += F.nll_loss(out, target, reduction="sum").item()
        pred = out.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        for c in range(num_classes):
            mask = target.eq(c)
            n = int(mask.sum().item())
            if n > 0:
                per_class_total[c] += n
                per_class_correct[c] += int(pred[mask].eq(target[mask]).sum().item())
        total += data.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    class_acc = np.full((num_classes,), np.nan, dtype=np.float64)
    for c in range(num_classes):
        if per_class_total[c] > 0:
            class_acc[c] = 100.0 * float(per_class_correct[c]) / float(per_class_total[c])
    worst_class_acc = float(np.nanmin(class_acc))
    return avg_loss, acc, class_acc, worst_class_acc


def fixed_split_indices(n_total, val_frac, split_seed):
    n_val = int(val_frac * n_total)
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def train_one_seed(args, seed, full_train_guided, full_train_plain, true_test, device, loader_kwargs):
    set_seed(seed)
    alignment_loss = normalize_alignment_loss_name(
        getattr(args, "alignment_loss", "forward_kl")
    )

    n_total = len(full_train_plain)
    train_idx, val_idx = fixed_split_indices(n_total, args.val_frac, args.split_seed)

    train_subset = utils.Subset(full_train_guided, train_idx)
    val_subset = utils.Subset(full_train_plain, val_idx)

    train_loader = utils.DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = utils.DataLoader(
        val_subset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = utils.DataLoader(
        true_test, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs
    )

    model = GradCAMWrap(Net()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_weights = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    kl_lambda_real = args.kl_lambda

    for epoch in range(1, args.epochs + 1):
        attention_active = (epoch >= args.attention_epoch) and (args.kl_lambda > 0.0)
        if epoch > args.attention_epoch and args.kl_lambda > 0.0:
            kl_lambda_real += args.kl_incr

        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0
        for data, target, gt_masks in train_loader:
            data = data.to(device)
            target = target.to(device)
            gt_masks = gt_masks.to(device)

            if not attention_active:
                optimizer.zero_grad()
                out = model(data)
                ce_loss = F.nll_loss(out, target)
                ce_loss.backward()
                optimizer.step()
                loss = ce_loss
            else:
                model.zero_grad()
                logits = model.base.logits(data)
                class_scores = logits[torch.arange(target.size(0), device=device), target]
                class_scores.sum().backward(retain_graph=True)
                sal = model.grad_cam()
                gt_small = F.interpolate(gt_masks, size=sal.shape[1:], mode="nearest").squeeze(1)
                attn_loss = compute_attn_loss(sal, gt_small, alignment_loss)

                optimizer.zero_grad()
                out = model(data)
                ce_loss = F.nll_loss(out, target)
                loss = ce_loss + kl_lambda_real * attn_loss
                loss.backward()
                optimizer.step()

            train_loss_sum += loss.item() * data.size(0)
            train_correct += out.argmax(dim=1).eq(target).sum().item()
            train_total += data.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)
        val_loss, val_acc = evaluate_classification(model, val_loader, device)

        if epoch >= args.attention_epoch:
            improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                best_weights = deepcopy(model.state_dict())

        if args.epoch_checkpoint_dir:
            epoch_dir = Path(args.epoch_checkpoint_dir).expanduser().resolve()
            epoch_dir.mkdir(parents=True, exist_ok=True)
            epoch_path = epoch_dir / f"decoy_r4rr_seed{seed}_epoch{epoch:02d}.pth"
            temporary = epoch_path.with_name(f".{epoch_path.name}.tmp")
            torch.save(
                {
                    "model_state_dict": deepcopy(model.state_dict()),
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "attention_active": bool(attention_active),
                    "effective_kl_lambda": float(kl_lambda_real if attention_active else 0.0),
                    "args": vars(args),
                },
                temporary,
            )
            os.replace(str(temporary), str(epoch_path))
            print(f"[EPOCH CKPT] seed={seed} epoch={epoch} path={epoch_path}")

        if args.print_every > 0 and (epoch % args.print_every == 0 or epoch == args.epochs):
            print(
                f"seed={seed} epoch={epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
                f"guided={'on' if attention_active else 'off'}"
            )

    if best_weights is None:
        # Safety fallback when attention_epoch > epochs.
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_epoch = args.epochs
        best_weights = deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    test_loss, test_acc, test_class_acc, test_worst_class_acc = evaluate_with_class_stats(
        model, test_loader, device, num_classes=10
    )
    test_balanced_class_acc = float(np.nanmean(test_class_acc))
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_balanced_class_acc": test_balanced_class_acc,
        "test_worst_class_acc": test_worst_class_acc,
        "test_class_acc": test_class_acc,
        "test_loss": test_loss,
        "state_dict": best_weights,
    }


def main():
    parser = argparse.ArgumentParser(description="Decoy guided run with CDEP-style single-LR backbone")
    parser.add_argument("--png-root", type=str, default=None)
    parser.add_argument("--teacher-map-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.031210590691245817)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--attention-epoch", type=int, default=7)
    parser.add_argument("--kl-lambda", type=float, default=495.60509512105125)
    parser.add_argument("--kl-incr", type=float, default=0.0)
    parser.add_argument(
        "--alignment-loss",
        choices=ALIGNMENT_LOSSES,
        default="forward_kl",
        help="Spatial teacher/student alignment objective.",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Optional directory for one best-validation checkpoint per seed.",
    )
    parser.add_argument(
        "--epoch-checkpoint-dir",
        type=str,
        default="",
        help="Optional directory for checkpoints from every completed epoch.",
    )
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()

    repro_root = Path(__file__).resolve().parents[2]
    default_png_root = repro_root / "third_party" / "CDEP" / "data" / "DecoyMNIST_png"
    png_root = args.png_root or str(default_png_root.resolve())

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}

    image_transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
    mask_transform = transforms.Compose(
        [
            ExpandWhite(thr=10, radius=3),
            EdgeExtract(thr=10, edge_width=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            Brighten(8.0),
        ]
    )

    full_train_guided = GuidedImageFolder(
        image_root=os.path.join(png_root, "train"),
        mask_root=args.teacher_map_path,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    full_train_plain = ImageFolder(os.path.join(png_root, "train"), transform=image_transform)
    true_test = ImageFolder(os.path.join(png_root, "test"), transform=image_transform)

    print("Running Decoy guided (CDEP-style single-LR backbone, val-acc selector)")
    print(f"device={device}")
    print(f"png_root={png_root}")
    print(f"teacher_map_path={args.teacher_map_path}")
    print(f"train={len(full_train_plain)} test={len(true_test)} split={int((1-args.val_frac)*100)}/{int(args.val_frac*100)}")
    print(
        f"optimizer=Adam lr={args.lr} weight_decay={args.weight_decay} epochs={args.epochs} "
        f"attention_epoch={args.attention_epoch} alignment_loss={args.alignment_loss} "
        f"alignment_weight={args.kl_lambda} kl_incr={args.kl_incr}"
    )

    rows = []
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(args.n_seeds):
        seed = args.seed_start + i
        row = train_one_seed(
            args=args,
            seed=seed,
            full_train_guided=full_train_guided,
            full_train_plain=full_train_plain,
            true_test=true_test,
            device=device,
            loader_kwargs=loader_kwargs,
        )
        rows.append(row)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / (
                f"decoy_r4rr_seed{seed}_bestval_{row['best_val_acc']:.2f}_"
                f"epoch{row['best_epoch']}_{run_ts}.pth"
            )
            torch.save(
                {
                    "model_state_dict": row["state_dict"],
                    "seed": int(seed),
                    "best_epoch": int(row["best_epoch"]),
                    "best_val_acc": float(row["best_val_acc"]),
                    "best_val_loss": float(row["best_val_loss"]),
                    "test_acc": float(row["test_acc"]),
                    "test_balanced_class_acc": float(row["test_balanced_class_acc"]),
                    "test_worst_class_acc": float(row["test_worst_class_acc"]),
                    "test_class_acc": [
                        float(x)
                        for x in np.asarray(row["test_class_acc"], dtype=np.float64).tolist()
                    ],
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[CKPT] seed={seed} path={ckpt_path}")
        print(
            f"seed={seed} best_epoch={row['best_epoch']} "
            f"best_val_acc={row['best_val_acc']:.2f}% test_acc={row['test_acc']:.2f}% "
            f"test_balanced_class_acc={row['test_balanced_class_acc']:.2f}% "
            f"test_worst_class_acc={row['test_worst_class_acc']:.2f}% "
            f"test_class_acc={np.array2string(row['test_class_acc'], precision=2, separator=',')}"
        )

    val_accs = np.asarray([r["best_val_acc"] for r in rows], dtype=np.float64)
    test_accs = np.asarray([r["test_acc"] for r in rows], dtype=np.float64)
    test_bals = np.asarray([r["test_balanced_class_acc"] for r in rows], dtype=np.float64)
    test_worsts = np.asarray([r["test_worst_class_acc"] for r in rows], dtype=np.float64)
    print("\nSummary over seeds")
    print(f"best_val_acc mean={val_accs.mean():.2f}% std={val_accs.std():.2f}%")
    print(f"test_acc     mean={test_accs.mean():.2f}% std={test_accs.std():.2f}%")
    print(f"test_balanced_class_acc mean={test_bals.mean():.2f}% std={test_bals.std():.2f}%")
    print(f"test_worst_class_acc mean={test_worsts.mean():.2f}% std={test_worsts.std():.2f}%")


if __name__ == "__main__":
    main()
