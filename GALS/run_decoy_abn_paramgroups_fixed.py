#!/usr/bin/env python3
"""ABN-style DecoyMNIST CNN with CDEP-style optimizer/split setup.

Same behavior as the MakeMNIST runner, plus optional best-checkpoint saving.
"""

from __future__ import print_function

import argparse
import os
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor


class ABNNet(nn.Module):
    def __init__(self):
        super(ABNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        # Attention branch over conv2 activations (8x8 feature map).
        self.att_conv = nn.Conv2d(50, 1, kernel_size=1, stride=1)
        self.abn_fc = nn.Linear(50, 10)

        # Main classifier branch.
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        feat = F.relu(self.conv2(x))

        att = torch.sigmoid(self.att_conv(feat))
        feat_att = feat * (1.0 + att)

        x_main = F.max_pool2d(feat_att, 2, 2)
        x_main = x_main.view(-1, 4 * 4 * 50)
        x_main = F.relu(self.fc1(x_main))
        logits_main = self.fc2(x_main)

        # ABN auxiliary classification on attention-weighted global pooled features.
        feat_aux = (feat * att).mean(dim=(2, 3))
        logits_aux = self.abn_fc(feat_aux)

        return F.log_softmax(logits_main, dim=1), logits_aux


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
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
        out_main, _ = model(data)
        pred = out_main.argmax(dim=1)
        loss_sum += F.nll_loss(out_main, target, reduction="sum").item()
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


def train_one_seed(args, seed, full_train, test_dataset, device, loader_kwargs):
    set_seed(seed)

    # Keep split deterministic/fixed across seeds (CDEP-style behavior).
    split_g = torch.Generator().manual_seed(0)
    n_total = len(full_train)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val
    train_subset, val_subset = utils.random_split(full_train, [n_train, n_val], generator=split_g)

    train_loader = utils.DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = utils.DataLoader(
        val_subset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = utils.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs
    )

    model = ABNNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            out_main, logits_aux = model(data)
            loss_main = F.nll_loss(out_main, target)
            loss_aux = F.cross_entropy(logits_aux, target)
            loss = loss_main + args.abn_cls_weight * loss_aux
            loss.backward()
            optimizer.step()

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

    assert best_weights is not None
    model.load_state_dict(best_weights)
    _, test_acc, test_class_acc, test_worst_class_acc = evaluate_with_class_stats(
        model, test_loader, device, num_classes=10
    )
    return best_val_acc, test_acc, test_worst_class_acc, test_class_acc, best_epoch, best_weights


def _save_ckpt(
    save_dir,
    seed,
    best_epoch,
    best_val_acc,
    test_acc,
    test_worst_class_acc,
    abn_cls_weight,
    state_dict,
):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(
        save_dir,
        f"decoy_abn_seed{seed}_bestval_{best_val_acc:.2f}_test_{test_acc:.2f}_epoch{best_epoch}_{ts}.pth",
    )
    payload = {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_worst_class_acc": float(test_worst_class_acc),
        "abn_cls_weight": float(abn_cls_weight),
        "state_dict": state_dict,
    }
    torch.save(payload, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="ABN DecoyMNIST CNN with CDEP-style optimizer/split")
    parser.add_argument("--png-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--abn-cls-weight", type=float, default=3.2547104257357056)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    png_root = args.png_root or os.path.join(repo_root, "MakeMNIST", "data", "DecoyMNIST_png")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}

    transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
    full_train = ImageFolder(os.path.join(png_root, "train"), transform=transform)
    test_dataset = ImageFolder(os.path.join(png_root, "test"), transform=transform)

    print("Running ABN DecoyMNIST with CDEP-style optimizer/split")
    print(f"device={device}")
    print(f"png_root={png_root}")
    print(f"train={len(full_train)} test={len(test_dataset)} split={1.0 - args.val_frac:.2f}/{args.val_frac:.2f}")
    print(
        f"optimizer=Adam lr={args.lr} weight_decay={args.weight_decay} "
        f"abn_cls_weight={args.abn_cls_weight}"
    )

    rows = []
    for i in range(args.n_seeds):
        seed = args.seed_start + i
        best_val_acc, test_acc, test_worst_class_acc, test_class_acc, best_epoch, best_weights = train_one_seed(
            args=args,
            seed=seed,
            full_train=full_train,
            test_dataset=test_dataset,
            device=device,
            loader_kwargs=loader_kwargs,
        )
        rows.append((seed, best_val_acc, test_acc, test_worst_class_acc, best_epoch))
        print(
            f"seed={seed} best_val_acc={best_val_acc:.2f}% "
            f"best_epoch={best_epoch} test_acc={test_acc:.2f}% "
            f"test_worst_class_acc={test_worst_class_acc:.2f}% "
            f"test_class_acc={np.array2string(test_class_acc, precision=2, separator=',')}"
        )
        if args.save_dir:
            ckpt_path = _save_ckpt(
                save_dir=args.save_dir,
                seed=seed,
                best_epoch=best_epoch,
                best_val_acc=best_val_acc,
                test_acc=test_acc,
                test_worst_class_acc=test_worst_class_acc,
                abn_cls_weight=args.abn_cls_weight,
                state_dict=best_weights,
            )
            print(f"[CKPT] seed={seed} path={ckpt_path}")

    vals = np.asarray([r[1] for r in rows], dtype=np.float64)
    tests = np.asarray([r[2] for r in rows], dtype=np.float64)
    test_worsts = np.asarray([r[3] for r in rows], dtype=np.float64)
    print("\nSummary over seeds")
    print(f"val_acc  mean={vals.mean():.2f}% std={vals.std():.2f}%")
    print(f"test_acc mean={tests.mean():.2f}% std={tests.std():.2f}%")
    print(f"test_worst_class_acc mean={test_worsts.mean():.2f}% std={test_worsts.std():.2f}%")


if __name__ == "__main__":
    main()
