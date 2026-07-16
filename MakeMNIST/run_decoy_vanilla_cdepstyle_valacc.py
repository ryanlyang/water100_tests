#!/usr/bin/env python3
"""CDEP-style DecoyMNIST vanilla run, but model selection by internal val accuracy.

This mirrors the old Decoy vanilla setup:
- same LeNet-style architecture
- same Decoy transform (grayscale, scale to [-1, 1])
- same 90/10 split from train set with fixed split seed=0
- Adam optimizer with weight_decay=1e-4

Difference from old script:
- selects best checkpoint by internal validation accuracy (instead of val loss).
"""

from __future__ import print_function

import argparse
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        loss_sum += F.nll_loss(out, target, reduction="sum").item()
        pred = out.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def train_one_seed(args, seed, full_train, true_test, device, loader_kwargs):
    set_seed(seed)

    # Keep split deterministic/fixed across seeds (CDEP-style behavior).
    split_g = torch.Generator().manual_seed(0)
    n_total = len(full_train)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    train_subset, val_subset = utils.random_split(full_train, [n_train, n_val], generator=split_g)

    train_loader = utils.DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = utils.DataLoader(
        val_subset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = utils.DataLoader(
        true_test, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs
    )

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_weights = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, target)
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

    model.load_state_dict(best_weights)
    test_loss, test_acc = evaluate(model, test_loader, device)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Decoy vanilla (CDEP-style) selected by val accuracy")
    parser.add_argument("--png-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    png_root = args.png_root or os.path.join(repo_root, "data", "DecoyMNIST_png")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}

    transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
    full_train = ImageFolder(os.path.join(png_root, "train"), transform=transform)
    true_test = ImageFolder(os.path.join(png_root, "test"), transform=transform)

    print("Running Decoy vanilla (CDEP-style, val-acc selector)")
    print(f"device={device}")
    print(f"png_root={png_root}")
    print(f"train={len(full_train)} test={len(true_test)} split=90/10")
    print(
        f"optimizer=Adam lr={args.lr} weight_decay={args.weight_decay} "
        f"epochs={args.epochs}"
    )

    rows = []
    for i in range(args.n_seeds):
        seed = args.seed_start + i
        row = train_one_seed(
            args=args,
            seed=seed,
            full_train=full_train,
            true_test=true_test,
            device=device,
            loader_kwargs=loader_kwargs,
        )
        rows.append(row)
        print(
            f"seed={seed} best_epoch={row['best_epoch']} "
            f"best_val_acc={row['best_val_acc']:.2f}% test_acc={row['test_acc']:.2f}%"
        )

    val_accs = np.asarray([r["best_val_acc"] for r in rows], dtype=np.float64)
    test_accs = np.asarray([r["test_acc"] for r in rows], dtype=np.float64)
    print("\nSummary over seeds")
    print(f"best_val_acc mean={val_accs.mean():.2f}% std={val_accs.std():.2f}%")
    print(f"test_acc     mean={test_accs.mean():.2f}% std={test_accs.std():.2f}%")


if __name__ == "__main__":
    main()
