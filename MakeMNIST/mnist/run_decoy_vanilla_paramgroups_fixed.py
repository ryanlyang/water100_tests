#!/usr/bin/env python3
"""Vanilla DecoyMNIST CNN with two LR parameter groups and fixed hyperparameters.

Runs N seeds and reports mean/std of:
- best validation accuracy
- test accuracy at the best-validation checkpoint
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


def get_param_groups(model, base_lr, classifier_lr):
    base_params = []
    classifier_params = []
    for name, p in model.named_parameters():
        if name.startswith("fc2.") or ".fc2." in name:
            classifier_params.append(p)
        else:
            base_params.append(p)
    return [
        {"params": base_params, "lr": base_lr},
        {"params": classifier_params, "lr": classifier_lr},
    ]


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
        correct += out.argmax(dim=1).eq(target).sum().item()
        total += data.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def train_one_seed(args, seed, full_train, test_dataset, device, loader_kwargs):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)

    n_total = len(full_train)
    n_val = max(1, int(args.val_frac * n_total))
    n_train = n_total - n_val
    train_subset, val_subset = utils.random_split(full_train, [n_train, n_val], generator=g)

    train_loader = utils.DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = utils.DataLoader(
        val_subset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = utils.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs
    )

    model = Net().to(device)
    optimizer = optim.SGD(
        get_param_groups(model, args.base_lr, args.classifier_lr),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    best_weights = None
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

        _, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())

        if args.print_every > 0 and (epoch % args.print_every == 0 or epoch == args.epochs):
            print(f"seed={seed} epoch={epoch}/{args.epochs} val_acc={val_acc:.2f}%")

    model.load_state_dict(best_weights)
    _, test_acc = evaluate(model, test_loader, device)
    return best_val_acc, test_acc, best_epoch


def main():
    parser = argparse.ArgumentParser(description="Vanilla DecoyMNIST CNN with base/classifier LR groups")
    parser.add_argument("--png-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--val-frac", type=float, default=0.16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-lr", type=float, default=0.031210590691245817)
    parser.add_argument("--classifier-lr", type=float, default=0.0008517287145349147)
    parser.add_argument("--momentum", type=float, default=0.8914661939990524)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=5)
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
    test_dataset = ImageFolder(os.path.join(png_root, "test"), transform=transform)

    print("Running vanilla DecoyMNIST with fixed hyperparameters")
    print(f"device={device}")
    print(f"png_root={png_root}")
    print(f"train={len(full_train)} test={len(test_dataset)} val_frac={args.val_frac}")
    print(
        f"base_lr={args.base_lr} classifier_lr={args.classifier_lr} "
        f"momentum={args.momentum} weight_decay={args.weight_decay}"
    )

    rows = []
    for i in range(args.n_seeds):
        seed = args.seed_start + i
        best_val_acc, test_acc, best_epoch = train_one_seed(
            args=args,
            seed=seed,
            full_train=full_train,
            test_dataset=test_dataset,
            device=device,
            loader_kwargs=loader_kwargs,
        )
        rows.append((seed, best_val_acc, test_acc, best_epoch))
        print(
            f"seed={seed} best_val_acc={best_val_acc:.2f}% "
            f"best_epoch={best_epoch} test_acc={test_acc:.2f}%"
        )

    vals = np.asarray([r[1] for r in rows], dtype=np.float64)
    tests = np.asarray([r[2] for r in rows], dtype=np.float64)
    print("\nSummary over seeds")
    print(f"val_acc  mean={vals.mean():.2f}% std={vals.std():.2f}%")
    print(f"test_acc mean={tests.mean():.2f}% std={tests.std():.2f}%")


if __name__ == "__main__":
    main()
