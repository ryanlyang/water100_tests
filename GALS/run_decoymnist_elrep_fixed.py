#!/usr/bin/env python3
"""Fixed-hyperparameter ElRep-style DecoyMNIST LeNet run.

This keeps the DecoyMNIST setup used by the other LeNet baselines:
- 28x28 grayscale inputs normalized to [-1, 1]
- fixed 90/10 train/val split
- Adam with lr=1e-3 and weight_decay=1e-4 by default
- validation-accuracy checkpoint selection

ElRep is applied as a nuclear + Frobenius penalty on the 256-d
penultimate LeNet representation before the final classifier.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        return F.relu(self.fc1(x))

    def forward_logits_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        logits = self.fc2(features)
        return logits, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_logits_features(x)
        return F.log_softmax(logits, dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def elrep_penalty(features: torch.Tensor, theta1: float, theta2: float) -> torch.Tensor:
    if theta1 == 0.0 and theta2 == 0.0:
        return features.new_tensor(0.0)
    singular_values = torch.linalg.svdvals(features.float())
    batch_size = max(int(features.shape[0]), 1)
    nuclear = float(theta1) * singular_values.abs().sum() / batch_size
    frobenius = float(theta2) * singular_values.square().sum() / batch_size
    return nuclear + frobenius


@torch.no_grad()
def evaluate_with_class_stats(
    model: nn.Module,
    loader: utils.DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Tuple[float, float, float, float, np.ndarray]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    class_total = np.zeros((num_classes,), dtype=np.int64)
    class_correct = np.zeros((num_classes,), dtype=np.int64)

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        log_probs = model(data)
        loss_sum += F.nll_loss(log_probs, target, reduction="sum").item()
        pred = log_probs.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

        for cls in range(num_classes):
            mask = target.eq(cls)
            n = int(mask.sum().item())
            if n > 0:
                class_total[cls] += n
                class_correct[cls] += int(pred[mask].eq(target[mask]).sum().item())

    class_acc = np.full((num_classes,), np.nan, dtype=np.float64)
    for cls in range(num_classes):
        if class_total[cls] > 0:
            class_acc[cls] = 100.0 * float(class_correct[cls]) / float(class_total[cls])

    avg_loss = loss_sum / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    balanced = float(np.nanmean(class_acc))
    worst = float(np.nanmin(class_acc))
    return avg_loss, acc, balanced, worst, class_acc


def _make_loaders(args, full_train, test_dataset, device: torch.device):
    use_cuda = device.type == "cuda"
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}

    split_g = torch.Generator().manual_seed(int(args.split_seed))
    n_total = len(full_train)
    n_val = int(float(args.val_frac) * n_total)
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
    return train_loader, val_loader, test_loader, n_train, n_val


def train_one_seed(
    args,
    seed: int,
    full_train: ImageFolder,
    test_dataset: ImageFolder,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(seed)
    train_loader, val_loader, test_loader, _, _ = _make_loaders(args, full_train, test_dataset, device)

    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_ce = 0.0
        running_elrep = 0.0
        running_total = 0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits, features = model.forward_logits_features(data)
            ce_loss = F.cross_entropy(logits, target)
            rep_loss = elrep_penalty(features, args.theta1, args.theta2)
            loss = ce_loss + rep_loss
            loss.backward()
            optimizer.step()

            running_ce += ce_loss.item() * data.size(0)
            running_elrep += rep_loss.item() * data.size(0)
            running_total += data.size(0)

        val_loss, val_acc, _, _, _ = evaluate_with_class_stats(model, val_loader, device)
        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        if args.print_every > 0 and (epoch % args.print_every == 0 or epoch == args.epochs):
            avg_ce = running_ce / max(running_total, 1)
            avg_elrep = running_elrep / max(running_total, 1)
            print(
                f"seed={seed} epoch={epoch}/{args.epochs} "
                f"train_ce={avg_ce:.4f} train_elrep={avg_elrep:.6f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("No best checkpoint selected.")

    model.load_state_dict(best_state)
    _, test_acc, test_balanced, test_worst, test_class_accs = evaluate_with_class_stats(
        model, test_loader, device
    )

    return {
        "seed": int(seed),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "test_acc": float(test_acc),
        "test_balanced_class_acc": float(test_balanced),
        "test_worst_class_acc": float(test_worst),
        "test_class_accs": test_class_accs,
        "state_dict": best_state,
    }


def _save_ckpt(save_dir: str, row: Dict[str, object], args) -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(save_dir) / (
        f"decoy_elrep_seed{int(row['seed'])}_bestval_{float(row['best_val_acc']):.2f}_"
        f"test_{float(row['test_acc']):.2f}_epoch{int(row['best_epoch'])}_{ts}.pth"
    )
    payload = {
        "seed": int(row["seed"]),
        "best_epoch": int(row["best_epoch"]),
        "best_val_acc": float(row["best_val_acc"]),
        "best_val_loss": float(row["best_val_loss"]),
        "test_acc": float(row["test_acc"]),
        "test_balanced_class_acc": float(row["test_balanced_class_acc"]),
        "test_worst_class_acc": float(row["test_worst_class_acc"]),
        "theta1": float(args.theta1),
        "theta2": float(args.theta2),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "state_dict": row["state_dict"],
    }
    torch.save(payload, str(path))
    return str(path)


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header = [
        "seed",
        "theta1",
        "theta2",
        "lr",
        "weight_decay",
        "best_epoch",
        "best_val_acc",
        "test_acc",
        "test_balanced_class_acc",
        "test_worst_class_acc",
        "test_class_accs",
        "checkpoint",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed ElRep DecoyMNIST LeNet run")
    parser.add_argument("--png-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--theta1", type=float, default=4.1741182912970346e-05)
    parser.add_argument("--theta2", type=float, default=2.5433427234421545e-06)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parent
    png_root = Path(args.png_root) if args.png_root else repo_root / "MakeMNIST" / "data" / "DecoyMNIST_png"

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
    full_train = ImageFolder(str(png_root / "train"), transform=transform)
    test_dataset = ImageFolder(str(png_root / "test"), transform=transform)

    n_total = len(full_train)
    n_val = int(float(args.val_frac) * n_total)
    n_train = n_total - n_val

    print("Running ElRep DecoyMNIST with CDEP-style LeNet setup")
    print(f"device={device}")
    print(f"png_root={png_root}")
    print(f"train={len(full_train)} test={len(test_dataset)} split={n_train}/{n_val}")
    print(
        f"optimizer=Adam lr={args.lr} weight_decay={args.weight_decay} "
        f"theta1={args.theta1} theta2={args.theta2}"
    )
    print("selection_metric=val_acc, report=test_acc/test_worst_class_acc")

    rows: List[Dict[str, object]] = []
    for i in range(int(args.n_seeds)):
        seed = int(args.seed_start) + i
        result = train_one_seed(args, seed, full_train, test_dataset, device)
        ckpt_path = ""
        if args.save_dir:
            ckpt_path = _save_ckpt(args.save_dir, result, args)
            print(f"[CKPT] seed={seed} path={ckpt_path}", flush=True)

        row = {
            "seed": seed,
            "theta1": float(args.theta1),
            "theta2": float(args.theta2),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "best_epoch": int(result["best_epoch"]),
            "best_val_acc": float(result["best_val_acc"]),
            "test_acc": float(result["test_acc"]),
            "test_balanced_class_acc": float(result["test_balanced_class_acc"]),
            "test_worst_class_acc": float(result["test_worst_class_acc"]),
            "test_class_accs": np.array2string(
                result["test_class_accs"], precision=2, separator=","
            ),
            "checkpoint": ckpt_path,
        }
        rows.append(row)
        print(
            f"seed={seed} best_val_acc={row['best_val_acc']:.2f}% "
            f"best_epoch={row['best_epoch']} test_acc={row['test_acc']:.2f}% "
            f"test_balanced_class_acc={row['test_balanced_class_acc']:.2f}% "
            f"test_worst_class_acc={row['test_worst_class_acc']:.2f}% "
            f"test_class_acc={row['test_class_accs']}",
            flush=True,
        )

    _write_csv(args.output_csv, rows)
    if args.output_csv:
        print(f"[CSV] wrote {args.output_csv}")

    vals = np.asarray([float(r["best_val_acc"]) for r in rows], dtype=np.float64)
    tests = np.asarray([float(r["test_acc"]) for r in rows], dtype=np.float64)
    test_bals = np.asarray([float(r["test_balanced_class_acc"]) for r in rows], dtype=np.float64)
    test_worsts = np.asarray([float(r["test_worst_class_acc"]) for r in rows], dtype=np.float64)

    print("\nSummary over seeds")
    print(f"val_acc                 mean={vals.mean():.2f}% std={vals.std():.2f}%")
    print(f"test_acc                mean={tests.mean():.2f}% std={tests.std():.2f}%")
    print(f"test_balanced_class_acc mean={test_bals.mean():.2f}% std={test_bals.std():.2f}%")
    print(f"test_worst_class_acc    mean={test_worsts.mean():.2f}% std={test_worsts.std():.2f}%")


if __name__ == "__main__":
    main()
