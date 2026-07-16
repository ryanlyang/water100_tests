#!/usr/bin/env python3
"""AFR-style DecoyMNIST run with LeNet and fixed hyperparameters.

Design goals:
- Keep DecoyMNIST LeNet/CDEP training setup for stage-1:
  Adam, lr=1e-3, weight_decay=1e-4, fixed 90/10 train/val split.
- Apply AFR-style stage-2 on frozen features:
  sample weighting with exp(-gamma * p_true) and L2 regularization to the
  stage-1 head weights (reg_coeff).
- Select checkpoint by validation worst-class accuracy, then report test
  metrics at that selected epoch.
"""

from __future__ import annotations

import argparse
import csv
import math
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
        x = F.relu(self.fc1(x))
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward_logits(x), dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if piece:
            out.append(int(piece))
    return out


@torch.no_grad()
def eval_stage1(model: nn.Module, loader: utils.DataLoader, device: torch.device) -> Tuple[float, float]:
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


@torch.no_grad()
def extract_features_logits_labels(
    model: LeNet, loader: utils.DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    feats: List[torch.Tensor] = []
    logits: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for data, target in loader:
        data = data.to(device)
        z = model.forward_features(data)
        logit = model.fc2(z)
        feats.append(z.detach().cpu())
        logits.append(logit.detach().cpu())
        labels.append(target.detach().cpu())
    return torch.cat(feats, dim=0), torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def _class_metrics(
    logits: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> Tuple[float, float, float, List[float]]:
    preds = logits.argmax(dim=1)
    total_acc = 100.0 * (preds.eq(labels).float().mean().item())
    class_accs: List[float] = []
    for cls in range(n_classes):
        mask = labels.eq(cls)
        if mask.any():
            acc = 100.0 * preds[mask].eq(labels[mask]).float().mean().item()
            class_accs.append(acc)
        else:
            class_accs.append(float("nan"))
    finite_accs = [a for a in class_accs if math.isfinite(a)]
    if finite_accs:
        balanced = float(np.mean(finite_accs))
        worst = float(np.min(finite_accs))
    else:
        balanced = float("nan")
        worst = float("nan")
    return total_acc, balanced, worst, class_accs


@torch.no_grad()
def _compute_afr_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    rebalance_classes: bool,
) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    weights = torch.exp(-float(gamma) * p_true)
    if rebalance_classes:
        num = float(len(labels))
        for cls in labels.unique():
            cls = int(cls.item())
            mask = labels.eq(cls)
            cls_count = float(mask.sum().item())
            if cls_count > 0:
                weights[mask] *= num / cls_count
    weights /= weights.sum().clamp_min(1e-12)
    return weights


def train_stage1(
    model: LeNet,
    train_loader: utils.DataLoader,
    val_loader: utils.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    print_every: int,
) -> Dict[str, object]:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, target)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = eval_stage1(model, val_loader, device)
        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        if print_every > 0 and (epoch % print_every == 0 or epoch == epochs):
            print(
                f"[S1] epoch={epoch}/{epochs} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.2f}",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_state": best_state,
    }


def train_stage2_head(
    stage1_model: LeNet,
    train_loader_eval: utils.DataLoader,
    val_loader_eval: utils.DataLoader,
    test_loader_eval: utils.DataLoader,
    device: torch.device,
    stage2_epochs: int,
    stage2_lr: float,
    stage2_weight_decay: float,
    stage2_batch_size: int,
    gamma: float,
    reg_coeff: float,
    rebalance_classes: bool,
    print_every: int,
) -> Dict[str, object]:
    train_feats, train_logits, train_labels = extract_features_logits_labels(
        stage1_model, train_loader_eval, device
    )
    val_feats, _, val_labels = extract_features_logits_labels(stage1_model, val_loader_eval, device)
    test_feats, _, test_labels = extract_features_logits_labels(stage1_model, test_loader_eval, device)

    weights = _compute_afr_weights(
        logits=train_logits,
        labels=train_labels,
        gamma=gamma,
        rebalance_classes=rebalance_classes,
    )
    feat_dim = int(train_feats.shape[1])
    n_classes = int(stage1_model.fc2.out_features)

    head = nn.Linear(feat_dim, n_classes, bias=True).to(device)
    with torch.no_grad():
        head.weight.copy_(stage1_model.fc2.weight.detach())
        head.bias.copy_(stage1_model.fc2.bias.detach())

    init_w = stage1_model.fc2.weight.detach().to(device).clone()
    init_b = stage1_model.fc2.bias.detach().to(device).clone()

    train_ds = utils.TensorDataset(train_feats, train_labels, weights)
    if stage2_batch_size <= 0:
        stage2_batch_size = len(train_ds)
    train_loader = utils.DataLoader(train_ds, batch_size=stage2_batch_size, shuffle=True)

    optimizer = optim.Adam(head.parameters(), lr=stage2_lr, weight_decay=stage2_weight_decay)

    def evaluate_head(split_feats: torch.Tensor, split_labels: torch.Tensor) -> Tuple[float, float, float, List[float]]:
        with torch.no_grad():
            logits = head(split_feats.to(device)).cpu()
        return _class_metrics(logits=logits, labels=split_labels, n_classes=n_classes)

    best = {
        "best_epoch": 0,
        "val_acc": float("nan"),
        "val_balanced_acc": float("nan"),
        "val_worst_class_acc": float("-inf"),
        "val_class_accs": [],
        "test_acc": float("nan"),
        "test_balanced_acc": float("nan"),
        "test_worst_class_acc": float("nan"),
        "test_class_accs": [],
        "best_head_state": deepcopy(head.state_dict()),
    }

    # Epoch 0 baseline (warm-start head before stage-2 updates).
    v_acc, v_bal, v_worst, v_cls = evaluate_head(val_feats, val_labels)
    t_acc, t_bal, t_worst, t_cls = evaluate_head(test_feats, test_labels)
    best.update(
        {
            "val_acc": v_acc,
            "val_balanced_acc": v_bal,
            "val_worst_class_acc": v_worst,
            "val_class_accs": v_cls,
            "test_acc": t_acc,
            "test_balanced_acc": t_bal,
            "test_worst_class_acc": t_worst,
            "test_class_accs": t_cls,
        }
    )

    for epoch in range(1, stage2_epochs + 1):
        head.train()
        for feat_b, y_b, w_b in train_loader:
            feat_b = feat_b.to(device)
            y_b = y_b.to(device)
            w_b = w_b.to(device)

            optimizer.zero_grad()
            logits = head(feat_b)
            ce = F.cross_entropy(logits, y_b, reduction="none")
            weighted_ce = (w_b * ce).sum() / w_b.sum().clamp_min(1e-12)
            reg = (head.weight - init_w).pow(2).sum() + (head.bias - init_b).pow(2).sum()
            loss = weighted_ce + float(reg_coeff) * reg
            loss.backward()
            optimizer.step()

        v_acc, v_bal, v_worst, v_cls = evaluate_head(val_feats, val_labels)
        t_acc, t_bal, t_worst, t_cls = evaluate_head(test_feats, test_labels)

        improved = v_worst > float(best["val_worst_class_acc"])
        if improved:
            best.update(
                {
                    "best_epoch": epoch,
                    "val_acc": v_acc,
                    "val_balanced_acc": v_bal,
                    "val_worst_class_acc": v_worst,
                    "val_class_accs": v_cls,
                    "test_acc": t_acc,
                    "test_balanced_acc": t_bal,
                    "test_worst_class_acc": t_worst,
                    "test_class_accs": t_cls,
                    "best_head_state": deepcopy(head.state_dict()),
                }
            )

        if print_every > 0 and (epoch % print_every == 0 or epoch == stage2_epochs):
            print(
                f"[S2] epoch={epoch}/{stage2_epochs} "
                f"val_acc={v_acc:.2f} val_bal={v_bal:.2f} val_worst_class={v_worst:.2f} "
                f"test_acc={t_acc:.2f}",
                flush=True,
            )

    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AFR-style fixed run on DecoyMNIST with LeNet")
    parser.add_argument("--png-root", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--stage1-epochs", type=int, default=19)
    parser.add_argument("--stage2-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=1000)
    parser.add_argument("--stage2-batch-size", type=int, default=-1)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--reg-coeff", type=float, default=0.0)
    parser.add_argument("--rebalance-classes", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    return parser


def _save_ckpt(path: Path, payload: Dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return str(path)


def run(args: argparse.Namespace) -> None:
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": int(args.num_workers), "pin_memory": use_cuda}

    transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
    png_root = Path(args.png_root).expanduser().resolve()
    full_train = ImageFolder(str(png_root / "train"), transform=transform)
    true_test = ImageFolder(str(png_root / "test"), transform=transform)

    n_total = len(full_train)
    n_val = int(float(args.val_frac) * n_total)
    n_train = n_total - n_val
    split_g = torch.Generator().manual_seed(int(args.split_seed))
    train_subset, val_subset = utils.random_split(full_train, [n_train, n_val], generator=split_g)

    train_loader = utils.DataLoader(
        train_subset, batch_size=int(args.batch_size), shuffle=True, **loader_kwargs
    )
    train_loader_eval = utils.DataLoader(
        train_subset, batch_size=int(args.eval_batch_size), shuffle=False, **loader_kwargs
    )
    val_loader_eval = utils.DataLoader(
        val_subset, batch_size=int(args.eval_batch_size), shuffle=False, **loader_kwargs
    )
    test_loader_eval = utils.DataLoader(
        true_test, batch_size=int(args.eval_batch_size), shuffle=False, **loader_kwargs
    )

    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")

    print("AFR-style DecoyMNIST LeNet fixed run")
    print(f"device={device}")
    print(f"png_root={png_root}")
    print(f"split={n_train}/{n_val} train/val, test={len(true_test)}")
    print(f"seeds={seeds}")
    print(
        f"stage1: epochs={args.stage1_epochs} Adam lr={args.lr} wd={args.weight_decay} "
        "(CDEP setup)"
    )
    print(
        f"stage2: epochs={args.stage2_epochs} Adam lr={args.stage2_lr} wd={args.stage2_weight_decay} "
        f"gamma={args.gamma} reg={args.reg_coeff} rebalance_classes={args.rebalance_classes}"
    )
    print(
        "selection_metric=val_worst_class_acc, "
        "report=test_acc/test_worst_class_acc at selected epoch"
    )

    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        set_seed(seed)
        model = LeNet().to(device)

        s1 = train_stage1(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader_eval,
            device=device,
            epochs=int(args.stage1_epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            print_every=int(args.print_every),
        )
        s2 = train_stage2_head(
            stage1_model=model,
            train_loader_eval=train_loader_eval,
            val_loader_eval=val_loader_eval,
            test_loader_eval=test_loader_eval,
            device=device,
            stage2_epochs=int(args.stage2_epochs),
            stage2_lr=float(args.stage2_lr),
            stage2_weight_decay=float(args.stage2_weight_decay),
            stage2_batch_size=int(args.stage2_batch_size),
            gamma=float(args.gamma),
            reg_coeff=float(args.reg_coeff),
            rebalance_classes=bool(args.rebalance_classes),
            print_every=int(args.print_every),
        )

        stage1_ckpt = ""
        stage2_ckpt = ""
        if save_dir is not None:
            stage1_path = save_dir / f"decoy_afr_seed{seed}_stage1_{run_ts}.pth"
            stage1_payload = {
                "seed": int(seed),
                "stage": "stage1",
                "best_epoch": int(s1["best_epoch"]),
                "best_val_acc": float(s1["best_val_acc"]),
                "model_state_dict": deepcopy(s1["best_state"]),
            }
            stage1_ckpt = _save_ckpt(stage1_path, stage1_payload)

            stage2_model_state = deepcopy(s1["best_state"])
            stage2_model_state["fc2.weight"] = s2["best_head_state"]["weight"].detach().cpu()
            stage2_model_state["fc2.bias"] = s2["best_head_state"]["bias"].detach().cpu()
            stage2_path = save_dir / f"decoy_afr_seed{seed}_stage2_{run_ts}.pth"
            stage2_payload = {
                "seed": int(seed),
                "stage": "stage2",
                "gamma": float(args.gamma),
                "reg_coeff": float(args.reg_coeff),
                "best_epoch": int(s2["best_epoch"]),
                "val_worst_class_acc": float(s2["val_worst_class_acc"]),
                "test_acc_at_val_worst": float(s2["test_acc"]),
                "test_worst_class_acc_at_val_worst": float(s2["test_worst_class_acc"]),
                "head_state_dict": deepcopy(s2["best_head_state"]),
                "model_state_dict": stage2_model_state,
            }
            stage2_ckpt = _save_ckpt(stage2_path, stage2_payload)
            print(f"[CKPT] seed={seed} stage1={stage1_ckpt}")
            print(f"[CKPT] seed={seed} stage2={stage2_ckpt}")

        row = {
            "seed": seed,
            "gamma": float(args.gamma),
            "reg_coeff": float(args.reg_coeff),
            "stage1_best_epoch": int(s1["best_epoch"]),
            "stage1_best_val_acc": float(s1["best_val_acc"]),
            "stage2_best_epoch": int(s2["best_epoch"]),
            "val_acc": float(s2["val_acc"]),
            "val_balanced_acc": float(s2["val_balanced_acc"]),
            "val_worst_class_acc": float(s2["val_worst_class_acc"]),
            "test_acc_at_val_worst": float(s2["test_acc"]),
            "test_balanced_acc_at_val_worst": float(s2["test_balanced_acc"]),
            "test_worst_class_acc_at_val_worst": float(s2["test_worst_class_acc"]),
            "val_class_accs": str([round(x, 4) for x in s2["val_class_accs"]]),
            "test_class_accs": str([round(x, 4) for x in s2["test_class_accs"]]),
            "stage1_checkpoint": stage1_ckpt,
            "stage2_checkpoint": stage2_ckpt,
        }
        rows.append(row)
        print(
            f"[BEST] seed={seed} gamma={args.gamma} reg={args.reg_coeff} "
            f"val_worst_class={row['val_worst_class_acc']:.4f} "
            f"test_acc@val={row['test_acc_at_val_worst']:.4f} "
            f"test_worst_class@val={row['test_worst_class_acc_at_val_worst']:.4f}"
        )

    test_accs = np.asarray([float(r["test_acc_at_val_worst"]) for r in rows], dtype=np.float64)
    test_worsts = np.asarray(
        [float(r["test_worst_class_acc_at_val_worst"]) for r in rows], dtype=np.float64
    )
    print("\n===== Summary (across seeds) =====")
    print(f"test_acc@val_worst:        {test_accs.mean():.4f} +/- {test_accs.std():.4f}")
    print(f"test_worst_class@val_worst:{test_worsts.mean():.4f} +/- {test_worsts.std():.4f}")
    print(
        "Note: DecoyMNIST worst metric here is worst-class accuracy "
        "(no explicit spurious-group column provided)."
    )

    if args.output_csv:
        out_csv = Path(args.output_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        header = list(rows[0].keys()) if rows else []
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_csv}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
