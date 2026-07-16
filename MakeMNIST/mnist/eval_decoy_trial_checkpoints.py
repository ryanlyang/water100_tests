#!/usr/bin/env python3
"""Evaluate saved DecoyMNIST trial checkpoints and export trial_eval.csv."""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))


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

    def forward(self, x):
        return self.base(x)


def make_model(device):
    return GradCAMWrap(Net()).to(device)


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return 100.0 * correct / max(total, 1)


def load_state_dict_from_ckpt(path, device):
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint payload type: {type(payload)} at {path}")


def parse_float(row, key):
    val = row.get(key, "")
    if val in ("", None):
        return np.nan
    return float(val)


def parse_int(row, key):
    val = row.get(key, "")
    if val in ("", None):
        return -1
    return int(float(val))


def main():
    parser = argparse.ArgumentParser(description="Evaluate DecoyMNIST per-trial checkpoints.")
    parser.add_argument("--summary-csv", required=True, help="CSV from run_decoy_param_optuna.py trial logging.")
    parser.add_argument("--png-root", default=os.path.join(REPO_ROOT, "data", "DecoyMNIST_png"))
    parser.add_argument("--output-csv", default=None, help="Output CSV path (default: alongside summary CSV).")
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--skip-missing", action="store_true", default=False,
                        help="Skip rows with missing checkpoints instead of failing.")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(os.path.dirname(args.summary_csv), "trial_eval.csv")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": True} if use_cuda else {}

    transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
    test_dataset = ImageFolder(os.path.join(args.png_root, "test"), transform=transform)
    test_loader = utils.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    with open(args.summary_csv, "r", newline="") as f:
        summary_rows = list(csv.DictReader(f))
    if not summary_rows:
        raise RuntimeError(f"No rows found in summary CSV: {args.summary_csv}")

    rows_out = []
    for idx, row in enumerate(summary_rows):
        trial_id = row.get("trial_id", str(idx))
        ckpt_val_path = row.get("ckpt_val_path", "")
        ckpt_optim_path = row.get("ckpt_optim_path", "")

        if not ckpt_val_path or not ckpt_optim_path:
            msg = f"Missing checkpoint path(s) for trial_id={trial_id}"
            if args.skip_missing:
                print(f"[skip] {msg}")
                continue
            raise FileNotFoundError(msg)
        if not os.path.exists(ckpt_val_path) or not os.path.exists(ckpt_optim_path):
            msg = f"Checkpoint file missing for trial_id={trial_id}"
            if args.skip_missing:
                print(f"[skip] {msg}")
                continue
            raise FileNotFoundError(msg)

        model = make_model(device)

        model.load_state_dict(load_state_dict_from_ckpt(ckpt_val_path, device), strict=True)
        test_acc_val_sel = evaluate_accuracy(model, test_loader, device)

        model.load_state_dict(load_state_dict_from_ckpt(ckpt_optim_path, device), strict=True)
        test_acc_optim_sel = evaluate_accuracy(model, test_loader, device)

        out_row = {
            "trial_id": trial_id,
            "best_val_acc": parse_float(row, "best_val_acc"),
            "best_val_epoch": parse_int(row, "best_val_epoch"),
            "best_log_optim": parse_float(row, "best_log_optim"),
            "best_optim_value": parse_float(row, "best_optim_value"),
            "best_optim_epoch": parse_int(row, "best_optim_epoch"),
            "test_acc_valSel": test_acc_val_sel,
            "test_acc_optimSel": test_acc_optim_sel,
            "selection_gain": test_acc_optim_sel - test_acc_val_sel,
            "ckpt_val_path": ckpt_val_path,
            "ckpt_optim_path": ckpt_optim_path,
        }
        rows_out.append(out_row)
        print(
            f"trial={trial_id}  test_acc_valSel={test_acc_val_sel:.2f}%  "
            f"test_acc_optimSel={test_acc_optim_sel:.2f}%  "
            f"gain={out_row['selection_gain']:.2f}"
        )

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        fieldnames = [
            "trial_id",
            "best_val_acc",
            "best_val_epoch",
            "best_log_optim",
            "best_optim_value",
            "best_optim_epoch",
            "test_acc_valSel",
            "test_acc_optimSel",
            "selection_gain",
            "ckpt_val_path",
            "ckpt_optim_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    gains = np.array([r["selection_gain"] for r in rows_out], dtype=np.float32)
    print(f"\nWrote: {args.output_csv}")
    print(f"rows={len(rows_out)}  mean_gain={float(gains.mean()):.3f}  median_gain={float(np.median(gains)):.3f}")


if __name__ == "__main__":
    main()
