"""Shared Optuna runner for MNIST-method sweeps on PNG datasets.

Methods: vanilla, cdep, rrr, eg
Datasets: color, decoy

Objective: plain validation accuracy (maximize)
Phase 1: Optuna sweep (capped by --n-trials)
Phase 2: re-run best hyperparameters for --n-seeds
"""

from __future__ import print_function

import argparse
import os
import random
import time
from copy import deepcopy

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, Normalize, ToTensor


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
import sys
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
import score_funcs  # noqa: E402


DATASET_CHOICES = ("color", "decoy")
METHOD_CHOICES = ("vanilla", "cdep", "rrr", "eg")


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DecoyNet(nn.Module):
    def __init__(self):
        super(DecoyNet, self).__init__()
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


def _compute_mean_std(dataset, batch_size=512):
    loader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = 0
    sum_ = torch.zeros(3)
    sumsq = torch.zeros(3)
    for data, _ in loader:
        b = data.size(0)
        total += b * data.size(2) * data.size(3)
        sum_ += data.sum(dim=(0, 2, 3))
        sumsq += (data ** 2).sum(dim=(0, 2, 3))
    mean = sum_ / total
    std = torch.sqrt(sumsq / total - mean ** 2)
    return mean, std


def _compute_color_prob(raw_train_dataset, batch_size=512):
    loader = utils.DataLoader(raw_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    nonzero = torch.zeros(28, 28)
    n = 0
    for data, _ in loader:
        nonzero += (data.sum(dim=1) > 0).float().sum(dim=0)
        n += data.size(0)
    prob = (nonzero / max(n, 1)).view(-1).numpy()
    prob = prob / prob.sum()
    return prob


def _make_color_blobs():
    blobs = np.zeros((28 * 28, 28, 28), dtype=np.float32)
    for i in range(28):
        for j in range(28):
            blobs[i * 28 + j, i, j] = 1.0
    return blobs


def _make_decoy_blob():
    blob = np.zeros((28, 28), dtype=np.float32)
    size = 5
    blob[:size, :size] = 1.0
    blob[-size:, :size] = 1.0
    blob[:size, -size:] = 1.0
    blob[-size:, -size:] = 1.0
    return blob


def build_dataset_bundle(dataset_name, png_root):
    if dataset_name == "color":
        train_dir = os.path.join(png_root, "train")
        test_dir = os.path.join(png_root, "test")
        raw_transform = Compose([ToTensor(), Lambda(lambda x: x * 255.0)])
        raw_train = ImageFolder(train_dir, transform=raw_transform)
        mean, std = _compute_mean_std(raw_train)
        transform = Compose([raw_transform, Normalize(mean.tolist(), std.tolist())])
        full_train = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
        artifacts = {
            "blobs_np": _make_color_blobs(),
            "prob": _compute_color_prob(raw_train),
            "num_blobs_default": 8,
            "eg_samples_default": 50,
        }
        return full_train, test_dataset, artifacts

    if dataset_name == "decoy":
        train_dir = os.path.join(png_root, "train")
        test_dir = os.path.join(png_root, "test")
        transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
        full_train = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
        artifacts = {
            "blob_np": _make_decoy_blob(),
            "eg_samples_default": 200,
        }
        return full_train, test_dataset, artifacts

    raise ValueError("Unknown dataset: {}".format(dataset_name))


def make_model(dataset_name, device):
    if dataset_name == "color":
        return ColorNet().to(device)
    if dataset_name == "decoy":
        return DecoyNet().to(device)
    raise ValueError("Unknown dataset: {}".format(dataset_name))


def evaluate_acc(model, loader, device):
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


def compute_regularizer_loss(method, dataset_name, model, data, target, params, artifacts, device):
    if method == "vanilla":
        return torch.zeros(1, device=device).squeeze()

    reg_rate = float(params["regularizer_rate"])
    add_loss = torch.zeros(1, device=device)

    if dataset_name == "color":
        num_blobs = int(params.get("num_blobs", artifacts["num_blobs_default"]))
        blob_idxs = np.random.choice(28 * 28, size=num_blobs, p=artifacts["prob"])
    else:
        blob_idxs = None

    if method == "cdep":
        if dataset_name == "color":
            for idx in blob_idxs:
                add_loss += score_funcs.cdep(model, data, artifacts["blobs_np"][idx], model_type="mnist")
        else:
            add_loss += score_funcs.cdep(model, data, artifacts["blob_np"], model_type="mnist")
        return reg_rate * add_loss

    if method == "rrr":
        if dataset_name == "color":
            for idx in blob_idxs:
                seg = artifacts["blobs_torch"][idx]
                add_loss += score_funcs.gradient_sum(data, target, seg, model, F.nll_loss)
        else:
            add_loss += score_funcs.gradient_sum(data, target, artifacts["blob_torch"], model, F.nll_loss)
        return reg_rate * add_loss

    if method == "eg":
        eg_samples = int(params.get("eg_num_samples", artifacts["eg_samples_default"]))
        if dataset_name == "color":
            for j in range(len(data)):
                eg_map = score_funcs.eg_scores_2d(model, data, j, target, eg_samples)
                for idx in blob_idxs:
                    add_loss += (eg_map * artifacts["blobs_torch"][idx]).sum()
        else:
            for j in range(len(data)):
                eg_map = score_funcs.eg_scores_2d(model, data, j, target, eg_samples)
                add_loss += (eg_map * artifacts["blob_torch"]).sum()
        return reg_rate * add_loss

    raise ValueError("Unknown method: {}".format(method))


def train_one_run(dataset_name, method, full_train, test_dataset, artifacts, params, run_seed, args, device):
    set_seed(run_seed)
    g = torch.Generator().manual_seed(run_seed)

    n_total = len(full_train)
    n_val = max(1, int(args.val_frac * n_total))
    n_train = n_total - n_val
    train_subset, val_subset = utils.random_split(full_train, [n_train, n_val], generator=g)

    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": bool(device.type == "cuda")}
    train_loader = utils.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = utils.DataLoader(val_subset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)
    test_loader = utils.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)

    model = make_model(dataset_name, device)
    optimizer = optim.Adam(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))

    runtime_artifacts = dict(artifacts)
    if method in ("rrr", "eg"):
        if dataset_name == "color":
            runtime_artifacts["blobs_torch"] = torch.from_numpy(artifacts["blobs_np"]).float().to(device)
        else:
            runtime_artifacts["blob_torch"] = torch.from_numpy(artifacts["blob_np"]).float().to(device)

    best_val_acc = -1.0
    best_weights = None

    for _ in range(args.epochs):
        model.train()
        for data, target in train_loader:
            model.train()
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            out = model(data)
            ce_loss = F.nll_loss(out, target)
            reg_loss = compute_regularizer_loss(
                method=method,
                dataset_name=dataset_name,
                model=model,
                data=data,
                target=target,
                params=params,
                artifacts=runtime_artifacts,
                device=device,
            )
            loss = ce_loss + reg_loss
            loss.backward()
            optimizer.step()

        val_acc = evaluate_acc(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)
    test_acc = evaluate_acc(model, test_loader, device)
    return best_val_acc, test_acc, best_weights


def sample_hparams(trial, dataset_name, method):
    if method == "vanilla":
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

    if method == "cdep":
        reg_lo, reg_hi = (10.0, 5000.0) if dataset_name == "color" else (0.1, 500.0)
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "regularizer_rate": trial.suggest_float("regularizer_rate", reg_lo, reg_hi, log=True),
        }
        if dataset_name == "color":
            params["num_blobs"] = trial.suggest_int("num_blobs", 4, 12)
        return params

    if method == "rrr":
        reg_lo, reg_hi = (1e-4, 100.0) if dataset_name == "color" else (1e-5, 50.0)
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "regularizer_rate": trial.suggest_float("regularizer_rate", reg_lo, reg_hi, log=True),
        }
        if dataset_name == "color":
            params["num_blobs"] = trial.suggest_int("num_blobs", 4, 12)
        return params

    if method == "eg":
        reg_lo, reg_hi = (1e-4, 50.0) if dataset_name == "color" else (1e-5, 20.0)
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "regularizer_rate": trial.suggest_float("regularizer_rate", reg_lo, reg_hi, log=True),
        }
        if dataset_name == "color":
            params["num_blobs"] = trial.suggest_int("num_blobs", 4, 10)
            params["eg_num_samples"] = trial.suggest_int("eg_num_samples", 20, 80)
        else:
            params["eg_num_samples"] = trial.suggest_int("eg_num_samples", 50, 250)
        return params

    raise ValueError("Unknown method: {}".format(method))


def run_pipeline(dataset_name, method, args):
    if dataset_name == "color":
        png_root = args.png_root or os.path.join(REPO_ROOT, "data", "ColorMNIST_png")
        model_dir = os.path.join(REPO_ROOT, "models", "ColorMNIST_test")
    else:
        png_root = args.png_root or os.path.join(REPO_ROOT, "data", "DecoyMNIST_png")
        model_dir = os.path.join(REPO_ROOT, "models", "DecoyMNIST")
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    print("dataset={}, method={}, device={}".format(dataset_name, method, device))
    print("png_root={}".format(png_root))

    full_train, test_dataset, artifacts = build_dataset_bundle(dataset_name, png_root)
    print("full_train={}, test={}".format(len(full_train), len(test_dataset)))

    db_path = args.db_path or os.path.join(model_dir, "{}.db".format(args.study_name))
    storage = "sqlite:///{}".format(db_path)
    print("optuna_storage={}".format(storage))

    def objective(trial):
        params = sample_hparams(trial, dataset_name, method)
        best_val_acc, test_acc, _ = train_one_run(
            dataset_name=dataset_name,
            method=method,
            full_train=full_train,
            test_dataset=test_dataset,
            artifacts=artifacts,
            params=params,
            run_seed=args.objective_seed,
            args=args,
            device=device,
        )
        print("trial={} params={} val_acc={:.2f} test_acc={:.2f}".format(
            trial.number, params, best_val_acc, test_acc))
        return best_val_acc

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner(),
    )

    before = len(study.trials)
    remaining = max(0, args.n_trials - before)
    print("trial_cap={}, existing={}, running={}".format(args.n_trials, before, remaining))
    if remaining > 0:
        t0 = time.time()
        study.optimize(objective, n_trials=remaining)
        print("phase1_hours={:.2f}, total_trials={}".format((time.time() - t0) / 3600.0, len(study.trials)))
    else:
        print("phase1 skipped (already at cap)")

    best = study.best_trial
    bp = best.params
    print("best_trial={} best_val_acc={:.2f} params={}".format(best.number, best.value, bp))

    results = []
    for i in range(args.n_seeds):
        seed = args.seed_start + i
        best_val_acc, test_acc, _ = train_one_run(
            dataset_name=dataset_name,
            method=method,
            full_train=full_train,
            test_dataset=test_dataset,
            artifacts=artifacts,
            params=bp,
            run_seed=seed,
            args=args,
            device=device,
        )
        row = {"seed": seed, "val_acc": best_val_acc, "test_acc": test_acc}
        results.append(row)
        print("seed={} val_acc={:.2f} test_acc={:.2f}".format(seed, best_val_acc, test_acc))

    vals = np.array([r["val_acc"] for r in results], dtype=np.float32)
    tests = np.array([r["test_acc"] for r in results], dtype=np.float32)
    print("summary val_acc mean={:.2f} std={:.2f}".format(float(vals.mean()), float(vals.std())))
    print("summary test_acc mean={:.2f} std={:.2f}".format(float(tests.mean()), float(tests.std())))


def main_fixed(dataset_name, method_name, default_study_name):
    parser = argparse.ArgumentParser(description="Optuna sweep for {} on {}".format(method_name, dataset_name))
    parser.add_argument("--png-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=(256 if dataset_name == "color" else 64))
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--val-frac", type=float, default=0.16)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--study-name", type=str, default=default_study_name)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--objective-seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    run_pipeline(dataset_name=dataset_name, method=method_name, args=args)

