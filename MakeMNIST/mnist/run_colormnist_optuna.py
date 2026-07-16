"""Optuna hyperparameter sweep for guided LeNet on ColorMNIST.

Phase 1 (24 h): Optuna explores kl_lambda, attention_epoch, lr, lr2,
                step_size, gamma.  kl_incr is always kl_lambda / 10.
                Objective = best optim_value seen during training.

Phase 2: Take the best trial's hyperparameters and train 10 seeds,
         printing test accuracy on the best-optim-value checkpoint each time.
"""

from __future__ import print_function
import argparse
import time
import os
import sys
import re
import random
import pickle as pkl
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda, Normalize
import cv2
from PIL import Image

import optuna

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, ".."))
sys.path.append(os.path.join(_here, "ColorMNIST"))
from params_save import S

model_path = os.path.join(_repo_root, "models", "ColorMNIST_test")
os.makedirs(model_path, exist_ok=True)
torch.backends.cudnn.deterministic = True


# ── CLI args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Optuna sweep for guided ColorMNIST LeNet")
parser.add_argument("--png-root", type=str, default=None)
parser.add_argument("--gt-path", type=str, required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--test-batch-size", type=int, default=1000)
parser.add_argument("--momentum", type=float, default=0.98)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--beta", type=float, default=0.3)
parser.add_argument("--val-frac", type=float, default=0.16)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--log-interval", type=int, default=100)
parser.add_argument("--sweep-hours", type=float, default=24.0,
                    help="How many hours to run the Optuna sweep (default: 24)")
parser.add_argument("--n-seeds", type=int, default=10,
                    help="Number of seeds for final evaluation (default: 10)")
parser.add_argument("--study-name", type=str, default="colormnist_guided")
parser.add_argument("--db-path", type=str, default=None,
                    help="SQLite path for Optuna storage (default: auto)")

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
loader_kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}
png_root = args.png_root or os.path.join(_repo_root, "data", "ColorMNIST_png")


# ── Model ────────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        gap = self.gap(x).view(x.size(0), -1)
        out = self.classifier(gap)
        return out


def make_cam_model(num_classes=10):
    base = Net(num_classes)

    class CAMWrap(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.features = None
            self.base.conv2.register_forward_hook(self._hook_fn)

        def _hook_fn(self, module, inp, out):
            self.features = out

        def forward(self, x):
            out = self.base(x)
            return out, self.features

    return CAMWrap(base)


# ── Mask transforms ─────────────────────────────────────────────────────
class ExpandWhite:
    def __init__(self, thr=10, radius=3):
        self.thr = thr
        self.radius = radius

    def __call__(self, mask):
        arr = np.array(mask)
        white = (arr > self.thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (2 * self.radius + 1, 2 * self.radius + 1))
        dil = cv2.dilate(white, k, iterations=1)
        return Image.fromarray((dil * 255).astype(np.uint8))


class EdgeExtract:
    def __init__(self, thr=10, edge_width=1):
        self.thr = thr
        self.edge_width = edge_width

    def __call__(self, mask):
        arr = np.array(mask)
        white = (arr > self.thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (2 * self.edge_width + 1, 2 * self.edge_width + 1))
        edge = cv2.morphologyEx(white, cv2.MORPH_GRADIENT, k)
        return Image.fromarray((edge * 255).astype(np.uint8))


class Brighten:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, mask):
        return torch.clamp(mask * self.factor, 0.0, 1.0)


# ── Dataset ──────────────────────────────────────────────────────────────
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
                path = os.path.join(self.mask_root, stem + ext)
                if os.path.exists(path):
                    return path
        tried = [os.path.join(self.mask_root, stem + ext)
                 for stem in candidates for ext in self._mask_exts]
        raise FileNotFoundError(
            f"Mask not found for base='{base}', class='{class_name}'. Tried: {tried}")

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


# ── Loss helpers ─────────────────────────────────────────────────────────
def compute_loss(outputs, labels, cams, gt_masks, kl_lambda, only_ce):
    ce_loss = F.cross_entropy(outputs, labels)
    cam_flat = cams.view(cams.size(0), -1)
    gt_flat = gt_masks.view(gt_masks.size(0), -1)
    log_p = F.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    kl_div = nn.KLDivLoss(reduction="batchmean")
    attn_loss = kl_div(log_p, gt_prob)
    if only_ce:
        return ce_loss, attn_loss
    else:
        return ce_loss + kl_lambda * attn_loss, attn_loss


def compute_attn_losses(cams, gt_masks):
    cam_flat = cams.view(cams.size(0), -1)
    gt_flat = gt_masks.view(gt_masks.size(0), -1)
    log_cam = F.log_softmax(cam_flat, dim=1)
    cam_prob = F.softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    log_gt = torch.log(gt_prob + 1e-8)
    kl_div = nn.KLDivLoss(reduction="batchmean")
    forward_kl = kl_div(log_cam, gt_prob)
    reverse_kl = kl_div(log_gt, cam_prob)
    return forward_kl, reverse_kl


# ── Normalization ────────────────────────────────────────────────────────
def _compute_mean_std(dataset, batch_size=512):
    loader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = 0
    sum_ = torch.zeros(3)
    sumsq = torch.zeros(3)
    for data, *_ in loader:
        b = data.size(0)
        total += b * data.size(2) * data.size(3)
        sum_ += data.sum(dim=(0, 2, 3))
        sumsq += (data ** 2).sum(dim=(0, 2, 3))
    mean = sum_ / total
    std = torch.sqrt(sumsq / total - mean ** 2)
    return mean, std


# ── Build datasets once (shared across all trials) ──────────────────────
print("Loading datasets ...")
base_transform = Compose([ToTensor(), Lambda(lambda x: x * 255.0)])
_raw = ImageFolder(os.path.join(png_root, "train"), transform=base_transform)
_mean, _std = _compute_mean_std(_raw)
full_transform = Compose([base_transform, Normalize(_mean.tolist(), _std.tolist())])
del _raw

mask_transform = transforms.Compose([
    ExpandWhite(thr=10, radius=3),
    EdgeExtract(thr=10, edge_width=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    Brighten(8.0),
])

full_train = GuidedImageFolder(
    image_root=os.path.join(png_root, "train"),
    mask_root=args.gt_path,
    image_transform=full_transform,
    mask_transform=mask_transform,
)

test_dataset = ImageFolder(os.path.join(png_root, "test"), transform=full_transform)
test_loader = utils.DataLoader(test_dataset, batch_size=args.test_batch_size,
                               shuffle=False, **loader_kwargs)
print(f"Full train: {len(full_train)}, Test: {len(test_dataset)}")


# ═════════════════════════════════════════════════════════════════════════
#  Core training function (returns best optim_value and test acc)
# ═════════════════════════════════════════════════════════════════════════
def run_training(seed, kl_lambda, attention_epoch, lr, lr2, step_size, gamma,
                 epochs=None, trial=None, verbose=True):
    """Train one run.  Returns (best_optim_value, test_acc, best_weights)."""
    if epochs is None:
        epochs = args.epochs
    kl_incr = kl_lambda / 10.0

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Split train / val
    g = torch.Generator()
    g.manual_seed(seed)
    n_total = len(full_train)
    n_val = max(1, int(args.val_frac * n_total))
    n_train = n_total - n_val
    train_subset, val_subset = utils.random_split(full_train, [n_train, n_val], generator=g)

    train_loader = utils.DataLoader(train_subset, batch_size=args.batch_size,
                                    shuffle=True, **loader_kwargs)
    val_loader = utils.DataLoader(val_subset, batch_size=args.batch_size,
                                  shuffle=False, **loader_kwargs)

    model = make_cam_model(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_model_weights = None
    best_optim = -100.0
    kl_lambda_real = kl_lambda

    for epoch in range(1, epochs + 1):
        attention_active = (epoch >= attention_epoch) and (kl_lambda > 0)

        # Restart optimizer at attention epoch
        if epoch == attention_epoch and kl_lambda > 0:
            if verbose:
                print(f"  *** Attention epoch {epoch}: restarting optimizer & scheduler ***")
            optimizer = optim.SGD(model.parameters(), lr=lr2, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            best_model_weights = deepcopy(model.state_dict())
            best_optim = -100.0
            kl_lambda_real = kl_lambda

        if epoch > attention_epoch and kl_lambda > 0:
            kl_lambda_real += kl_incr

        # ── Train ────────────────────────────────────────────────────
        model.train()
        for data, target, gt_masks in train_loader:
            data, target = data.to(device), target.to(device)
            gt_masks = gt_masks.to(device)
            optimizer.zero_grad()
            outputs, feats = model(data)

            weights = model.base.classifier.weight[target]
            cams = torch.relu(torch.einsum("bc,bchw->bhw", weights, feats))
            flat = cams.view(cams.size(0), -1)
            mn, _ = flat.min(dim=1, keepdim=True)
            mx, _ = flat.max(dim=1, keepdim=True)
            sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)
            gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                     mode="nearest").squeeze(1)

            if not attention_active:
                loss, _ = compute_loss(outputs, target, sal_norm, gt_small, 0, True)
            else:
                loss, _ = compute_loss(outputs, target, sal_norm, gt_small,
                                       kl_lambda_real, False)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        running_corrects = 0
        running_attn_rev = 0.0
        total = 0
        with torch.no_grad():
            for data, target, gt_masks in val_loader:
                data, target = data.to(device), target.to(device)
                gt_masks = gt_masks.to(device)
                outputs, feats = model(data)
                preds = outputs.argmax(dim=1)

                weights = model.base.classifier.weight[target]
                cams = torch.relu(torch.einsum("bc,bchw->bhw", weights, feats))
                flat = cams.view(cams.size(0), -1)
                mn, _ = flat.min(dim=1, keepdim=True)
                mx, _ = flat.max(dim=1, keepdim=True)
                sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)
                gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                         mode="nearest").squeeze(1)
                _, rev_kl = compute_attn_losses(sal_norm, gt_small)

                running_corrects += preds.eq(target).sum().item()
                running_attn_rev += rev_kl.item() * data.size(0)
                total += data.size(0)

        val_acc = running_corrects / total
        val_rev = running_attn_rev / total
        optim_num = val_acc * np.exp(-args.beta * val_rev)

        if epoch >= attention_epoch and optim_num > best_optim:
            best_optim = optim_num
            best_model_weights = deepcopy(model.state_dict())

        if epoch < attention_epoch:
            best_model_weights = deepcopy(model.state_dict())

        if verbose:
            print(f"  Epoch {epoch:>2d}  val_acc={100*val_acc:.1f}%  "
                  f"rev_kl={val_rev:.4f}  optim={optim_num:.4f}"
                  f"{'  *best*' if optim_num >= best_optim else ''}")

        # Optuna pruning (only after attention epoch when guidance has started)
        if trial is not None and epoch >= attention_epoch:
            trial.report(best_optim, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # ── Test with best-optim checkpoint ──────────────────────────────
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs, _ = model(data)
            correct += outputs.argmax(dim=1).eq(target).sum().item()
            total += data.size(0)
    test_acc = 100.0 * correct / total

    if verbose:
        print(f"  >> best optim_value = {best_optim:.4f},  "
              f"test_acc @ best_optim = {test_acc:.1f}%")

    return best_optim, test_acc, best_model_weights


# ═════════════════════════════════════════════════════════════════════════
#  Phase 1: Optuna sweep
# ═════════════════════════════════════════════════════════════════════════
def objective(trial):
    kl_lambda = trial.suggest_float("kl_lambda", 1.0, 500.0, log=True)
    attention_epoch = trial.suggest_int("attention_epoch", 1, 25)
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    lr2 = trial.suggest_float("lr2", 1e-5, 5e-2, log=True)
    step_size = trial.suggest_int("step_size", 2, 15)
    gamma = trial.suggest_float("gamma", 0.05, 0.5, log=True)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: kl_lambda={kl_lambda:.2f}, "
          f"attn_epoch={attention_epoch}, lr={lr:.5f}, lr2={lr2:.5f}, "
          f"step_size={step_size}, gamma={gamma:.3f}")
    print(f"{'='*60}")

    best_optim, test_acc, _ = run_training(
        seed=0,
        kl_lambda=kl_lambda,
        attention_epoch=attention_epoch,
        lr=lr,
        lr2=lr2,
        step_size=step_size,
        gamma=gamma,
        trial=trial,
        verbose=True,
    )
    print(f"Trial {trial.number} finished: optim={best_optim:.4f}, test_acc={test_acc:.1f}%")
    return best_optim


if __name__ == "__main__":
    sweep_seconds = args.sweep_hours * 3600

    db_path = args.db_path or os.path.join(model_path, f"{args.study_name}.db")
    storage = f"sqlite:///{db_path}"
    print(f"\nOptuna storage: {storage}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    print(f"\n{'#'*60}")
    print(f"  Phase 1: Optuna sweep ({args.sweep_hours:.1f} h)")
    print(f"{'#'*60}\n")
    t0 = time.time()
    study.optimize(objective, timeout=sweep_seconds)
    elapsed = time.time() - t0
    print(f"\nSweep finished after {elapsed/3600:.2f} h, "
          f"{len(study.trials)} trials attempted.")

    best = study.best_trial
    print(f"\nBest trial #{best.number}: optim_value = {best.value:.4f}")
    print(f"  Params: {best.params}")

    # ═════════════════════════════════════════════════════════════════
    #  Phase 2: 10 seeds with best hyperparameters
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print(f"  Phase 2: {args.n_seeds} seeds with best hyperparameters")
    print(f"{'#'*60}\n")

    bp = best.params
    seed_results = []
    for i in range(args.n_seeds):
        seed = i
        print(f"\n--- Seed {seed} ---")
        best_optim, test_acc, best_weights = run_training(
            seed=seed,
            kl_lambda=bp["kl_lambda"],
            attention_epoch=bp["attention_epoch"],
            lr=bp["lr"],
            lr2=bp["lr2"],
            step_size=bp["step_size"],
            gamma=bp["gamma"],
            verbose=True,
        )
        seed_results.append({
            "seed": seed,
            "optim_value": best_optim,
            "test_acc": test_acc,
        })

        # Save each seed's model via params_save
        s = S(args.epochs)
        s.regularizer_rate = bp["kl_lambda"]
        s.num_blobs = 0
        s.seed = seed
        s.dataset = "Color"
        s.method = "Guided"
        s.acc_test = test_acc
        s.model_weights = best_weights
        np.random.seed()
        pid = "".join(["%s" % np.random.randint(0, 9) for _ in range(20)])
        os.makedirs(model_path, exist_ok=True)
        pkl.dump(s._dict(), open(os.path.join(model_path, pid + ".pkl"), "wb"))
        print(f"  Saved model to {os.path.join(model_path, pid + '.pkl')}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Best hyperparameters (from trial #{best.number}):")
    for k, v in bp.items():
        print(f"  {k}: {v}")
    print(f"\nResults across {args.n_seeds} seeds:")
    print(f"{'Seed':>6s}  {'Optim':>8s}  {'Test Acc':>10s}")
    optims = []
    accs = []
    for r in seed_results:
        print(f"{r['seed']:>6d}  {r['optim_value']:>8.4f}  {r['test_acc']:>9.1f}%")
        optims.append(r["optim_value"])
        accs.append(r["test_acc"])
    print(f"\nOptim  — mean: {np.mean(optims):.4f}, std: {np.std(optims):.4f}")
    print(f"TestAcc — mean: {np.mean(accs):.1f}%, std: {np.std(accs):.1f}%")
