"""Optuna hyperparameter sweep for Grad-CAM guided LeNet on DecoyMNIST.

Phase 1 (100 trials): Optuna explores kl_lambda and attention_epoch,
                      lr (base), lr2 (classifier), lr2_mult (post-attention LR multiplier).
                      kl_incr is fixed.
                      StepLR settings are fixed.
                      Objective = best optim_value seen during training.
                      Adam throughout, reset at attention epoch with lr2 + StepLR.
                      Uses Grad-CAM on original FC LeNet (can learn decoy shortcut).
                      Pruning disabled.

Phase 2: Take the best trial's hyperparameters and train 5 seeds each on:
         - Original GT_PATH (WeCLIP)
         - OpenAI XCiT GT path
         - OpenCLIP GT path
"""

from __future__ import print_function
import argparse
import time
import os
import sys
import re
import random
import csv
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
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale
import cv2
from PIL import Image

import optuna

LOG_OPTIM_EPS = 1e-12

# -- GT paths for multi-mask evaluation ---------------------------------------
GT_PATHS = {
    "WeCLIP": "/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap",
    "OpenAI_XCiT": "/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist_openai_xcit/val/prediction_cmap",
    "OpenCLIP": "/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist_openclip/val/prediction_cmap",
}

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, ".."))
sys.path.append(os.path.join(_here, "DecoyMNIST"))
from params_save import S

model_path = os.path.join(_repo_root, "models", "DecoyMNIST")
os.makedirs(model_path, exist_ok=True)
torch.backends.cudnn.deterministic = True


# -- CLI args -----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Optuna sweep for Grad-CAM guided DecoyMNIST LeNet")
parser.add_argument("--png-root", type=str, default=None)
parser.add_argument("--gt-path", type=str, required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--test-batch-size", type=int, default=1000)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--beta", type=float, default=10.0)
parser.add_argument("--val-frac", type=float, default=0.16)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--log-interval", type=int, default=100)
parser.add_argument("--n-trials", type=int, default=100,
                    help="Number of Optuna trials to run (default: 100)")
parser.add_argument("--n-seeds", type=int, default=5,
                    help="Number of seeds for final evaluation per GT path (default: 5)")
parser.add_argument("--objective-seed", type=int, default=42,
                    help="Seed used for trial training/evaluation during Optuna objective.")
parser.add_argument("--optuna-seed", type=int, default=None,
                    help="Seed for Optuna TPE sampler (default: random).")
parser.add_argument("--study-name", type=str, default="decoymnist_gradcam")
parser.add_argument("--db-path", type=str, default=None,
                    help="SQLite path for Optuna storage (default: auto)")
parser.add_argument("--val-saliency", type=str, default="igrad",
                    choices=["cam", "igrad"],
                    help="Saliency used for validation metric: Grad-CAM or Integrated Gradients.")
parser.add_argument("--val-metric", type=str, default="fwd_kl",
                    choices=["rev_kl", "fwd_kl", "outside"],
                    help="Validation metric against GT masks.")
parser.add_argument("--save-trial-checkpoints", action="store_true", default=False,
                    help="During Phase 1, save best-val and best-optim checkpoints per trial.")
parser.add_argument("--trial-artifacts-dir", type=str, default=None,
                    help="Output directory for per-trial checkpoints and summary CSV.")
parser.add_argument("--trial-summary-csv", type=str, default=None,
                    help="Per-trial summary CSV path (default: <trial-artifacts-dir>/trial_summary.csv).")
parser.add_argument("--selector-start", type=str, default="attention",
                    choices=["attention", "all"],
                    help="Epoch eligibility for checkpoint selectors.")
parser.add_argument("--checkpoint-selector", type=str, default="optim",
                    choices=["optim", "val_acc"],
                    help="Checkpoint selector used for trial objective and final test checkpoint.")
parser.add_argument("--skip-phase2", action="store_true", default=False,
                    help="Skip Phase 2 seed reruns (useful for trial-scatter sweeps).")
parser.add_argument("--kl-lambda-min", type=float, default=1.0)
parser.add_argument("--kl-lambda-max", type=float, default=500.0)
parser.add_argument("--attention-epoch-min", type=int, default=1)
parser.add_argument("--attention-epoch-max", type=int, default=29)
parser.add_argument("--lr-min", type=float, default=1e-5)
parser.add_argument("--lr-max", type=float, default=5e-2)
parser.add_argument("--lr2-min", type=float, default=1e-5)
parser.add_argument("--lr2-max", type=float, default=5e-2)
parser.add_argument("--lr2-mult-min", type=float, default=1e-3)
parser.add_argument("--lr2-mult-max", type=float, default=1.0)
parser.add_argument("--kl-incr-fixed", type=float, default=0.0,
                    help="Fixed KL increment added after attention epoch.")
parser.add_argument("--manual-kl-sweep", action="store_true", default=False,
                    help="Bypass Optuna and run a fixed-parameter KL-lambda log sweep.")
parser.add_argument("--manual-kl-min", type=float, default=0.01)
parser.add_argument("--manual-kl-max", type=float, default=100.0)
parser.add_argument("--manual-num-points", type=int, default=50)
parser.add_argument("--manual-attention-epoch", type=int, default=15)
parser.add_argument("--manual-lr", type=float, default=1e-5)
parser.add_argument("--manual-lr2", type=float, default=5e-5)
parser.add_argument("--manual-lr2-mult", type=float, default=0.1)
parser.add_argument("--ig-steps", type=int, default=16,
                    help="Integrated Gradients interpolation steps for val saliency.")

args = parser.parse_args()
if args.kl_lambda_min > args.kl_lambda_max:
    raise ValueError("kl-lambda-min must be <= kl-lambda-max")
if args.attention_epoch_min > args.attention_epoch_max:
    raise ValueError("attention-epoch-min must be <= attention-epoch-max")
if args.lr_min > args.lr_max:
    raise ValueError("lr-min must be <= lr-max")
if args.lr2_min > args.lr2_max:
    raise ValueError("lr2-min must be <= lr2-max")
if args.lr2_mult_min > args.lr2_mult_max:
    raise ValueError("lr2-mult-min must be <= lr2-mult-max")
if args.manual_kl_min > args.manual_kl_max:
    raise ValueError("manual-kl-min must be <= manual-kl-max")
if args.manual_kl_min <= 0.0:
    raise ValueError("manual-kl-min must be > 0 for log spacing")
if args.manual_num_points < 1:
    raise ValueError("manual-num-points must be >= 1")
if args.ig_steps < 1:
    raise ValueError("ig-steps must be >= 1")
if args.beta != 10.0:
    print("Forcing beta=10.0 for Decoy log-optim objective.")
args.beta = 10.0
if args.val_saliency != "igrad" or args.val_metric != "fwd_kl":
    print("Forcing validation strategy to val_saliency=igrad (Integrated Gradients) and val_metric=fwd_kl.")
args.val_saliency = "igrad"
args.val_metric = "fwd_kl"
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
loader_kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
png_root = args.png_root or os.path.join(_repo_root, "data", "DecoyMNIST_png")

trial_artifacts_dir = None
trial_ckpt_dir = None
trial_summary_csv = None
if args.save_trial_checkpoints:
    trial_artifacts_dir = args.trial_artifacts_dir or os.path.join(
        model_path, f"{args.study_name}_trial_artifacts"
    )
    trial_ckpt_dir = os.path.join(trial_artifacts_dir, "checkpoints")
    os.makedirs(trial_ckpt_dir, exist_ok=True)
    trial_summary_csv = args.trial_summary_csv or os.path.join(trial_artifacts_dir, "trial_summary.csv")


# -- Model: original FC LeNet (can learn decoy shortcut) ---------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -- Grad-CAM wrapper --------------------------------------------------------
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

    def grad_cam(self, targets):
        """Compute Grad-CAM saliency. Call AFTER backward populates self.gradients."""
        weights = self.gradients.mean(dim=(2, 3))  # (B, C)
        cams = torch.einsum('bc,bchw->bhw', weights, self.features)
        cams = torch.relu(cams)
        flat = cams.view(cams.size(0), -1)
        mn, _ = flat.min(dim=1, keepdim=True)
        mx, _ = flat.max(dim=1, keepdim=True)
        sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)
        return sal_norm


def make_gradcam_model():
    return GradCAMWrap(Net())


def _get_param_groups(model, base_lr, classifier_lr):
    base_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if ".fc2" in name or name.startswith("fc2") or ".classifier" in name or name.startswith("classifier"):
            classifier_params.append(param)
        else:
            base_params.append(param)
    if not classifier_params:
        classifier_params = list(model.parameters())
        base_params = []
    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr})
    param_groups.append({"params": classifier_params, "lr": classifier_lr})
    return param_groups


# -- Mask transforms ----------------------------------------------------------
class Brighten:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, mask):
        return torch.clamp(mask * self.factor, 0.0, 1.0)


# -- Dataset ------------------------------------------------------------------
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


# -- Loss helpers --------------------------------------------------------------
def compute_attn_loss(cams, gt_masks):
    """Forward KL: KL(Mask || CAM)."""
    cam_flat = cams.view(cams.size(0), -1)
    gt_flat = gt_masks.view(gt_masks.size(0), -1)
    log_p = F.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    kl_div = nn.KLDivLoss(reduction="batchmean")
    return kl_div(log_p, gt_prob)


def compute_attn_losses(cams, gt_masks):
    """Both forward and reverse KL for validation."""
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


def compute_outside_mass(saliency, gt_masks):
    """Fraction of saliency mass outside the GT mask. Lower is better."""
    sal = saliency.view(saliency.size(0), -1)
    gt = gt_masks.view(gt_masks.size(0), -1)
    sal = sal / (sal.sum(dim=1, keepdim=True) + 1e-8)
    outside = sal * (1.0 - gt)
    return outside.sum(dim=1).mean()


def input_grad_saliency(model, data, target):
    """Backward-compatible alias: now uses Integrated Gradients saliency."""
    return integrated_grad_saliency(model, data, target, steps=args.ig_steps)


def integrated_grad_saliency(model, data, target, steps=16):
    """Compute Integrated Gradients saliency, normalized to [0,1]."""
    x = data.detach()
    baseline = torch.zeros_like(x)
    delta = x - baseline
    total_grads = torch.zeros_like(x)

    for step in range(1, steps + 1):
        alpha = float(step) / float(steps)
        x_step = (baseline + alpha * delta).detach().requires_grad_(True)
        logits = model.base.logits(x_step)
        class_scores = logits[torch.arange(len(target), device=x_step.device), target]
        grads = torch.autograd.grad(
            class_scores.sum(), x_step, retain_graph=False, create_graph=False
        )[0]
        total_grads += grads

    avg_grads = total_grads / float(steps)
    ig_attr = (delta * avg_grads).detach().abs().sum(dim=1)  # B,H,W
    flat = ig_attr.view(ig_attr.size(0), -1)
    mn, _ = flat.min(dim=1, keepdim=True)
    mx, _ = flat.max(dim=1, keepdim=True)
    sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(ig_attr)
    return sal_norm


def save_trial_checkpoint(path, model_state_dict, epoch, selector_name, trial_id):
    payload = {
        "trial_id": trial_id,
        "epoch": epoch,
        "selector": selector_name,
        "model_state_dict": model_state_dict,
    }
    torch.save(payload, path)


def append_trial_summary_row(csv_path, row):
    fieldnames = [
        "trial_id",
        "seed",
        "kl_lambda",
        "kl_incr",
        "attention_epoch",
        "lr",
        "lr2",
        "lr2_mult",
        "selector_start_mode",
        "selector_start_epoch",
        "best_val_acc",
        "best_val_epoch",
        "best_log_optim",
        "best_optim_value",
        "best_optim_epoch",
        "selected_selector",
        "selected_epoch",
        "selected_value",
        "ckpt_val_path",
        "ckpt_optim_path",
    ]
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# -- Build datasets once (shared across all trials) ---------------------------
print("Loading datasets ...")
image_transform = Compose([Grayscale(num_output_channels=1), ToTensor(),
                           Lambda(lambda x: x * 2.0 - 1.0)])

mask_transform = transforms.Compose([
    # ExpandWhite(thr=10, radius=3),
    # EdgeExtract(thr=10, edge_width=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    Brighten(8.0),
])

full_train = GuidedImageFolder(
    image_root=os.path.join(png_root, "train"),
    mask_root=args.gt_path,
    image_transform=image_transform,
    mask_transform=mask_transform,
)

test_dataset = ImageFolder(os.path.join(png_root, "test"), transform=image_transform)
test_loader = utils.DataLoader(test_dataset, batch_size=args.test_batch_size,
                               shuffle=False, **loader_kwargs)
print(f"Full train: {len(full_train)}, Test: {len(test_dataset)}")


# =============================================================================
#  Core training function (returns best optim_value and test acc)
# =============================================================================
def run_training(seed, kl_lambda, kl_incr, attention_epoch, lr, lr2, lr2_mult,
                 epochs=None, trial=None, verbose=True, trial_id=None):
    """Train one run. Returns (best_optim_value, test_acc, best_weights, info)."""
    if epochs is None:
        epochs = args.epochs

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

    model = make_gradcam_model().to(device)

    # Pre-attention optimizer uses separate base/classifier learning rates.
    optimizer = optim.Adam(_get_param_groups(model, lr, lr2), weight_decay=args.weight_decay)
    scheduler = None

    best_optim_weights = None
    best_log_optim = -np.inf
    best_optim = 0.0
    best_optim_epoch = -1
    best_val_acc = -1.0
    best_val_epoch = -1
    best_val_weights = None
    ckpt_val_path = None
    ckpt_optim_path = None
    kl_lambda_real = kl_lambda
    selector_start_epoch = attention_epoch if args.selector_start == "attention" else 1
    last_val_acc = 0.0
    last_log_optim = -np.inf

    for epoch in range(1, epochs + 1):
        attention_active = (epoch >= attention_epoch) and (kl_lambda > 0)

        # Reset optimizer at attention epoch
        if epoch == attention_epoch and kl_lambda > 0:
            if verbose:
                post_base_lr = lr * lr2_mult
                post_classifier_lr = lr2 * lr2_mult
                print(
                    f"  *** Attention epoch {epoch}: resetting optimizer & beginning guidance "
                    f"(base_lr={post_base_lr:.5g}, cls_lr={post_classifier_lr:.5g}) ***"
                )
            optimizer = optim.Adam(
                _get_param_groups(model, lr * lr2_mult, lr2 * lr2_mult),
                weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
            kl_lambda_real = kl_lambda

        if epoch > attention_epoch and kl_lambda > 0:
            kl_lambda_real += kl_incr

        # -- Train --------------------------------------------------------
        model.train()
        for data, target, gt_masks in train_loader:
            data, target = data.to(device), target.to(device)
            gt_masks = gt_masks.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            ce_loss = F.nll_loss(outputs, target)

            if not attention_active:
                # Pure CE training
                ce_loss.backward()
                optimizer.step()
            else:
                # Two-pass Grad-CAM guided training
                # Pass 1: backward class scores to get conv2 gradients for Grad-CAM
                model.zero_grad()
                logits = model.base.logits(data)
                class_scores = logits[torch.arange(len(target), device=device), target]
                class_scores.sum().backward(retain_graph=True)

                sal_norm = model.grad_cam(target)
                gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                         mode="nearest").squeeze(1)
                attn_loss = compute_attn_loss(sal_norm, gt_small)

                # Pass 2: actual training loss = CE + KL * lambda
                optimizer.zero_grad()
                outputs2 = model(data)
                ce_loss = F.nll_loss(outputs2, target)
                total_loss = ce_loss + kl_lambda_real * attn_loss
                total_loss.backward()
                optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # -- Validate -----------------------------------------------------
        model.eval()
        running_corrects = 0
        running_attn_rev = 0.0
        total = 0

        for data, target, gt_masks in val_loader:
            data, target = data.to(device), target.to(device)
            gt_masks = gt_masks.to(device)

            if args.val_saliency == "cam":
                # Grad-CAM needs gradients even during validation
                logits = model.base.logits(data)
                class_scores = logits[torch.arange(len(target), device=device), target]
                model.zero_grad()
                class_scores.sum().backward()
                sal_norm = model.grad_cam(target)
            else:
                # Integrated Gradients (IG)
                sal_norm = integrated_grad_saliency(model, data, target, steps=args.ig_steps)

            gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                     mode="nearest").squeeze(1)

            with torch.no_grad():
                outputs = model(data)
                preds = outputs.argmax(dim=1)

                if args.val_metric == "rev_kl":
                    _, metric_val = compute_attn_losses(sal_norm.detach(), gt_small)
                elif args.val_metric == "fwd_kl":
                    metric_val, _ = compute_attn_losses(sal_norm.detach(), gt_small)
                else:
                    metric_val = compute_outside_mass(sal_norm.detach(), gt_small)

                running_corrects += preds.eq(target).sum().item()
                running_attn_rev += metric_val.item() * data.size(0)
                total += data.size(0)

        val_acc = running_corrects / total
        val_metric = running_attn_rev / total
        if val_acc <= 0.0:
            log_optim = -np.inf
        else:
            log_optim = np.log(max(val_acc, LOG_OPTIM_EPS)) - args.beta * val_metric
        optim_num = float(np.exp(np.clip(log_optim, -700.0, 50.0)))
        last_val_acc = val_acc
        last_log_optim = log_optim

        improved_val = False
        improved_optim = False

        if epoch >= selector_start_epoch and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_weights = deepcopy(model.state_dict())
            improved_val = True
            if args.save_trial_checkpoints and trial_id is not None and trial_ckpt_dir is not None:
                ckpt_val_path = os.path.join(trial_ckpt_dir, f"trial_{trial_id}_bestVal.ckpt")
                save_trial_checkpoint(
                    path=ckpt_val_path,
                    model_state_dict=best_val_weights,
                    epoch=epoch,
                    selector_name="best_val_acc",
                    trial_id=trial_id,
                )

        if epoch >= selector_start_epoch and log_optim > best_log_optim:
            best_log_optim = log_optim
            best_optim = optim_num
            best_optim_epoch = epoch
            best_optim_weights = deepcopy(model.state_dict())
            improved_optim = True
            if args.save_trial_checkpoints and trial_id is not None and trial_ckpt_dir is not None:
                ckpt_optim_path = os.path.join(trial_ckpt_dir, f"trial_{trial_id}_bestOptim.ckpt")
                save_trial_checkpoint(
                    path=ckpt_optim_path,
                    model_state_dict=best_optim_weights,
                    epoch=epoch,
                    selector_name="best_optim_value",
                    trial_id=trial_id,
                )

        if verbose:
            print(f"  Epoch {epoch:>2d}  val_acc={100*val_acc:.1f}%  "
                  f"{args.val_metric}={val_metric:.4f}  "
                  f"log_optim={log_optim:.4f}  optim={optim_num:.6f}"
                  f"{'  *best_val*' if improved_val else ''}"
                  f"{'  *best_optim*' if improved_optim else ''}")

        # Report to Optuna (no pruning)
        if trial is not None:
            if args.checkpoint_selector == "val_acc":
                trial.report(best_val_acc, epoch)
            else:
                trial.report(best_log_optim, epoch)

    if best_val_weights is None:
        best_val_acc = last_val_acc
        best_val_epoch = epochs
        best_val_weights = deepcopy(model.state_dict())

    if best_optim_weights is None:
        best_log_optim = last_log_optim
        best_optim = float(np.exp(np.clip(best_log_optim, -700.0, 50.0)))
        best_optim_epoch = epochs
        best_optim_weights = deepcopy(model.state_dict())

    # -- Test with selected checkpoint ----------------------------------
    if args.checkpoint_selector == "val_acc":
        selected_name = "best_val_acc"
        selected_epoch = best_val_epoch
        selected_value = best_val_acc
        selected_weights = best_val_weights
    else:
        selected_name = "best_log_optim"
        selected_epoch = best_optim_epoch
        selected_value = best_log_optim
        selected_weights = best_optim_weights

    model.load_state_dict(selected_weights)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            correct += outputs.argmax(dim=1).eq(target).sum().item()
            total += data.size(0)
    test_acc = 100.0 * correct / total

    if verbose:
        print(
            f"  >> best_val_acc={best_val_acc:.4f} (epoch {best_val_epoch}), "
            f"best_log_optim={best_log_optim:.4f} (epoch {best_optim_epoch})"
        )
        if args.checkpoint_selector == "val_acc":
            print(f"  >> selected checkpoint: val_acc @ epoch {selected_epoch}, "
                  f"test_acc={test_acc:.1f}%")
        else:
            print(f"  >> selected checkpoint: log_optim @ epoch {selected_epoch}, "
                  f"test_acc={test_acc:.1f}%")

    info = {
        "best_val_acc": float(best_val_acc),
        "best_val_epoch": int(best_val_epoch),
        "best_log_optim": float(best_log_optim),
        "best_optim_value": float(best_optim),
        "best_optim_epoch": int(best_optim_epoch),
        "selected_selector": selected_name,
        "selected_epoch": int(selected_epoch),
        "selected_value": float(selected_value),
        "selector_start_mode": args.selector_start,
        "selector_start_epoch": int(selector_start_epoch),
        "ckpt_val_path": ckpt_val_path or "",
        "ckpt_optim_path": ckpt_optim_path or "",
    }

    return best_optim, test_acc, best_optim_weights, info


# =============================================================================
#  Phase 1: Optuna sweep
# =============================================================================
def suggest_float_or_const(trial, name, low, high, log=True):
    if low == high:
        return float(trial.suggest_categorical(name, [float(low)]))
    return trial.suggest_float(name, low, high, log=log)


def suggest_int_or_const(trial, name, low, high):
    if low == high:
        return int(trial.suggest_categorical(name, [int(low)]))
    return trial.suggest_int(name, low, high)


def objective(trial):
    kl_lambda = suggest_float_or_const(trial, "kl_lambda", args.kl_lambda_min, args.kl_lambda_max, log=True)
    kl_incr = args.kl_incr_fixed
    attention_epoch = suggest_int_or_const(trial, "attention_epoch", args.attention_epoch_min, args.attention_epoch_max)
    lr = suggest_float_or_const(trial, "lr", args.lr_min, args.lr_max, log=True)
    lr2 = suggest_float_or_const(trial, "lr2", args.lr2_min, args.lr2_max, log=True)
    lr2_mult = suggest_float_or_const(trial, "lr2_mult", args.lr2_mult_min, args.lr2_mult_max, log=True)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: kl_lambda={kl_lambda:.2f}, kl_incr={kl_incr:.2f} (fixed), "
          f"attn_epoch={attention_epoch}, lr={lr:.5f}, lr2={lr2:.5f}, lr2_mult={lr2_mult:.3f}")
    print(f"{'='*60}")

    best_optim, test_acc, _, info = run_training(
        seed=args.objective_seed,
        kl_lambda=kl_lambda,
        kl_incr=kl_incr,
        attention_epoch=attention_epoch,
        lr=lr,
        lr2=lr2,
        lr2_mult=lr2_mult,
        trial=trial,
        verbose=True,
        trial_id=trial.number,
    )
    if args.save_trial_checkpoints and trial_summary_csv is not None:
        append_trial_summary_row(
            trial_summary_csv,
            {
                "trial_id": trial.number,
                "seed": args.objective_seed,
                "kl_lambda": kl_lambda,
                "kl_incr": kl_incr,
                "attention_epoch": attention_epoch,
                "lr": lr,
                "lr2": lr2,
                "lr2_mult": lr2_mult,
                "selector_start_mode": info["selector_start_mode"],
                "selector_start_epoch": info["selector_start_epoch"],
                "best_val_acc": info["best_val_acc"],
                "best_val_epoch": info["best_val_epoch"],
                "best_log_optim": info["best_log_optim"],
                "best_optim_value": info["best_optim_value"],
                "best_optim_epoch": info["best_optim_epoch"],
                "selected_selector": info["selected_selector"],
                "selected_epoch": info["selected_epoch"],
                "selected_value": info["selected_value"],
                "ckpt_val_path": info["ckpt_val_path"],
                "ckpt_optim_path": info["ckpt_optim_path"],
            },
        )
    print(
        f"Trial {trial.number} finished: val_acc={info['best_val_acc']:.4f}, "
        f"log_optim={info['best_log_optim']:.4f}, "
        f"selected={info['selected_selector']}@{info['selected_epoch']}, "
        f"test_acc={test_acc:.1f}%"
    )
    if args.checkpoint_selector == "val_acc":
        return info["best_val_acc"]
    return info["best_log_optim"]


def build_dataset_with_gt(gt_path):
    """Build GuidedImageFolder with a specific GT path."""
    return GuidedImageFolder(
        image_root=os.path.join(png_root, "train"),
        mask_root=gt_path,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )


def run_manual_kl_sweep():
    print(f"\n{'#'*60}")
    print("  Manual KL sweep mode (no Optuna)")
    print(f"{'#'*60}")
    print(f"kl_lambda: logspace({args.manual_kl_min}, {args.manual_kl_max}), points={args.manual_num_points}")
    print(
        "fixed params: "
        f"attention_epoch={args.manual_attention_epoch}, "
        f"lr={args.manual_lr}, lr2={args.manual_lr2}, "
        f"lr2_mult={args.manual_lr2_mult}, kl_incr={args.kl_incr_fixed}"
    )
    print(f"selector_start={args.selector_start}")
    if args.save_trial_checkpoints and trial_summary_csv is not None:
        print(f"trial_summary_csv={trial_summary_csv}")
        if trial_ckpt_dir is not None:
            print(f"trial_ckpt_dir={trial_ckpt_dir}")
    else:
        print("WARNING: --save-trial-checkpoints is off; no selector-scatter artifacts will be saved.")

    kl_values = np.logspace(
        np.log10(args.manual_kl_min),
        np.log10(args.manual_kl_max),
        num=args.manual_num_points,
        dtype=np.float64,
    )

    for idx, kl_lambda in enumerate(kl_values):
        print(f"\n{'='*60}")
        print(f"Manual run {idx + 1}/{len(kl_values)}: kl_lambda={kl_lambda:.8f}")
        print(f"{'='*60}")
        best_optim, test_acc, _, info = run_training(
            seed=42,
            kl_lambda=float(kl_lambda),
            kl_incr=args.kl_incr_fixed,
            attention_epoch=args.manual_attention_epoch,
            lr=args.manual_lr,
            lr2=args.manual_lr2,
            lr2_mult=args.manual_lr2_mult,
            trial=None,
            verbose=True,
            trial_id=f"manual_{idx}",
        )
        if args.save_trial_checkpoints and trial_summary_csv is not None:
            append_trial_summary_row(
                trial_summary_csv,
                {
                    "trial_id": f"manual_{idx}",
                    "seed": 42,
                    "kl_lambda": float(kl_lambda),
                    "kl_incr": args.kl_incr_fixed,
                    "attention_epoch": args.manual_attention_epoch,
                    "lr": args.manual_lr,
                    "lr2": args.manual_lr2,
                    "lr2_mult": args.manual_lr2_mult,
                    "selector_start_mode": info["selector_start_mode"],
                    "selector_start_epoch": info["selector_start_epoch"],
                    "best_val_acc": info["best_val_acc"],
                    "best_val_epoch": info["best_val_epoch"],
                    "best_log_optim": info["best_log_optim"],
                    "best_optim_value": info["best_optim_value"],
                    "best_optim_epoch": info["best_optim_epoch"],
                    "selected_selector": info["selected_selector"],
                    "selected_epoch": info["selected_epoch"],
                    "selected_value": info["selected_value"],
                    "ckpt_val_path": info["ckpt_val_path"],
                    "ckpt_optim_path": info["ckpt_optim_path"],
                },
            )
        print(
            f"Manual run {idx + 1} done: best_log_optim={info['best_log_optim']:.4f}, "
            f"best_optim={best_optim:.6f}, "
            f"test_acc(best_optim)={test_acc:.2f}%"
        )

    print("\nManual KL sweep complete.")
    if args.save_trial_checkpoints and trial_summary_csv is not None:
        print(f"Per-trial summary CSV: {trial_summary_csv}")
        if trial_ckpt_dir is not None:
            print(f"Per-trial checkpoint dir: {trial_ckpt_dir}")


if __name__ == "__main__":
    if args.manual_kl_sweep:
        run_manual_kl_sweep()
        sys.exit(0)

    db_path = args.db_path or os.path.join(model_path, f"{args.study_name}.db")
    storage = f"sqlite:///{db_path}"
    print(f"\nOptuna storage: {storage}")

    sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),  # No pruning
    )

    print(f"\n{'#'*60}")
    print(f"  Phase 1: Optuna sweep (cap={args.n_trials} trials, no pruning)")
    print(f"{'#'*60}\n")
    trials_before = len(study.trials)
    remaining_trials = max(0, args.n_trials - trials_before)
    if remaining_trials > 0:
        t0 = time.time()
        study.optimize(objective, n_trials=remaining_trials)
        elapsed = time.time() - t0
        print(f"\nSweep finished after {elapsed/3600:.2f} h, "
              f"{len(study.trials)} total trials attempted.")
    else:
        print(f"Study already has {trials_before} trials (cap={args.n_trials}); skipping Phase 1 optimization.")

    best = study.best_trial
    print(f"\nBest trial #{best.number}: optim_value = {best.value:.4f}")
    print(f"  Params: {best.params}")

    bp = best.params

    if args.skip_phase2:
        print("\nSkipping Phase 2 (--skip-phase2 set).")
        if args.save_trial_checkpoints and trial_summary_csv is not None:
            print(f"Per-trial summary CSV: {trial_summary_csv}")
            if trial_ckpt_dir is not None:
                print(f"Per-trial checkpoint dir: {trial_ckpt_dir}")
        sys.exit(0)

    # =================================================================
    #  Phase 2: Evaluate best hyperparameters on all GT paths
    # =================================================================
    # Use supplied --gt-path as first, then add the other two from GT_PATHS
    gt_paths_to_eval = [("Original", args.gt_path)]
    for name, path in GT_PATHS.items():
        if path != args.gt_path:
            gt_paths_to_eval.append((name, path))

    all_results = {}  # {gt_name: [seed_results]}

    for gt_name, gt_path in gt_paths_to_eval:
        print(f"\n{'#'*60}")
        print(f"  Phase 2: {args.n_seeds} seeds with best hyperparameters")
        print(f"  GT Path: {gt_name}")
        print(f"  ({gt_path})")
        print(f"{'#'*60}\n")

        # Rebuild dataset with this GT path
        full_train = build_dataset_with_gt(gt_path)
        print(f"Rebuilt dataset with {len(full_train)} samples using {gt_name} masks")

        seed_results = []
        for i in range(args.n_seeds):
            seed = i
            print(f"\n--- Seed {seed} ({gt_name}) ---")
            best_optim, test_acc, best_weights, _ = run_training(
                seed=seed,
                kl_lambda=bp["kl_lambda"],
                kl_incr=args.kl_incr_fixed,
                attention_epoch=bp["attention_epoch"],
                lr=bp["lr"],
                lr2=bp["lr2"],
                lr2_mult=bp.get("lr2_mult", 1.0),
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
            s.dataset = "Decoy"
            s.method = f"GradCAM_Guided_{gt_name}"
            s.model_weights = best_weights
            np.random.seed()
            pid = "".join(["%s" % np.random.randint(0, 9) for _ in range(20)])
            os.makedirs(model_path, exist_ok=True)
            pkl.dump(s._dict(), open(os.path.join(model_path, pid + ".pkl"), "wb"))
            print(f"  Saved model to {os.path.join(model_path, pid + '.pkl')}")

        all_results[gt_name] = seed_results

    # -- Summary ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Best hyperparameters (from trial #{best.number}):")
    for k, v in bp.items():
        print(f"  {k}: {v}")

    for gt_name, seed_results in all_results.items():
        print(f"\n--- {gt_name} GT Path ---")
        print(f"{'Seed':>6s}  {'Optim':>8s}  {'Test Acc':>10s}")
        optims = []
        accs = []
        for r in seed_results:
            print(f"{r['seed']:>6d}  {r['optim_value']:>8.4f}  {r['test_acc']:>9.1f}%")
            optims.append(r["optim_value"])
            accs.append(r["test_acc"])
        print(f"  Optim   -- mean: {np.mean(optims):.4f}, std: {np.std(optims):.4f}")
        print(f"  TestAcc -- mean: {np.mean(accs):.1f}%, std: {np.std(accs):.1f}%")

    # Final comparison table
    print(f"\n{'='*60}")
    print("  COMPARISON ACROSS GT PATHS")
    print(f"{'='*60}")
    print(f"{'GT Path':<15s}  {'Mean Acc':>10s}  {'Std':>8s}")
    for gt_name, seed_results in all_results.items():
        accs = [r["test_acc"] for r in seed_results]
        print(f"{gt_name:<15s}  {np.mean(accs):>9.1f}%  {np.std(accs):>7.1f}%")
