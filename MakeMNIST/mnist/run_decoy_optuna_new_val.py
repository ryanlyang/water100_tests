"""Optuna hyperparameter sweep for Grad-CAM guided LeNet on DecoyMNIST.

Phase 1 (100 trials): Optuna explores kl_lambda, kl_incr, attention_epoch,
                      lr, lr2. StepLR settings are fixed.
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
parser.add_argument("--beta", type=float, default=0.3)
parser.add_argument("--val-frac", type=float, default=0.16)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--log-interval", type=int, default=100)
parser.add_argument("--n-trials", type=int, default=100,
                    help="Number of Optuna trials to run (default: 100)")
parser.add_argument("--n-seeds", type=int, default=5,
                    help="Number of seeds for final evaluation per GT path (default: 5)")
parser.add_argument("--study-name", type=str, default="decoymnist_gradcam")
parser.add_argument("--db-path", type=str, default=None,
                    help="SQLite path for Optuna storage (default: auto)")
parser.add_argument("--val-saliency", type=str, default="cam",
                    choices=["cam", "igrad"],
                    help="Saliency used for validation metric: Grad-CAM or input gradients.")
parser.add_argument("--val-metric", type=str, default="rev_kl",
                    choices=["rev_kl", "fwd_kl", "outside"],
                    help="Validation metric against GT masks.")

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
loader_kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
png_root = args.png_root or os.path.join(_repo_root, "data", "DecoyMNIST_png")


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
    """Compute |d(logit_target)/d(input)| saliency, normalized to [0,1]."""
    data = data.requires_grad_(True)
    model.zero_grad()
    logits = model.base.logits(data)
    class_scores = logits[torch.arange(len(target), device=device), target]
    class_scores.sum().backward()
    grads = data.grad.detach().abs().sum(dim=1)  # B,H,W
    flat = grads.view(grads.size(0), -1)
    mn, _ = flat.min(dim=1, keepdim=True)
    mx, _ = flat.max(dim=1, keepdim=True)
    sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(grads)
    return sal_norm


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
def run_training(seed, kl_lambda, kl_incr, attention_epoch, lr, lr2,
                 epochs=None, trial=None, verbose=True):
    """Train one run.  Returns (best_optim_value, test_acc, best_weights)."""
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

    # Adam throughout, reset at attention epoch with lr2 + StepLR (fixed)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = None

    best_model_weights = None
    best_optim = -100.0
    kl_lambda_real = kl_lambda

    for epoch in range(1, epochs + 1):
        attention_active = (epoch >= attention_epoch) and (kl_lambda > 0)

        # Reset optimizer at attention epoch
        if epoch == attention_epoch and kl_lambda > 0:
            if verbose:
                print(f"  *** Attention epoch {epoch}: resetting optimizer & beginning guidance ***")
            optimizer = optim.Adam(model.parameters(), lr=lr2, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
            best_model_weights = deepcopy(model.state_dict())
            best_optim = -100.0
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
                # Input gradients (RRR-style)
                sal_norm = input_grad_saliency(model, data, target)

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
        optim_num = val_acc * np.exp(-args.beta * val_metric)

        if epoch >= attention_epoch and optim_num > best_optim:
            best_optim = optim_num
            best_model_weights = deepcopy(model.state_dict())

        if epoch < attention_epoch:
            best_model_weights = deepcopy(model.state_dict())

        if verbose:
            print(f"  Epoch {epoch:>2d}  val_acc={100*val_acc:.1f}%  "
                  f"{args.val_metric}={val_metric:.4f}  optim={optim_num:.4f}"
                  f"{'  *best*' if optim_num >= best_optim else ''}")

        # Report to Optuna (no pruning)
        if trial is not None:
            trial.report(best_optim, epoch)

    # -- Test with best-optim checkpoint ----------------------------------
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

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
        print(f"  >> best optim_value = {best_optim:.4f},  "
              f"test_acc @ best_optim = {test_acc:.1f}%")

    return best_optim, test_acc, best_model_weights


# =============================================================================
#  Phase 1: Optuna sweep
# =============================================================================
def objective(trial):
    kl_lambda = trial.suggest_float("kl_lambda", 1.0, 500.0, log=True)
    kl_incr = trial.suggest_float("kl_incr", 0.1, 50.0, log=True)
    attention_epoch = trial.suggest_int("attention_epoch", 1, 25)
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    lr2 = trial.suggest_float("lr2", 1e-5, 5e-2, log=True)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: kl_lambda={kl_lambda:.2f}, kl_incr={kl_incr:.2f}, "
          f"attn_epoch={attention_epoch}, lr={lr:.5f}, lr2={lr2:.5f}")
    print(f"{'='*60}")

    best_optim, test_acc, _ = run_training(
        seed=42,
        kl_lambda=kl_lambda,
        kl_incr=kl_incr,
        attention_epoch=attention_epoch,
        lr=lr,
        lr2=lr2,
        trial=trial,
        verbose=True,
    )
    print(f"Trial {trial.number} finished: optim={best_optim:.4f}, test_acc={test_acc:.1f}%")
    return best_optim


def build_dataset_with_gt(gt_path):
    """Build GuidedImageFolder with a specific GT path."""
    return GuidedImageFolder(
        image_root=os.path.join(png_root, "train"),
        mask_root=gt_path,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )


if __name__ == "__main__":
    db_path = args.db_path or os.path.join(model_path, f"{args.study_name}.db")
    storage = f"sqlite:///{db_path}"
    print(f"\nOptuna storage: {storage}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
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
            best_optim, test_acc, best_weights = run_training(
                seed=seed,
                kl_lambda=bp["kl_lambda"],
                kl_incr=bp["kl_incr"],
                attention_epoch=bp["attention_epoch"],
                lr=bp["lr"],
                lr2=bp["lr2"],
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
