#!/usr/bin/env python3
"""Run DecoyMNIST Grad-CAM guided training and report val + test every epoch."""
from __future__ import print_function

import argparse
import os
import random
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale
from PIL import Image


# -- Model --------------------------------------------------------------------
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

    def logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
        weights = self.gradients.mean(dim=(2, 3))
        cams = torch.einsum("bc,bchw->bhw", weights, self.features)
        cams = torch.relu(cams)
        flat = cams.view(cams.size(0), -1)
        mn, _ = flat.min(dim=1, keepdim=True)
        mx, _ = flat.max(dim=1, keepdim=True)
        sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)
        return sal_norm


def make_gradcam_model():
    return GradCAMWrap(Net())


class Brighten:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, mask):
        return torch.clamp(mask * self.factor, 0.0, 1.0)


class GuidedImageFolder(utils.Dataset):
    def __init__(self, image_root, mask_root, image_transform=None, mask_transform=None):
        self.images = ImageFolder(image_root, transform=image_transform)
        self.mask_root = mask_root
        self.mask_transform = mask_transform
        self._mask_exts = (".png", ".jpg", ".jpeg")

    def _resolve_mask_path(self, base, class_name):
        candidates = [f"{class_name}_{base}", base]
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


def compute_attn_loss(cams, gt_masks):
    cam_flat = cams.view(cams.size(0), -1)
    gt_flat = gt_masks.view(gt_masks.size(0), -1)
    log_p = F.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    kl_div = nn.KLDivLoss(reduction="batchmean")
    return kl_div(log_p, gt_prob)


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


def compute_outside_mass(saliency, gt_masks):
    sal = saliency.view(saliency.size(0), -1)
    gt = gt_masks.view(gt_masks.size(0), -1)
    sal = sal / (sal.sum(dim=1, keepdim=True) + 1e-8)
    outside = sal * (1.0 - gt)
    return outside.sum(dim=1).mean()


def input_grad_saliency(model, data, target, device):
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss_sum += F.nll_loss(outputs, target, reduction="sum").item()
        correct += outputs.argmax(dim=1).eq(target).sum().item()
        total += data.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="DecoyMNIST: val + test every epoch")
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--png-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--val-frac", type=float, default=0.16)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kl-lambda", type=float, default=100.0)
    parser.add_argument("--attention-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr2", type=float, default=5e-4)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--val-saliency", type=str, default="cam",
                        choices=["cam", "igrad"])
    parser.add_argument("--val-metric", type=str, default="rev_kl",
                        choices=["rev_kl", "fwd_kl", "outside"])
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, ".."))
    png_root = args.png_root or os.path.join(_repo_root, "data", "DecoyMNIST_png")

    seed_everything(args.seed)

    image_transform = Compose([Grayscale(num_output_channels=1), ToTensor(),
                               Lambda(lambda x: x * 2.0 - 1.0)])
    mask_transform = transforms.Compose([
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

    g = torch.Generator().manual_seed(args.seed)
    n_total = len(full_train)
    n_val = max(1, int(args.val_frac * n_total))
    n_train = n_total - n_val
    train_subset, val_subset = utils.random_split(full_train, [n_train, n_val], generator=g)
    train_loader = utils.DataLoader(train_subset, batch_size=args.batch_size,
                                    shuffle=True, **loader_kwargs)
    val_loader = utils.DataLoader(val_subset, batch_size=args.batch_size,
                                  shuffle=False, **loader_kwargs)

    model = make_gradcam_model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    kl_incr = args.kl_lambda / 10.0
    kl_lambda_real = args.kl_lambda
    best_optim = -100.0
    best_weights = None

    since = time.time()

    for epoch in range(1, args.epochs + 1):
        attention_active = (epoch >= args.attention_epoch) and (args.kl_lambda > 0)

        if epoch == args.attention_epoch and args.kl_lambda > 0:
            print(f"*** Attention epoch {epoch}: resetting optimizer & beginning guidance ***")
            optimizer = optim.Adam(model.parameters(), lr=args.lr2, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            best_optim = -100.0
            best_weights = deepcopy(model.state_dict())
            kl_lambda_real = args.kl_lambda

        if epoch > args.attention_epoch and args.kl_lambda > 0:
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
                ce_loss.backward()
                optimizer.step()
            else:
                model.zero_grad()
                logits = model.base.logits(data)
                class_scores = logits[torch.arange(len(target), device=device), target]
                class_scores.sum().backward(retain_graph=True)

                sal_norm = model.grad_cam(target)
                gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                         mode="nearest").squeeze(1)
                attn_loss = compute_attn_loss(sal_norm, gt_small)

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
        running_rev_kl = 0.0
        running_fwd_kl = 0.0
        running_outside = 0.0
        total = 0

        for data, target, gt_masks in val_loader:
            data, target = data.to(device), target.to(device)
            gt_masks = gt_masks.to(device)

            if args.val_saliency == "cam":
                logits = model.base.logits(data)
                class_scores = logits[torch.arange(len(target), device=device), target]
                model.zero_grad()
                class_scores.sum().backward()
                sal_norm = model.grad_cam(target)
            else:
                sal_norm = input_grad_saliency(model, data, target, device)

            gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                     mode="nearest").squeeze(1)

            with torch.no_grad():
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                fwd_kl, rev_kl = compute_attn_losses(sal_norm.detach(), gt_small)
                outside = compute_outside_mass(sal_norm.detach(), gt_small)

                running_corrects += preds.eq(target).sum().item()
                running_fwd_kl += fwd_kl.item() * data.size(0)
                running_rev_kl += rev_kl.item() * data.size(0)
                running_outside += outside.item() * data.size(0)
                total += data.size(0)

        val_acc = running_corrects / total
        val_fwd = running_fwd_kl / total
        val_rev = running_rev_kl / total
        val_out = running_outside / total
        if args.val_metric == "rev_kl":
            metric_for_optim = val_rev
        elif args.val_metric == "fwd_kl":
            metric_for_optim = val_fwd
        else:
            metric_for_optim = val_out
        optim_num = val_acc * np.exp(-args.beta * metric_for_optim)

        if epoch >= args.attention_epoch and optim_num > best_optim:
            best_optim = optim_num
            best_weights = deepcopy(model.state_dict())
        if epoch < args.attention_epoch:
            best_weights = deepcopy(model.state_dict())

        test_loss, test_acc = evaluate_test(model, test_loader, device)

        print(f"Epoch {epoch:>2d}  val_acc={100*val_acc:.1f}%  "
              f"rev_kl={val_rev:.4f}  outside={val_out:.4f}  "
              f"optim={optim_num:.4f}  test_loss={test_loss:.4f}  "
              f"test_acc={test_acc:.2f}%"
              f"{'  *best*' if optim_num >= best_optim else ''}")

    # -- Final eval with best weights ------------------------------------
    if best_weights is not None:
        model.load_state_dict(best_weights)
    final_loss, final_acc = evaluate_test(model, test_loader, device)
    elapsed = time.time() - since
    print(f"\nFinal @ best optim: test_loss={final_loss:.4f}  test_acc={final_acc:.2f}%")
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.checkpoint_dir, f"decoy_valtest_{ts}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
