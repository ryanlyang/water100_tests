from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from os.path import join as oj
import torch.utils.data as utils
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda, Normalize
import numpy as np
import cv2
from PIL import Image
import os
import sys
import re
import pickle as pkl
from copy import deepcopy
import random

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, ".."))

sys.path.append(os.path.join(_here, "ColorMNIST"))
from params_save import S

model_path = os.path.join(_repo_root, "models", "ColorMNIST_test")
os.makedirs(model_path, exist_ok=True)
torch.backends.cudnn.deterministic = True


def save(p, out_name):
    os.makedirs(model_path, exist_ok=True)
    pkl.dump(s._dict(), open(os.path.join(model_path, out_name + '.pkl'), 'wb'))


# ---------------------------------------------------------------------------
# Model: same conv backbone as original Net (20/50 channels), but with
# AdaptiveAvgPool + single Linear instead of fc1+fc2, so CAMs work.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Mask transforms (from guided strategy)
# ---------------------------------------------------------------------------
class ExpandWhite(object):
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


class EdgeExtract(object):
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


class Brighten(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, mask):
        return torch.clamp(mask * self.factor, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset that pairs each image with its ground-truth mask
# ---------------------------------------------------------------------------
class GuidedImageFolder(utils.Dataset):
    def __init__(self, image_root, mask_root, image_transform=None, mask_transform=None):
        self.images = ImageFolder(image_root, transform=image_transform)
        self.mask_root = mask_root
        self.mask_transform = mask_transform
        self._mask_exts = (".png", ".jpg", ".jpeg")

    def _resolve_mask_path(self, base, class_name):
        # Try: {class}_{base}.ext  (e.g. 1_012687_y1.png)
        # Also try: {base}.ext directly, and stripping _lbl suffixes
        candidates = [
            f"{class_name}_{base}",
            base,
        ]
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
        raise FileNotFoundError(f"Mask not found for base='{base}', class='{class_name}'. Tried: {tried}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx]
        path, _ = self.images.samples[idx]
        base = os.path.splitext(os.path.basename(path))[0]
        # class name from the parent directory (ImageFolder convention)
        class_name = os.path.basename(os.path.dirname(path))
        mask_path = self._resolve_mask_path(base, class_name)
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, label, mask


# ---------------------------------------------------------------------------
# Loss functions (from guided strategy)
# ---------------------------------------------------------------------------
def compute_loss(outputs, labels, cams, gt_masks, kl_lambda, only_ce):
    ce_loss = F.cross_entropy(outputs, labels)
    B, Hf, Wf = cams.shape
    cam_flat = cams.view(B, -1)
    gt_flat = gt_masks.view(B, -1)
    log_p = F.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    kl_div = nn.KLDivLoss(reduction='batchmean')
    attn_loss = kl_div(log_p, gt_prob)
    if only_ce:
        return ce_loss, attn_loss
    else:
        return ce_loss + kl_lambda * attn_loss, attn_loss


def compute_attn_losses(cams, gt_masks):
    B, Hf, Wf = cams.shape
    cam_flat = cams.view(B, -1)
    gt_flat = gt_masks.view(B, -1)
    log_cam = F.log_softmax(cam_flat, dim=1)
    cam_prob = F.softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    log_gt = torch.log(gt_prob + 1e-8)
    kl_div = nn.KLDivLoss(reduction='batchmean')
    forward_kl = kl_div(log_cam, gt_prob)   # KL(Mask || CAM)
    reverse_kl = kl_div(log_gt, cam_prob)   # KL(CAM || Mask)
    return forward_kl, reverse_kl


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ColorMNIST Guided LeNet Training')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.98, metavar='M',
                    help='SGD momentum (default: 0.98)')
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--step-size', type=int, default=7,
                    help='StepLR decay every N epochs (default: 7)')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='StepLR decay factor (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--png-root', type=str, default=None,
                    help='root folder for PNGs (default: data/ColorMNIST_png)')
parser.add_argument('--gt-path', type=str, required=True,
                    help='folder with ground-truth mask PNGs')
parser.add_argument('--attention-epoch', type=int, default=11,
                    help='epoch at which to restart optimizer and begin guidance (default: 11)')
parser.add_argument('--kl-lambda', type=float, default=160.0,
                    help='weight for attention KL loss (default: 160.0)')
parser.add_argument('--kl-incr', type=float, default=None,
                    help='KL lambda increase per epoch after attention_epoch (default: kl_lambda/10)')
parser.add_argument('--lr2', type=float, default=None,
                    help='learning rate after attention_epoch restart (default: same as --lr)')
parser.add_argument('--beta', type=float, default=0.1,
                    help='weight for reverse KL in optim_num (default: 0.3)')
parser.add_argument('--val-frac', type=float, default=0.16,
                    help='fraction of training data for internal val (default: 0.16)')

args = parser.parse_args()
if args.kl_incr is None:
    args.kl_incr = args.kl_lambda / 10.0
if args.lr2 is None:
    args.lr2 = args.lr

s = S(args.epochs)
s.regularizer_rate = args.kl_lambda
s.num_blobs = 0
s.seed = args.seed

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
png_root = args.png_root or oj(_repo_root, "data", "ColorMNIST_png")

# Compute normalization stats from raw training images
base_transform = Compose([ToTensor(), Lambda(lambda x: x * 255.0)])
train_dataset_raw = ImageFolder(oj(png_root, "train"), transform=base_transform)
mean, std = _compute_mean_std(train_dataset_raw)
norm = Normalize(mean.tolist(), std.tolist())
full_transform = Compose([base_transform, norm])

mask_transform = transforms.Compose([
    ExpandWhite(thr=10, radius=4),
    EdgeExtract(thr=10, edge_width=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    Brighten(8.0),
])

# Full guided dataset (train images + masks)
full_train = GuidedImageFolder(
    image_root=oj(png_root, "train"),
    mask_root=args.gt_path,
    image_transform=full_transform,
    mask_transform=mask_transform,
)

# Internal train/val split
g = torch.Generator()
g.manual_seed(args.seed)
n_total = len(full_train)
n_val = max(1, int(args.val_frac * n_total))
n_train = n_total - n_val
train_subset, val_subset = torch.utils.data.random_split(full_train, [n_train, n_val], generator=g)

# Test set (plain ImageFolder, no masks needed)
test_dataset = ImageFolder(oj(png_root, "test"), transform=full_transform)

kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
train_loader = utils.DataLoader(train_subset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = utils.DataLoader(val_subset,
        batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = utils.DataLoader(test_dataset,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

print(f"Train: {n_train}, Val: {n_val}, Test: {len(test_dataset)}")
print(f"Attention epoch: {args.attention_epoch}, KL lambda: {args.kl_lambda}, "
      f"KL incr: {args.kl_incr}, beta: {args.beta}")

# ---------------------------------------------------------------------------
# Model + optimizer
# ---------------------------------------------------------------------------
model = make_cam_model(num_classes=10).to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, device, loader, optimizer, epoch, kl_lambda_real, attention_active):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_attn = 0.0
    total = 0

    for batch_idx, (data, target, gt_masks) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        gt_masks = gt_masks.to(device)

        optimizer.zero_grad()
        outputs, feats = model(data)
        _, preds = torch.max(outputs, 1)

        # Compute CAMs
        weights = model.base.classifier.weight[target]  # (B, 50)
        cams = torch.einsum('bc,bchw->bhw', weights, feats)
        cams = torch.relu(cams)

        # Min-max normalize
        flat = cams.view(cams.size(0), -1)
        mn, _ = flat.min(dim=1, keepdim=True)
        mx, _ = flat.max(dim=1, keepdim=True)
        sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)

        gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                 mode='nearest').squeeze(1)

        if not attention_active:
            # Before attention epoch: CE only, but still track attn loss
            loss, attn_loss = compute_loss(outputs, target, sal_norm, gt_small, 0, True)
        else:
            loss, attn_loss = compute_loss(outputs, target, sal_norm, gt_small,
                                           kl_lambda_real, False)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        running_corrects += preds.eq(target).sum().item()
        running_attn += attn_loss.item() * data.size(0)
        total += data.size(0)

        if batch_idx % args.log_interval == 0:
            acc = 100. * preds.eq(target).sum().item() / len(target)
            s.losses_train.append(loss.item())
            s.accs_train.append(acc)
            s.cd.append(attn_loss.item())

    epoch_loss = running_loss / total
    epoch_acc = 100. * running_corrects / total
    epoch_attn = running_attn / total
    print(f'Epoch {epoch} Train: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.1f}%, '
          f'Attn: {epoch_attn:.4f}')


def validate(model, device, loader, epoch):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    running_attn_fwd = 0.0
    running_attn_rev = 0.0
    total = 0

    with torch.no_grad():
        for data, target, gt_masks in loader:
            data, target = data.to(device), target.to(device)
            gt_masks = gt_masks.to(device)

            outputs, feats = model(data)
            _, preds = torch.max(outputs, 1)
            ce_loss = F.cross_entropy(outputs, target, reduction='sum')

            # Compute CAMs
            weights = model.base.classifier.weight[target]
            cams = torch.einsum('bc,bchw->bhw', weights, feats)
            cams = torch.relu(cams)
            flat = cams.view(cams.size(0), -1)
            mn, _ = flat.min(dim=1, keepdim=True)
            mx, _ = flat.max(dim=1, keepdim=True)
            sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)

            gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                     mode='nearest').squeeze(1)

            fwd_kl, rev_kl = compute_attn_losses(sal_norm, gt_small)

            running_loss += ce_loss.item()
            running_corrects += preds.eq(target).sum().item()
            running_attn_fwd += fwd_kl.item() * data.size(0)
            running_attn_rev += rev_kl.item() * data.size(0)
            total += data.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    epoch_attn_fwd = running_attn_fwd / total
    epoch_attn_rev = running_attn_rev / total

    # optim_num: accuracy weighted by how well CAMs match masks
    optim_num = epoch_acc * np.exp(-args.beta * epoch_attn_rev)

    s.losses_dev.append(epoch_loss)
    s.accs_dev.append(100. * epoch_acc)

    print(f'Epoch {epoch} Val: Loss: {epoch_loss:.4f}, Acc: {100.*epoch_acc:.1f}%, '
          f'Attn_fwd: {epoch_attn_fwd:.4f}, Attn_rev: {epoch_attn_rev:.4f}, '
          f'Optim: {optim_num:.4f}')
    return optim_num


@torch.no_grad()
def evaluate_test(model, device, loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        outputs, _ = model(data)
        test_loss += F.cross_entropy(outputs, target, reduction='sum').item()
        preds = outputs.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += data.size(0)

    test_loss /= total
    acc = 100. * correct / total
    s.acc_test = acc
    s.loss_test = test_loss
    print(f'Epoch {epoch} Test: Loss: {test_loss:.4f}, Acc: {correct}/{total} ({acc:.1f}%)')
    return test_loss, acc


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
best_model_weights = None
best_optim = -100.0
kl_lambda_real = args.kl_lambda

for epoch in range(1, args.epochs + 1):
    attention_active = (epoch >= args.attention_epoch) and (args.kl_lambda > 0)

    # Restart optimizer at attention epoch
    if epoch == args.attention_epoch and args.kl_lambda > 0:
        print(f'\n*** Attention epoch {epoch}: restarting optimizer & scheduler ***')
        optimizer = optim.SGD(model.parameters(), lr=args.lr2, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                              gamma=args.gamma)
        best_model_weights = deepcopy(model.state_dict())
        best_optim = -100.0
        kl_lambda_real = args.kl_lambda

    # Increase KL weight after attention epoch
    if epoch > args.attention_epoch and args.kl_lambda > 0:
        kl_lambda_real += args.kl_incr

    train_epoch(model, device, train_loader, optimizer, epoch,
                kl_lambda_real, attention_active)
    scheduler.step()

    optim_num = validate(model, device, val_loader, epoch)

    # Model selection by optim_num (only after attention epoch)
    if epoch >= args.attention_epoch and optim_num > best_optim:
        best_optim = optim_num
        best_model_weights = deepcopy(model.state_dict())

    # Before attention epoch, always keep latest as best
    if epoch < args.attention_epoch:
        best_model_weights = deepcopy(model.state_dict())

    # Run test each epoch for monitoring
    evaluate_test(model, device, test_loader, epoch)
    print()

# Load best model and do final test
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

print('=== Final evaluation with best model ===')
s.dataset = "Color"
evaluate_test(model, device, test_loader, args.epochs)
s.method = "Guided"

np.random.seed()
pid = ''.join(["%s" % np.random.randint(0, 9) for num in range(0, 20)])
save(s, pid)
