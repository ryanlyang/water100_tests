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
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale
import numpy as np
import cv2
from PIL import Image
import os
import sys
import re
import pickle as pkl
from copy import deepcopy
import time
import random

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, ".."))

sys.path.append(os.path.join(_here, "DecoyMNIST"))
from params_save import S

model_path = os.path.join(_repo_root, "models", "DecoyMNIST")
os.makedirs(model_path, exist_ok=True)
torch.backends.cudnn.deterministic = True


def save(p, out_name):
    os.makedirs(model_path, exist_ok=True)
    pkl.dump(s._dict(), open(os.path.join(model_path, out_name + '.pkl'), 'wb'))


# ---------------------------------------------------------------------------
# Model: original DecoyMNIST LeNet with FC layers (can learn decoy shortcut)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Grad-CAM wrapper: hooks conv2 for features + gradients
# ---------------------------------------------------------------------------
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
        out = self.base(x)
        return out

    def grad_cam(self, targets):
        """Compute Grad-CAM for given target classes.
        Call AFTER a backward pass that populates self.gradients.
        Returns (B, H, W) saliency map, min-max normalized per sample.
        """
        # Channel weights: global average pool of gradients (B, C)
        weights = self.gradients.mean(dim=(2, 3))  # (B, C)
        # Weighted combination of feature maps
        cams = torch.einsum('bc,bchw->bhw', weights, self.features)
        cams = torch.relu(cams)
        # Min-max normalize per sample
        flat = cams.view(cams.size(0), -1)
        mn, _ = flat.min(dim=1, keepdim=True)
        mx, _ = flat.max(dim=1, keepdim=True)
        sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)
        return sal_norm


def make_gradcam_model():
    base = Net()
    return GradCAMWrap(base)


# ---------------------------------------------------------------------------
# Mask transforms
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
        class_name = os.path.basename(os.path.dirname(path))
        mask_path = self._resolve_mask_path(base, class_name)
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, label, mask


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def compute_attn_loss(cams, gt_masks):
    """Forward KL: KL(Mask || CAM)."""
    cam_flat = cams.view(cams.size(0), -1)
    gt_flat = gt_masks.view(gt_masks.size(0), -1)
    log_p = F.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    kl_div = nn.KLDivLoss(reduction='batchmean')
    return kl_div(log_p, gt_prob)


def compute_attn_losses(cams, gt_masks):
    """Both forward and reverse KL for validation."""
    cam_flat = cams.view(cams.size(0), -1)
    gt_flat = gt_masks.view(gt_masks.size(0), -1)
    log_cam = F.log_softmax(cam_flat, dim=1)
    cam_prob = F.softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    log_gt = torch.log(gt_prob + 1e-8)
    kl_div = nn.KLDivLoss(reduction='batchmean')
    forward_kl = kl_div(log_cam, gt_prob)
    reverse_kl = kl_div(log_gt, cam_prob)
    return forward_kl, reverse_kl


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch DecoyMNIST Grad-CAM Guided LeNet')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr2', type=float, default=None,
                    help='learning rate after attention_epoch restart (default: same as --lr)')
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--step-size', type=int, default=7,
                    help='StepLR decay every N epochs (default: 7)')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='StepLR decay factor (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--png-root', type=str, default=None,
                    help='root folder for PNGs (default: data/DecoyMNIST_png)')
parser.add_argument('--gt-path', type=str, required=True,
                    help='folder with ground-truth mask PNGs')
parser.add_argument('--attention-epoch', type=int, default=11,
                    help='epoch at which to restart optimizer and begin guidance (default: 11)')
parser.add_argument('--kl-lambda', type=float, default=160.0,
                    help='weight for attention KL loss (default: 160.0)')
parser.add_argument('--kl-incr', type=float, default=None,
                    help='KL lambda increase per epoch after attention_epoch (default: kl_lambda/10)')
parser.add_argument('--beta', type=float, default=0.3,
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
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
png_root = args.png_root or oj(_repo_root, "data", "DecoyMNIST_png")
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
    image_root=oj(png_root, "train"),
    mask_root=args.gt_path,
    image_transform=image_transform,
    mask_transform=mask_transform,
)

g = torch.Generator()
g.manual_seed(args.seed)
n_total = len(full_train)
n_val = max(1, int(args.val_frac * n_total))
n_train = n_total - n_val
train_subset, val_subset = torch.utils.data.random_split(full_train, [n_train, n_val], generator=g)

test_dataset = ImageFolder(oj(png_root, "test"), transform=image_transform)

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
# Model + optimizer: Adam throughout, reset at attention epoch
# ---------------------------------------------------------------------------
model = make_gradcam_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = None


# ---------------------------------------------------------------------------
# Training with Grad-CAM guidance
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

        # Forward pass
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        ce_loss = F.nll_loss(outputs, target)

        if not attention_active:
            # Pure CE — still compute Grad-CAM for monitoring
            ce_loss.backward()

            # Compute Grad-CAM (uses gradients from ce_loss backward)
            with torch.no_grad():
                sal_norm = model.grad_cam(target)
                gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                         mode='nearest').squeeze(1)
                attn_loss = compute_attn_loss(sal_norm, gt_small)

            optimizer.step()
        else:
            # Need Grad-CAM gradients, so: backward class scores to get
            # conv2 gradients, compute attn loss, then do the real backward.
            #
            # Step 1: Get Grad-CAM by backpropagating class scores through conv2
            # We use the logits (pre-softmax) for Grad-CAM, not the log_softmax output
            model.zero_grad()
            logits = model.base.logits(data)
            # Gather the target class scores
            class_scores = logits[torch.arange(len(target), device=device), target]
            class_scores.sum().backward(retain_graph=True)

            # Now model.gradients is populated — compute Grad-CAM
            sal_norm = model.grad_cam(target)
            gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                     mode='nearest').squeeze(1)
            attn_loss = compute_attn_loss(sal_norm, gt_small)

            # Step 2: Actual training loss = CE + KL * lambda
            optimizer.zero_grad()
            outputs2 = model(data)
            ce_loss = F.nll_loss(outputs2, target)
            total_loss = ce_loss + kl_lambda_real * attn_loss
            total_loss.backward()
            optimizer.step()

        running_loss += ce_loss.item() * data.size(0)
        running_corrects += preds.eq(target).sum().item()
        running_attn += attn_loss.item() * data.size(0)
        total += data.size(0)

        if batch_idx % args.log_interval == 0:
            acc = 100. * preds.eq(target).sum().item() / len(target)
            s.losses_train.append(ce_loss.item())
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

    for data, target, gt_masks in loader:
        data, target = data.to(device), target.to(device)
        gt_masks = gt_masks.to(device)

        # Need gradients for Grad-CAM even during validation
        data.requires_grad_(False)
        logits = model.base.logits(data)
        class_scores = logits[torch.arange(len(target), device=device), target]
        model.zero_grad()
        class_scores.sum().backward()

        sal_norm = model.grad_cam(target)
        gt_small = F.interpolate(gt_masks, size=sal_norm.shape[1:],
                                 mode='nearest').squeeze(1)

        with torch.no_grad():
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            ce_loss = F.nll_loss(outputs, target, reduction='sum')

            fwd_kl, rev_kl = compute_attn_losses(sal_norm.detach(), gt_small)

            running_loss += ce_loss.item()
            running_corrects += preds.eq(target).sum().item()
            running_attn_fwd += fwd_kl.item() * data.size(0)
            running_attn_rev += rev_kl.item() * data.size(0)
            total += data.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    epoch_attn_fwd = running_attn_fwd / total
    epoch_attn_rev = running_attn_rev / total

    optim_num = epoch_acc * np.exp(-args.beta * epoch_attn_rev)

    s.losses_test.append(epoch_loss)
    s.accs_test.append(100. * epoch_acc)

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
        outputs = model(data)
        test_loss += F.nll_loss(outputs, target, reduction='sum').item()
        preds = outputs.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += data.size(0)

    test_loss /= total
    acc = 100. * correct / total
    print(f'Epoch {epoch} Test: Loss: {test_loss:.4f}, Acc: {correct}/{total} ({acc:.1f}%)')
    return test_loss, acc


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
best_model_weights = None
best_optim = -100.0
kl_lambda_real = args.kl_lambda

start = time.time()

for epoch in range(1, args.epochs + 1):
    attention_active = (epoch >= args.attention_epoch) and (args.kl_lambda > 0)

    if epoch == args.attention_epoch and args.kl_lambda > 0:
        print(f'\n*** Attention epoch {epoch}: resetting optimizer & beginning guidance ***')
        optimizer = optim.Adam(model.parameters(), lr=args.lr2, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                              gamma=args.gamma)
        best_model_weights = deepcopy(model.state_dict())
        best_optim = -100.0
        kl_lambda_real = args.kl_lambda

    if epoch > args.attention_epoch and args.kl_lambda > 0:
        kl_lambda_real += args.kl_incr

    train_epoch(model, device, train_loader, optimizer, epoch,
                kl_lambda_real, attention_active)
    if scheduler is not None:
        scheduler.step()

    optim_num = validate(model, device, val_loader, epoch)

    if epoch >= args.attention_epoch and optim_num > best_optim:
        best_optim = optim_num
        best_model_weights = deepcopy(model.state_dict())

    if epoch < args.attention_epoch:
        best_model_weights = deepcopy(model.state_dict())

    evaluate_test(model, device, test_loader, epoch)
    print()

end = time.time()
s.time_per_epoch = (end - start) / (epoch)

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

print('=== Final evaluation with best model ===')
s.dataset = "Decoy"
test_loss, test_acc = evaluate_test(model, device, test_loader, args.epochs)
s.method = "GradCAM_Guided"
s.model_weights = best_model_weights

np.random.seed()
pid = ''.join(["%s" % np.random.randint(0, 9) for num in range(0, 20)])
save(s, pid)
