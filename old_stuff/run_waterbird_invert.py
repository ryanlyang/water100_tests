import os
import time
import copy
import argparse
from datetime import datetime
import random

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
import pandas as pd




batch_size = 96
num_epochs = 200
base_lr = 0.01
classifier_lr = 0.01
momentum = 0.9
weight_decay = 1e-5

checkpoint_dir = "ResNet_Checkpoints"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0



def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class ExpandWhite(object):
    def __init__(self, thr: int = 10, radius: int = 3):
        self.thr = thr
        self.radius = radius
    def __call__(self, mask: Image.Image) -> Image.Image:
        arr = np.array(mask)
        white = (arr > self.thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * self.radius + 1, 2 * self.radius + 1))
        dil = cv2.dilate(white, k, iterations=1)
        return Image.fromarray((dil * 255).astype(np.uint8))


class EdgeExtract(object):
    def __init__(self, thr: int = 10, edge_width: int = 1):
        self.thr = thr
        self.edge_width = edge_width
    def __call__(self, mask: Image.Image) -> Image.Image:
        arr = np.array(mask)
        white = (arr > self.thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * self.edge_width + 1, 2 * self.edge_width + 1))
        edge = cv2.morphologyEx(white, cv2.MORPH_GRADIENT, k)
        return Image.fromarray((edge * 255).astype(np.uint8))


class Brighten(object):
    def __init__(self, factor: float):
        self.factor = factor
    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.clamp(mask * self.factor, 0.0, 1.0)



def mask_name_from_path(image_path):
    folder = os.path.basename(os.path.dirname(image_path))
    folder_prefix = folder.replace('.', '_')
    base = os.path.splitext(os.path.basename(image_path))[0]
    return f"{folder_prefix}_{base}.png"


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        feats = x
        gap = self.gap(feats).view(feats.size(0), -1)
        out = self.classifier(gap)
        return out


class ResNetCAM(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.base = models.resnet50(pretrained=pretrained)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        self.classifier = self.base.fc
        self.features = None
        self.base.layer4.register_forward_hook(self._hook_fn)
    def _hook_fn(self, module, inp, out):
        self.features = out
    def forward(self, x):
        out = self.base(x)
        return out, self.features


def make_cam_model(num_classes, model_name="resnet50", pretrained=True):
    if model_name == "lenet":
        base = LeNet(num_classes)
        class CAMWrap(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model
                self.features = None
                self.classifier = base_model.classifier
                self.base.conv2.register_forward_hook(self._hook_fn)
            def _hook_fn(self, module, inp, out):
                self.features = out
            def forward(self, x):
                out = self.base(x)
                return out, self.features
        return CAMWrap(base)
    if model_name == "resnet50":
        return ResNetCAM(num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model_name: {model_name}")



class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


class GuidedImageFolder(Dataset):
    def __init__(self, image_root: str, mask_root: str, image_transform=None, mask_transform=None):
        self.images = datasets.ImageFolder(image_root, transform=image_transform)
        self.mask_root = mask_root
        self.mask_transform = mask_transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img, label = self.images[idx]
        path, _ = self.images.samples[idx]
        mask_path = os.path.join(self.mask_root, mask_name_from_path(path))
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, label, mask, path


GROUP_NAMES = ['Land_on_Land', 'Land_on_Water', 'Water_on_Land', 'Water_on_Water']


class WaterbirdsMetadataDataset(Dataset):
    SPLIT_MAP = {'train': 0, 'val': 1, 'test': 2}

    def __init__(self, data_root, split, image_transform=None, mask_root=None,
                 mask_transform=None, return_mask=False, return_path=True, return_group=False):
        self.data_root = data_root
        self.image_transform = image_transform
        self.mask_root = mask_root
        self.mask_transform = mask_transform
        self.return_mask = return_mask
        self.return_path = return_path
        self.return_group = return_group

        metadata_path = os.path.join(self.data_root, 'metadata.csv')
        df = pd.read_csv(metadata_path)
        split_id = self.SPLIT_MAP[split]
        df = df[df['split'] == split_id]

        self.paths = [os.path.join(self.data_root, p) for p in df['img_filename'].values]
        self.labels = df['y'].astype(int).values
        self.places = df['place'].astype(int).values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = int(self.labels[idx])
        img = Image.open(path).convert('RGB')
        if self.image_transform is not None:
            img = self.image_transform(img)

        output = [img, label]

        if self.return_mask:
            mask_path = os.path.join(self.mask_root, mask_name_from_path(path))
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
            output.append(mask)

        if self.return_path:
            output.append(path)

        if self.return_group:
            group = int(label * 2 + self.places[idx])
            output.append(group)

        return tuple(output)



def compute_loss(outputs, labels, cams, gt_masks, kl_lambda, only_attn):
    ce_loss = nn.functional.cross_entropy(outputs, labels)
    B, Hf, Wf = cams.shape
    cam_flat = cams.view(B, -1)
    gt_flat = gt_masks.view(B, -1)
    log_p = nn.functional.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    kl_div = nn.KLDivLoss(reduction='batchmean')
    attn_loss = kl_div(log_p, gt_prob)
    if only_attn:
        return attn_loss, attn_loss
    else:
        return ce_loss + kl_lambda * attn_loss, attn_loss



def _get_param_groups(model, base_lr, classifier_lr):
    base_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if '.classifier' in name or '.fc' in name:
            classifier_params.append(param)
        else:
            base_params.append(param)
    if not classifier_params:
        classifier_params = list(model.parameters())
        base_params = []
    param_groups = []
    if base_params:
        param_groups.append({'params': base_params, 'lr': base_lr})
    param_groups.append({'params': classifier_params, 'lr': classifier_lr})
    return param_groups


def train_model(model, dataloaders, dataset_sizes,
                attention_epoch, kl_lambda_start, num_epochs,
                base_lr, classifier_lr, kl_incr, use_attention, num_classes):
    best_wts = copy.deepcopy(model.state_dict())
    best_optim = -100.0
    best_epoch = -1
    since = time.time()

    param_groups = _get_param_groups(model, base_lr, classifier_lr)
    opt = optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    sch = None

    kl_lambda_real = kl_lambda_start

    for epoch in range(num_epochs):
        # restart at attention_epoch
        if use_attention and epoch == attention_epoch:
            print(f"*** Attention epoch {epoch} reached: restarting optimizer ***")
            param_groups = _get_param_groups(model, base_lr, classifier_lr)
            opt = optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
            best_wts = copy.deepcopy(model.state_dict())
            best_optim = -100.0

        # increase KL after attention_epoch
        if use_attention and epoch > attention_epoch:
            kl_lambda_real += kl_incr

        print(f"Epoch {epoch + 1}/{num_epochs}")

        for phase in ['train', 'val']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_attn_loss = 0.0
            class_correct = np.zeros(num_classes, dtype=np.int64)
            class_total = np.zeros(num_classes, dtype=np.int64)

            for batch in dataloaders[phase]:
                if len(batch) == 4:
                    inputs, labels, gt_masks, paths = batch
                    gt_masks = gt_masks.to(device)
                    has_masks = True
                elif len(batch) == 3:
                    inputs, labels, paths = batch
                    gt_masks = None
                    has_masks = False
                else:
                    raise RuntimeError("Unexpected batch format.")

                # attention used on both train and val
                use_attention_this_batch = use_attention and has_masks

                inputs, labels = inputs.to(device), labels.to(device).long()
                if is_train:
                    opt.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs, feats = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    if use_attention_this_batch and has_masks:
                        weights = model.classifier.weight[labels]
                        cams = torch.einsum('bc,bchw->bhw', weights, feats)
                        cams = torch.relu(cams)

                        flat = cams.view(cams.size(0), -1)
                        mn, _ = flat.min(dim=1, keepdim=True)
                        mx, _ = flat.max(dim=1, keepdim=True)
                        sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)

                        gt_small = nn.functional.interpolate(
                            gt_masks, size=sal_norm.shape[1:], mode='nearest'
                        ).squeeze(1)

                        if epoch < attention_epoch:
                            loss_tuple = compute_loss(outputs, labels, sal_norm, gt_small, 333, True)
                        else:
                            loss_tuple = compute_loss(outputs, labels, sal_norm, gt_small, kl_lambda_real, False)

                        loss = loss_tuple[0]
                        attn_loss = loss_tuple[1]
                    else:
                        loss = nn.functional.cross_entropy(outputs, labels)
                        attn_loss = torch.tensor(0.0, device=outputs.device)

                    if is_train:
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_attn_loss += attn_loss.item() * inputs.size(0)
                if phase == 'val':
                    labels_cpu = labels.detach().cpu().numpy()
                    preds_cpu = preds.detach().cpu().numpy()
                    for cls in range(num_classes):
                        cls_mask = labels_cpu == cls
                        if np.any(cls_mask):
                            class_correct[cls] += np.sum(preds_cpu[cls_mask] == labels_cpu[cls_mask])
                            class_total[cls] += np.sum(cls_mask)

            if is_train and sch is not None:
                sch.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_attn_loss = running_attn_loss / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Attn_Loss: {epoch_attn_loss:.4f}")

            if phase == 'val':
                class_acc = class_correct / np.maximum(class_total, 1)
                balanced_acc = class_acc.mean()
                print(f"{phase} Balanced Acc: {balanced_acc:.4f}")
                optim_num = balanced_acc
                print(f"{phase} Optim Num: {optim_num:.4f}")
                if (not use_attention or epoch >= attention_epoch) and (optim_num > best_optim):
                    best_optim = optim_num
                    best_epoch = epoch
                    best_wts = copy.deepcopy(model.state_dict())

    print()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # load best val weights before returning
    model.load_state_dict(best_wts)
    return model, best_optim, best_epoch




@torch.no_grad()
def evaluate_test(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    group_correct = np.zeros(len(GROUP_NAMES), dtype=np.int64)
    group_total = np.zeros(len(GROUP_NAMES), dtype=np.int64)
    have_groups = False
    for batch in test_loader:
        if len(batch) == 4:
            images, labels, paths, groups = batch
            have_groups = True
        else:
            images, labels, paths = batch
            groups = None
        images = images.to(device)
        labels = labels.to(device).long()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        if groups is not None:
            groups = groups.to(device).long()
            for g in torch.unique(groups):
                g = int(g.item())
                g_mask = groups == g
                group_correct[g] += (preds[g_mask] == labels[g_mask]).sum().item()
                group_total[g] += g_mask.sum().item()
    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    if have_groups:
        group_acc = group_correct / np.maximum(group_total, 1)
        per_group = 100.0 * group_acc.mean()
        worst_group = 100.0 * group_acc.min()
        return avg_loss, acc, group_acc * 100.0, per_group, worst_group
    return avg_loss, acc, None, None, None




def run_single(args, attn_epoch, kl_value):
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_attention = attn_epoch < num_epochs and kl_value > 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'eval': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    mask_transforms = {
        'train': transforms.Compose([
            # ExpandWhite(thr=10, radius=3),
            # EdgeExtract(thr=10, edge_width=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Brighten(8.0),
        ])
    }

    seed_everything(SEED)
    g = torch.Generator(); g.manual_seed(SEED)

    metadata_path = os.path.join(args.data_path, 'metadata.csv')
    if os.path.exists(metadata_path):
        train_dataset = WaterbirdsMetadataDataset(
            data_root=args.data_path,
            split='train',
            image_transform=data_transforms['train'],
            mask_root=args.gt_path,
            mask_transform=mask_transforms['train'],
            return_mask=use_attention,
            return_path=True,
            return_group=False
        )
        val_dataset = WaterbirdsMetadataDataset(
            data_root=args.data_path,
            split='val',
            image_transform=data_transforms['eval'],
            return_mask=False,
            return_path=True,
            return_group=False
        )
        test_dataset = WaterbirdsMetadataDataset(
            data_root=args.data_path,
            split='test',
            image_transform=data_transforms['eval'],
            return_mask=False,
            return_path=True,
            return_group=True
        )
        num_classes = len(np.unique(train_dataset.labels))
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=4, worker_init_fn=seed_worker, generator=g),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, worker_init_fn=seed_worker, generator=g),
        }
        dataset_sizes = {
            'train': len(train_dataset),
            'val': len(val_dataset),
        }
    else:
        # Fallback to ImageFolder split if metadata is unavailable.
        full_train = GuidedImageFolder(
            image_root=os.path.join(args.data_path, 'train'),
            mask_root=args.gt_path,
            image_transform=data_transforms['train'],
            mask_transform=mask_transforms['train'],
        )
        n_total = len(full_train)
        n_val_in = max(1, int(0.16 * n_total))
        n_train = n_total - n_val_in
        train_subset, val_subset = random_split(full_train, [n_train, n_val_in], generator=g)
        test_dataset = ImageFolderWithPaths(
            root=os.path.join(args.data_path, 'test'),
            transform=data_transforms['eval']
        )
        num_classes = len(full_train.images.classes)
        dataloaders = {
            'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                num_workers=4, worker_init_fn=seed_worker, generator=g),
            'val': DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                              num_workers=4, worker_init_fn=seed_worker, generator=g),
        }
        dataset_sizes = {
            'train': len(train_subset),
            'val': len(val_subset),
        }

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, worker_init_fn=seed_worker, generator=g)

    model = make_cam_model(num_classes, model_name="resnet50", pretrained=True).to(device)

    save_checkpoints = os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "n")
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n=== RUN: kl_lambda={kl_value}, attention_epoch={attn_epoch} ===", flush=True)
    best_model, best_score, best_epoch = train_model(
        model, dataloaders, dataset_sizes,
        attn_epoch, kl_value, num_epochs,
        base_lr=base_lr, classifier_lr=classifier_lr,
        kl_incr=(kl_value / 10), use_attention=use_attention,
        num_classes=num_classes
    )
    print(f"\n[VAL] Best Balanced Acc: {best_score:.4f} at epoch {best_epoch}")

    # Evaluate once on TEST with the best val weights
    test_loss, test_acc, group_acc, per_group, worst_group = evaluate_test(best_model, test_loader)
    print(f"\n[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")
    if group_acc is not None:
        for name, acc in zip(GROUP_NAMES, group_acc):
            print(f"[TEST] {name}: {acc:.2f}%")
        print(f"[TEST] Per Group: {per_group:.2f}%  Worst Group: {worst_group:.2f}%")

    # Save best model (named with hyperparams) unless disabled.
    if save_checkpoints:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"resnet50_final_kl{int(kl_value)}_attn{attn_epoch}_{ts}.pth"
        save_path = os.path.join(checkpoint_dir, save_name)
        torch.save(best_model.state_dict(), save_path)
    else:
        save_path = "NONE"
        print("[RUN DONE] Checkpoint saving disabled via SAVE_CHECKPOINTS=0", flush=True)

    print(f"[RUN DONE] kl={kl_value} attn={attn_epoch} | best_balanced_val_acc={best_score:.4f} "
          f"| test_acc={test_acc:.2f}% | saved: {save_path}", flush=True)
    return best_score, test_acc, per_group, worst_group, save_path



def main():
    global SEED, base_lr, classifier_lr
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Dataset root (expects metadata.csv or train/ and test/ subdirs)')
    parser.add_argument('gt_path', help='Folder with ground-truth mask PNGs (for train only)')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--attention_epoch', type=int, default=num_epochs,
                        help='Epoch at which to restart training (>= num_epochs disables attention)')
    parser.add_argument('--kl_lambda', type=float, default=0.0, help='Weight for attention KL loss')
    parser.add_argument('--base_lr', type=float, default=base_lr, help='Base learning rate')
    parser.add_argument('--classifier_lr', type=float, default=classifier_lr, help='Classifier learning rate')
    parser.add_argument('--sweep', action='store_true',
                        help='Run the full hyperparameter sweep (kl 100..300 step 20; attn 5..25 step 2)')
    args = parser.parse_args()

    SEED = args.seed
    base_lr = args.base_lr
    classifier_lr = args.classifier_lr

    if not args.sweep:
        run_single(args, args.attention_epoch, args.kl_lambda)
        return

    kl_values = list(range(100, 301, 20))
    attn_values = list(range(5, 26, 2))

    best_overall = (-1.0, None, None, None)  # (best_optim, kl, attn, test_acc)
    for kl in kl_values:
        for attn in attn_values:
            try:
                score, test_acc, per_group, worst_group, _ = run_single(args, attn, kl)
                if score > best_overall[0]:
                    best_overall = (score, kl, attn, test_acc, per_group, worst_group)
            except Exception as e:
                print(f"[SWEEP ERROR] kl={kl} attn={attn} -> {e}", flush=True)

    print("\n=== SWEEP COMPLETE ===")
    if best_overall[1] is not None:
        print(f"Best by val Optim Num: optim={best_overall[0]:.4f}, "
              f"kl={best_overall[1]}, attn={best_overall[2]}, "
              f"test_acc={best_overall[3]:.2f}%, "
              f"per_group={best_overall[4]:.2f}%, "
              f"worst_group={best_overall[5]:.2f}%", flush=True)
    else:
        print("No successful runs.", flush=True)


if __name__ == '__main__':
    main()
