#!/usr/bin/env python3
import argparse
import copy
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

GROUP_NAMES = ["Land_on_Land", "Land_on_Water", "Water_on_Land", "Water_on_Water"]


def _gals_repo_root() -> Path:
    return Path(__file__).resolve().parent


def _try_import_clip():
    try:
        import clip  # type: ignore
        return clip
    except Exception:
        repo_root = str(_gals_repo_root())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from CLIP.clip import clip  # type: ignore
        return clip


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


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


class WaterbirdsMetadataDataset(Dataset):
    SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

    def __init__(self, data_root, split, image_transform=None, return_path=True, return_group=False):
        self.data_root = data_root
        self.image_transform = image_transform
        self.return_path = return_path
        self.return_group = return_group

        metadata_path = os.path.join(self.data_root, "metadata.csv")
        df = pd.read_csv(metadata_path)
        split_id = self.SPLIT_MAP[split]
        df = df[df["split"] == split_id]

        self.paths = [os.path.join(self.data_root, p) for p in df["img_filename"].values]
        self.labels = df["y"].astype(int).values
        self.places = df["place"].astype(int).values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = int(self.labels[idx])
        img = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)

        output = [img, label]
        if self.return_path:
            output.append(path)
        if self.return_group:
            group = int(label * 2 + self.places[idx])
            output.append(group)
        return tuple(output)


class CLIPRN50CAM(nn.Module):
    """CLIP RN50 visual backbone + GAP classifier head for vanilla training."""

    def __init__(self, num_classes: int, clip_model_name: str = "RN50", pretrained: bool = True):
        super().__init__()
        if clip_model_name != "RN50":
            raise ValueError(f"CLIPRN50CAM currently supports clip_model_name='RN50' only, got: {clip_model_name}")
        if not pretrained:
            raise ValueError("CLIPRN50CAM requires pretrained=True (random-init CLIP RN50 is unsupported).")

        clip_lib = _try_import_clip()
        clip_model, _ = clip_lib.load(clip_model_name, device="cpu", jit=False)
        clip_model = clip_model.float()
        visual = clip_model.visual

        self.conv1 = visual.conv1
        self.bn1 = visual.bn1
        self.conv2 = visual.conv2
        self.bn2 = visual.bn2
        self.conv3 = visual.conv3
        self.bn3 = visual.bn3
        self.avgpool = visual.avgpool
        self.relu = visual.relu
        self.layer1 = visual.layer1
        self.layer2 = visual.layer2
        self.layer3 = visual.layer3
        self.layer4 = visual.layer4

        feat_dim = int(self.layer4[-1].bn3.num_features)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feat_dim, num_classes)

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x: torch.Tensor):
        x = x.to(dtype=self.conv1.weight.dtype)
        x = self._stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feats = self.layer4(x)
        logits = self.classifier(self.gap(feats).flatten(1))
        return logits


def _make_mobilenet_v3_large(num_classes: int, pretrained: bool) -> nn.Module:
    if hasattr(models, "MobileNet_V3_Large_Weights"):
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    else:
        model = models.mobilenet_v3_large(pretrained=pretrained)
    if not isinstance(model.classifier, nn.Sequential) or not isinstance(model.classifier[-1], nn.Linear):
        raise TypeError("Expected torchvision MobileNetV3-Large classifier to end with nn.Linear.")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def make_model(model_name: str, num_classes: int, pretrained: bool, clip_model: str = "RN50"):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "clip_rn50":
        return CLIPRN50CAM(num_classes=num_classes, clip_model_name=clip_model, pretrained=pretrained)
    if model_name == "mobilenet_v3_large":
        return _make_mobilenet_v3_large(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model_name: {model_name}")


def configure_tune_mode(model: nn.Module, tune_mode: str) -> None:
    """
    Configure trainable parameters:
    - full: train everything
    - layer4_head: train layer4/last MobileNet feature blocks + classifier/fc only
    - last_blocks_head: train final feature blocks + classifier/fc only
    - linear_probe: train classifier/fc only
    """
    mode = str(tune_mode).strip().lower()
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    for p in model.parameters():
        p.requires_grad = False

    head = None
    if hasattr(model, "classifier"):
        head = model.classifier
    elif hasattr(model, "fc"):
        head = model.fc
    if head is None:
        raise AttributeError(f"Model does not expose `.classifier` or `.fc`; cannot apply tune_mode={mode}")
    for p in head.parameters():
        p.requires_grad = True

    if mode == "linear_probe":
        return

    if mode in {"layer4_head", "last_blocks_head"}:
        if hasattr(model, "layer4"):
            for p in model.layer4.parameters():
                p.requires_grad = True
            return
        if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
            for block in model.features[-3:]:
                for p in block.parameters():
                    p.requires_grad = True
            return
        raise AttributeError(
            f"Model does not expose `.layer4` or MobileNet-style `.features`; cannot apply tune_mode={mode}"
        )

    raise ValueError(f"Unsupported tune_mode: {tune_mode}")


def _count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def forward_resnet_with_features(model: nn.Module, x: torch.Tensor):
    """Return logits and pooled penultimate features for torchvision ResNet models."""
    required = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc")
    if not all(hasattr(model, name) for name in required):
        raise TypeError("ElRep runner currently expects a torchvision-style ResNet with a final `.fc` head.")
    z = model.conv1(x)
    z = model.bn1(z)
    z = model.relu(z)
    z = model.maxpool(z)
    z = model.layer1(z)
    z = model.layer2(z)
    z = model.layer3(z)
    z = model.layer4(z)
    features = torch.flatten(model.avgpool(z), 1)
    logits = model.fc(features)
    return logits, features


def elrep_penalty(features: torch.Tensor, theta1: float, theta2: float) -> torch.Tensor:
    """Nuclear + Frobenius penalties from ElRep, scaled by batch size."""
    if theta1 == 0.0 and theta2 == 0.0:
        return features.new_tensor(0.0)
    singular_values = torch.linalg.svdvals(features.float())
    batch_size = max(int(features.shape[0]), 1)
    nuclear = float(theta1) * singular_values.abs().sum() / batch_size
    frobenius = float(theta2) * singular_values.square().sum() / batch_size
    return nuclear + frobenius


def evaluate_test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    group_correct = np.zeros(len(GROUP_NAMES), dtype=np.int64)
    group_total = np.zeros(len(GROUP_NAMES), dtype=np.int64)
    have_groups = False

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                images, labels, _, groups = batch
                have_groups = True
            else:
                images, labels, _ = batch
                groups = None

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            if groups is not None:
                groups = groups.to(device, non_blocking=True).long()
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


def train_model(model, dataloaders, dataset_sizes, num_classes, num_epochs, optimizer, device, theta1, theta2):
    criterion = nn.CrossEntropyLoss()
    best_wts = copy.deepcopy(model.state_dict())
    best_balanced = -1.0
    best_epoch = -1
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}", flush=True)

        for phase in ["train", "val"]:
            is_train = phase == "train"
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_ce_loss = 0.0
            running_elrep_loss = 0.0
            running_corrects = 0
            class_correct = np.zeros(num_classes, dtype=np.int64)
            class_total = np.zeros(num_classes, dtype=np.int64)

            for batch in dataloaders[phase]:
                inputs, labels = batch[0], batch[1]
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_train):
                    outputs, features = forward_resnet_with_features(model, inputs)
                    ce_loss = criterion(outputs, labels)
                    rep_loss = elrep_penalty(features, theta1, theta2) if is_train else features.new_tensor(0.0)
                    loss = ce_loss + rep_loss
                    preds = outputs.argmax(dim=1)

                    if is_train:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_ce_loss += ce_loss.item() * inputs.size(0)
                running_elrep_loss += rep_loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

                labels_cpu = labels.detach().cpu().numpy()
                preds_cpu = preds.detach().cpu().numpy()
                for cls in range(num_classes):
                    mask = labels_cpu == cls
                    if np.any(mask):
                        class_correct[cls] += np.sum(preds_cpu[mask] == labels_cpu[mask])
                        class_total[cls] += np.sum(mask)

            epoch_loss = running_loss / max(dataset_sizes[phase], 1)
            epoch_ce_loss = running_ce_loss / max(dataset_sizes[phase], 1)
            epoch_elrep_loss = running_elrep_loss / max(dataset_sizes[phase], 1)
            epoch_acc = running_corrects / max(dataset_sizes[phase], 1)
            class_acc = class_correct / np.maximum(class_total, 1)
            balanced_acc = float(np.mean(class_acc))

            print(
                f"{phase} Loss: {epoch_loss:.4f} CE: {epoch_ce_loss:.4f} "
                f"ElRep: {epoch_elrep_loss:.4f} Acc: {epoch_acc:.4f} Balanced Acc: {balanced_acc:.4f}",
                flush=True,
            )

            if phase == "val" and balanced_acc > best_balanced:
                best_balanced = balanced_acc
                best_epoch = epoch
                best_wts = copy.deepcopy(model.state_dict())

    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s", flush=True)
    model.load_state_dict(best_wts)
    return model, best_balanced, best_epoch


def build_dataloaders(data_path, batch_size, num_workers, generator, model_name: str = "resnet50"):
    if model_name == "clip_rn50":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    metadata_path = os.path.join(data_path, "metadata.csv")
    if os.path.exists(metadata_path):
        train_ds = WaterbirdsMetadataDataset(data_path, "train", tfm, return_path=True, return_group=False)
        val_ds = WaterbirdsMetadataDataset(data_path, "val", tfm, return_path=True, return_group=False)
        test_ds = WaterbirdsMetadataDataset(data_path, "test", tfm, return_path=True, return_group=True)
        num_classes = int(len(np.unique(train_ds.labels)))
    else:
        train_full = ImageFolderWithPaths(root=os.path.join(data_path, "train"), transform=tfm)
        n_total = len(train_full)
        n_val = max(1, int(0.16 * n_total))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=generator)
        test_ds = ImageFolderWithPaths(root=os.path.join(data_path, "test"), transform=tfm)
        num_classes = len(train_full.classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_ds), "val": len(val_ds)}
    return dataloaders, dataset_sizes, test_loader, num_classes


def run_single(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    dataloaders, dataset_sizes, test_loader, num_classes = build_dataloaders(
        args.data_path, args.batch_size, args.num_workers, g, model_name=args.model
    )

    model = make_model(
        args.model,
        num_classes,
        pretrained=args.pretrained,
        clip_model=getattr(args, "clip_model", "RN50"),
    ).to(device)
    configure_tune_mode(model, tune_mode=getattr(args, "tune_mode", "full"))

    base_lr = getattr(args, "base_lr", None)
    classifier_lr = getattr(args, "classifier_lr", None)
    if base_lr is None and classifier_lr is None:
        base_lr = args.lr
        classifier_lr = args.lr
    elif base_lr is None:
        base_lr = args.lr
    elif classifier_lr is None:
        classifier_lr = args.lr

    base_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ("fc" in name) or ("classifier" in name):
            head_params.append(param)
        else:
            base_params.append(param)

    if not head_params:
        head_params = [p for p in model.parameters() if p.requires_grad]
        base_params = []

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": float(base_lr)})
    param_groups.append({"params": head_params, "lr": float(classifier_lr)})

    optimizer = optim.SGD(
        param_groups,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    print(
        f"\n=== RUN: model={args.model} clip_model={getattr(args, 'clip_model', 'N/A')} "
        f"tune_mode={getattr(args, 'tune_mode', 'full')} trainable_params={_count_trainable_params(model):,} "
        f"epochs={args.num_epochs} base_lr={base_lr} classifier_lr={classifier_lr} "
        f"momentum={args.momentum} wd={args.weight_decay} nesterov={args.nesterov} "
        f"theta1={args.theta1} theta2={args.theta2} seed={args.seed} ===",
        flush=True,
    )

    best_model, best_balanced_val, best_epoch = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        num_classes=num_classes,
        num_epochs=args.num_epochs,
        optimizer=optimizer,
        device=device,
        theta1=args.theta1,
        theta2=args.theta2,
    )

    print(f"\n[VAL] Best Balanced Acc: {best_balanced_val:.4f} at epoch {best_epoch}", flush=True)

    test_loss, test_acc, group_acc, per_group, worst_group = evaluate_test(best_model, test_loader, device)
    print(f"\n[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%", flush=True)
    if group_acc is not None:
        for name, acc in zip(GROUP_NAMES, group_acc):
            print(f"[TEST] {name}: {acc:.2f}%", flush=True)
        print(f"[TEST] Per Group: {per_group:.2f}%  Worst Group: {worst_group:.2f}%", flush=True)

    save_checkpoints = os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "n")
    if save_checkpoints:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = f"elrep_{args.model}_theta1{args.theta1:g}_theta2{args.theta2:g}_seed{args.seed}_{ts}.pth"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        torch.save(best_model.state_dict(), ckpt_path)
    else:
        ckpt_path = "NONE"
        print("[RUN DONE] Checkpoint saving disabled via SAVE_CHECKPOINTS=0", flush=True)

    print(
        f"[RUN DONE] best_balanced_val_acc={best_balanced_val:.4f} | "
        f"test_acc={test_acc:.2f}% | saved: {ckpt_path}",
        flush=True,
    )
    return best_balanced_val, test_acc, per_group, worst_group, ckpt_path


def parse_args():
    p = argparse.ArgumentParser(
        description="ElRep Waterbirds trainer: ERM + nuclear/Frobenius penalties on ResNet-50 penultimate features."
    )
    p.add_argument("data_path", help="Waterbirds dataset root (expects metadata.csv or train/test folders)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=["resnet50"], default="resnet50")
    p.add_argument("--clip-model", default="RN50", help="Used when --model clip_rn50.")
    p.add_argument("--tune-mode", choices=["full", "layer4_head", "last_blocks_head", "linear_probe"], default="full")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--base-lr", type=float, default=None)
    p.add_argument("--classifier-lr", type=float, default=None)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--theta1", type=float, default=1e-4, help="Nuclear-norm representation penalty weight")
    p.add_argument("--theta2", type=float, default=1e-5, help="Frobenius-norm representation penalty weight")
    p.add_argument("--nesterov", action="store_true", default=False)
    p.add_argument("--no-nesterov", action="store_false", dest="nesterov")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-dir", default="ElRep_Checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_single(args)
