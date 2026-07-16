#!/usr/bin/env python3
import argparse
import copy
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import interpolate_embeddings


# Import shared data/utility helpers from the canonical Waterbirds runner.
REPRO_ROOT = Path(__file__).resolve().parents[4]
WATERBIRDS_TRAIN_ROOT = REPRO_ROOT / "r4rr" / "train"
if str(WATERBIRDS_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(WATERBIRDS_TRAIN_ROOT))
import r4rr_waterbirds as base  # noqa: E402


batch_size = 2
num_epochs = 200
base_lr = 2e-4
classifier_lr = 2e-4
lr2_mult = 1.0
momentum = 0.9
weight_decay = 1e-2
img_size = 640
kl_grid = 0

checkpoint_dir = "ViT_LGMStyle_SGD_640_Checkpoints"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0
fusion_beta = 0.85


class ViTLGMStyleMaps(nn.Module):
    """
    ViT student with ViT-native spatial maps inspired by LGM-ViT.

    Evidence extraction:
    - keep the standard ViT CLS-token classifier
    - explicitly expose the last-block attention weights
    - expose final patch embeddings before encoder LayerNorm
    - build student spatial map from `attn`, `embed`, or `fusion`
    """
    MAP_SOURCES = ("attn", "embed", "fusion")

    def __init__(
        self,
        num_classes: int,
        model_name: str = "vit_b_16",
        pretrained: bool = True,
        image_size: int = 640,
    ):
        super().__init__()
        if model_name != "vit_b_16":
            raise ValueError(f"Unsupported vit model_name: {model_name}")

        self.base = vit_b_16(weights=None, image_size=image_size)

        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            state_dict = weights.get_state_dict(progress=True)
            # torchvision ViT pretrained checkpoints are 224x224; interpolate pos embeddings for 640.
            state_dict = interpolate_embeddings(
                image_size=image_size,
                patch_size=16,
                model_state=state_dict,
                interpolation_mode="bicubic",
                reset_heads=True,
            )
            self.base.load_state_dict(state_dict, strict=False)

        if not hasattr(self.base.heads, "head"):
            raise RuntimeError("Unexpected torchvision ViT head structure")
        in_features = int(self.base.heads.head.in_features)
        self.base.heads.head = nn.Linear(in_features, num_classes)
        self.classifier = self.base.heads.head

        self.last_attn = None
        self.patch_tokens = None

    def forward(self, x):
        self.last_attn = None
        self.patch_tokens = None

        x = self.base._process_input(x)
        n = x.shape[0]
        batch_class_token = self.base.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.base.encoder.dropout(x + self.base.encoder.pos_embedding)
        num_layers = len(self.base.encoder.layers)
        for idx, layer in enumerate(self.base.encoder.layers):
            if idx < num_layers - 1:
                x = layer(x)
                continue

            residual = x
            y = layer.ln_1(x)
            mha = layer.self_attention
            cur_batch, seq_len, hidden_dim = y.shape
            num_heads = mha.num_heads
            head_dim = hidden_dim // num_heads
            scale = head_dim ** -0.5

            qkv = F.linear(y, mha.in_proj_weight, mha.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(cur_batch, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(cur_batch, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(cur_batch, seq_len, num_heads, head_dim).transpose(1, 2)

            attn_scores = torch.matmul(q * scale, k.transpose(-2, -1))
            attn_probs = torch.softmax(attn_scores, dim=-1)
            if mha.dropout > 0.0:
                attn_used = F.dropout(attn_probs, p=mha.dropout, training=self.training)
            else:
                attn_used = attn_probs

            attn_out = torch.matmul(attn_used, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(cur_batch, seq_len, hidden_dim)
            attn_out = F.linear(attn_out, mha.out_proj.weight, mha.out_proj.bias)
            x = layer.dropout(attn_out)
            x = x + residual
            z = layer.ln_2(x)
            z = layer.mlp(z)
            x = x + z

            self.last_attn = attn_used

        self.patch_tokens = x[:, 1:, :]
        x = self.base.encoder.ln(x)
        logits = self.base.heads(x[:, 0])
        aux = {"attn_weights": self.last_attn, "patch_tokens": self.patch_tokens}
        return logits, aux


def _reshape_patch_scores(patch_scores: torch.Tensor) -> torch.Tensor:
    num_tokens = patch_scores.size(1)
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise RuntimeError(f"Patch token count {num_tokens} is not a perfect square.")
    return patch_scores.view(patch_scores.size(0), side, side)


def _compute_attention_map(attn_weights: torch.Tensor) -> torch.Tensor:
    if attn_weights is None:
        raise RuntimeError("Attention weights were not captured; attention map computation is unavailable.")
    if attn_weights.ndim != 4 or attn_weights.size(-1) <= 1:
        raise RuntimeError(f"Unexpected attention tensor shape: {tuple(attn_weights.shape)}")
    attn_cls = attn_weights[:, :, 0, 1:]
    return _reshape_patch_scores(attn_cls.mean(dim=1))


def _compute_embedding_map(patch_tokens: torch.Tensor) -> torch.Tensor:
    if patch_tokens is None:
        raise RuntimeError("Patch tokens were not captured; embedding map computation is unavailable.")
    if patch_tokens.ndim != 3 or patch_tokens.size(1) < 1:
        raise RuntimeError(f"Unexpected patch token tensor shape: {tuple(patch_tokens.shape)}")

    batch_size_local, num_tokens, channels = patch_tokens.shape
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise RuntimeError(f"Patch token count {num_tokens} is not a perfect square.")
    patch_grid = patch_tokens.transpose(1, 2).contiguous().view(batch_size_local, channels, side, side)
    return patch_grid.mean(dim=1)


def _compute_vit_lgmstyle_map(aux: dict, map_source: str, fusion_beta_val: float) -> torch.Tensor:
    if map_source not in ViTLGMStyleMaps.MAP_SOURCES:
        raise ValueError(f"Unsupported map_source: {map_source}")

    attn_map = embed_map = None
    if map_source in ("attn", "fusion"):
        attn_map = _compute_attention_map(aux.get("attn_weights"))
    if map_source in ("embed", "fusion"):
        embed_map = _compute_embedding_map(aux.get("patch_tokens"))

    if map_source == "attn":
        return attn_map
    if map_source == "embed":
        return embed_map
    return fusion_beta_val * attn_map + (1.0 - fusion_beta_val) * embed_map



def _get_param_groups(model, base_lr_val, classifier_lr_val):
    base_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".heads.head" in name or ".classifier" in name:
            classifier_params.append(param)
        else:
            base_params.append(param)

    if not classifier_params:
        classifier_params = [p for p in model.parameters() if p.requires_grad]
        base_params = []

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr_val})
    param_groups.append({"params": classifier_params, "lr": classifier_lr_val})
    return param_groups



def train_model(
    model,
    dataloaders,
    dataset_sizes,
    attention_epoch,
    kl_lambda_start,
    num_epochs_val,
    base_lr_val,
    classifier_lr_val,
    lr2_mult_val,
    momentum_val,
    kl_incr,
    use_attention,
    num_classes,
    map_source,
    fusion_beta_val,
    kl_grid_size=0,
):
    best_wts = copy.deepcopy(model.state_dict())
    best_optim = -100.0
    best_epoch = -1
    since = time.time()

    param_groups = _get_param_groups(model, base_lr_val, classifier_lr_val)
    opt = optim.SGD(param_groups, momentum=momentum_val, weight_decay=weight_decay)

    kl_lambda_real = kl_lambda_start

    for epoch in range(num_epochs_val):
        if use_attention and epoch == attention_epoch:
            base_lr_after = base_lr_val * lr2_mult_val
            classifier_lr_after = classifier_lr_val * lr2_mult_val
            print(
                f"*** Attention epoch {epoch} reached: restarting SGD "
                f"(lr2_mult={lr2_mult_val}, base_lr={base_lr_after}, classifier_lr={classifier_lr_after}, "
                f"momentum={momentum_val}) ***"
            )
            param_groups = _get_param_groups(model, base_lr_after, classifier_lr_after)
            opt = optim.SGD(param_groups, momentum=momentum_val, weight_decay=weight_decay)
            best_wts = copy.deepcopy(model.state_dict())
            best_optim = -100.0

        if use_attention and epoch > attention_epoch:
            kl_lambda_real += kl_incr

        print(f"Epoch {epoch + 1}/{num_epochs_val}")

        for phase in ["train", "val"]:
            is_train = phase == "train"
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_attn_loss = 0.0
            class_correct = np.zeros(num_classes, dtype=np.int64)
            class_total = np.zeros(num_classes, dtype=np.int64)

            for batch in dataloaders[phase]:
                if len(batch) == 4:
                    inputs, labels, gt_masks, _paths = batch
                    gt_masks = gt_masks.to(device)
                    has_masks = True
                elif len(batch) == 3:
                    inputs, labels, _paths = batch
                    gt_masks = None
                    has_masks = False
                else:
                    raise RuntimeError("Unexpected batch format.")

                use_attention_this_batch = (
                    is_train and use_attention and has_masks and epoch >= attention_epoch
                )

                inputs = inputs.to(device)
                labels = labels.to(device).long()

                if is_train:
                    opt.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs, aux = model(inputs)
                    preds = outputs.argmax(dim=1)

                    if use_attention_this_batch:
                        sal_norm = _compute_vit_lgmstyle_map(
                            aux,
                            map_source=map_source,
                            fusion_beta_val=fusion_beta_val,
                        )
                        if kl_grid_size and kl_grid_size > 0:
                            # Optional parity mode: align on a fixed spatial grid (e.g., 7x7 like ResNet layer4).
                            sal_kl = nn.functional.adaptive_avg_pool2d(
                                sal_norm.unsqueeze(1), (kl_grid_size, kl_grid_size)
                            ).squeeze(1)
                            gt_small = nn.functional.adaptive_avg_pool2d(
                                gt_masks, (kl_grid_size, kl_grid_size)
                            ).squeeze(1)
                        else:
                            sal_kl = sal_norm
                            gt_small = nn.functional.interpolate(
                                gt_masks, size=sal_norm.shape[1:], mode="nearest"
                            ).squeeze(1)

                        loss, attn_loss = base.compute_loss(
                            outputs,
                            labels,
                            sal_kl,
                            gt_small,
                            kl_lambda_real,
                            only_ce=False,
                        )
                    else:
                        loss = nn.functional.cross_entropy(outputs, labels)
                        attn_loss = torch.tensor(0.0, device=outputs.device)

                    if is_train:
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_attn_loss += attn_loss.item() * inputs.size(0)

                if phase == "val":
                    labels_cpu = labels.detach().cpu().numpy()
                    preds_cpu = preds.detach().cpu().numpy()
                    for cls in range(num_classes):
                        cls_mask = labels_cpu == cls
                        if np.any(cls_mask):
                            class_correct[cls] += np.sum(preds_cpu[cls_mask] == labels_cpu[cls_mask])
                            class_total[cls] += np.sum(cls_mask)

            epoch_loss = running_loss / max(dataset_sizes[phase], 1)
            epoch_acc = running_corrects.double() / max(dataset_sizes[phase], 1)
            epoch_attn_loss = running_attn_loss / max(dataset_sizes[phase], 1)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Attn_Loss: {epoch_attn_loss:.4f}")

            if phase == "val":
                class_acc = class_correct / np.maximum(class_total, 1)
                balanced_acc = class_acc.mean()
                print(f"{phase} Balanced Acc: {balanced_acc:.4f}")
                if (not use_attention or epoch >= attention_epoch) and (balanced_acc > best_optim):
                    best_optim = balanced_acc
                    best_epoch = epoch
                    best_wts = copy.deepcopy(model.state_dict())

    print()
    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    model.load_state_dict(best_wts)
    return model, best_optim, best_epoch



def run_single(args, attn_epoch, kl_value, kl_increment=None):
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_attention = attn_epoch < num_epochs and kl_value > 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }
    mask_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                base.Brighten(8.0),
            ]
        )
    }

    base.seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    num_workers = base.get_num_workers(default=4)

    metadata_path = os.path.join(args.data_path, "metadata.csv")
    if os.path.exists(metadata_path):
        train_dataset = base.WaterbirdsMetadataDataset(
            data_root=args.data_path,
            split="train",
            image_transform=data_transforms["train"],
            mask_root=args.teacher_map_path,
            mask_transform=mask_transforms["train"],
            return_mask=use_attention,
            return_path=True,
            return_group=False,
        )
        val_dataset = base.WaterbirdsMetadataDataset(
            data_root=args.data_path,
            split="val",
            image_transform=data_transforms["eval"],
            return_mask=False,
            return_path=True,
            return_group=False,
        )
        test_dataset = base.WaterbirdsMetadataDataset(
            data_root=args.data_path,
            split="test",
            image_transform=data_transforms["eval"],
            return_mask=False,
            return_path=True,
            return_group=True,
        )
        num_classes = len(np.unique(train_dataset.labels))
        dataloaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=base.seed_worker,
                generator=g,
            ),
            "val": DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=base.seed_worker,
                generator=g,
            ),
        }
        dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    else:
        full_train = base.GuidedImageFolder(
            image_root=os.path.join(args.data_path, "train"),
            mask_root=args.teacher_map_path,
            image_transform=data_transforms["train"],
            mask_transform=mask_transforms["train"],
        )
        n_total = len(full_train)
        n_val_in = max(1, int(0.16 * n_total))
        n_train = n_total - n_val_in
        train_subset, val_subset = random_split(full_train, [n_train, n_val_in], generator=g)
        test_dataset = base.ImageFolderWithPaths(
            root=os.path.join(args.data_path, "test"),
            transform=data_transforms["eval"],
        )
        num_classes = len(full_train.images.classes)
        dataloaders = {
            "train": DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=base.seed_worker,
                generator=g,
            ),
            "val": DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=base.seed_worker,
                generator=g,
            ),
        }
        dataset_sizes = {"train": len(train_subset), "val": len(val_subset)}

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=base.seed_worker,
        generator=g,
    )

    model = ViTLGMStyleMaps(
        num_classes=num_classes,
        model_name=args.vit_model,
        pretrained=args.pretrained,
        image_size=img_size,
    ).to(device)

    save_checkpoints = os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in (
        "0",
        "false",
        "no",
        "n",
    )
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(
        f"\n=== RUN: model={args.vit_model}, cam_mode=lgmstyle_native_map, "
        f"map_source={args.map_source}, fusion_beta={args.fusion_beta}, "
        f"optimizer=sgd, momentum={momentum}, "
        f"kl_lambda={kl_value}, attention_epoch={attn_epoch}, "
        f"base_lr={base_lr}, classifier_lr={classifier_lr}, weight_decay={weight_decay}, "
        f"batch_size={batch_size}, num_epochs={num_epochs}, img_size={img_size}, "
        f"kl_grid={kl_grid if kl_grid > 0 else 'native'} ===",
        flush=True,
    )

    if kl_increment is None:
        kl_increment = kl_value / 10

    best_model, best_score, best_epoch = train_model(
        model,
        dataloaders,
        dataset_sizes,
        attn_epoch,
        kl_value,
        num_epochs,
        base_lr_val=base_lr,
        classifier_lr_val=classifier_lr,
        lr2_mult_val=lr2_mult,
        momentum_val=momentum,
        kl_incr=kl_increment,
        use_attention=use_attention,
        num_classes=num_classes,
        map_source=args.map_source,
        fusion_beta_val=args.fusion_beta,
        kl_grid_size=kl_grid,
    )
    print(f"\n[VAL] Best Balanced Acc: {best_score:.4f} at epoch {best_epoch}")

    test_loss, test_acc, group_acc, per_group, worst_group = base.evaluate_test(best_model, test_loader)
    print(f"\n[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")
    if group_acc is not None:
        for name, acc in zip(base.GROUP_NAMES, group_acc):
            print(f"[TEST] {name}: {acc:.2f}%")
        print(f"[TEST] Per Group: {per_group:.2f}%  Worst Group: {worst_group:.2f}%")

    if save_checkpoints:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{args.vit_model}_final_kl{int(kl_value)}_attn{attn_epoch}_{ts}.pth"
        save_path = os.path.join(checkpoint_dir, save_name)
        torch.save(best_model.state_dict(), save_path)
    else:
        save_path = "NONE"
        print("[RUN DONE] Checkpoint saving disabled via SAVE_CHECKPOINTS=0", flush=True)

    print(
        f"[RUN DONE] model={args.vit_model} map_source={args.map_source} fusion_beta={args.fusion_beta} "
        f"kl={kl_value} attn={attn_epoch} lr2_mult={lr2_mult} "
        f"kl_incr={kl_increment} | best_balanced_val_acc={best_score:.4f} | test_acc={test_acc:.2f}% "
        f"| saved: {save_path}",
        flush=True,
    )
    return best_score, test_acc, per_group, worst_group, save_path



def main():
    global SEED, base_lr, classifier_lr, lr2_mult, momentum, weight_decay, batch_size, num_epochs, img_size, kl_grid, fusion_beta

    p = argparse.ArgumentParser(
        description="R4RR Waterbirds ViT runner with LGM-style maps and SGD."
    )
    p.add_argument("data_path", help="Waterbirds dataset root (expects metadata.csv or train/test folders)")
    p.add_argument("teacher_map_path", help="Folder with teacher-map PNGs")

    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--attention_epoch", "--attention-epoch", dest="attention_epoch", type=int, default=num_epochs)
    p.add_argument("--kl_lambda", "--kl-lambda", dest="kl_lambda", type=float, default=0.0)
    p.add_argument("--kl_increment", "--kl-increment", dest="kl_increment", type=float, default=None)

    p.add_argument("--base_lr", "--base-lr", dest="base_lr", type=float, default=base_lr)
    p.add_argument("--classifier_lr", "--classifier-lr", dest="classifier_lr", type=float, default=classifier_lr)
    p.add_argument("--lr2_mult", "--lr2-mult", dest="lr2_mult", type=float, default=lr2_mult)
    p.add_argument("--momentum", type=float, default=momentum)
    p.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=weight_decay)
    p.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=batch_size)
    p.add_argument("--num_epochs", "--num-epochs", dest="num_epochs", type=int, default=num_epochs)
    p.add_argument("--img_size", "--img-size", dest="img_size", type=int, default=img_size)
    p.add_argument(
        "--kl_grid",
        "--kl-grid",
        dest="kl_grid",
        type=int,
        default=kl_grid,
        help="If >0, pool student/teacher maps to fixed grid for KL (e.g., 7). 0 keeps native token grid.",
    )
    p.add_argument("--map_source", "--map-source", dest="map_source", choices=ViTLGMStyleMaps.MAP_SOURCES, default="fusion")
    p.add_argument("--fusion_beta", "--fusion-beta", dest="fusion_beta", type=float, default=fusion_beta)

    p.add_argument("--vit_model", "--vit-model", dest="vit_model", choices=["vit_b_16"], default="vit_b_16")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")

    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run a simple built-in sweep (kl 100..300 step 20; attn 5..25 step 2).",
    )

    args = p.parse_args()

    SEED = int(args.seed)
    base_lr = float(args.base_lr)
    classifier_lr = float(args.classifier_lr)
    lr2_mult = float(args.lr2_mult)
    momentum = float(args.momentum)
    weight_decay = float(args.weight_decay)
    batch_size = int(args.batch_size)
    num_epochs = int(args.num_epochs)
    img_size = int(args.img_size)
    kl_grid = int(args.kl_grid)
    fusion_beta = float(args.fusion_beta)

    print(
        f"[MAP] Using LGM-style ViT map source: {args.map_source} (fusion_beta={args.fusion_beta})",
        flush=True,
    )

    if not args.sweep:
        run_single(args, int(args.attention_epoch), float(args.kl_lambda), args.kl_increment)
        return

    kl_values = list(range(100, 301, 20))
    attn_values = list(range(5, 26, 2))

    best_overall = (-1.0, None, None, None)
    for kl in kl_values:
        for attn in attn_values:
            try:
                score, test_acc, per_group, worst_group, _ = run_single(args, attn, kl)
                if score > best_overall[0]:
                    best_overall = (score, kl, attn, test_acc, per_group, worst_group)
            except Exception as exc:
                print(f"[SWEEP ERROR] kl={kl} attn={attn} -> {exc}", flush=True)

    print("\n=== SWEEP COMPLETE ===")
    if best_overall[1] is not None:
        print(
            f"Best by val balanced acc: balanced_acc={best_overall[0]:.4f}, "
            f"kl={best_overall[1]}, attn={best_overall[2]}, "
            f"test_acc={best_overall[3]:.2f}%, per_group={best_overall[4]:.2f}%, "
            f"worst_group={best_overall[5]:.2f}%",
            flush=True,
        )
    else:
        print("No successful runs.", flush=True)


if __name__ == "__main__":
    main()
