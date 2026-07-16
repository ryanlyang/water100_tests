#!/usr/bin/env python3
import argparse
import copy
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


RUNNER_ROOT = Path(__file__).resolve().parent
if str(RUNNER_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNNER_ROOT))
import waterbirds as lgm  # noqa: E402

REPRO_ROOT = Path(__file__).resolve().parents[4]
TRAIN_ROOT = REPRO_ROOT / "r4rr" / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))
import r4rr_redmeat as redmeat  # noqa: E402


batch_size = 32
num_epochs = 150
base_lr = 2e-4
classifier_lr = 2e-4
lr2_mult = 1.0
momentum = 0.9
weight_decay = 1e-5
img_size = 224
kl_grid = 14
fusion_beta = 0.85
checkpoint_dir = "RedMeat_ViT_LGMStyle_Checkpoints"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0


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

    param_groups = lgm._get_param_groups(model, base_lr_val, classifier_lr_val)
    opt = optim.SGD(param_groups, momentum=momentum_val, weight_decay=weight_decay)
    kl_lambda_real = float(kl_lambda_start)

    for epoch in range(num_epochs_val):
        if use_attention and epoch == attention_epoch:
            base_lr_after = base_lr_val * lr2_mult_val
            classifier_lr_after = classifier_lr_val * lr2_mult_val
            print(
                f"*** Attention epoch {epoch} reached: restarting SGD "
                f"(lr2_mult={lr2_mult_val}, base_lr={base_lr_after}, "
                f"classifier_lr={classifier_lr_after}, momentum={momentum_val}) ***",
                flush=True,
            )
            param_groups = lgm._get_param_groups(model, base_lr_after, classifier_lr_after)
            opt = optim.SGD(param_groups, momentum=momentum_val, weight_decay=weight_decay)
            best_wts = copy.deepcopy(model.state_dict())
            best_optim = -100.0

        if use_attention and epoch > attention_epoch:
            kl_lambda_real += kl_incr

        print(f"Epoch {epoch + 1}/{num_epochs_val}", flush=True)

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
                    raise RuntimeError(f"Unexpected batch format: len={len(batch)}")

                use_attention_this_batch = is_train and use_attention and has_masks and epoch >= attention_epoch
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                if is_train:
                    opt.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs, aux = model(inputs)
                    preds = outputs.argmax(dim=1)

                    if use_attention_this_batch:
                        sal_norm = lgm._compute_vit_lgmstyle_map(
                            aux,
                            map_source=map_source,
                            fusion_beta_val=fusion_beta_val,
                        )
                        if kl_grid_size and kl_grid_size > 0:
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

                        loss, attn_loss = redmeat.base.compute_loss(
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
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Attn_Loss: {epoch_attn_loss:.4f}", flush=True)

            if phase == "val":
                class_acc = class_correct / np.maximum(class_total, 1)
                balanced_acc = float(class_acc.mean())
                print(f"{phase} Balanced Acc: {balanced_acc:.4f}", flush=True)
                if (not use_attention or epoch >= attention_epoch) and balanced_acc > best_optim:
                    best_optim = balanced_acc
                    best_epoch = epoch
                    best_wts = copy.deepcopy(model.state_dict())

    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s", flush=True)
    model.load_state_dict(best_wts)
    return model, best_optim, best_epoch


def run_single(args, attn_epoch, kl_value, kl_increment=None):
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    redmeat.device = device

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
                redmeat.base.Brighten(8.0),
            ]
        )
    }

    redmeat.base.seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    num_workers = redmeat.base.get_num_workers(default=4)

    train_dataset = redmeat.RedMeatMetadataDataset(
        data_root=args.data_path,
        split="train",
        image_transform=data_transforms["train"],
        mask_root=args.teacher_map_path,
        mask_transform=mask_transforms["train"],
        return_mask=use_attention,
        return_path=True,
        classes=args.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )
    val_dataset = redmeat.RedMeatMetadataDataset(
        data_root=args.data_path,
        split="val",
        image_transform=data_transforms["eval"],
        return_mask=False,
        return_path=True,
        classes=train_dataset.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )
    test_dataset = redmeat.RedMeatMetadataDataset(
        data_root=args.data_path,
        split="test",
        image_transform=data_transforms["eval"],
        return_mask=False,
        return_path=True,
        classes=train_dataset.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )

    num_classes = len(train_dataset.classes)
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=redmeat.base.seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=redmeat.base.seed_worker,
            generator=g,
        ),
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=redmeat.base.seed_worker,
        generator=g,
    )

    model = lgm.ViTLGMStyleMaps(
        num_classes=num_classes,
        model_name=args.vit_model,
        pretrained=args.pretrained,
        image_size=img_size,
    ).to(device)

    save_checkpoints = os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "n")
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(
        f"\n=== RUN: dataset=redmeat, model={args.vit_model}, cam_mode=lgmstyle_native_map, "
        f"map_source={args.map_source}, fusion_beta={args.fusion_beta}, optimizer=sgd, "
        f"momentum={momentum}, kl_lambda={kl_value}, attention_epoch={attn_epoch}, "
        f"base_lr={base_lr}, classifier_lr={classifier_lr}, lr2_mult={lr2_mult}, "
        f"weight_decay={weight_decay}, batch_size={batch_size}, num_epochs={num_epochs}, "
        f"img_size={img_size}, kl_grid={kl_grid if kl_grid > 0 else 'native'} ===",
        flush=True,
    )

    if kl_increment is None:
        kl_increment = kl_value / 10.0

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
    print(f"\n[VAL] Best Balanced Acc: {best_score:.4f} at epoch {best_epoch}", flush=True)

    test_loss, test_acc, class_acc, per_group, worst_group = redmeat.evaluate_test(best_model, test_loader, num_classes)
    print(f"\n[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%", flush=True)
    for cls_name, acc in zip(train_dataset.classes, class_acc):
        print(f"[TEST] {cls_name}: {acc:.2f}%", flush=True)
    print(f"[TEST] Per-class mean: {per_group:.2f}%  Worst-class: {worst_group:.2f}%", flush=True)

    if save_checkpoints:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{args.vit_model}_redmeat_lgmstyle_kl{int(kl_value)}_attn{attn_epoch}_{ts}.pth"
        save_path = os.path.join(checkpoint_dir, save_name)
        torch.save(best_model.state_dict(), save_path)
    else:
        save_path = "NONE"
        print("[RUN DONE] Checkpoint saving disabled via SAVE_CHECKPOINTS=0", flush=True)

    print(
        f"[RUN DONE] dataset=redmeat model={args.vit_model} map_source={args.map_source} "
        f"fusion_beta={args.fusion_beta} kl={kl_value} attn={attn_epoch} "
        f"lr2_mult={lr2_mult} kl_incr={kl_increment} | best_balanced_val_acc={best_score:.4f} "
        f"| test_acc={test_acc:.2f}% | per_class={per_group:.2f}% | worst_class={worst_group:.2f}% "
        f"| saved: {save_path}",
        flush=True,
    )
    return float(best_score), float(test_acc), float(per_group), float(worst_group), save_path


def main():
    global SEED, base_lr, classifier_lr, lr2_mult, momentum, weight_decay, batch_size, num_epochs, img_size, kl_grid, fusion_beta, checkpoint_dir

    p = argparse.ArgumentParser(description="R4RR RedMeat ViT runner with LGM-style maps and SGD.")
    p.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")
    p.add_argument("teacher_map_path", help="Teacher-map root")
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
    p.add_argument("--kl_grid", "--kl-grid", dest="kl_grid", type=int, default=kl_grid)
    p.add_argument("--map_source", "--map-source", dest="map_source", choices=lgm.ViTLGMStyleMaps.MAP_SOURCES, default="fusion")
    p.add_argument("--fusion_beta", "--fusion-beta", dest="fusion_beta", type=float, default=fusion_beta)
    p.add_argument("--vit_model", "--vit-model", dest="vit_model", choices=["vit_b_16"], default="vit_b_16")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default=checkpoint_dir)
    p.add_argument("--split-col", "--split_col", dest="split_col", default="split")
    p.add_argument("--label-col", "--label_col", dest="label_col", default="label")
    p.add_argument("--path-col", "--path_col", dest="path_col", default="abs_file_path")
    p.add_argument(
        "--classes",
        default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon",
        help="Comma-separated class list. Empty string = infer from metadata.",
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
    checkpoint_dir = str(args.checkpoint_dir)

    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()] if args.classes else None
    args.classes = classes

    print(f"[MAP] Using LGM-style ViT map source: {args.map_source} (fusion_beta={args.fusion_beta})", flush=True)
    run_single(args, int(args.attention_epoch), float(args.kl_lambda), args.kl_increment)


if __name__ == "__main__":
    main()
