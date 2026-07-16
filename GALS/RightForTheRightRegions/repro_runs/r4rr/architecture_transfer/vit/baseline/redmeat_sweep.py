#!/usr/bin/env python3
import argparse
import copy
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


BASELINE_ROOT = Path(__file__).resolve().parent
R4RR_VIT_ROOT = BASELINE_ROOT.parent / "r4rr"
if str(R4RR_VIT_ROOT) not in sys.path:
    sys.path.insert(0, str(R4RR_VIT_ROOT))
import waterbirds as lgm  # noqa: E402

REPRO_ROOT = Path(__file__).resolve().parents[4]
TRAIN_ROOT = REPRO_ROOT / "r4rr" / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))
import r4rr_redmeat as redmeat  # noqa: E402


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def write_row(csv_path, row, header):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_model_ce(
    model,
    dataloaders,
    dataset_sizes,
    num_epochs_val,
    base_lr_val,
    classifier_lr_val,
    momentum_val,
    weight_decay_val,
    num_classes,
):
    best_wts = copy.deepcopy(model.state_dict())
    best_bal = -100.0
    best_epoch = -1
    since = time.time()

    param_groups = lgm._get_param_groups(model, base_lr_val, classifier_lr_val)
    opt = optim.SGD(param_groups, momentum=momentum_val, weight_decay=weight_decay_val)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs_val):
        print(f"Epoch {epoch + 1}/{num_epochs_val}", flush=True)
        for phase in ["train", "val"]:
            is_train = phase == "train"
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            class_correct = np.zeros(num_classes, dtype=np.int64)
            class_total = np.zeros(num_classes, dtype=np.int64)

            for batch in dataloaders[phase]:
                if len(batch) != 3:
                    raise RuntimeError(f"Unexpected batch format for RedMeat CE baseline: len={len(batch)}")

                inputs, labels, _paths = batch
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                if is_train:
                    opt.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs, _aux = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    if is_train:
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

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
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}", flush=True)

            if phase == "val":
                class_acc = class_correct / np.maximum(class_total, 1)
                balanced_acc = float(class_acc.mean())
                print(f"{phase} Balanced Acc: {balanced_acc:.4f}", flush=True)
                if balanced_acc > best_bal:
                    best_bal = balanced_acc
                    best_epoch = epoch
                    best_wts = copy.deepcopy(model.state_dict())

    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s", flush=True)

    model.load_state_dict(best_wts)
    return model, best_bal, best_epoch, int(elapsed)


def _classes_from_arg(classes_arg):
    if not classes_arg:
        return None
    classes = [c.strip() for c in str(classes_arg).split(",") if c.strip()]
    return classes or None


def run_single_trial(args, trial_number, base_lr, classifier_lr, momentum):
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    redmeat.device = device

    redmeat.base.seed_everything(int(args.seed))
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    num_workers = redmeat.base.get_num_workers(default=4)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((int(args.img_size), int(args.img_size))),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((int(args.img_size), int(args.img_size))),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

    classes = _classes_from_arg(getattr(args, "classes", None))
    train_dataset = redmeat.RedMeatMetadataDataset(
        data_root=args.data_path,
        split="train",
        image_transform=data_transforms["train"],
        return_mask=False,
        return_path=True,
        classes=classes,
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
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=redmeat.base.seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=redmeat.base.seed_worker,
            generator=g,
        ),
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=redmeat.base.seed_worker,
        generator=g,
    )

    model = lgm.ViTLGMStyleMaps(
        num_classes=num_classes,
        model_name=args.vit_model,
        pretrained=args.pretrained,
        image_size=int(args.img_size),
    ).to(device)

    print(
        f"\n=== TRIAL {trial_number}: RedMeat ViT baseline CE-only | "
        f"base_lr={base_lr} classifier_lr={classifier_lr} momentum={momentum} "
        f"weight_decay={args.weight_decay} batch={args.batch_size} epochs={args.num_epochs} "
        f"img_size={args.img_size} classes={train_dataset.classes} ===",
        flush=True,
    )

    best_model, best_balanced, best_epoch, train_seconds = train_model_ce(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        num_epochs_val=int(args.num_epochs),
        base_lr_val=float(base_lr),
        classifier_lr_val=float(classifier_lr),
        momentum_val=float(momentum),
        weight_decay_val=float(args.weight_decay),
        num_classes=int(num_classes),
    )

    test_loss, test_acc, class_acc, per_group, worst_group = redmeat.evaluate_test(
        best_model, test_loader, num_classes
    )
    print(f"[TRIAL {trial_number}] VAL best balanced acc={best_balanced:.4f} (epoch {best_epoch})", flush=True)
    print(f"[TRIAL {trial_number}] TEST acc={test_acc:.2f}% loss={test_loss:.4f}", flush=True)
    print(f"[TRIAL {trial_number}] TEST per_class={per_group:.2f}% worst_class={worst_group:.2f}%", flush=True)

    row = {
        "trial": int(trial_number),
        "base_lr": float(base_lr),
        "classifier_lr": float(classifier_lr),
        "momentum": float(momentum),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "num_epochs": int(args.num_epochs),
        "img_size": int(args.img_size),
        "best_balanced_val_acc": float(best_balanced),
        "best_epoch": int(best_epoch),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "per_group": float(per_group),
        "worst_group": float(worst_group),
        "seconds": int(train_seconds),
    }
    return row


def main():
    p = argparse.ArgumentParser(
        description=(
            "Optuna sweep for RedMeat ViT ERM baseline (CE-only), using the same "
            "ViT/SGD/two-LR setup as the Waterbirds vanilla baseline with R4RR disabled."
        )
    )
    p.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")

    p.add_argument("--n-trials", "--n_trials", dest="n_trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", "--output_csv", dest="output_csv", default="guided_redmeat_vit_baseline_optuna50.csv")

    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=150)
    p.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--vit-model", "--vit_model", dest="vit_model", choices=["vit_b_16"], default="vit_b_16")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")

    p.add_argument("--split-col", "--split_col", dest="split_col", default="split")
    p.add_argument("--label-col", "--label_col", dest="label_col", default="label")
    p.add_argument("--path-col", "--path_col", dest="path_col", default="abs_file_path")
    p.add_argument("--classes", default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon")

    # Same vanilla baseline sweep ranges used for Waterbirds.
    p.add_argument("--base-lr-min", "--base_lr_min", dest="base_lr_min", type=float, default=1e-5)
    p.add_argument("--base-lr-max", "--base_lr_max", dest="base_lr_max", type=float, default=5e-2)
    p.add_argument("--cls-lr-min", "--cls_lr_min", dest="cls_lr_min", type=float, default=1e-5)
    p.add_argument("--cls-lr-max", "--cls_lr_max", dest="cls_lr_max", type=float, default=5e-2)
    p.add_argument("--momentum-min", "--momentum_min", dest="momentum_min", type=float, default=0.85)
    p.add_argument("--momentum-max", "--momentum_max", dest="momentum_max", type=float, default=0.95)

    args = p.parse_args()

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError(f"Optuna import failed: {exc}") from exc

    header = [
        "trial",
        "base_lr",
        "classifier_lr",
        "momentum",
        "weight_decay",
        "batch_size",
        "num_epochs",
        "img_size",
        "best_balanced_val_acc",
        "best_epoch",
        "test_acc",
        "test_loss",
        "per_group",
        "worst_group",
        "seconds",
    ]

    print(
        "[SWEEP CONFIG] RedMeat baseline CE-only (R4RR off) | "
        f"n_trials={args.n_trials}, seed={args.seed}, "
        f"base_lr=[{args.base_lr_min},{args.base_lr_max}] (log), "
        f"classifier_lr=[{args.cls_lr_min},{args.cls_lr_max}] (log), "
        f"momentum=[{args.momentum_min},{args.momentum_max}], "
        f"fixed: batch={args.batch_size}, epochs={args.num_epochs}, "
        f"img_size={args.img_size}, weight_decay={args.weight_decay}",
        flush=True,
    )

    best_row = None
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        nonlocal best_row
        base_lr = float(trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
        classifier_lr = float(trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
        momentum = float(trial.suggest_float("momentum", args.momentum_min, args.momentum_max))

        row = run_single_trial(
            args=args,
            trial_number=int(trial.number),
            base_lr=base_lr,
            classifier_lr=classifier_lr,
            momentum=momentum,
        )
        write_row(args.output_csv, row, header)

        if best_row is None or row["best_balanced_val_acc"] > best_row["best_balanced_val_acc"]:
            best_row = row

        print(
            f"[TRIAL {trial.number}] done: best_balanced_val_acc={row['best_balanced_val_acc']:.4f} "
            f"(base_lr={base_lr:.6g}, cls_lr={classifier_lr:.6g}, momentum={momentum:.6g})",
            flush=True,
        )
        return row["best_balanced_val_acc"]

    study.optimize(objective, n_trials=int(args.n_trials), catch=(Exception,))

    print("\n[SWEEP DONE]", flush=True)
    if best_row is None:
        print("No successful trials.", flush=True)
        return

    print("Best trial row:", flush=True)
    for k in header:
        print(f"  {k}: {best_row[k]}", flush=True)


if __name__ == "__main__":
    main()
