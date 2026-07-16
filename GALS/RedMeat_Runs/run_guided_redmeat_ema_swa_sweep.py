#!/usr/bin/env python3
"""Optuna sweep for EMA/SWA on the fixed Guided RedMeat recipe.

This script intentionally locks the core guided hyperparameters to a known setting
and only sweeps averaging strategy parameters (EMA/SWA) at a single seed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
GALS_ROOT = SCRIPT_DIR.parent
if str(GALS_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(GALS_ROOT))

from RedMeat_Runs import run_guided_redmeat as rgm  # noqa: E402


# Locked guided recipe from the user-provided best row.
LOCKED_MODEL_NAME = "resnet50"
LOCKED_CLIP_MODEL = "RN50"
LOCKED_TUNE_MODE = "full"
LOCKED_PRETRAINED = True

LOCKED_ATTENTION_EPOCH = 2
LOCKED_KL_LAMBDA = 11.440224326405463
LOCKED_KL_INCR = 0.0
LOCKED_BASE_LR = 0.002404864319394485
LOCKED_CLASSIFIER_LR = 0.00023328334547057482
LOCKED_LR2_MULT = 1.5668544555186086

LOCKED_CLASSES = "prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon"
LOCKED_DEFAULT_GT_PATH = "/workspace/data/results_redmeat_openclip_dinovit/val/prediction_cmap"


def _parse_csv_list(text: str) -> Sequence[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _write_row(path: str, row: Dict[str, object], header: Sequence[str]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rgm.base.seed_everything(seed)


def _clone_state_dict(state_dict: Dict[str, torch.Tensor], to_cpu: bool) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        t = v.detach().clone()
        if to_cpu:
            t = t.cpu()
        out[k] = t
    return out


def _update_ema_state(
    ema_state: Optional[Dict[str, torch.Tensor]], model_state: Dict[str, torch.Tensor], decay: float
) -> Dict[str, torch.Tensor]:
    if ema_state is None:
        return _clone_state_dict(model_state, to_cpu=False)

    for k, src in model_state.items():
        dst = ema_state[k]
        src_detached = src.detach()
        if torch.is_floating_point(dst):
            dst.mul_(decay).add_(src_detached, alpha=1.0 - decay)
        else:
            dst.copy_(src_detached)
    return ema_state


def _update_swa_state(
    swa_state: Optional[Dict[str, torch.Tensor]], model_state: Dict[str, torch.Tensor], swa_n: int
) -> Tuple[Dict[str, torch.Tensor], int]:
    if swa_state is None:
        return _clone_state_dict(model_state, to_cpu=False), 1

    alpha = 1.0 / float(swa_n + 1)
    beta = 1.0 - alpha
    for k, src in model_state.items():
        dst = swa_state[k]
        src_detached = src.detach()
        if torch.is_floating_point(dst):
            dst.mul_(beta).add_(src_detached, alpha=alpha)
        else:
            dst.copy_(src_detached)
    return swa_state, swa_n + 1


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, num_classes: int) -> Dict[str, float]:
    loss, acc, _class_acc, balanced, worst = rgm.evaluate_test(model, loader, num_classes)
    return {
        "loss": float(loss),
        "acc": float(acc),
        "balanced_acc": float(balanced),
        "worst_group": float(worst),
    }


def _build_loaders(args: argparse.Namespace):
    # Locked backbone is resnet50, so use ImageNet normalization.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_tf = {
        "train": transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
        "eval": transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
    }
    mask_tf = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), rgm.base.Brighten(8.0)]
    )

    class_list = _parse_csv_list(args.classes) if args.classes else None

    train_dataset = rgm.RedMeatMetadataDataset(
        data_root=args.data_path,
        split="train",
        image_transform=image_tf["train"],
        mask_root=args.gt_path,
        mask_transform=mask_tf,
        return_mask=True,
        return_path=True,
        classes=class_list,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )
    val_dataset = rgm.RedMeatMetadataDataset(
        data_root=args.data_path,
        split="val",
        image_transform=image_tf["eval"],
        return_mask=False,
        return_path=True,
        classes=train_dataset.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )
    test_dataset = rgm.RedMeatMetadataDataset(
        data_root=args.data_path,
        split="test",
        image_transform=image_tf["eval"],
        return_mask=False,
        return_path=True,
        classes=train_dataset.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )

    num_classes = len(train_dataset.classes)

    g = torch.Generator()
    g.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        worker_init_fn=rgm.base.seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        worker_init_fn=rgm.base.seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        worker_init_fn=rgm.base.seed_worker,
        generator=g,
    )

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    return dataloaders, dataset_sizes, val_loader, test_loader, num_classes


def _run_one_trial(
    trial,
    args: argparse.Namespace,
    dataloaders,
    dataset_sizes,
    val_loader,
    test_loader,
    num_classes: int,
    device: torch.device,
):
    model = rgm.make_redmeat_cam_model(
        num_classes=num_classes,
        model_name=LOCKED_MODEL_NAME,
        pretrained=LOCKED_PRETRAINED,
        clip_model=LOCKED_CLIP_MODEL,
    ).to(device)
    rgm.configure_tune_mode(model, tune_mode=LOCKED_TUNE_MODE)

    eval_model = copy.deepcopy(model).to(device)

    avg_method = trial.suggest_categorical("avg_method", ["ema", "swa"])

    ema_decay = None
    ema_start_epoch = None
    ema_update_interval = None
    swa_start_frac = None
    swa_start_epoch = None
    swa_freq = None

    if avg_method == "ema":
        ema_decay = float(trial.suggest_float("ema_decay", args.ema_decay_min, args.ema_decay_max))
        ema_start_epoch = int(trial.suggest_int("ema_start_epoch", args.ema_start_min, args.ema_start_max))
        ema_update_interval = int(
            trial.suggest_int("ema_update_interval", args.ema_update_interval_min, args.ema_update_interval_max)
        )
    else:
        swa_start_frac = float(trial.suggest_float("swa_start_frac", args.swa_start_frac_min, args.swa_start_frac_max))
        swa_start_epoch = int(np.floor(swa_start_frac * float(args.num_epochs)))
        swa_start_epoch = max(LOCKED_ATTENTION_EPOCH, min(int(args.num_epochs) - 1, swa_start_epoch))
        swa_freq = int(trial.suggest_int("swa_freq", args.swa_freq_min, args.swa_freq_max))

    use_attention = LOCKED_ATTENTION_EPOCH < int(args.num_epochs) and LOCKED_KL_LAMBDA > 0.0

    param_groups = rgm.base._get_param_groups(model, LOCKED_BASE_LR, LOCKED_CLASSIFIER_LR)
    optimizer = optim.SGD(param_groups, momentum=rgm.momentum, weight_decay=rgm.weight_decay)

    kl_lambda_real = float(LOCKED_KL_LAMBDA)

    best_val_bal = -1e9
    best_epoch = -1
    best_state_cpu = _clone_state_dict(model.state_dict(), to_cpu=True)

    ema_state = None
    swa_state = None
    swa_n = 0

    t0 = time.time()

    for epoch in range(int(args.num_epochs)):
        if use_attention and epoch == LOCKED_ATTENTION_EPOCH:
            base_lr_after = LOCKED_BASE_LR * LOCKED_LR2_MULT
            classifier_lr_after = LOCKED_CLASSIFIER_LR * LOCKED_LR2_MULT
            print(
                f"[TRIAL {trial.number}] restart @epoch={epoch} "
                f"(base_lr={base_lr_after:.6g}, cls_lr={classifier_lr_after:.6g})",
                flush=True,
            )
            param_groups = rgm.base._get_param_groups(model, base_lr_after, classifier_lr_after)
            optimizer = optim.SGD(param_groups, momentum=rgm.momentum, weight_decay=rgm.weight_decay)

            # Keep selection logic aligned with the baseline script, which resets at restart.
            best_val_bal = -1e9
            best_epoch = -1
            best_state_cpu = _clone_state_dict(model.state_dict(), to_cpu=True)

            ema_state = None
            swa_state = None
            swa_n = 0

        if use_attention and epoch > LOCKED_ATTENTION_EPOCH:
            kl_lambda_real += float(LOCKED_KL_INCR)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_attn = 0.0

        for step_idx, batch in enumerate(dataloaders["train"]):
            if len(batch) != 4:
                raise RuntimeError("Expected train batch (images, labels, masks, paths)")

            inputs, labels, gt_masks, _paths = batch
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            gt_masks = gt_masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs, feats = model(inputs)
            _, preds = torch.max(outputs, 1)

            weights = model.classifier.weight[labels]
            cams = torch.einsum("bc,bchw->bhw", weights, feats)
            cams = torch.relu(cams)

            flat = cams.view(cams.size(0), -1)
            mn, _ = flat.min(dim=1, keepdim=True)
            mx, _ = flat.max(dim=1, keepdim=True)
            sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)

            gt_small = nn.functional.interpolate(gt_masks, size=sal_norm.shape[1:], mode="nearest").squeeze(1)

            if epoch < LOCKED_ATTENTION_EPOCH:
                loss, attn_loss = rgm.base.compute_loss(outputs, labels, sal_norm, gt_small, 333, True)
            else:
                loss, attn_loss = rgm.base.compute_loss(outputs, labels, sal_norm, gt_small, kl_lambda_real, False)

            loss.backward()
            optimizer.step()

            if avg_method == "ema" and ema_start_epoch is not None and epoch >= ema_start_epoch:
                if (step_idx % max(1, int(ema_update_interval))) == 0:
                    ema_state = _update_ema_state(ema_state, model.state_dict(), float(ema_decay))

            running_loss += float(loss.item()) * inputs.size(0)
            running_corrects += int(torch.sum(preds == labels.data).item())
            running_attn += float(attn_loss.item()) * inputs.size(0)

        if avg_method == "swa" and swa_start_epoch is not None and epoch >= swa_start_epoch:
            if ((epoch - swa_start_epoch) % max(1, int(swa_freq))) == 0:
                swa_state, swa_n = _update_swa_state(swa_state, model.state_dict(), swa_n)

        # Evaluate the selected averaged candidate on validation.
        if avg_method == "ema" and ema_state is not None:
            eval_model.load_state_dict(ema_state, strict=True)
            val_metrics = _evaluate(eval_model, val_loader, num_classes)
            candidate_state = ema_state
        elif avg_method == "swa" and swa_state is not None and swa_n > 0:
            eval_model.load_state_dict(swa_state, strict=True)
            val_metrics = _evaluate(eval_model, val_loader, num_classes)
            candidate_state = swa_state
        else:
            val_metrics = _evaluate(model, val_loader, num_classes)
            candidate_state = None

        eligible = (not use_attention) or (epoch >= LOCKED_ATTENTION_EPOCH)
        if eligible and val_metrics["balanced_acc"] > best_val_bal:
            best_val_bal = float(val_metrics["balanced_acc"])
            best_epoch = int(epoch)
            if candidate_state is None:
                best_state_cpu = _clone_state_dict(model.state_dict(), to_cpu=True)
            else:
                best_state_cpu = _clone_state_dict(candidate_state, to_cpu=True)

        if ((epoch + 1) % int(args.log_every) == 0) or (epoch == int(args.num_epochs) - 1):
            train_loss = running_loss / max(1, dataset_sizes["train"])
            train_acc = running_corrects / max(1, dataset_sizes["train"])
            train_attn = running_attn / max(1, dataset_sizes["train"])
            print(
                f"[TRIAL {trial.number}] E {epoch + 1}/{args.num_epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_attn={train_attn:.4f} "
                f"val_bal={val_metrics['balanced_acc']:.4f} val_acc={val_metrics['acc']:.2f} "
                f"best_val_bal={best_val_bal:.4f}",
                flush=True,
            )

    # Final eval at best checkpoint selected by val balanced accuracy.
    eval_model.load_state_dict(best_state_cpu, strict=True)
    val_best_metrics = _evaluate(eval_model, val_loader, num_classes)
    test_metrics = _evaluate(eval_model, test_loader, num_classes)

    ckpt_path = ""
    if args.save_best_checkpoint:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_name = f"redmeat_guided_{avg_method}_trial{trial.number}_seed{args.seed}.pth"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        torch.save(best_state_cpu, ckpt_path)

    row = {
        "trial": int(trial.number),
        "seed": int(args.seed),
        "gt_path": args.gt_path,
        "model_name": LOCKED_MODEL_NAME,
        "clip_model": LOCKED_CLIP_MODEL,
        "tune_mode": LOCKED_TUNE_MODE,
        "pretrained": int(bool(LOCKED_PRETRAINED)),
        "attention_epoch": int(LOCKED_ATTENTION_EPOCH),
        "kl_lambda": float(LOCKED_KL_LAMBDA),
        "kl_incr": float(LOCKED_KL_INCR),
        "base_lr": float(LOCKED_BASE_LR),
        "classifier_lr": float(LOCKED_CLASSIFIER_LR),
        "lr2_mult": float(LOCKED_LR2_MULT),
        "num_epochs": int(args.num_epochs),
        "avg_method": avg_method,
        "ema_decay": "" if ema_decay is None else float(ema_decay),
        "ema_start_epoch": "" if ema_start_epoch is None else int(ema_start_epoch),
        "ema_update_interval": "" if ema_update_interval is None else int(ema_update_interval),
        "swa_start_frac": "" if swa_start_frac is None else float(swa_start_frac),
        "swa_start_epoch": "" if swa_start_epoch is None else int(swa_start_epoch),
        "swa_freq": "" if swa_freq is None else int(swa_freq),
        "best_epoch": int(best_epoch),
        "best_balanced_val_acc": float(best_val_bal),
        "val_acc_at_best": float(val_best_metrics["acc"]),
        "val_worst_group_at_best": float(val_best_metrics["worst_group"]),
        "test_acc": float(test_metrics["acc"]),
        "test_balanced_acc": float(test_metrics["balanced_acc"]),
        "test_worst_group": float(test_metrics["worst_group"]),
        "seconds": int(time.time() - t0),
        "checkpoint": ckpt_path if ckpt_path else "NONE",
    }

    print(
        f"[TRIAL {trial.number}] DONE method={avg_method} "
        f"best_val_bal={row['best_balanced_val_acc']:.4f} "
        f"test_acc={row['test_acc']:.2f} test_bal={row['test_balanced_acc']:.2f} "
        f"test_worst={row['test_worst_group']:.2f}",
        flush=True,
    )

    return row


def main() -> None:
    p = argparse.ArgumentParser(
        description="Optuna EMA/SWA sweep on fixed Guided RedMeat hyperparameters (seed 0 by default)."
    )
    p.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")
    p.add_argument("--gt-path", default=LOCKED_DEFAULT_GT_PATH, help="OpenCLIP GT path for guidance masks.")

    p.add_argument("--split-col", default="split")
    p.add_argument("--label-col", default="label")
    p.add_argument("--path-col", default="abs_file_path")
    p.add_argument("--classes", default=LOCKED_CLASSES)

    p.add_argument("--seed", type=int, default=0, help="Training + Optuna seed (default: 0)")
    p.add_argument("--n-trials", type=int, default=20, help="Small/efficient default sweep size.")
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--output-csv", default="guided_redmeat_ema_swa_sweep.csv")

    p.add_argument("--num-epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--no-cuda", action="store_true")

    # EMA search space.
    p.add_argument("--ema-decay-min", type=float, default=0.995)
    p.add_argument("--ema-decay-max", type=float, default=0.99995)
    p.add_argument("--ema-start-min", type=int, default=2)
    p.add_argument("--ema-start-max", type=int, default=30)
    p.add_argument("--ema-update-interval-min", type=int, default=1)
    p.add_argument("--ema-update-interval-max", type=int, default=4)

    # SWA search space.
    p.add_argument("--swa-start-frac-min", type=float, default=0.45)
    p.add_argument("--swa-start-frac-max", type=float, default=0.9)
    p.add_argument("--swa-freq-min", type=int, default=1)
    p.add_argument("--swa-freq-max", type=int, default=5)

    p.add_argument("--save-best-checkpoint", action="store_true", default=True)
    p.add_argument("--no-save-best-checkpoint", action="store_false", dest="save_best_checkpoint")
    p.add_argument("--checkpoint-dir", default="RedMeat_Guided_Checkpoints")

    args = p.parse_args()

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"Missing data_path: {args.data_path}")
    if not os.path.isdir(args.gt_path):
        raise FileNotFoundError(f"Missing gt_path: {args.gt_path}")
    if int(args.n_trials) < 1:
        raise ValueError("--n-trials must be >= 1")
    if int(args.num_epochs) < 1:
        raise ValueError("--num-epochs must be >= 1")

    args.ema_start_min = max(0, int(args.ema_start_min))
    args.ema_start_max = max(args.ema_start_min, int(args.ema_start_max))

    args.swa_start_frac_min = max(0.0, min(0.99, float(args.swa_start_frac_min)))
    args.swa_start_frac_max = max(args.swa_start_frac_min, min(0.99, float(args.swa_start_frac_max)))

    args.ema_update_interval_min = max(1, int(args.ema_update_interval_min))
    args.ema_update_interval_max = max(args.ema_update_interval_min, int(args.ema_update_interval_max))
    args.swa_freq_min = max(1, int(args.swa_freq_min))
    args.swa_freq_max = max(args.swa_freq_min, int(args.swa_freq_max))

    _seed_everything(int(args.seed))

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    rgm.device = device
    rgm.base.device = device

    print("[INFO] Fixed Guided Hyperparameters", flush=True)
    print(f"  model_name={LOCKED_MODEL_NAME} clip_model={LOCKED_CLIP_MODEL} tune_mode={LOCKED_TUNE_MODE}", flush=True)
    print(
        f"  attention_epoch={LOCKED_ATTENTION_EPOCH} kl_lambda={LOCKED_KL_LAMBDA} kl_incr={LOCKED_KL_INCR}",
        flush=True,
    )
    print(
        f"  base_lr={LOCKED_BASE_LR} classifier_lr={LOCKED_CLASSIFIER_LR} lr2_mult={LOCKED_LR2_MULT}",
        flush=True,
    )
    print(f"  seed={args.seed} num_epochs={args.num_epochs} device={device}", flush=True)

    dataloaders, dataset_sizes, val_loader, test_loader, num_classes = _build_loaders(args)

    header = [
        "trial",
        "seed",
        "gt_path",
        "model_name",
        "clip_model",
        "tune_mode",
        "pretrained",
        "attention_epoch",
        "kl_lambda",
        "kl_incr",
        "base_lr",
        "classifier_lr",
        "lr2_mult",
        "num_epochs",
        "avg_method",
        "ema_decay",
        "ema_start_epoch",
        "ema_update_interval",
        "swa_start_frac",
        "swa_start_epoch",
        "swa_freq",
        "best_epoch",
        "best_balanced_val_acc",
        "val_acc_at_best",
        "val_worst_group_at_best",
        "test_acc",
        "test_balanced_acc",
        "test_worst_group",
        "seconds",
        "checkpoint",
    ]

    try:
        import optuna
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Optuna is required for this script. Import failed: {exc}")

    sampler = (
        optuna.samplers.TPESampler(seed=int(args.seed))
        if args.sampler == "tpe"
        else optuna.samplers.RandomSampler(seed=int(args.seed))
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)

    best_row_holder = {"row": None}

    def objective(trial):
        row = _run_one_trial(
            trial=trial,
            args=args,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            device=device,
        )
        _write_row(args.output_csv, row, header)

        best_row = best_row_holder["row"]
        if best_row is None or float(row["best_balanced_val_acc"]) > float(best_row["best_balanced_val_acc"]):
            best_row_holder["row"] = row

        return float(row["best_balanced_val_acc"])

    study.optimize(objective, n_trials=int(args.n_trials))

    best_row = best_row_holder["row"]
    if best_row is None:
        raise RuntimeError("No successful trials completed.")

    print("[BEST] EMA/SWA sweep best trial", flush=True)
    for k in header:
        print(f"  {k}: {best_row[k]}", flush=True)


if __name__ == "__main__":
    main()
