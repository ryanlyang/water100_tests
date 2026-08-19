#!/usr/bin/env python3
"""Measure CLIP-LR after progressively adapting the CLIP RN50 image encoder.

The image encoder is evaluated at fixed epoch checkpoints. At every checkpoint
it is frozen, features are extracted once, and two logistic-regression probes
are evaluated:

1. the original optimized C held fixed; and
2. C retuned using validation macro-class accuracy only.

The temporary fine-tuning head is discarded before either probe is fitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from RedMeat_Runs import run_clip_lr_sweep_redmeat as clip_lr


CLASS_NAMES = (
    "prime_rib",
    "pork_chop",
    "steak",
    "baby_back_ribs",
    "filet_mignon",
)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
RESULT_FIELDS = (
    "seed",
    "finetune_epoch",
    "protocol",
    "c_selection",
    "C",
    "c_trials",
    "clip_model",
    "unfreeze_scope",
    "encoder_lr",
    "head_lr",
    "weight_decay",
    "train_loss",
    "train_acc",
    "val_acc",
    "val_avg_group_acc",
    "val_worst_group_acc",
    "val_group_accs",
    "test_acc",
    "test_avg_group_acc",
    "test_worst_group_acc",
    "test_group_accs",
    "feature_dim",
    "seconds",
)
TRIAL_FIELDS = (
    "seed",
    "finetune_epoch",
    "trial",
    "C",
    "val_acc",
    "val_avg_group_acc",
    "val_worst_group_acc",
    "val_group_accs",
    "state",
)


class FineTuneDataset(Dataset):
    def __init__(self, samples: Sequence[object], transform) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as source:
            image = self.transform(source.convert("RGB"))
        return image, int(sample.label)


def parse_epoch_list(text: str) -> List[int]:
    epochs = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not epochs or epochs[0] != 0:
        raise ValueError("Evaluation epochs must include epoch 0")
    if any(epoch < 0 for epoch in epochs):
        raise ValueError("Evaluation epochs must be nonnegative")
    return epochs


def atomic_write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    values = np.full(num_classes, np.nan, dtype=np.float64)
    for class_index in range(num_classes):
        selected = y_true == class_index
        if np.any(selected):
            values[class_index] = 100.0 * float(np.mean(y_pred[selected] == y_true[selected]))
    return values


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, object]:
    per_class = class_accuracy(y_true, y_pred, num_classes)
    return {
        "acc": 100.0 * float(np.mean(y_pred == y_true)),
        "avg_group_acc": float(np.nanmean(per_class)),
        "worst_group_acc": float(np.nanmin(per_class)),
        "group_accs": json.dumps([float(value) for value in per_class]),
    }


def fit_logistic_regression(C: float, X_train: np.ndarray, y_train: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(
        random_state=int(seed),
        C=float(C),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        max_iter=5000,
        n_jobs=1,
        verbose=0,
    )
    clip_lr._safe_fit(classifier, X_train, y_train)
    return classifier


def evaluate_probe(
    C: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    num_classes: int,
) -> Dict[str, object]:
    classifier = fit_logistic_regression(C, X_train, y_train, seed)
    val = metric_dict(y_val, classifier.predict(X_val), num_classes)
    test = metric_dict(y_test, classifier.predict(X_test), num_classes)
    return {
        "val_acc": val["acc"],
        "val_avg_group_acc": val["avg_group_acc"],
        "val_worst_group_acc": val["worst_group_acc"],
        "val_group_accs": val["group_accs"],
        "test_acc": test["acc"],
        "test_avg_group_acc": test["avg_group_acc"],
        "test_worst_group_acc": test["worst_group_acc"],
        "test_group_accs": test["group_accs"],
    }


def tune_c(
    *,
    epoch: int,
    seed: int,
    sweep_seed: int,
    baseline_c: float,
    c_min: float,
    c_max: float,
    n_trials: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
) -> Tuple[float, List[Dict[str, object]]]:
    import optuna

    if n_trials < 1:
        raise ValueError("c_trials must be positive")
    sampler = optuna.samplers.TPESampler(seed=int(sweep_seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if c_min <= baseline_c <= c_max:
        study.enqueue_trial({"C": float(baseline_c)})

    trial_rows: List[Dict[str, object]] = []

    def objective(trial) -> float:
        C = float(trial.suggest_float("C", float(c_min), float(c_max), log=True))
        classifier = fit_logistic_regression(C, X_train, y_train, seed)
        val = metric_dict(y_val, classifier.predict(X_val), num_classes)
        trial_rows.append(
            {
                "seed": seed,
                "finetune_epoch": epoch,
                "trial": trial.number,
                "C": C,
                "val_acc": val["acc"],
                "val_avg_group_acc": val["avg_group_acc"],
                "val_worst_group_acc": val["worst_group_acc"],
                "val_group_accs": val["group_accs"],
                "state": "COMPLETE",
            }
        )
        return float(val["avg_group_acc"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=int(n_trials))
    return float(study.best_params["C"]), trial_rows


def configure_trainable(model: nn.Module, scope: str) -> List[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if scope == "layer4_attnpool":
        modules = (model.visual.layer4, model.visual.attnpool)
    elif scope == "full_visual":
        modules = (model.visual,)
    else:
        raise ValueError(f"Unsupported unfreeze scope: {scope}")
    parameters: List[nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
            parameters.append(parameter)
    return parameters


def set_finetune_mode(model: nn.Module, head: nn.Module, scope: str) -> None:
    # Keep frozen BatchNorm statistics fixed. Only explicitly unfrozen modules
    # enter training mode.
    model.eval()
    if scope == "layer4_attnpool":
        model.visual.layer4.train()
        model.visual.attnpool.train()
    else:
        model.visual.train()
    head.train()


def extract_features(samples, model, preprocess, device: str, batch_size: int, workers: int):
    X, y = clip_lr._extract_features(
        list(samples), model, preprocess, device, int(batch_size), int(workers)
    )
    X = np.ascontiguousarray(clip_lr._l2_normalize(X), dtype=np.float64)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, copy=False)
    return X, y


def result_row(
    *,
    args,
    epoch: int,
    protocol: str,
    c_selection: str,
    C: float,
    train_loss: float,
    train_acc: float,
    feature_dim: int,
    metrics: Dict[str, object],
    seconds: int,
) -> Dict[str, object]:
    return {
        "seed": args.seed,
        "finetune_epoch": epoch,
        "protocol": protocol,
        "c_selection": c_selection,
        "C": C,
        "c_trials": 0 if protocol == "fixed_c" or epoch == 0 else args.c_trials,
        "clip_model": args.clip_model,
        "unfreeze_scope": args.unfreeze_scope,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "feature_dim": feature_dim,
        "seconds": seconds,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--clip-model", default="RN50")
    parser.add_argument("--eval-epochs", default="0,1,2,4,8,16")
    parser.add_argument(
        "--unfreeze-scope",
        choices=("layer4_attnpool", "full_visual"),
        default="full_visual",
    )
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--baseline-c", type=float, default=1.329346323656201)
    parser.add_argument("--c-min", type=float, default=1e-2)
    parser.add_argument("--c-max", type=float, default=1e2)
    parser.add_argument("--c-trials", type=int, default=25)
    parser.add_argument("--c-sweep-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = time.time()
    eval_epochs = parse_epoch_list(args.eval_epochs)
    if args.clip_model != "RN50":
        raise ValueError("This controlled study is fixed to OpenAI CLIP RN50")
    if args.c_min <= 0 or args.c_max <= args.c_min:
        raise ValueError("Require 0 < c_min < c_max")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    trials_path = output_dir / "c_trials.csv"
    config_path = output_dir / "run_config.json"
    config = {
        "seed": args.seed,
        "data_root": str(args.data_root.expanduser().resolve()),
        "clip_model": args.clip_model,
        "eval_epochs": eval_epochs,
        "unfreeze_scope": args.unfreeze_scope,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "feature_batch_size": args.feature_batch_size,
        "num_workers": args.num_workers,
        "baseline_c": args.baseline_c,
        "c_min": args.c_min,
        "c_max": args.c_max,
        "c_trials": args.c_trials,
        "c_sweep_seed": args.c_sweep_seed,
    }
    if config_path.is_file() and results_path.is_file():
        old_config = json.load(config_path.open(encoding="utf-8"))
        with results_path.open(newline="", encoding="utf-8") as handle:
            old_rows = list(csv.DictReader(handle))
        expected = {(epoch, protocol) for epoch in eval_epochs for protocol in ("fixed_c", "retuned_c")}
        observed = {
            (int(row["finetune_epoch"]), row["protocol"])
            for row in old_rows
            if int(row.get("seed", -1)) == args.seed
        }
        if old_config == config and observed == expected:
            print(f"[RESUME] Complete result already exists: {results_path}", flush=True)
            return
        if old_config != config:
            raise RuntimeError(
                f"Existing output uses a different configuration: {config_path}. "
                "Choose a new output directory."
            )
    atomic_write_json(config_path, config)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    data_root = args.data_root.expanduser().resolve()
    classes, train_samples, val_samples, test_samples = clip_lr._build_splits(
        dataset_path=str(data_root),
        split_col="split",
        label_col="label",
        path_col="abs_file_path",
        train_value="train",
        val_value="val",
        test_value="test",
        classes=list(CLASS_NAMES),
    )
    if tuple(classes) != CLASS_NAMES:
        raise RuntimeError(f"Class order mismatch: {classes}")

    clip_module = clip_lr._try_import_clip()
    try:
        model, eval_preprocess = clip_module.load(args.clip_model, device=str(device), jit=False)
    except TypeError:
        model, eval_preprocess = clip_module.load(args.clip_model, device=str(device))
    model.float()
    model.to(device)

    encoder_parameters = configure_trainable(model, args.unfreeze_scope)
    feature_dim = int(model.visual.output_dim)
    head = nn.Linear(feature_dim, len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(args.encoder_lr)},
            {"params": head.parameters(), "lr": float(args.head_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        FineTuneDataset(train_samples, train_transform),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=int(args.num_workers) > 0,
    )

    result_rows: List[Dict[str, object]] = []
    all_trial_rows: List[Dict[str, object]] = []
    last_train_loss = float("nan")
    last_train_acc = float("nan")

    def evaluate_epoch(epoch: int) -> None:
        nonlocal result_rows, all_trial_rows
        checkpoint_started = time.time()
        model.eval()
        print(f"[FEATURES] epoch={epoch} extracting train/val/test", flush=True)
        X_train, y_train = extract_features(
            train_samples, model, eval_preprocess, str(device), args.feature_batch_size, args.num_workers
        )
        X_val, y_val = extract_features(
            val_samples, model, eval_preprocess, str(device), args.feature_batch_size, args.num_workers
        )
        X_test, y_test = extract_features(
            test_samples, model, eval_preprocess, str(device), args.feature_batch_size, args.num_workers
        )

        fixed_metrics = evaluate_probe(
            args.baseline_c,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            args.seed,
            len(CLASS_NAMES),
        )
        result_rows.append(
            result_row(
                args=args,
                epoch=epoch,
                protocol="fixed_c",
                c_selection="original_optimized",
                C=args.baseline_c,
                train_loss=last_train_loss,
                train_acc=last_train_acc,
                feature_dim=feature_dim,
                metrics=fixed_metrics,
                seconds=int(time.time() - checkpoint_started),
            )
        )

        if epoch == 0:
            tuned_c = float(args.baseline_c)
            tuned_metrics = dict(fixed_metrics)
            c_selection = "original_optimized_epoch0"
        else:
            tuned_c, epoch_trials = tune_c(
                epoch=epoch,
                seed=args.seed,
                sweep_seed=args.c_sweep_seed,
                baseline_c=args.baseline_c,
                c_min=args.c_min,
                c_max=args.c_max,
                n_trials=args.c_trials,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_classes=len(CLASS_NAMES),
            )
            all_trial_rows.extend(epoch_trials)
            atomic_write_csv(trials_path, all_trial_rows, TRIAL_FIELDS)
            tuned_metrics = evaluate_probe(
                tuned_c,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                args.seed,
                len(CLASS_NAMES),
            )
            c_selection = "retuned_val_avg_group"

        result_rows.append(
            result_row(
                args=args,
                epoch=epoch,
                protocol="retuned_c",
                c_selection=c_selection,
                C=tuned_c,
                train_loss=last_train_loss,
                train_acc=last_train_acc,
                feature_dim=feature_dim,
                metrics=tuned_metrics,
                seconds=int(time.time() - checkpoint_started),
            )
        )
        atomic_write_csv(results_path, result_rows, RESULT_FIELDS)
        print(
            f"[RESULT] seed={args.seed} epoch={epoch} "
            f"fixed_C={args.baseline_c:.8g} fixed_test={fixed_metrics['test_avg_group_acc']:.2f} "
            f"fixed_worst={fixed_metrics['test_worst_group_acc']:.2f} "
            f"retuned_C={tuned_c:.8g} retuned_test={tuned_metrics['test_avg_group_acc']:.2f} "
            f"retuned_worst={tuned_metrics['test_worst_group_acc']:.2f}",
            flush=True,
        )

    evaluate_epoch(0)
    max_epoch = max(eval_epochs)
    for epoch in range(1, max_epoch + 1):
        set_finetune_mode(model, head, args.unfreeze_scope)
        loss_sum = 0.0
        correct = 0
        count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            features = F.normalize(model.encode_image(images).float(), dim=1)
            logits = head(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            batch_count = int(labels.numel())
            loss_sum += float(loss.detach()) * batch_count
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += batch_count
        last_train_loss = loss_sum / max(count, 1)
        last_train_acc = 100.0 * correct / max(count, 1)
        print(
            f"[TRAIN] seed={args.seed} epoch={epoch}/{max_epoch} "
            f"loss={last_train_loss:.6f} acc={last_train_acc:.2f}",
            flush=True,
        )
        if epoch in eval_epochs:
            evaluate_epoch(epoch)

    print(f"[DONE] results={results_path}", flush=True)
    print(f"[DONE] c_trials={trials_path}", flush=True)
    print(f"[TIME] seconds={int(time.time() - started)}", flush=True)


if __name__ == "__main__":
    main()
