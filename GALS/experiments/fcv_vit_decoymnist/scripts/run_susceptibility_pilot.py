#!/usr/bin/env python3
"""Train one ViT-S/16 candidate and measure DecoyMNIST shortcut reliance."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_susceptibility import (  # noqa: E402
    DiagnosticDataset,
    OriginalDataset,
    WEIGHT_DECAY,
    atomic_json,
    build_model,
    build_transform,
    discover_samples,
    enumerate_runs,
    evaluate_diagnostics,
    flatten_samples,
    get_run,
    seed_everything,
    seed_worker,
    split_fingerprint,
    stratified_train_holdout,
    train_one_epoch,
    verify_encoding,
    warmup_cosine_factor,
)


DEFAULT_DATA = "/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png"
DEFAULT_OUTPUT = "/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def _flatten(prefix: str, metrics: dict) -> dict:
    row = {}
    for view in ("original", "digit_only", "patch_only"):
        row[f"{prefix}_{view}_accuracy"] = metrics[f"{view}_accuracy"]
        row[f"{prefix}_{view}_balanced_class_accuracy"] = metrics[
            f"{view}_balanced_class_accuracy"
        ]
        row[f"{prefix}_{view}_worst_class_accuracy"] = metrics[
            f"{view}_worst_class_accuracy"
        ]
        row[f"{prefix}_{view}_class_accuracies"] = json.dumps(
            metrics[f"{view}_class_accuracies"], separators=(",", ":")
        )
    return row


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup_epochs must satisfy 0 <= warmup < epochs")
    run = get_run(args.run_index)
    seed_everything(run.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_root = Path(args.output_root).expanduser().resolve()
    run_root = output_root / "runs" / run.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    metrics_path = run_root / "metrics.csv"
    summary_path = run_root / "run_summary.json"

    train_by_label = discover_samples(args.data_root, "train")
    test_by_label = discover_samples(args.data_root, "test")
    verify_encoding(train_by_label, "train", per_class=100)
    verify_encoding(test_by_label, "test", per_class=100)
    train_samples, validation_samples = stratified_train_holdout(
        train_by_label,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    test_samples = flatten_samples(test_by_label)
    holdout_sha256 = split_fingerprint(train_samples, validation_samples)

    transform = build_transform()
    train_dataset = OriginalDataset(train_samples, transform)
    biased_validation_dataset = DiagnosticDataset(
        validation_samples, "train", transform
    )
    reversed_test_dataset = DiagnosticDataset(test_samples, "test", transform)
    generator = torch.Generator().manual_seed(run.seed)
    common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        # Three diagnostic loaders are used sequentially.  Keeping all worker
        # pools resident would unnecessarily oversubscribe each Slurm task.
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    biased_validation_loader = DataLoader(
        biased_validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    reversed_test_loader = DataLoader(
        reversed_test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )

    print(
        f"[RUN] {run.run_id} lr={run.learning_rate:g} wd={WEIGHT_DECAY:g} "
        f"seed={run.seed} epochs={args.epochs}",
        flush=True,
    )
    print(
        f"[DATA] train={len(train_dataset)} biased_val={len(biased_validation_dataset)} "
        f"reversed_test={len(reversed_test_dataset)} split_sha256={holdout_sha256}",
        flush=True,
    )
    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=run.learning_rate, weight_decay=WEIGHT_DECAY
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_factor(
            step, total_steps=total_steps, warmup_steps=warmup_steps
        ),
    )

    rows: list[dict] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        lr_start = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        biased_metrics = evaluate_diagnostics(
            model, biased_validation_loader, device
        )
        reversed_metrics = evaluate_diagnostics(model, reversed_test_loader, device)
        row = {
            "run_index": run.run_index,
            "run_id": run.run_id,
            "epoch": epoch,
            "seed": run.seed,
            "learning_rate": run.learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "lr_epoch_start": lr_start,
            "lr_epoch_end": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.time() - epoch_started,
            "split_sha256": holdout_sha256,
        }
        row.update(_flatten("biased_val", biased_metrics))
        row.update(_flatten("reversed_test", reversed_metrics))
        row["biased_val_to_reversed_test_gap"] = (
            row["biased_val_original_accuracy"]
            - row["reversed_test_original_accuracy"]
        )
        row["biased_val_patch_erasure_drop"] = (
            row["biased_val_original_accuracy"]
            - row["biased_val_digit_only_accuracy"]
        )
        row["reversed_test_patch_erasure_change"] = (
            row["reversed_test_digit_only_accuracy"]
            - row["reversed_test_original_accuracy"]
        )
        row["susceptibility_gate_passed"] = bool(
            row["biased_val_original_accuracy"] >= 0.95
            and row["biased_val_to_reversed_test_gap"] >= 0.10
            and (
                row["biased_val_patch_erasure_drop"] >= 0.10
                or row["biased_val_patch_only_accuracy"] >= 0.80
            )
        )
        rows.append(row)
        _write_rows(metrics_path, rows)
        print(
            f"[EPOCH {epoch:02d}] train={row['train_accuracy']*100:.2f}% "
            f"biased={row['biased_val_original_accuracy']*100:.2f}% "
            f"digit_only={row['biased_val_digit_only_accuracy']*100:.2f}% "
            f"patch_only={row['biased_val_patch_only_accuracy']*100:.2f}% "
            f"reversed_test={row['reversed_test_original_accuracy']*100:.2f}% "
            f"gap={row['biased_val_to_reversed_test_gap']*100:.2f}pp",
            flush=True,
        )

    final = rows[-1]
    susceptible = any(bool(row["susceptibility_gate_passed"]) for row in rows)
    atomic_json(
        summary_path,
        {
            "artifact_type": "decoymnist_vit_susceptibility_run",
            "complete": True,
            "run_index": run.run_index,
            "run_id": run.run_id,
            "learning_rate": run.learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "seed": run.seed,
            "epochs": args.epochs,
            "model_name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
            "pretrained": not args.no_pretrained,
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "train_count": len(train_dataset),
            "biased_validation_count": len(biased_validation_dataset),
            "reversed_test_count": len(reversed_test_dataset),
            "validation_fraction": args.validation_fraction,
            "split_seed": args.split_seed,
            "split_sha256": holdout_sha256,
            "diagnostic_definition": {
                "original": "unmodified PNG",
                "digit_only": "detected 5x5 class-coded corner patch set to zero",
                "patch_only": "all pixels except detected 5x5 corner patch set to zero",
            },
            "susceptibility_gate": {
                "biased_validation_accuracy_min": 0.95,
                "biased_to_reversed_test_gap_min": 0.10,
                "patch_erasure_drop_min_or_patch_only_accuracy_min": [0.10, 0.80],
                "passed_any_epoch": susceptible,
                "passed_final_epoch": bool(final["susceptibility_gate_passed"]),
            },
            "final_epoch_metrics": final,
            "metrics_path": str(metrics_path),
            "checkpoint_saved": False,
            "seconds": time.time() - started,
        },
    )
    print(f"[DONE] susceptibility_gate={susceptible} metrics={metrics_path}")
    print("[DONE] no model checkpoint was saved")


if __name__ == "__main__":
    main()
