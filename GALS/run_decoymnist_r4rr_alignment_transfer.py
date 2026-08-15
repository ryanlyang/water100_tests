#!/usr/bin/env python3
"""Transfer the best WB100 alignment-loss trial to DecoyMNIST over five seeds."""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from run_r4rr_alignment_best5 import LOSSES, resolve_sweep_csv, select_best_row


EPOCHS = 19
SOURCE_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
METRICS = (
    "best_val_acc",
    "test_acc",
    "test_balanced_class_acc",
    "test_worst_class_acc",
)


def exposure_scaled_epoch(source_epoch):
    scaled = int(math.floor(float(source_epoch) * EPOCHS / SOURCE_EPOCHS + 0.5))
    return max(1, min(EPOCHS, scaled))


def load_train_module():
    train_dir = Path(__file__).resolve().parent / "RightForTheRightRegions/repro_runs/r4rr/train"
    sys.path.insert(0, str(train_dir))
    import r4rr_decoy_fixed as train

    return train


def build_datasets(train, args, use_cuda):
    image_transform = train.Compose(
        [
            train.Grayscale(num_output_channels=1),
            train.ToTensor(),
            train.Lambda(lambda image: image * 2.0 - 1.0),
        ]
    )
    mask_transform = train.transforms.Compose(
        [
            train.ExpandWhite(thr=10, radius=3),
            train.EdgeExtract(thr=10, edge_width=1),
            train.transforms.Resize((28, 28)),
            train.transforms.ToTensor(),
            train.Brighten(8.0),
        ]
    )
    guided = train.GuidedImageFolder(
        image_root=os.path.join(args.png_root, "train"),
        mask_root=args.teacher_map_path,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    plain = train.ImageFolder(os.path.join(args.png_root, "train"), transform=image_transform)
    test = train.ImageFolder(os.path.join(args.png_root, "test"), transform=image_transform)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}
    return guided, plain, test, loader_kwargs


def load_completed(path, expected):
    if not os.path.isfile(path):
        return {}
    completed = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                same = (
                    row.get("alignment_loss") == expected["alignment_loss"]
                    and os.path.realpath(row.get("source_sweep_csv", ""))
                    == os.path.realpath(expected["source_sweep_csv"])
                    and int(row["source_best_trial"]) == expected["source_best_trial"]
                    and int(row["source_attention_epoch"]) == expected["source_attention_epoch"]
                    and int(row["attention_epoch"]) == expected["attention_epoch"]
                    and math.isclose(
                        float(row["alignment_weight"]),
                        expected["alignment_weight"],
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                )
                for metric in METRICS:
                    same = same and math.isfinite(float(row[metric]))
                if same:
                    completed[int(row["seed"])] = row
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def existing_source_sweep(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source_sweep_csv")
            if source and os.path.isfile(source):
                return source
    return None


def append_row(path, row, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_summary(path, rows, metadata):
    fields = [
        "alignment_loss",
        "source_sweep_csv",
        "source_best_trial",
        "source_attention_epoch",
        "attention_epoch",
        "alignment_weight",
        "metric",
        "mean",
        "std",
        "n",
    ]
    summary = []
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary.append(
            {
                **metadata,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "n": int(values.size),
            }
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    os.replace(temporary, path)
    print("\n===== DECOYMNIST FIVE-SEED SUMMARY (population std) =====", flush=True)
    for row in summary:
        print(
            f"{row['metric']}: mean={row['mean']:.4f} std={row['std']:.4f} n={row['n']}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-loss", choices=LOSSES, required=True)
    parser.add_argument("--sweep-csv")
    parser.add_argument("--sweep-log-dir", required=True)
    parser.add_argument("--min-sweep-trials", type=int, default=50)
    parser.add_argument("--png-root", required=True)
    parser.add_argument("--teacher-map-path", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sweep_csv = args.sweep_csv
    if not sweep_csv:
        sweep_csv = existing_source_sweep(args.output_csv)
        if sweep_csv:
            print(f"[RESUME] Reusing source sweep recorded in output CSV: {sweep_csv}", flush=True)
        else:
            sweep_csv = resolve_sweep_csv(
                args.sweep_log_dir, "wb100", args.alignment_loss, args.min_sweep_trials
            )
    best, row_count = select_best_row(
        sweep_csv, args.alignment_loss, args.min_sweep_trials
    )
    source_attention_epoch = int(best["attention_epoch"])
    attention_epoch = exposure_scaled_epoch(source_attention_epoch)
    sweep_csv = os.path.realpath(sweep_csv)
    metadata = {
        "alignment_loss": args.alignment_loss,
        "source_sweep_csv": sweep_csv,
        "source_best_trial": int(best["trial"]),
        "source_attention_epoch": source_attention_epoch,
        "attention_epoch": attention_epoch,
        "alignment_weight": float(best["kl_lambda"]),
    }
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds requested: {seeds}")

    print(f"[SELECT] wb100_sweep={sweep_csv} valid_rows={row_count}", flush=True)
    print(
        f"[TRANSFER] loss={args.alignment_loss} source_trial={best['trial']} "
        f"source_attention_epoch={source_attention_epoch}/200 -> "
        f"decoy_attention_epoch={attention_epoch}/19 alignment_weight={best['kl_lambda']:.9g}",
        flush=True,
    )
    print(
        f"[FIXED] LeNet Grad-CAM, Adam lr={LR} weight_decay={WEIGHT_DECAY} epochs={EPOCHS}",
        flush=True,
    )

    train = load_train_module()
    torch = train.torch
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    guided, plain, test, loader_kwargs = build_datasets(train, args, use_cuda)
    train_args = argparse.Namespace(
        val_frac=0.1,
        split_seed=0,
        batch_size=64,
        test_batch_size=1000,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        attention_epoch=attention_epoch,
        kl_lambda=float(best["kl_lambda"]),
        kl_incr=0.0,
        alignment_loss=args.alignment_loss,
        print_every=5,
        epoch_checkpoint_dir="",
    )
    fields = [
        "alignment_loss",
        "seed",
        "source_sweep_csv",
        "source_best_trial",
        "source_attention_epoch",
        "source_lr2_mult",
        "attention_epoch",
        "alignment_weight",
        "epochs",
        "lr",
        "weight_decay",
        *METRICS,
        "best_epoch",
        "test_class_accs",
        "seconds",
    ]
    completed = load_completed(args.output_csv, metadata)
    for seed in seeds:
        if seed in completed:
            print(f"[RESUME] seed={seed} already complete; skipping", flush=True)
            continue
        started = time.time()
        result = train.train_one_seed(
            args=train_args,
            seed=seed,
            full_train_guided=guided,
            full_train_plain=plain,
            true_test=test,
            device=device,
            loader_kwargs=loader_kwargs,
        )
        row = {
            **metadata,
            "seed": seed,
            "source_lr2_mult": float(best["lr2_mult"]),
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "best_val_acc": result["best_val_acc"],
            "test_acc": result["test_acc"],
            "test_balanced_class_acc": result["test_balanced_class_acc"],
            "test_worst_class_acc": result["test_worst_class_acc"],
            "best_epoch": result["best_epoch"],
            "test_class_accs": np.array2string(
                result["test_class_acc"], precision=4, separator=",", max_line_width=10000
            ),
            "seconds": int(time.time() - started),
        }
        append_row(args.output_csv, row, fields)
        completed[seed] = row
        print(
            f"[SEED DONE] seed={seed} val={result['best_val_acc']:.2f}% "
            f"test={result['test_acc']:.2f}% mean_class={result['test_balanced_class_acc']:.2f}% "
            f"worst_class={result['test_worst_class_acc']:.2f}%",
            flush=True,
        )

    missing = [seed for seed in seeds if seed not in completed]
    if missing:
        raise RuntimeError(f"Missing completed seeds: {missing}")
    write_summary(args.summary_csv, [completed[seed] for seed in seeds], metadata)
    print(f"[DONE] per-seed CSV: {args.output_csv}", flush=True)
    print(f"[DONE] summary CSV: {args.summary_csv}", flush=True)


if __name__ == "__main__":
    main()
