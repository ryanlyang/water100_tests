#!/bin/bash -l
# Fixed five-seed rerun for the best Waterbirds-100 vanilla MobileNetV2 trial.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/vanilla100_mobilenetv2_fixed5_trial10_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/vanilla100_mobilenetv2_fixed5_trial10_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME=${ENV_NAME:-gals_a100}
conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
DATA_PATH=/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2

SEED_LIST=${SEED_LIST:-"0 1 2 3 4"}
BASE_LR=${BASE_LR:-0.03757772653269032}
CLASSIFIER_LR=${CLASSIFIER_LR:-9.265434364396105e-05}
MOMENTUM=${MOMENTUM:-0.8808375186178569}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
NESTEROV=${NESTEROV:-0}
NUM_EPOCHS=${NUM_EPOCHS:-200}
NUM_WORKERS=${NUM_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-96}

OUT_CSV=${OUT_CSV:-$LOG_DIR/vanilla100_mobilenetv2_fixed5_trial10_${SLURM_JOB_ID}.csv}
SUMMARY_CSV=${SUMMARY_CSV:-$LOG_DIR/vanilla100_mobilenetv2_fixed5_trial10_summary_${SLURM_JOB_ID}.csv}
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/Vanilla_MobileNetV2_Checkpoints}
export BASE_LR CLASSIFIER_LR MOMENTUM WEIGHT_DECAY NESTEROV NUM_EPOCHS NUM_WORKERS BATCH_SIZE CKPT_DIR

cd "$REPO_ROOT/GALS"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/GALS:${PYTHONPATH:-}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "[ERROR] Missing DATA_PATH: $DATA_PATH" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "Backbone: mobilenet_v2 pretrained=1 tune_mode=full"
echo "Fixed hyperparams from vanilla WB100 MobileNetV2 trial 10:"
echo "  base_lr=$BASE_LR"
echo "  classifier_lr=$CLASSIFIER_LR"
echo "  momentum=$MOMENTUM"
echo "  weight_decay=$WEIGHT_DECAY nesterov=$NESTEROV"
echo "Seeds: $SEED_LIST"
echo "Epochs: $NUM_EPOCHS batch_size=$BATCH_SIZE workers=$NUM_WORKERS"
echo "Output CSV: $OUT_CSV"
echo "Summary CSV: $SUMMARY_CSV"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
which python

python - "$DATA_PATH" "$OUT_CSV" "$SUMMARY_CSV" "$SEED_LIST" <<'PY'
import argparse
import csv
import os
import sys
import time

import numpy as np

import run_vanilla_waterbird_clip as rvw

data_path, out_csv, summary_csv, seed_list_s = sys.argv[1:5]
seeds = [int(s) for s in seed_list_s.split()]

base_lr = float(os.environ["BASE_LR"])
classifier_lr = float(os.environ["CLASSIFIER_LR"])
momentum = float(os.environ["MOMENTUM"])
weight_decay = float(os.environ["WEIGHT_DECAY"])
nesterov = os.environ.get("NESTEROV", "0").lower() in {"1", "true", "yes", "y", "on"}
num_epochs = int(os.environ["NUM_EPOCHS"])
num_workers = int(os.environ["NUM_WORKERS"])
batch_size = int(os.environ["BATCH_SIZE"])
ckpt_dir = os.environ["CKPT_DIR"]

header = [
    "seed",
    "model",
    "tune_mode",
    "pretrained",
    "base_lr",
    "classifier_lr",
    "weight_decay",
    "momentum",
    "nesterov",
    "best_balanced_val_acc",
    "test_acc",
    "per_group",
    "worst_group",
    "checkpoint",
    "seconds",
]

rows = []
for seed in seeds:
    args = argparse.Namespace(
        data_path=data_path,
        seed=seed,
        model="mobilenet_v2",
        clip_model="RN50",
        tune_mode="full",
        pretrained=True,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=base_lr,
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        num_workers=num_workers,
        checkpoint_dir=ckpt_dir,
    )
    t0 = time.time()
    best_balanced_val, test_acc, per_group, worst_group, ckpt = rvw.run_single(args)
    row = {
        "seed": seed,
        "model": "mobilenet_v2",
        "tune_mode": "full",
        "pretrained": 1,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "nesterov": nesterov,
        "best_balanced_val_acc": best_balanced_val,
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "checkpoint": ckpt,
        "seconds": int(time.time() - t0),
    }
    rows.append(row)
    exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(
        f"[SEED DONE] seed={seed} best_val={best_balanced_val:.4f} "
        f"test_acc={test_acc:.2f}% per_group={per_group:.2f}% worst_group={worst_group:.2f}%",
        flush=True,
    )

metrics = ["best_balanced_val_acc", "test_acc", "per_group", "worst_group"]
summary_header = ["metric", "mean", "std", "n"]
summary_rows = []
print("\n===== SUMMARY OVER SEEDS =====", flush=True)
for metric in metrics:
    vals = np.array([float(r[metric]) for r in rows], dtype=float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    summary_rows.append({"metric": metric, "mean": mean, "std": std, "n": len(vals)})
    print(f"{metric}: mean={mean:.4f} std={std:.4f} n={len(vals)}", flush=True)

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_header)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"[DONE] wrote {out_csv}", flush=True)
print(f"[DONE] wrote {summary_csv}", flush=True)
PY
