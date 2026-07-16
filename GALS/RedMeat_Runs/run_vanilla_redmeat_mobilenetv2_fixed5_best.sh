#!/bin/bash -l
# Fixed five-seed rerun for the best vanilla RedMeat MobileNetV2 trial.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/vanilla_redmeat_mobilenetv2_fixed5_best_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/vanilla_redmeat_mobilenetv2_fixed5_best_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV_CANDIDATES=(
  "${SCRIPT_DIR}/common_env.sh"
  "${SBATCH_SUBMIT_DIR:-}/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/Food101/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
)
COMMON_ENV=""
for candidate in "${COMMON_ENV_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    COMMON_ENV="$candidate"
    break
  fi
done
if [[ -z "$COMMON_ENV" ]]; then
  echo "[ERROR] Could not locate common_env.sh" >&2
  exit 2
fi
source "$COMMON_ENV"

redmeat_set_defaults
redmeat_activate_env

REPO_ROOT="$PROJECT_ROOT"
GALS_REPO="$GALS_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"

SEED_LIST=${SEED_LIST:-"0 1 2 3 4"}
MODEL_NAME=${MODEL_NAME:-mobilenet_v2}
CLIP_MODEL=${CLIP_MODEL:-RN50}
TUNE_MODE=${TUNE_MODE:-full}
PRETRAINED=${PRETRAINED:-1}
BASE_LR=${BASE_LR:-0.002798860574410339}
CLASSIFIER_LR=${CLASSIFIER_LR:-2.6678559411334764e-05}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
MOMENTUM=${MOMENTUM:-0.9168142621559325}
NESTEROV=${NESTEROV:-0}
NUM_EPOCHS=${NUM_EPOCHS:-150}
NUM_WORKERS=${NUM_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-96}

OUT_CSV=${OUT_CSV:-$LOG_DIR/vanilla_redmeat_mobilenetv2_fixed5_best_${SLURM_JOB_ID}.csv}
SUMMARY_CSV=${SUMMARY_CSV:-$LOG_DIR/vanilla_redmeat_mobilenetv2_fixed5_best_summary_${SLURM_JOB_ID}.csv}
CKPT_DIR=${CKPT_DIR:-$GALS_REPO/Vanilla_MobileNetV2_RedMeat_Checkpoints}

export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}

cd "$GALS_REPO"
export PYTHONPATH="$REPO_ROOT:$GALS_REPO:${PYTHONPATH:-}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing DATASET_ROOT: $DATASET_ROOT" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "Backbone: $MODEL_NAME pretrained=$PRETRAINED tune_mode=$TUNE_MODE"
echo "Fixed hyperparams from vanilla RedMeat MobileNetV2 best run:"
echo "  base_lr=$BASE_LR"
echo "  classifier_lr=$CLASSIFIER_LR"
echo "  weight_decay=$WEIGHT_DECAY"
echo "  momentum=$MOMENTUM"
echo "  nesterov=$NESTEROV"
echo "Seeds: $SEED_LIST"
echo "Epochs: $NUM_EPOCHS batch_size=$BATCH_SIZE workers=$NUM_WORKERS"
echo "Output CSV: $OUT_CSV"
echo "Summary CSV: $SUMMARY_CSV"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
which python

python - "$DATASET_ROOT" "$OUT_CSV" "$SUMMARY_CSV" "$SEED_LIST" \
  "$MODEL_NAME" "$CLIP_MODEL" "$TUNE_MODE" "$PRETRAINED" \
  "$BASE_LR" "$CLASSIFIER_LR" "$WEIGHT_DECAY" "$MOMENTUM" "$NESTEROV" \
  "$NUM_EPOCHS" "$NUM_WORKERS" "$BATCH_SIZE" "$CKPT_DIR" <<'PY'
import argparse
import csv
import os
import sys
import time

import numpy as np

import RedMeat_Runs.run_vanilla_redmeat as rvr

(
    data_path,
    out_csv,
    summary_csv,
    seed_list_s,
    model_name,
    clip_model,
    tune_mode,
    pretrained_s,
    base_lr_s,
    classifier_lr_s,
    weight_decay_s,
    momentum_s,
    nesterov_s,
    num_epochs_s,
    num_workers_s,
    batch_size_s,
    ckpt_dir,
) = sys.argv[1:18]

seeds = [int(s) for s in seed_list_s.split()]
pretrained = str(pretrained_s).lower() in {"1", "true", "yes", "y", "on"}
base_lr = float(base_lr_s)
classifier_lr = float(classifier_lr_s)
weight_decay = float(weight_decay_s)
momentum = float(momentum_s)
nesterov = str(nesterov_s).lower() in {"1", "true", "yes", "y", "on"}
num_epochs = int(num_epochs_s)
num_workers = int(num_workers_s)
batch_size = int(batch_size_s)
classes = "prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon"

header = [
    "seed",
    "model",
    "clip_model",
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
        model=model_name,
        clip_model=clip_model,
        tune_mode=tune_mode,
        pretrained=pretrained,
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
        classes=classes,
    )
    t0 = time.time()
    best_balanced_val, test_acc, per_group, worst_group, ckpt = rvr.run_single(args)
    row = {
        "seed": seed,
        "model": model_name,
        "clip_model": clip_model,
        "tune_mode": tune_mode,
        "pretrained": int(pretrained),
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
