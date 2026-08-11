#!/bin/bash -l
# Train one R4RR seed and evaluate Pointing Game at every epoch.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=r4rr_decoy_epoch_pg
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
PNG_ROOT="${PNG_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"
MNIST_ROOT="${MNIST_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data}"
R4RR_MAPS="${R4RR_MAPS:-/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap}"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-19}"
ATTENTION_EPOCH="${ATTENTION_EPOCH:-7}"
KL_LAMBDA="${KL_LAMBDA:-495.61}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
RUN_DIR="${RUN_DIR:-$LOG_DIR/decoy_r4rr_epochwise_pointing_seed${SEED}}"

CHECKPOINT_DIR="$RUN_DIR/checkpoints"
BEST_DIR="$RUN_DIR/best_checkpoint"
EPOCH_DIR="$RUN_DIR/epochs"
TRAIN_SCRIPT="$GALS_ROOT/RightForTheRightRegions/repro_runs/r4rr/train/r4rr_decoy_fixed.py"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$BEST_DIR" "$EPOCH_DIR"
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$GALS_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled

cd "$GALS_ROOT"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[RUN] seed=$SEED epochs=$EPOCHS attention_epoch=$ATTENTION_EPOCH"
echo "[RUN] lr=$LR weight_decay=$WEIGHT_DECAY kl_lambda=$KL_LAMBDA"
echo "[RUN] output=$RUN_DIR"
which python

[[ -d "$PNG_ROOT/train" && -d "$PNG_ROOT/test" ]] || {
  echo "[ERROR] Missing DecoyMNIST data: $PNG_ROOT" >&2
  exit 2
}
[[ -d "$MNIST_ROOT/MNIST" ]] || {
  echo "[ERROR] Missing clean torchvision MNIST: $MNIST_ROOT/MNIST" >&2
  exit 2
}
[[ -d "$R4RR_MAPS" ]] || {
  echo "[ERROR] Missing R4RR teacher maps: $R4RR_MAPS" >&2
  exit 2
}

checkpoint_set_is_valid() {
  python - "$CHECKPOINT_DIR" "$BEST_DIR" "$SEED" "$EPOCHS" \
    "$ATTENTION_EPOCH" "$KL_LAMBDA" "$LR" "$WEIGHT_DECAY" "$R4RR_MAPS" <<'PY'
import math
import os
import sys
from pathlib import Path

import torch

(
    checkpoint_dir,
    best_dir,
    seed_text,
    epochs_text,
    attention_epoch_text,
    kl_lambda_text,
    lr_text,
    weight_decay_text,
    teacher_maps,
) = sys.argv[1:]
seed, epochs = int(seed_text), int(epochs_text)
valid = len(list(Path(best_dir).glob("*.pth"))) >= 1
for epoch in range(1, epochs + 1):
    path = Path(checkpoint_dir) / f"decoy_r4rr_seed{seed}_epoch{epoch:02d}.pth"
    if not path.is_file():
        valid = False
        break
    try:
        payload = torch.load(str(path), map_location="cpu")
        settings = payload.get("args", {})
        valid = valid and int(payload.get("seed", -1)) == seed
        valid = valid and int(payload.get("epoch", -1)) == epoch
        valid = valid and int(settings.get("epochs", -1)) == epochs
        valid = valid and int(settings.get("attention_epoch", -1)) == int(attention_epoch_text)
        valid = valid and math.isclose(float(settings.get("kl_lambda", "nan")), float(kl_lambda_text))
        valid = valid and math.isclose(float(settings.get("lr", "nan")), float(lr_text))
        valid = valid and math.isclose(
            float(settings.get("weight_decay", "nan")), float(weight_decay_text)
        )
        valid = valid and os.path.realpath(settings.get("teacher_map_path", "")) == os.path.realpath(
            teacher_maps
        )
    except Exception:
        valid = False
        break
raise SystemExit(0 if valid else 1)
PY
}

DID_TRAIN=0
if [[ "$FORCE_RETRAIN" == "0" ]] && checkpoint_set_is_valid; then
  echo "[RESUME] Complete epoch checkpoint set exists; skipping training."
else
  echo "[TRAIN] Saving every epoch from one R4RR trajectory."
  python -u "$TRAIN_SCRIPT" \
    --png-root "$PNG_ROOT" \
    --teacher-map-path "$R4RR_MAPS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --attention-epoch "$ATTENTION_EPOCH" \
    --kl-lambda "$KL_LAMBDA" \
    --kl-incr 0.0 \
    --n-seeds 1 \
    --seed-start "$SEED" \
    --split-seed 0 \
    --num-workers "$NUM_WORKERS" \
    --print-every 1 \
    --save-dir "$BEST_DIR" \
    --epoch-checkpoint-dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$RUN_DIR/training.log"
  DID_TRAIN=1
fi

for epoch in $(seq 1 "$EPOCHS"); do
  epoch_pad="$(printf '%02d' "$epoch")"
  checkpoint="$CHECKPOINT_DIR/decoy_r4rr_seed${SEED}_epoch${epoch_pad}.pth"
  output_dir="$EPOCH_DIR/epoch_${epoch_pad}"
  summary="$output_dir/pointing_game_summary.csv"
  mkdir -p "$output_dir"

  if [[ "$DID_TRAIN" == "0" ]] && python - "$summary" "$checkpoint" "$SEED" <<'PY'
import csv, os, sys
summary, checkpoint, seed = sys.argv[1:]
try:
    rows = list(csv.DictReader(open(summary, newline="", encoding="utf-8")))
    row = rows[0] if len(rows) == 1 else None
    valid = (
        row is not None
        and row.get("dataset") == "decoymnist"
        and row.get("method") == "r4rr"
        and int(row.get("seed", -1)) == int(seed)
        and int(row.get("mask_protocol_version", -1)) == 2
        and row.get("primary_pg_protocol") == "native_resolution_overlap"
        and os.path.realpath(row.get("checkpoint", "")) == os.path.realpath(checkpoint)
        and int(row.get("pg_native_total", 0)) == 10000
        and int(row.get("errors", 1)) == 0
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
  then
    echo "[RESUME] epoch=$epoch valid Pointing Game result exists."
    continue
  fi

  echo "[POINTING] epoch=$epoch checkpoint=$checkpoint"
  python -u decoymnist_pointing_game_eval.py \
    --png-root "$PNG_ROOT" \
    --mnist-root "$MNIST_ROOT" \
    --checkpoint "$checkpoint" \
    --method r4rr \
    --seed "$SEED" \
    --split test \
    --target-mode label \
    --mask-threshold 0 \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device cuda:0 \
    --output-dir "$output_dir" \
    2>&1 | tee "$output_dir/pointing_game.log"
done

python -u summarize_decoymnist_r4rr_epochwise_pointing.py \
  --run-dir "$RUN_DIR" \
  --seed "$SEED" \
  --epochs "$EPOCHS"

echo "[DONE] $RUN_DIR/epochwise_pointing_game.csv"
