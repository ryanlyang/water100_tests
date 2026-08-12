#!/usr/bin/env bash
#SBATCH --job-name=in9_clip_lr_sweep
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_clip_lr_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_clip_lr_sweep_%j.err

set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/manifest.csv}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUN_TAG="${RUN_TAG:-main}"
RUN_ROOT="${RUN_ROOT:-${LOG_ROOT}/sweeps/clip_lr/${RUN_TAG}}"
TARGET_COMPLETE_TRIALS="${TARGET_COMPLETE_TRIALS:-50}"
MAX_HOURS="${MAX_HOURS:-94}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"

mkdir -p "$LOG_ROOT" "$RUN_ROOT"
cd "$REPO"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gals_a100
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[$(date)] Host: $(hostname)"
echo "Method: OpenAI CLIP RN50 + logistic regression"
echo "Manifest: $MANIFEST"
echo "Run root: $RUN_ROOT"
echo "Target completed C trials: $TARGET_COMPLETE_TRIALS"
echo "Objective: Original validation macro class accuracy"
echo "Official variants used for tuning: NO"

srun --unbuffered "$PYTHON_BIN" -u ImageNet9_Runs/sweep_imagenet9_clip_lr.py \
  --manifest "$MANIFEST" \
  --feature-cache "$RUN_ROOT/openai_clip_rn50_features.npz" \
  --study-db "$RUN_ROOT/optuna.sqlite3" \
  --study-name "imagenet9_clip_lr_${RUN_TAG}" \
  --output-csv "$RUN_ROOT/trials.csv" \
  --summary-json "$RUN_ROOT/summary.json" \
  --target-complete-trials "$TARGET_COMPLETE_TRIALS" \
  --sweep-seed "${SWEEP_SEED:-0}" \
  --train-seed "${TRAIN_SEED:-0}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --c-min 1e-2 \
  --c-max 1e2 \
  --max-hours "$MAX_HOURS" \
  --device cuda:0

echo "[$(date)] ImageNet-9 CLIP-LR sweep invocation finished."
