#!/usr/bin/env bash
#SBATCH --job-name=in9_gals_abn
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_gals_abn_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_gals_abn_%j.err

set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/manifest.csv}"
MAP_ROOT="${MAP_ROOT:-${DATA_ROOT}/gals_maps/clip_vit_b32_transformer_v1}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUN_TAG="${RUN_TAG:-main}"
RUN_ROOT="${RUN_ROOT:-${LOG_ROOT}/sweeps/gals_abn/${RUN_TAG}}"
TARGET_COMPLETE_TRIALS="${TARGET_COMPLETE_TRIALS:-50}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-96}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
SWEEP_SEED="${SWEEP_SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-0}"
MAX_HOURS="${MAX_HOURS:-70}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"

ABN_CHECKPOINT="${ABN_CHECKPOINT:-}"
if [[ -z "$ABN_CHECKPOINT" ]]; then
  candidates=(
    "$REPO/weights/resnet50_abn_imagenet.pth.tar"
    "/home/ryreu/guided_cnn/waterbirds/old_Waterbird_Runs/GALS/weights/resnet50_abn_imagenet.pth.tar"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      ABN_CHECKPOINT="$candidate"
      break
    fi
  done
fi
if [[ ! -f "$ABN_CHECKPOINT" ]]; then
  echo "[ERROR] ABN checkpoint not found: ${ABN_CHECKPOINT:-NONE}" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$RUN_ROOT/trials"
cd "$REPO"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gals_a100
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DISABLED=true

echo "[$(date)] Host: $(hostname)"
echo "Method: GALS ABN (OpenAI CLIP ViT-B/32 maps)"
echo "Manifest: $MANIFEST"
echo "Map root: $MAP_ROOT"
echo "ABN checkpoint: $ABN_CHECKPOINT"
echo "Run root: $RUN_ROOT"
echo "Study: imagenet9_gals_abn_${RUN_TAG}"
echo "Target completed trials: $TARGET_COMPLETE_TRIALS"
echo "Epochs: $EPOCHS batch_size=$BATCH_SIZE workers=$NUM_WORKERS"
echo "Maximum driver runtime: $MAX_HOURS hours (Slurm walltime: 3 days)"
echo "Objective: Original validation macro class accuracy"
echo "Ranges: base_lr=[1e-3,1e-1] classifier_lr=[1e-4,1e-2] abn_att_weight=[1e-2,1e2] (log)"
echo "Fixed: ABN classification weight=1 criterion=L1 mode=suppress_outside"
echo "Official Backgrounds Challenge variants used for tuning: NO"
"$PYTHON_BIN" --version

srun --unbuffered "$PYTHON_BIN" -u ImageNet9_Runs/sweep_imagenet9_baseline.py \
  --method gals_abn \
  --manifest "$MANIFEST" \
  --gals-map-root "$MAP_ROOT" \
  --abn-checkpoint "$ABN_CHECKPOINT" \
  --study-db "$RUN_ROOT/optuna.sqlite3" \
  --study-name "imagenet9_gals_abn_${RUN_TAG}" \
  --output-csv "$RUN_ROOT/trials.csv" \
  --summary-json "$RUN_ROOT/summary.json" \
  --trial-logs "$RUN_ROOT/trials" \
  --target-complete-trials "$TARGET_COMPLETE_TRIALS" \
  --sweep-seed "$SWEEP_SEED" \
  --train-seed "$TRAIN_SEED" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --weight-decay 1e-5 \
  --fixed-momentum 0.9 \
  --max-hours "$MAX_HOURS" \
  --device cuda:0 \
  --python "$PYTHON_BIN"

echo "[$(date)] ImageNet-9 GALS ABN sweep invocation finished."
