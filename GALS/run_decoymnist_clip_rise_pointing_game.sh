#!/bin/bash -l
# Evaluate deterministic CLIP-ZS or CLIP-LR on DecoyMNIST with RISE.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/%x_%j.err

set -Eeuo pipefail

METHOD="${METHOD:?Submit with METHOD=clip_zs|clip_lr}"
case "$METHOD" in clip_zs|clip_lr) ;; *) exit 2 ;; esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
METHOD_DIR="$RUN_ROOT/$METHOD"
PNG_ROOT="${PNG_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"
MNIST_ROOT="${MNIST_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data}"
CLIP_MODEL="${CLIP_MODEL:-RN50}"
CLIP_C="${CLIP_C:-0.2515000498909345}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CLIP_FEATURE_BATCH_SIZE="${CLIP_FEATURE_BATCH_SIZE:-128}"
RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-$RUN_ROOT/rise_masks/decoymnist_gals_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_28x28.npy}"
PG_DIR="$METHOD_DIR/seed_0/pointing_game"

mkdir -p "$LOG_DIR" "$PG_DIR" "$(dirname "$RISE_MASKS_PATH")"
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$GALS_ROOT:${PYTHONPATH:-}"
cd "$GALS_ROOT"

python -u decoymnist_clip_rise_pointing_game_eval.py \
  --png-root "$PNG_ROOT" \
  --mnist-root "$MNIST_ROOT" \
  --method "$METHOD" \
  --seed 0 \
  --clip-model "$CLIP_MODEL" \
  --clip-c "$CLIP_C" \
  --clip-feature-batch-size "$CLIP_FEATURE_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --rise-num-masks "$RISE_NUM_MASKS" \
  --rise-grid-size "$RISE_GRID_SIZE" \
  --rise-p1 "$RISE_P1" \
  --rise-seed "$RISE_SEED" \
  --rise-masks-path "$RISE_MASKS_PATH" \
  --device cuda:0 \
  --output-dir "$PG_DIR" \
  2>&1 | tee "$METHOD_DIR/seed_0/pointing_game_rise.log"

python -u summarize_decoymnist_rise_pointing_game_5seed.py \
  --method-dir "$METHOD_DIR" --seeds 0
