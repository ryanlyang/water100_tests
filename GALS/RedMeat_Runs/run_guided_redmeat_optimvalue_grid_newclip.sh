#!/bin/bash -l
# Random guided RedMeat runs + saliency/loss optim_value grid correlation search.
#
# Default:
# - 30 random trials
# - GT_NEWCLIP as mask root
# - 5 saliency methods x 4 losses = 20 strategies
#
# Usage:
#   sbatch RedMeat_Runs/run_guided_redmeat_optimvalue_grid_newclip.sh
# Optional overrides:
#   sbatch --export=ALL,N_TRIALS=20,OPTIM_BETA=1.0 RedMeat_Runs/run_guided_redmeat_optimvalue_grid_newclip.sh

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_optim_grid_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_optim_grid_%j.err
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
GT_NEWCLIP_ROOT=${GT_NEWCLIP_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_dinovit/val/prediction_cmap/}

N_TRIALS=${N_TRIALS:-30}
SWEEP_SEED=${SWEEP_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
TRAIN_SEED_MODE=${TRAIN_SEED_MODE:-fixed}
OPTIM_BETA=${OPTIM_BETA:-1.0}
IG_STEPS=${IG_STEPS:-16}
SMOOTHGRAD_SAMPLES=${SMOOTHGRAD_SAMPLES:-8}
SMOOTHGRAD_NOISE_STD=${SMOOTHGRAD_NOISE_STD:-0.10}
NUM_EPOCHS=${NUM_EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-96}
NUM_WORKERS=${NUM_WORKERS:-4}

OUT_DIR=${OUT_DIR:-$LOG_DIR/guided_redmeat_optim_grid_${SLURM_JOB_ID}}

cd "$GALS_REPO"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing DATASET_ROOT: $DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -d "$GT_NEWCLIP_ROOT" ]]; then
  echo "[ERROR] Missing GT_NEWCLIP_ROOT: $GT_NEWCLIP_ROOT" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "GT_NEWCLIP: $GT_NEWCLIP_ROOT"
echo "Trials: $N_TRIALS"
echo "Output dir: $OUT_DIR"
which python

srun --unbuffered python -u RedMeat_Runs/run_guided_redmeat_optimvalue_grid.py \
  --data-path "$DATASET_ROOT" \
  --gt-path "$GT_NEWCLIP_ROOT" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --train-seed "$TRAIN_SEED" \
  --train-seed-mode "$TRAIN_SEED_MODE" \
  --num-epochs "$NUM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --optim-beta "$OPTIM_BETA" \
  --ig-steps "$IG_STEPS" \
  --smoothgrad-samples "$SMOOTHGRAD_SAMPLES" \
  --smoothgrad-noise-std "$SMOOTHGRAD_NOISE_STD" \
  --output-dir "$OUT_DIR" \
  --continue-on-error
