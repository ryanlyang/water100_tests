#!/bin/bash -l
# Guided RedMeat sweep using CLIP-RN50 initialized backbone.
# Fixed to OpenAI+DINOvIT WeCLIP masks as primary GT root.
#
# Usage:
#   sbatch RedMeat_Runs/run_guided_redmeat_cliprn50_openai_dinovit_sweep_100.sh
#
# Optional overrides:
#   sbatch --export=ALL,SWEEP_SEED=1,POST_SEEDS=0 \
#     RedMeat_Runs/run_guided_redmeat_cliprn50_openai_dinovit_sweep_100.sh

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_cliprn50_openai_dinovit_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_cliprn50_openai_dinovit_sweep_%j.err
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

# Requested setup.
PRIMARY_GT_ROOT=${PRIMARY_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_dinovit/val/prediction_cmap/}
N_TRIALS=${N_TRIALS:-100}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
NUM_EPOCHS=${NUM_EPOCHS:-150}

SWEEP_OUT=${SWEEP_OUT:-$LOG_DIR/guided_redmeat_cliprn50_openai_dinovit_sweep_${SLURM_JOB_ID}.csv}
POST_OUT=${POST_OUT:-$LOG_DIR/guided_redmeat_cliprn50_openai_dinovit_best5_${SLURM_JOB_ID}.csv}
POST_SUMMARY_OUT=${POST_SUMMARY_OUT:-$LOG_DIR/guided_redmeat_cliprn50_openai_dinovit_best5_${SLURM_JOB_ID}_summary.csv}

# Force CLIP-backbone guided model.
GUIDED_MODEL_NAME=clip_rn50
GUIDED_CLIP_MODEL=RN50
GUIDED_PRETRAINED=1

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$GALS_REPO"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing DATASET_ROOT: $DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -d "$PRIMARY_GT_ROOT" ]]; then
  echo "[ERROR] Missing PRIMARY_GT_ROOT: $PRIMARY_GT_ROOT" >&2
  exit 2
fi

python -c "import optuna" 2>/dev/null || { echo "[INFO] Installing optuna..."; pip install -q optuna; }

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "Primary GT masks: $PRIMARY_GT_ROOT"
echo "Trials: $N_TRIALS"
echo "Sampler: $SAMPLER (seed=$SWEEP_SEED)"
echo "Backbone: $GUIDED_MODEL_NAME ($GUIDED_CLIP_MODEL, pretrained=$GUIDED_PRETRAINED)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-1}"
echo "Output CSV: $SWEEP_OUT"
echo "Post seeds: $POST_SEEDS (start=$POST_SEED_START)"
echo "Post output CSV: $POST_OUT"
echo "Post summary CSV: $POST_SUMMARY_OUT"
which python

srun --unbuffered python -u RedMeat_Runs/run_guided_redmeat_sweep.py \
  "$DATASET_ROOT" \
  "$PRIMARY_GT_ROOT" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --sampler "$SAMPLER" \
  --num-epochs "$NUM_EPOCHS" \
  --output-csv "$SWEEP_OUT" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$POST_OUT" \
  --post-summary-csv "$POST_SUMMARY_OUT" \
  --model-name "$GUIDED_MODEL_NAME" \
  --clip-model "$GUIDED_CLIP_MODEL" \
  --pretrained
