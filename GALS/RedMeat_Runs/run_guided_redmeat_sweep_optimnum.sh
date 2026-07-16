#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_sweep_optimnum_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_sweep_optimnum_%j.err
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

PRIMARY_GT_ROOT=${PRIMARY_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_dinovit/val/prediction_cmap/}
ALT_GT_ROOT_1=${ALT_GT_ROOT_1:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_xcit/val/prediction_cmap/}
ALT_GT_ROOT_2=${ALT_GT_ROOT_2:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_dinovit/val/prediction_cmap/}
ALT_GT_ROOT_3=${ALT_GT_ROOT_3:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_siglip2_dinovit/val/prediction_cmap/}

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
OPTIM_BETA=${OPTIM_BETA:-1}

SWEEP_OUT=${SWEEP_OUT:-$LOG_DIR/guided_redmeat_sweep_optimnum_${SLURM_JOB_ID}.csv}
POST_OUT=${POST_OUT:-$LOG_DIR/guided_redmeat_sweep_optimnum_best5_${SLURM_JOB_ID}.csv}

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}

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

ALT_ARGS=()
for p in "$ALT_GT_ROOT_1" "$ALT_GT_ROOT_2" "$ALT_GT_ROOT_3"; do
  if [[ -n "$p" && -d "$p" ]]; then
    ALT_ARGS+=(--alt-gt-path "$p")
  else
    echo "[INFO] Skipping missing alt GT root: $p"
  fi
done

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "Primary GT masks: $PRIMARY_GT_ROOT"
echo "Trials: $N_TRIALS"
echo "Output CSV: $SWEEP_OUT"
echo "Post seeds: $POST_SEEDS (start=$POST_SEED_START)"
echo "Post output CSV: $POST_OUT"
which python

srun --unbuffered python -u RedMeat_Runs/run_guided_redmeat_sweep_optimnum.py \
  "$DATASET_ROOT" \
  "$PRIMARY_GT_ROOT" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --sampler "$SAMPLER" \
  --num-epochs 150 \
  --optim-beta "$OPTIM_BETA" \
  --output-csv "$SWEEP_OUT" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$POST_OUT" \
  "${ALT_ARGS[@]}"
