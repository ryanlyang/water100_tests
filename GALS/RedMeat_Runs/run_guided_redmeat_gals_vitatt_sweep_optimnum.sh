#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_galsvit_sweep_optimnum_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_galsvit_sweep_optimnum_%j.err
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
redmeat_prepare_food_layout "$DATA_ROOT" "$DATA_DIR"

REPO_ROOT="$GALS_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"
ATT_ROOT=${ATT_ROOT:-$DATASET_ROOT/clip_vit_attention}

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
OPTIM_BETA=${OPTIM_BETA:-1}

SWEEP_OUT=${SWEEP_OUT:-$LOG_DIR/guided_redmeat_galsvit_sweep_optimnum_${SLURM_JOB_ID}.csv}
POST_OUT=${POST_OUT:-$LOG_DIR/guided_redmeat_galsvit_sweep_optimnum_best5_${SLURM_JOB_ID}.csv}

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}

cd "$REPO_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing DATASET_ROOT: $DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -d "$ATT_ROOT" ]]; then
  echo "[ERROR] Missing ATT_ROOT: $ATT_ROOT" >&2
  echo "Run: sbatch RedMeat_Runs/run_generate_redmeat_vit_attentions.sh" >&2
  exit 2
fi

python -c "import optuna" 2>/dev/null || { echo "[INFO] Installing optuna..."; pip install -q optuna; }

echo "[$(date)] Host: $(hostname)"
echo "Repo: $PROJECT_ROOT"
echo "Data: $DATASET_ROOT"
echo "GALS ViT attentions: $ATT_ROOT"
echo "Trials: $N_TRIALS"
echo "Output CSV: $SWEEP_OUT"
echo "Post seeds: $POST_SEEDS (start=$POST_SEED_START)"
echo "Post output CSV: $POST_OUT"
which python

srun --unbuffered python -u RedMeat_Runs/run_guided_redmeat_gals_vitatt_sweep_optimnum.py \
  "$DATASET_ROOT" \
  "$ATT_ROOT" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --sampler "$SAMPLER" \
  --num-epochs 150 \
  --optim-beta "$OPTIM_BETA" \
  --output-csv "$SWEEP_OUT" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$POST_OUT"
