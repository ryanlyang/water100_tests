#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_abn_vit_sweep_optimnum_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_abn_vit_sweep_optimnum_%j.err
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
ATT_DIR=${ATT_DIR:-clip_vit_attention}

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
KEEP=${KEEP:-best}
MAX_HOURS=${MAX_HOURS:-}
TUNE_WEIGHT_DECAY=${TUNE_WEIGHT_DECAY:-0}
BASE_LR_MIN=${BASE_LR_MIN:-1e-3}
BASE_LR_MAX=${BASE_LR_MAX:-1e-1}
CLS_LR_MIN=${CLS_LR_MIN:-1e-4}
CLS_LR_MAX=${CLS_LR_MAX:-1e-2}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
POST_KEEP=${POST_KEEP:-all}
ABN_ATT_WEIGHT_MIN=${ABN_ATT_WEIGHT_MIN:-1e-2}
ABN_ATT_WEIGHT_MAX=${ABN_ATT_WEIGHT_MAX:-1e2}
OPTIM_BETA=${OPTIM_BETA:-1}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$DATASET_ROOT/$ATT_DIR" ]]; then
  echo "[ERROR] Missing ViT attention maps at: $DATASET_ROOT/$ATT_DIR" >&2
  exit 2
fi

python -c "import optuna" 2>/dev/null || { echo "[INFO] Installing optuna..."; pip install -q optuna; }

OUT_CSV="$LOG_DIR/redmeat_abn_vit_sweep_optimnum_${SLURM_JOB_ID}.csv"
TRIAL_LOGS="$LOG_DIR/redmeat_abn_vit_sweep_optimnum_logs_${SLURM_JOB_ID}"

ARGS=(--method abn_att
  --config RedMeat_Runs/configs/redmeat_abn_vit.yaml
  --data-root "$DATA_ROOT"
  --dataset-dir "$DATA_DIR"
  --n-trials "$N_TRIALS"
  --seed "$SWEEP_SEED"
  --train-seed "$TRAIN_SEED"
  --sampler "$SAMPLER"
  --keep "$KEEP"
  --output-csv "$OUT_CSV"
  --logs-dir "$TRIAL_LOGS"
  --base-lr-min "$BASE_LR_MIN"
  --base-lr-max "$BASE_LR_MAX"
  --cls-lr-min "$CLS_LR_MIN"
  --cls-lr-max "$CLS_LR_MAX"
  --abn-att-weight-min "$ABN_ATT_WEIGHT_MIN"
  --abn-att-weight-max "$ABN_ATT_WEIGHT_MAX"
  --optim-beta "$OPTIM_BETA"
  --post-seeds "$POST_SEEDS"
  --post-seed-start "$POST_SEED_START"
  --post-keep "$POST_KEEP"
)

if [[ "$TUNE_WEIGHT_DECAY" -eq 1 ]]; then
  ARGS+=(--tune-weight-decay)
fi
if [[ -n "${MAX_HOURS:-}" ]]; then
  ARGS+=(--max-hours "$MAX_HOURS")
fi

srun --unbuffered python -u RedMeat_Runs/run_gals_sweep_redmeat_optimnum.py "${ARGS[@]}"
