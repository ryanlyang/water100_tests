#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_abn_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_abn_sweep_%j.err
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
ABN_WEIGHTS_PATH="$REPO_ROOT/weights/resnet50_abn_imagenet.pth.tar"
if [[ ! -f "$ABN_WEIGHTS_PATH" ]]; then
  echo "[ERROR] Missing ABN pretrained weights: $ABN_WEIGHTS_PATH" >&2
  exit 2
fi

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
KEEP=${KEEP:-best}
MAX_HOURS=${MAX_HOURS:-}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
POST_KEEP=${POST_KEEP:-all}
BASE_LR_MIN=${BASE_LR_MIN:-5e-5}
BASE_LR_MAX=${BASE_LR_MAX:-1e-1}
CLS_LR_MIN=${CLS_LR_MIN:-5e-5}
CLS_LR_MAX=${CLS_LR_MAX:-1e-1}
ABN_CLS_WEIGHT_MIN=${ABN_CLS_WEIGHT_MIN:-1e-2}
ABN_CLS_WEIGHT_MAX=${ABN_CLS_WEIGHT_MAX:-1e2}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -c "import optuna" 2>/dev/null || { echo "[INFO] Installing optuna..."; pip install -q optuna; }

OUT_CSV="$LOG_DIR/redmeat_abn_sweep_${SLURM_JOB_ID}.csv"
TRIAL_LOGS="$LOG_DIR/redmeat_abn_sweep_logs_${SLURM_JOB_ID}"

ARGS=(--method abn_cls
  --config RedMeat_Runs/configs/redmeat_abn.yaml
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
  --abn-cls-weight-min "$ABN_CLS_WEIGHT_MIN"
  --abn-cls-weight-max "$ABN_CLS_WEIGHT_MAX"
  --post-seeds "$POST_SEEDS"
  --post-seed-start "$POST_SEED_START"
  --post-keep "$POST_KEEP"
)

if [[ -n "${MAX_HOURS:-}" ]]; then
  ARGS+=(--max-hours "$MAX_HOURS")
fi

srun --unbuffered python -u RedMeat_Runs/run_gals_sweep_redmeat.py "${ARGS[@]}"
