#!/bin/bash -l
# Vanilla MobileNetV2 sweep for RedMeat.
# Defaults: 50 Optuna trials, then rerun best hyperparams on seeds 0-4.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_vanilla_mobilenetv2_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_vanilla_mobilenetv2_sweep_%j.err
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

REPO_ROOT="$PROJECT_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
NUM_WORKERS=${NUM_WORKERS:-4}

BASE_LR_MIN=${BASE_LR_MIN:-1e-5}
BASE_LR_MAX=${BASE_LR_MAX:-5e-2}
CLS_LR_MIN=${CLS_LR_MIN:-1e-5}
CLS_LR_MAX=${CLS_LR_MAX:-5e-2}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
MOMENTUM=${MOMENTUM:-0.9}
MOMENTUM_MIN=${MOMENTUM_MIN:-0.85}
MOMENTUM_MAX=${MOMENTUM_MAX:-0.95}
NESTEROV=${NESTEROV:-0}

OUT_CSV=${OUT_CSV:-$LOG_DIR/redmeat_vanilla_mobilenetv2_sweep_${SLURM_JOB_ID}.csv}
POST_OUT_CSV=${POST_OUT_CSV:-$LOG_DIR/redmeat_vanilla_mobilenetv2_best5_${SLURM_JOB_ID}.csv}
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/Vanilla_MobileNetV2_RedMeat_Checkpoints}

export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"

cd "$GALS_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing DATASET_ROOT: $DATASET_ROOT" >&2
  exit 2
fi

python -c "import optuna" 2>/dev/null || { echo "[INFO] Installing optuna..."; pip install -q optuna; }

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "Backbone: mobilenet_v2 pretrained=1 tune_mode=full"
echo "Trials: $N_TRIALS (sampler=$SAMPLER sweep_seed=$SWEEP_SEED train_seed=$TRAIN_SEED)"
echo "Epochs: 150 workers=$NUM_WORKERS"
echo "Ranges: base_lr=[$BASE_LR_MIN,$BASE_LR_MAX] cls_lr=[$CLS_LR_MIN,$CLS_LR_MAX] momentum=[$MOMENTUM_MIN,$MOMENTUM_MAX]"
echo "Output CSV: $OUT_CSV"
echo "Post CSV: $POST_OUT_CSV"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
which python

ARGS=(
  "$DATASET_ROOT"
  --n-trials "$N_TRIALS"
  --seed "$SWEEP_SEED"
  --train-seed "$TRAIN_SEED"
  --sampler "$SAMPLER"
  --model mobilenet_v2
  --tune-mode full
  --pretrained
  --num-epochs 150
  --num-workers "$NUM_WORKERS"
  --base-lr-min "$BASE_LR_MIN" --base-lr-max "$BASE_LR_MAX"
  --cls-lr-min "$CLS_LR_MIN" --cls-lr-max "$CLS_LR_MAX"
  --weight-decay "$WEIGHT_DECAY"
  --momentum "$MOMENTUM"
  --momentum-min "$MOMENTUM_MIN" --momentum-max "$MOMENTUM_MAX"
  --output-csv "$OUT_CSV"
  --post-seeds "$POST_SEEDS"
  --post-seed-start "$POST_SEED_START"
  --post-output-csv "$POST_OUT_CSV"
  --checkpoint-dir "$CKPT_DIR"
)
if [[ "$NESTEROV" -eq 1 ]]; then
  ARGS+=(--nesterov)
fi

srun --unbuffered python -u RedMeat_Runs/run_vanilla_redmeat_sweep.py "${ARGS[@]}"

