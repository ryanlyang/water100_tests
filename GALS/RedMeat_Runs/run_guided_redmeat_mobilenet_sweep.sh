#!/bin/bash -l
# Guided KL / R4RR MobileNetV3-Large sweep for RedMeat.
# Defaults: 50 Optuna trials, then rerun the best setting on seeds 0-4.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_mobilenet_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_mobilenet_sweep_%j.err
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

PRIMARY_GT_ROOT=${PRIMARY_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_dinovit/val/prediction_cmap/}
ALT_GT_ROOT_1=${ALT_GT_ROOT_1:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_dinovit/val/prediction_cmap/}
ALT_GT_ROOT_2=${ALT_GT_ROOT_2:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_xcit/val/prediction_cmap/}
ALT_GT_ROOT_3=${ALT_GT_ROOT_3:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_siglip2_dinovit/val/prediction_cmap/}

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
GUIDED_MODEL_NAME=${GUIDED_MODEL_NAME:-mobilenet_v3_large}
GUIDED_TUNE_MODE=${GUIDED_TUNE_MODE:-full}
GUIDED_CLIP_MODEL=${GUIDED_CLIP_MODEL:-RN50}
GUIDED_PRETRAINED=${GUIDED_PRETRAINED:-1}

SWEEP_OUT=${SWEEP_OUT:-$LOG_DIR/guided_redmeat_mobilenet_sweep_${SLURM_JOB_ID}.csv}
POST_OUT=${POST_OUT:-$LOG_DIR/guided_redmeat_mobilenet_best5_${SLURM_JOB_ID}.csv}

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}
export GUIDED_NUM_WORKERS=${GUIDED_NUM_WORKERS:-0}

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
echo "Trials: $N_TRIALS (sampler=$SAMPLER sweep_seed=$SWEEP_SEED)"
echo "Guided backbone: $GUIDED_MODEL_NAME tune_mode=$GUIDED_TUNE_MODE pretrained=$GUIDED_PRETRAINED"
echo "Output CSV: $SWEEP_OUT"
echo "Post seeds: $POST_SEEDS (start=$POST_SEED_START)"
echo "Post output CSV: $POST_OUT"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS GUIDED_NUM_WORKERS=$GUIDED_NUM_WORKERS"
which python

MODEL_ARGS=(--model-name "$GUIDED_MODEL_NAME" --tune-mode "$GUIDED_TUNE_MODE" --clip-model "$GUIDED_CLIP_MODEL")
if [[ "$GUIDED_PRETRAINED" -eq 1 ]]; then
  MODEL_ARGS+=(--pretrained)
else
  MODEL_ARGS+=(--no-pretrained)
fi

srun --unbuffered python -u RedMeat_Runs/run_guided_redmeat_sweep.py \
  "$DATASET_ROOT" \
  "$PRIMARY_GT_ROOT" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --sampler "$SAMPLER" \
  --num-epochs 150 \
  --output-csv "$SWEEP_OUT" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$POST_OUT" \
  "${MODEL_ARGS[@]}" \
  "${ALT_ARGS[@]}"
