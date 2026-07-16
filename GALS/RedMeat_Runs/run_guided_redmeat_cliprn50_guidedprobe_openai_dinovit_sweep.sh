#!/bin/bash -l
# Guided RedMeat CLIP-RN50 sweep with CLIP-safe ranges and probe-style tune modes.
#
# Recommended first run:
#   sbatch RedMeat_Runs/run_guided_redmeat_cliprn50_guidedprobe_openai_dinovit_sweep.sh
#
# Optional overrides:
#   sbatch --export=ALL,TUNE_MODE=linear_probe,N_TRIALS=80 \
#     RedMeat_Runs/run_guided_redmeat_cliprn50_guidedprobe_openai_dinovit_sweep.sh

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_cliprn50_guidedprobe_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_cliprn50_guidedprobe_sweep_%j.err
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

N_TRIALS=${N_TRIALS:-100}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
NUM_EPOCHS=${NUM_EPOCHS:-150}

# CLIP-guided defaults
GUIDED_MODEL_NAME=${GUIDED_MODEL_NAME:-clip_rn50}
GUIDED_CLIP_MODEL=${GUIDED_CLIP_MODEL:-RN50}
GUIDED_PRETRAINED=${GUIDED_PRETRAINED:-1}
TUNE_MODE=${TUNE_MODE:-layer4_head}

# Use the same default sweep ranges as regular guided redmeat.
ATTN_MIN=${ATTN_MIN:-0}
ATTN_MAX=${ATTN_MAX:-$((NUM_EPOCHS - 1))}
KL_MIN=${KL_MIN:-1.0}
KL_MAX=${KL_MAX:-500.0}
BASE_LR_MIN=${BASE_LR_MIN:-1e-5}
BASE_LR_MAX=${BASE_LR_MAX:-5e-2}
CLS_LR_MIN=${CLS_LR_MIN:-1e-5}
CLS_LR_MAX=${CLS_LR_MAX:-5e-2}
LR2_MULT_MIN=${LR2_MULT_MIN:-0.1}
LR2_MULT_MAX=${LR2_MULT_MAX:-3.0}

SWEEP_OUT=${SWEEP_OUT:-$LOG_DIR/guided_redmeat_cliprn50_guidedprobe_sweep_${SLURM_JOB_ID}.csv}
POST_OUT=${POST_OUT:-$LOG_DIR/guided_redmeat_cliprn50_guidedprobe_best5_${SLURM_JOB_ID}.csv}
POST_SUMMARY_OUT=${POST_SUMMARY_OUT:-$LOG_DIR/guided_redmeat_cliprn50_guidedprobe_best5_${SLURM_JOB_ID}_summary.csv}

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
echo "Trials: $N_TRIALS (sampler=$SAMPLER seed=$SWEEP_SEED)"
echo "Backbone: $GUIDED_MODEL_NAME ($GUIDED_CLIP_MODEL pretrained=$GUIDED_PRETRAINED)"
echo "Tune mode: $TUNE_MODE"
echo "Ranges: attn=[$ATTN_MIN,$ATTN_MAX] kl=[$KL_MIN,$KL_MAX] base_lr=[$BASE_LR_MIN,$BASE_LR_MAX] cls_lr=[$CLS_LR_MIN,$CLS_LR_MAX] lr2_mult=[$LR2_MULT_MIN,$LR2_MULT_MAX]"
echo "Output CSV: $SWEEP_OUT"
echo "Post output CSV: $POST_OUT"
which python

MODEL_ARGS=(--model-name "$GUIDED_MODEL_NAME" --clip-model "$GUIDED_CLIP_MODEL" --tune-mode "$TUNE_MODE")
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
  --num-epochs "$NUM_EPOCHS" \
  --attn-min "$ATTN_MIN" \
  --attn-max "$ATTN_MAX" \
  --kl-min "$KL_MIN" \
  --kl-max "$KL_MAX" \
  --base-lr-min "$BASE_LR_MIN" \
  --base-lr-max "$BASE_LR_MAX" \
  --cls-lr-min "$CLS_LR_MIN" \
  --cls-lr-max "$CLS_LR_MAX" \
  --lr2-mult-min "$LR2_MULT_MIN" \
  --lr2-mult-max "$LR2_MULT_MAX" \
  --output-csv "$SWEEP_OUT" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$POST_OUT" \
  --post-summary-csv "$POST_SUMMARY_OUT" \
  "${MODEL_ARGS[@]}"
