#!/bin/bash -l
# Guided RedMeat sweep with:
# - 16 CPUs per task
# - attention_epoch locked to 0
# - lr2_mult locked to 1.0
# Everything else follows the standard guided RedMeat sweep defaults.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_sweep_attn0_lr2fixed_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_sweep_attn0_lr2fixed_%j.err
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

# Cluster defaults are populated by common_env.sh, but allow easy local/RunPod
# overrides before activation.
if [[ -d /workspace/Waterbird_Runs/GALS ]]; then
  export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/Waterbird_Runs}"
  export GALS_ROOT="${GALS_ROOT:-/workspace/Waterbird_Runs/GALS}"
fi
if [[ -d /workspace/data/food-101-redmeat ]]; then
  export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
  export DATA_DIR="${DATA_DIR:-food-101-redmeat}"
fi
export LOG_DIR="${LOG_DIR:-/workspace/logsRedMeat}"
mkdir -p "$LOG_DIR"

if [[ "${SKIP_ENV_ACTIVATE:-0}" == "1" ]]; then
  echo "[INFO] SKIP_ENV_ACTIVATE=1 -> using current Python environment."
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "[INFO] Using active virtualenv: $VIRTUAL_ENV"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  redmeat_activate_env
else
  echo "[WARN] No conda found; using current Python environment."
fi

REPO_ROOT="$PROJECT_ROOT"
GALS_REPO="$GALS_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"

# Keep the same defaults as the standard guided sweep.
PRIMARY_GT_ROOT=${PRIMARY_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_dinovit/val/prediction_cmap/}
ALT_GT_ROOT_1=${ALT_GT_ROOT_1:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_dinovit/val/prediction_cmap/}
ALT_GT_ROOT_2=${ALT_GT_ROOT_2:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_xcit/val/prediction_cmap/}
ALT_GT_ROOT_3=${ALT_GT_ROOT_3:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_siglip2_dinovit/val/prediction_cmap/}

N_TRIALS=${N_TRIALS:-150}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
GUIDED_MODEL_NAME=${GUIDED_MODEL_NAME:-resnet50}
GUIDED_CLIP_MODEL=${GUIDED_CLIP_MODEL:-RN50}
GUIDED_PRETRAINED=${GUIDED_PRETRAINED:-1}

# Locked hyperparameters requested.
ATTN_MIN=${ATTN_MIN:-0}
ATTN_MAX=${ATTN_MAX:-0}
LR2_MULT_MIN=${LR2_MULT_MIN:-1.0}
LR2_MULT_MAX=${LR2_MULT_MAX:-1.0}

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
SWEEP_OUT=${SWEEP_OUT:-$LOG_DIR/guided_redmeat_sweep_attn0_lr2fixed_${RUN_ID}.csv}
POST_OUT=${POST_OUT:-$LOG_DIR/guided_redmeat_sweep_attn0_lr2fixed_best5_${RUN_ID}.csv}

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}
CPU_COUNT="${SLURM_CPUS_PER_TASK:-${CPU_COUNT:-16}}"
export OMP_NUM_THREADS="$CPU_COUNT"
export MKL_NUM_THREADS="$CPU_COUNT"
export NUMEXPR_NUM_THREADS="$CPU_COUNT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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
echo "Guided backbone: $GUIDED_MODEL_NAME (clip_model=$GUIDED_CLIP_MODEL pretrained=$GUIDED_PRETRAINED)"
echo "Locked ranges: attention_epoch=[$ATTN_MIN,$ATTN_MAX], lr2_mult=[$LR2_MULT_MIN,$LR2_MULT_MAX]"
echo "Output CSV: $SWEEP_OUT"
echo "Post seeds: $POST_SEEDS (start=$POST_SEED_START)"
echo "Post output CSV: $POST_OUT"
which python

MODEL_ARGS=(--model-name "$GUIDED_MODEL_NAME" --clip-model "$GUIDED_CLIP_MODEL")
if [[ "$GUIDED_PRETRAINED" -eq 1 ]]; then
  MODEL_ARGS+=(--pretrained)
else
  MODEL_ARGS+=(--no-pretrained)
fi

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  RUNNER=(srun --unbuffered python -u)
else
  RUNNER=(python -u)
fi

"${RUNNER[@]}" RedMeat_Runs/run_guided_redmeat_sweep.py \
  "$DATASET_ROOT" \
  "$PRIMARY_GT_ROOT" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --sampler "$SAMPLER" \
  --num-epochs 150 \
  --attn-min "$ATTN_MIN" \
  --attn-max "$ATTN_MAX" \
  --lr2-mult-min "$LR2_MULT_MIN" \
  --lr2-mult-max "$LR2_MULT_MAX" \
  --output-csv "$SWEEP_OUT" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$POST_OUT" \
  "${MODEL_ARGS[@]}" \
  "${ALT_ARGS[@]}"
