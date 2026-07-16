#!/bin/bash -l
# Optuna sweep: CLIP-RN50 LR warm-start + 3-stage guided finetuning on RedMeat.
#
# Default run:
#   sbatch RedMeat_Runs/run_guided_redmeat_cliprn50_lrwarm_3stage_sweep.sh

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_cliprn50_lrwarm_3stage_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_cliprn50_lrwarm_3stage_%j.err
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
PRIMARY_GT_ROOT=${PRIMARY_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_dinovit/val/prediction_cmap/}

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
BATCH_SIZE=${BATCH_SIZE:-96}
NUM_WORKERS=${NUM_WORKERS:-4}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}

# Fixed CLIP+LR hyperparameters (best from your sweep).
LR_C=${LR_C:-5.302446323656201}
LR_PENALTY=${LR_PENALTY:-l2}
LR_SOLVER=${LR_SOLVER:-lbfgs}
LR_FIT_INTERCEPT=${LR_FIT_INTERCEPT:-0}
LR_MAX_ITER=${LR_MAX_ITER:-5000}

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
SWEEP_OUT=${SWEEP_OUT:-$LOG_DIR/guided_redmeat_cliprn50_lrwarm_3stage_sweep_${RUN_ID}.csv}
POST_OUT=${POST_OUT:-$LOG_DIR/guided_redmeat_cliprn50_lrwarm_3stage_best${POST_SEEDS}_${RUN_ID}.csv}

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

python -c "import optuna, sklearn" 2>/dev/null || {
  echo "[INFO] Installing missing python deps (optuna/sklearn)..."
  pip install -q optuna scikit-learn
}

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "GT masks: $PRIMARY_GT_ROOT"
echo "Trials: $N_TRIALS (sampler=$SAMPLER seed=$SWEEP_SEED)"
echo "Batch size: $BATCH_SIZE | num_workers: $NUM_WORKERS"
echo "LR warm-start params: C=$LR_C penalty=$LR_PENALTY solver=$LR_SOLVER fit_intercept=$LR_FIT_INTERCEPT"
echo "Sweep CSV: $SWEEP_OUT"
echo "Post CSV: $POST_OUT"
which python

ARGS=(
  "$DATASET_ROOT"
  "$PRIMARY_GT_ROOT"
  --n-trials "$N_TRIALS"
  --sampler "$SAMPLER"
  --seed "$SWEEP_SEED"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --clip-model RN50
  --lr-C "$LR_C"
  --lr-penalty "$LR_PENALTY"
  --lr-solver "$LR_SOLVER"
  --lr-max-iter "$LR_MAX_ITER"
  --output-csv "$SWEEP_OUT"
)

if [[ "$LR_FIT_INTERCEPT" -eq 1 ]]; then
  ARGS+=(--lr-fit-intercept)
fi

if [[ "$POST_SEEDS" -gt 0 ]]; then
  ARGS+=(--post-seeds "$POST_SEEDS" --post-seed-start "$POST_SEED_START" --post-output-csv "$POST_OUT")
fi

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
  RUNNER=(srun --unbuffered python -u)
else
  RUNNER=(python -u)
fi

"${RUNNER[@]}" RedMeat_Runs/run_guided_redmeat_cliprn50_lrwarm_3stage_sweep.py "${ARGS[@]}"
