#!/bin/bash -l
# Debug launcher for paper-style single Waterbirds runs:
# GALS_ViT, ABN, UpWeight, Vanilla, and CLIP+LR on WB95 + WB100.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/paper_single_runs_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/paper_single_runs_debug_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS"

resolve_repo_root() {
  local cand

  if [[ -n "${REPO_ROOT:-}" ]]; then
    cand="$REPO_ROOT"
    if [[ -f "$cand/run_waterbirds_paper_single_runs.py" ]]; then
      echo "$cand"
      return 0
    fi
  fi

  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cand="$SLURM_SUBMIT_DIR/GALS"
    if [[ -f "$cand/run_waterbirds_paper_single_runs.py" ]]; then
      echo "$cand"
      return 0
    fi

    cand="$SLURM_SUBMIT_DIR"
    if [[ -f "$cand/run_waterbirds_paper_single_runs.py" ]]; then
      echo "$cand"
      return 0
    fi
  fi

  cand="$SCRIPT_DIR"
  if [[ -f "$cand/run_waterbirds_paper_single_runs.py" ]]; then
    echo "$cand"
    return 0
  fi

  cand="$DEFAULT_REPO_ROOT"
  if [[ -f "$cand/run_waterbirds_paper_single_runs.py" ]]; then
    echo "$cand"
    return 0
  fi

  return 1
}

REPO_ROOT="$(resolve_repo_root || true)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "[ERROR] Could not locate repo root containing run_waterbirds_paper_single_runs.py." >&2
  echo "Checked REPO_ROOT, SLURM_SUBMIT_DIR/GALS, SLURM_SUBMIT_DIR, SCRIPT_DIR, and $DEFAULT_REPO_ROOT" >&2
  exit 2
fi

DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds}
WB95_DIR=${WB95_DIR:-waterbird_complete95_forest2water2}
WB100_DIR=${WB100_DIR:-waterbird_1.0_forest2water2}
WB95_PATH="$DATA_ROOT/$WB95_DIR"
WB100_PATH="$DATA_ROOT/$WB100_DIR"

CLIP_MODEL=${CLIP_MODEL:-RN50}
CLIP_DEVICE=${CLIP_DEVICE:-cuda}
CLIP_BATCH_SIZE=${CLIP_BATCH_SIZE:-256}
CLIP_NUM_WORKERS=${CLIP_NUM_WORKERS:-4}
CLIP_SEED=${CLIP_SEED:-0}
CLIP_C=${CLIP_C:-1.0}
CLIP_PENALTY=${CLIP_PENALTY:-l2}
CLIP_SOLVER=${CLIP_SOLVER:-lbfgs}
CLIP_MAX_ITER=${CLIP_MAX_ITER:-5000}
CLIP_FIT_INTERCEPT=${CLIP_FIT_INTERCEPT:-1}
AUX_LOSSES_ON_VAL=${AUX_LOSSES_ON_VAL:-0}
GALS_MODE=${GALS_MODE:-rn50}

JOB_TAG=${SLURM_JOB_ID:-manual}
RUN_PREFIX=${RUN_PREFIX:-paper_ref_debug}
RUN_LOG_DIR=${RUN_LOG_DIR:-$LOG_DIR/paper_single_runs_${JOB_TAG}}
OUT_CSV=${OUT_CSV:-$LOG_DIR/paper_single_runs_summary_${JOB_TAG}.csv}

if [[ ! -d "$WB95_PATH" ]]; then
  echo "[ERROR] Missing WB95 dataset path: $WB95_PATH" >&2
  exit 2
fi
if [[ ! -d "$WB100_PATH" ]]; then
  echo "[ERROR] Missing WB100 dataset path: $WB100_PATH" >&2
  exit 2
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data root: $DATA_ROOT"
echo "WB95: $WB95_PATH"
echo "WB100: $WB100_PATH"
echo "Logs dir: $RUN_LOG_DIR"
echo "Output CSV: $OUT_CSV"
echo "CLIP model/device: $CLIP_MODEL / $CLIP_DEVICE"
echo "AUX_LOSSES_ON_VAL: $AUX_LOSSES_ON_VAL"
echo "GALS_MODE: $GALS_MODE"
which python

ARGS=(
  --data-root "$DATA_ROOT"
  --wb95-dir "$WB95_DIR"
  --wb100-dir "$WB100_DIR"
  --logs-dir "$RUN_LOG_DIR"
  --output-csv "$OUT_CSV"
  --run-prefix "${RUN_PREFIX}_${JOB_TAG}"
  --clip-model "$CLIP_MODEL"
  --clip-device "$CLIP_DEVICE"
  --clip-batch-size "$CLIP_BATCH_SIZE"
  --clip-num-workers "$CLIP_NUM_WORKERS"
  --clip-seed "$CLIP_SEED"
  --clip-C "$CLIP_C"
  --clip-penalty "$CLIP_PENALTY"
  --clip-solver "$CLIP_SOLVER"
  --clip-max-iter "$CLIP_MAX_ITER"
  --gals-mode "$GALS_MODE"
)

if [[ "$CLIP_FIT_INTERCEPT" -eq 1 ]]; then
  ARGS+=(--clip-fit-intercept)
else
  ARGS+=(--clip-no-fit-intercept)
fi

if [[ "$AUX_LOSSES_ON_VAL" -eq 1 ]]; then
  ARGS+=(--aux-losses-on-val)
else
  ARGS+=(--no-aux-losses-on-val)
fi

srun --unbuffered python -u run_waterbirds_paper_single_runs.py "${ARGS[@]}"

echo "[DONE] Paper single runs complete."
echo "Summary CSV: $OUT_CSV"
