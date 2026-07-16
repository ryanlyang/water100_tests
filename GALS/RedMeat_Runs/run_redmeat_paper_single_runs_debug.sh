#!/bin/bash -l
# Debug launcher for paper-style single RedMeat runs:
# GALS (RN50 default), ABN, UpWeight, Vanilla, and CLIP+LR on food-101-redmeat.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/paper_single_runs_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/paper_single_runs_debug_%j.err
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

if [[ ! -d "${DATA_ROOT}/${DATA_DIR}" ]]; then
  echo "[ERROR] Missing dataset path: ${DATA_ROOT}/${DATA_DIR}" >&2
  exit 2
fi

JOB_TAG=${SLURM_JOB_ID:-manual}
RUN_PREFIX=${RUN_PREFIX:-paper_ref_redmeat_debug}
RUN_LOG_DIR=${RUN_LOG_DIR:-$LOG_DIR/paper_single_runs_${JOB_TAG}}
OUT_CSV=${OUT_CSV:-$LOG_DIR/paper_single_runs_summary_${JOB_TAG}.csv}

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

cd "$GALS_ROOT"
export PYTHONPATH="$GALS_ROOT:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $GALS_ROOT"
echo "Data root: $DATA_ROOT"
echo "Dataset: ${DATA_ROOT}/${DATA_DIR}"
echo "Logs dir: $RUN_LOG_DIR"
echo "Output CSV: $OUT_CSV"
echo "CLIP model/device: $CLIP_MODEL / $CLIP_DEVICE"
echo "AUX_LOSSES_ON_VAL: $AUX_LOSSES_ON_VAL"
echo "GALS_MODE: $GALS_MODE"
which python

python -c "import sklearn" >/dev/null 2>&1 || {
  echo "[INFO] Installing scikit-learn..."
  pip install -q scikit-learn
}

ARGS=(
  --data-root "$DATA_ROOT"
  --dataset-dir "$DATA_DIR"
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

srun --unbuffered python -u RedMeat_Runs/run_redmeat_paper_single_runs.py "${ARGS[@]}"

echo "[DONE] RedMeat paper single runs complete."
echo "Summary CSV: $OUT_CSV"
