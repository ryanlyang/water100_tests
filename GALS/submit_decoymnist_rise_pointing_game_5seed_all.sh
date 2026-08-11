#!/bin/bash
# Submit evaluation-only DecoyMNIST RISE Pointing Game jobs.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/run_decoymnist_rise_pointing_game_5seed_method.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
CHECKPOINT_RUN_ROOT="${CHECKPOINT_RUN_ROOT:-$LOG_DIR/pointing_game_5seed_gradcam}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
METHODS_CSV="${METHODS_CSV:-vanilla,elrep,upweight,abn,gals,afr,r4rr}"
SPLIT="${SPLIT:-test}"
TARGET_MODE="${TARGET_MODE:-label}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
RISE_IMAGE_BATCH_SIZE="${RISE_IMAGE_BATCH_SIZE:-16}"
RISE_MAX_MASKED_BATCH="${RISE_MAX_MASKED_BATCH:-8192}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-$RUN_ROOT/rise_masks/decoymnist_gals_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_28x28.npy}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT"

export LOG_DIR CHECKPOINT_RUN_ROOT RUN_ROOT SEEDS_CSV SPLIT TARGET_MODE
export MASK_THRESHOLD MAX_SAMPLES SAMPLE_SEED RISE_NUM_MASKS RISE_GRID_SIZE
export RISE_P1 RISE_SEED RISE_IMAGE_BATCH_SIZE RISE_MAX_MASKED_BATCH RISE_MASKS_PATH
export PNG_ROOT MNIST_ROOT ENV_NAME PROJECT_ROOT GALS_ROOT

IFS=',' read -r -a METHODS <<< "$METHODS_CSV"
JOB_FILE="$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
printf 'dataset,explainer,method,job_id\n' > "$JOB_FILE"

for method_raw in "${METHODS[@]}"; do
  method="${method_raw//[[:space:]]/}"
  [[ -n "$method" ]] || continue
  case "$method" in
    vanilla|elrep|upweight|abn|gals|afr|r4rr) ;;
    *) echo "[ERROR] Unsupported method in METHODS_CSV: $method" >&2; exit 2 ;;
  esac
  job_name="pgr5_decoy_${method}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] sbatch --job-name=$job_name --export=ALL,METHOD=$method $WORKER"
    job_id="DRY_RUN"
  else
    job_id="$(sbatch --parsable \
      --job-name="$job_name" \
      --export="ALL,METHOD=$method" \
      "$WORKER")"
    echo "[SUBMITTED] method=$method explainer=RISE job=$job_id"
  fi
  printf 'decoymnist,rise,%s,%s\n' "$method" "$job_id" >> "$JOB_FILE"
done

echo
echo "[DONE] submission record: $JOB_FILE"
echo "[DONE] result root: $RUN_ROOT"
echo "[INFO] Existing checkpoints are read from: $CHECKPOINT_RUN_ROOT"
echo "[INFO] No training is performed by these jobs."
echo "[INFO] After all selected jobs finish:"
echo "  cd $SCRIPT_DIR"
echo "  python summarize_decoymnist_rise_pointing_game_5seed.py --run-root $RUN_ROOT --seeds $SEEDS_CSV"
