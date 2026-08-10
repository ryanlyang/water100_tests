#!/bin/bash
# Submit seven DecoyMNIST Pointing Game jobs, one per trained method.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/run_decoymnist_pointing_game_5seed_method.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_gradcam}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
SPLIT="${SPLIT:-test}"
TARGET_MODE="${TARGET_MODE:-label}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT"

export LOG_DIR RUN_ROOT SEEDS_CSV SPLIT TARGET_MODE MASK_THRESHOLD MAX_SAMPLES SAMPLE_SEED
export PNG_ROOT MNIST_ROOT GALS_MAPS R4RR_MAPS ENV_NAME PROJECT_ROOT GALS_ROOT

METHODS=(vanilla elrep upweight abn gals afr r4rr)
JOB_FILE="$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
printf 'dataset,method,job_id\n' > "$JOB_FILE"

for method in "${METHODS[@]}"; do
  job_name="pg5_decoy_${method}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] sbatch --job-name=$job_name --export=ALL,METHOD=$method $WORKER"
    job_id="DRY_RUN"
  else
    job_id="$(sbatch --parsable \
      --job-name="$job_name" \
      --export="ALL,METHOD=$method" \
      "$WORKER")"
    echo "[SUBMITTED] method=$method job=$job_id"
  fi
  printf 'decoymnist,%s,%s\n' "$method" "$job_id" >> "$JOB_FILE"
done

echo
echo "[DONE] submission record: $JOB_FILE"
echo "[DONE] stable result root: $RUN_ROOT"
echo "[INFO] Each method job resumes independently at the seed/evaluation level."
echo "[INFO] After all seven jobs finish:"
echo "  cd $SCRIPT_DIR"
echo "  python summarize_decoymnist_pointing_game_5seed.py --run-root $RUN_ROOT --seeds $SEEDS_CSV"
