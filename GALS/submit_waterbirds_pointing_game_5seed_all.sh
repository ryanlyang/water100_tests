#!/bin/bash
# Submit 14 independent jobs: seven trained methods x two Waterbirds datasets.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/run_waterbirds_pointing_game_5seed_method.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_cam}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
SPLIT="${SPLIT:-val}"
TARGET_MODE="${TARGET_MODE:-label}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
EXISTING_CHECKPOINT_CSV="${EXISTING_CHECKPOINT_CSV:-}"
MASK_PROTOCOL="${MASK_PROTOCOL:-legacy}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT"

# --export=ALL carries these values into each job. DATASET and METHOD are added
# per submission without putting the comma-separated seed list in --export.
export LOG_DIR RUN_ROOT SEEDS_CSV SPLIT TARGET_MODE MAX_SAMPLES SAMPLE_SEED
export EXISTING_CHECKPOINT_CSV MASK_PROTOCOL
export WB95_MASK_ROOT WB100_MASK_ROOT

METHODS=(vanilla elrep upweight abn gals afr r4rr)
DATASETS=(95 100)
JOB_FILE="$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
printf 'dataset,method,job_id\n' > "$JOB_FILE"

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    job_name="pg5_wb${dataset}_${method}"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY RUN] sbatch --job-name=$job_name --export=ALL,DATASET=$dataset,METHOD=$method $WORKER"
      job_id="DRY_RUN"
    else
      job_id="$(sbatch --parsable \
        --job-name="$job_name" \
        --export="ALL,DATASET=$dataset,METHOD=$method" \
        "$WORKER")"
      echo "[SUBMITTED] dataset=$dataset method=$method job=$job_id"
    fi
    printf '%s,%s,%s\n' "$dataset" "$method" "$job_id" >> "$JOB_FILE"
  done
done

echo
echo "[DONE] submission record: $JOB_FILE"
echo "[DONE] stable result root: $RUN_ROOT"
echo "[INFO] Each job resumes from valid training manifests and per-seed result CSVs."
echo "[INFO] After all jobs finish, combine the 14 summaries with:"
echo "  cd $SCRIPT_DIR"
echo "  python summarize_waterbirds_pointing_game_5seed.py --run-root $RUN_ROOT --seeds $SEEDS_CSV"
