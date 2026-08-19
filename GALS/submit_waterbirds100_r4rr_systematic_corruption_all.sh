#!/bin/bash
# Submit the four Waterbirds-100 systematic and count-matched conditions.

set -Eeuo pipefail

# Avoid Slurm user-environment retrieval failures. Each worker activates Conda.
unset SBATCH_GET_USER_ENV SBATCH_EXPORT SBATCH_EXPORT_FILE
export SLURM_EXPORT_ENV=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/run_waterbirds100_r4rr_systematic_corruption_condition.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/r4rr_round2_systematic_teacher_corruption/waterbirds100}"
CORRUPTION_SEED="${CORRUPTION_SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"

CONDITIONS=(
  class_landbird
  random_matched_landbird
  class_waterbird
  random_matched_waterbird
)

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT"

submission_file="$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
printf 'dataset,condition,status,job_id\n' > "$submission_file"

condition_is_complete() {
  local condition="$1"
  local summary="$RUN_ROOT/$condition/summary.json"
  python3 - "$summary" "$condition" "$CORRUPTION_SEED" <<'PY'
import json
import sys

path, condition, corruption_seed = sys.argv[1:]
try:
    summary = json.load(open(path, "r", encoding="utf-8"))
    valid = (
        summary.get("protocol_version") == 1
        and summary.get("dataset") == "waterbirds100"
        and summary.get("condition") == condition
        and int(summary.get("corruption_seed", -1)) == int(corruption_seed)
        and summary.get("completed_seeds") == [0, 1, 2, 3, 4]
        and int(summary.get("n_completed", -1)) == 5
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

job_name_for_condition() {
  case "$1" in
    class_landbird) printf 'r4c_w100_land' ;;
    random_matched_landbird) printf 'r4c_w100_rland' ;;
    class_waterbird) printf 'r4c_w100_water' ;;
    random_matched_waterbird) printf 'r4c_w100_rwater' ;;
  esac
}

for condition in "${CONDITIONS[@]}"; do
  job_name="$(job_name_for_condition "$condition")"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] sbatch --job-name=$job_name --export=ALL,CONDITION=$condition $WORKER"
    status="DRY_RUN"
    job_id="DRY_RUN"
  elif condition_is_complete "$condition"; then
    echo "[SKIP-COMPLETE] condition=$condition"
    status="COMPLETE"
    job_id=""
  else
    queued_job_ids="$(squeue -h -u "$USER" -n "$job_name" -o '%A' | paste -sd ';' -)"
    if [[ -n "$queued_job_ids" ]]; then
      echo "[SKIP-QUEUED] condition=$condition jobs=$queued_job_ids"
      status="QUEUED"
      job_id="$queued_job_ids"
    else
      job_id="$(sbatch --parsable \
        --job-name="$job_name" \
        --export="ALL,CONDITION=$condition" \
        "$WORKER")"
      echo "[SUBMITTED] condition=$condition job=$job_id"
      status="SUBMITTED"
    fi
  fi
  printf 'waterbirds100,%s,%s,%s\n' "$condition" "$status" "$job_id" >> "$submission_file"
done

echo
echo "[DONE] submission record: $submission_file"
echo "[DONE] stable result root: $RUN_ROOT"
echo "[INFO] Each condition job runs seeds 0-4 and resumes completed seeds."
echo "[INFO] Aggregate all completed conditions with:"
echo "  cd $SCRIPT_DIR"
echo "  python summarize_waterbirds100_r4rr_systematic_corruption.py --run-root $RUN_ROOT"

