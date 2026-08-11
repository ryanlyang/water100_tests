#!/bin/bash
# Submit one independent five-day, 50-trial RedMeat job per alternative loss.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_redmeat_r4rr_alignment_sweep.sh"
LOSSES=(reverse_kl jensen_shannon squared_l2 cosine)
SUBMISSION_CSV=${SUBMISSION_CSV:-/home/ryreu/guided_cnn/logsRedMeat/redmeat_r4rr_alignment_submissions_$(date +%Y%m%d_%H%M%S).csv}

if [[ ! -f "$RUNNER" ]]; then
  echo "[ERROR] Missing runner: $RUNNER" >&2
  exit 1
fi

mkdir -p "$(dirname "$SUBMISSION_CSV")"
printf 'alignment_loss,job_id\n' > "$SUBMISSION_CSV"

for loss in "${LOSSES[@]}"; do
  short="$loss"
  [[ "$loss" == "jensen_shannon" ]] && short="js"
  [[ "$loss" == "squared_l2" ]] && short="l2"
  job_id=$(sbatch --parsable \
    --job-name="r4rrmeat_${short}" \
    --export="ALL,ALIGNMENT_LOSS=${loss}" \
    "$RUNNER")
  echo "[SUBMITTED] loss=$loss job=$job_id"
  printf '%s,%s\n' "$loss" "$job_id" >> "$SUBMISSION_CSV"
done

echo "[DONE] Submission record: $SUBMISSION_CSV"
