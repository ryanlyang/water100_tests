#!/bin/bash
# Submit 3 datasets x 4 alternative alignment losses as 12 independent jobs.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_r4rr_alignment_best5.sh"
LOSSES=(reverse_kl jensen_shannon squared_l2 cosine)
DATASETS=(wb95 wb100 redmeat)
RUN_ROOT=${RUN_ROOT:-/home/ryreu/guided_cnn/logsWaterbird/r4rr_alignment_best5}
SUBMISSION_CSV=${SUBMISSION_CSV:-$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv}

[[ -f "$RUNNER" ]] || { echo "[ERROR] Missing runner: $RUNNER" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
printf 'dataset,alignment_loss,job_id,output_csv,summary_csv\n' > "$SUBMISSION_CSV"

for dataset in "${DATASETS[@]}"; do
  for loss in "${LOSSES[@]}"; do
    short="$loss"
    [[ "$loss" == "jensen_shannon" ]] && short="js"
    [[ "$loss" == "squared_l2" ]] && short="l2"
    output_csv="$RUN_ROOT/${dataset}_${loss}_best5.csv"
    summary_csv="$RUN_ROOT/${dataset}_${loss}_best5_summary.csv"
    job_id=$(sbatch --parsable \
      --job-name="a5_${dataset}_${short}" \
      --export="ALL,DATASET=${dataset},ALIGNMENT_LOSS=${loss},RUN_ROOT=${RUN_ROOT},OUTPUT_CSV=${output_csv},SUMMARY_CSV=${summary_csv}" \
      "$RUNNER")
    echo "[SUBMITTED] dataset=$dataset loss=$loss job=$job_id"
    printf '%s,%s,%s,%s,%s\n' "$dataset" "$loss" "$job_id" "$output_csv" "$summary_csv" >> "$SUBMISSION_CSV"
  done
done

echo "[DONE] Submission record: $SUBMISSION_CSV"
echo "[DONE] Stable result root: $RUN_ROOT"
echo "[INFO] Resubmission skips completed seeds in matching per-seed CSVs."
