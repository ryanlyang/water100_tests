#!/bin/bash
# Submit one five-seed DecoyMNIST transfer job per WB100 alignment-loss ablation.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_decoymnist_r4rr_alignment_transfer.sh"
LOSSES=(reverse_kl jensen_shannon squared_l2 cosine)
RUN_ROOT=${RUN_ROOT:-/home/ryreu/guided_cnn/logsMNIST/r4rr_alignment_transfer_best5}
SUBMISSION_CSV=${SUBMISSION_CSV:-$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv}

[[ -f "$RUNNER" ]] || { echo "[ERROR] Missing runner: $RUNNER" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
printf 'alignment_loss,job_id,output_csv,summary_csv\n' > "$SUBMISSION_CSV"

for loss in "${LOSSES[@]}"; do
  short="$loss"
  [[ "$loss" == "jensen_shannon" ]] && short="js"
  [[ "$loss" == "squared_l2" ]] && short="l2"
  output_csv="$RUN_ROOT/decoy_${loss}_wb100_transfer_best5.csv"
  summary_csv="$RUN_ROOT/decoy_${loss}_wb100_transfer_best5_summary.csv"
  job_id=$(sbatch --parsable \
    --job-name="d5_${short}" \
    --export="ALL,ALIGNMENT_LOSS=${loss},RUN_ROOT=${RUN_ROOT},OUTPUT_CSV=${output_csv},SUMMARY_CSV=${summary_csv}" \
    "$RUNNER")
  echo "[SUBMITTED] loss=$loss job=$job_id"
  printf '%s,%s,%s,%s\n' "$loss" "$job_id" "$output_csv" "$summary_csv" >> "$SUBMISSION_CSV"
done

echo "[DONE] Submission record: $SUBMISSION_CSV"
echo "[DONE] Stable result root: $RUN_ROOT"
echo "[INFO] Resubmission skips completed seeds for the same transferred WB100 trial."
