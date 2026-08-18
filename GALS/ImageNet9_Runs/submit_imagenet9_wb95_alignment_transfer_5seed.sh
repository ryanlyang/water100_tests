#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
RUNNER="$REPO/ImageNet9_Runs/run_imagenet9_wb95_alignment_transfer_5seed.sbatch"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
LOSSES=(reverse_kl jensen_shannon squared_l2 cosine)
mkdir -p "$LOG_ROOT"
cd "$REPO"

record="$LOG_ROOT/submitted_imagenet9_wb95_alignment_transfer_$(date +%Y%m%d_%H%M%S).csv"
echo "alignment_loss,job_name,job_id" > "$record"
for loss in "${LOSSES[@]}"; do
  case "$loss" in
    reverse_kl) short="rev" ;;
    jensen_shannon) short="js" ;;
    squared_l2) short="l2" ;;
    cosine) short="cos" ;;
  esac
  job_name="in9x95_${short}"
  if [[ "${FORCE_SUBMIT:-0}" != "1" ]] && \
     squeue -h -u "$USER" -n "$job_name" | grep -q .; then
    echo "$loss,$job_name,ALREADY_QUEUED" >> "$record"
    echo "[SKIP] already queued: $job_name"
    continue
  fi
  output="$(sbatch --parsable \
    --job-name="$job_name" \
    --export="ALL,ALIGNMENT_LOSS=${loss}" \
    "$RUNNER")"
  job_id="${output%%;*}"
  echo "$loss,$job_name,$job_id" >> "$record"
  echo "[SUBMITTED] alignment_loss=$loss job=$job_id"
done

echo "[DONE] submission record: $record"
echo "[INFO] Stable outputs: $LOG_ROOT/transfer/waterbirds95_alignment/<loss>/main"
echo "[INFO] Re-running this command resumes completed seeds and evaluations."
