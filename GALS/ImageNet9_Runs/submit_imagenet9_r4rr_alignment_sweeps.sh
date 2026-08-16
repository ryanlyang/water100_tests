#!/usr/bin/env bash
# Submit the four non-default ImageNet-9 R4RR alignment-loss sweeps.

set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUN_TAG="${RUN_TAG:-main}"
RUNNER="$REPO/ImageNet9_Runs/run_imagenet9_r4rr_alignment_sweep.sbatch"
LOSSES=(reverse_kl jensen_shannon squared_l2 cosine)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUBMISSION_CSV="$LOG_ROOT/submitted_imagenet9_r4rr_alignment_${TIMESTAMP}.csv"

mkdir -p "$LOG_ROOT"
cd "$REPO"

if [[ ! -f "$RUNNER" ]]; then
  echo "[ERROR] Missing runner: $RUNNER" >&2
  exit 1
fi

printf 'alignment_loss,job_id,run_root\n' > "$SUBMISSION_CSV"
for loss in "${LOSSES[@]}"; do
  case "$loss" in
    reverse_kl) short="revkl" ;;
    jensen_shannon) short="js" ;;
    squared_l2) short="l2" ;;
    cosine) short="cos" ;;
  esac
  job_id="$(
    sbatch --parsable \
      --job-name="in9r_${short}" \
      --export="ALL,ALIGNMENT_LOSS=${loss},RUN_TAG=${RUN_TAG},REPO=${REPO},LOG_ROOT=${LOG_ROOT}" \
      "$RUNNER"
  )"
  run_root="$LOG_ROOT/sweeps/r4rr_${loss}/${RUN_TAG}"
  printf '%s,%s,%s\n' "$loss" "$job_id" "$run_root" >> "$SUBMISSION_CSV"
  echo "[SUBMITTED] alignment_loss=$loss job=$job_id run_root=$run_root"
done

echo "[DONE] submission record: $SUBMISSION_CSV"
echo "[INFO] Resubmitting this command resumes each fixed-loss study to 50 completed trials."
