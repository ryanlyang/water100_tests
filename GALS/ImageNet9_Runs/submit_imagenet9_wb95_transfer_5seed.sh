#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
RUNNER="$REPO/ImageNet9_Runs/run_imagenet9_wb95_transfer_5seed.sbatch"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
METHODS="${METHODS:-erm upweight abn elrep gals afr clip_lr r4rr}"
mkdir -p "$LOG_ROOT"
cd "$REPO"

record="$LOG_ROOT/submitted_imagenet9_wb95_transfer_$(date +%Y%m%d_%H%M%S).csv"
echo "method,job_name,job_id" > "$record"
read -r -a method_array <<< "$METHODS"

for method in "${method_array[@]}"; do
  case "$method" in
    erm|upweight|abn|elrep|gals|afr|clip_lr|r4rr) ;;
    *) echo "[ERROR] Unsupported method: $method" >&2; exit 2 ;;
  esac
  job_name="in9x95_${method}"
  if [[ "${FORCE_SUBMIT:-0}" != "1" ]] && \
     squeue -h -u "$USER" -n "$job_name" | grep -q .; then
    echo "$method,$job_name,ALREADY_QUEUED" >> "$record"
    echo "[SKIP] already queued: $job_name"
    continue
  fi
  output="$(sbatch --parsable \
    --job-name="$job_name" \
    --export="ALL,METHOD=${method}" \
    "$RUNNER")"
  job_id="${output%%;*}"
  echo "$method,$job_name,$job_id" >> "$record"
  echo "[SUBMITTED] method=$method job=$job_id"
done

echo "[DONE] submission record: $record"
echo "[INFO] Stable outputs: $LOG_ROOT/transfer/waterbirds95/<method>/main"
echo "[INFO] Re-running this command resumes completed seeds and evaluations."
