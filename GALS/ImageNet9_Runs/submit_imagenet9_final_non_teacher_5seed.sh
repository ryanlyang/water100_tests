#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
RUNNER="$REPO/ImageNet9_Runs/run_imagenet9_final_5seed_method.sbatch"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
METHODS="${METHODS:-erm upweight abn elrep afr clip_lr}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
mkdir -p "$LOG_ROOT"
cd "$REPO"

read -r -a method_array <<< "$METHODS"
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  "$PYTHON_BIN" ImageNet9_Runs/check_imagenet9_final_sources.py \
    --log-root "$LOG_ROOT" \
    --methods "${method_array[@]}"
fi

record="$LOG_ROOT/submitted_imagenet9_final_5seed_$(date +%Y%m%d_%H%M%S).csv"
echo "method,job_name,job_id" > "$record"

for method in "${method_array[@]}"; do
  job_name="in9f_${method}"
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
echo "[INFO] Stable outputs: $LOG_ROOT/final/<method>/main"
echo "[INFO] Resubmitting this script is safe; completed seeds and evaluations are reused."
