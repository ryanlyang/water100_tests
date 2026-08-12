#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-smoke}"
REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
RUNNER="$REPO/ImageNet9_Runs/run_imagenet9_baseline_sweep.sbatch"
CLIP_LR_RUNNER="$REPO/ImageNet9_Runs/run_imagenet9_clip_lr_sweep.sbatch"
AFR_RUNNER="$REPO/ImageNet9_Runs/run_imagenet9_afr.sbatch"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
mkdir -p "$LOG_ROOT"

case "$MODE" in
  smoke)
    RUN_TAG="${RUN_TAG:-smoke_$(date +%Y%m%d_%H%M%S)}"
    SBATCH_OPTIONS=(--partition=debug --time=04:00:00)
    TARGET=1
    EPOCHS_VALUE=1
    MAX_HOURS_VALUE=3.5
    ;;
  sweep)
    RUN_TAG="${RUN_TAG:-main}"
    SBATCH_OPTIONS=(--partition=tier3 --time=4-00:00:00)
    TARGET=50
    EPOCHS_VALUE=20
    MAX_HOURS_VALUE=94
    ;;
  *)
    echo "Usage: $0 [smoke|sweep]" >&2
    exit 2
    ;;
esac

record="$LOG_ROOT/submitted_imagenet9_non_teacher_${MODE}_$(date +%Y%m%d_%H%M%S).csv"
echo "method,mode,run_tag,job_id" > "$record"

for method in erm upweight abn elrep; do
  job_name="in9_${method}_${MODE}"
  if [[ "${FORCE_SUBMIT:-0}" != "1" ]] && \
     squeue -h -u "$USER" -n "$job_name" | grep -q .; then
    echo "$method,$MODE,$RUN_TAG,ALREADY_QUEUED" >> "$record"
    echo "[SKIP] already queued: $job_name (set FORCE_SUBMIT=1 to override)"
    continue
  fi
  output="$(sbatch --parsable \
    "${SBATCH_OPTIONS[@]}" \
    --job-name="$job_name" \
    --export="ALL,METHOD=${method},RUN_TAG=${RUN_TAG},TARGET_COMPLETE_TRIALS=${TARGET},EPOCHS=${EPOCHS_VALUE},MAX_HOURS=${MAX_HOURS_VALUE}" \
    "$RUNNER")"
  job_id="${output%%;*}"
  echo "$method,$MODE,$RUN_TAG,$job_id" >> "$record"
  echo "[SUBMITTED] method=$method mode=$MODE run_tag=$RUN_TAG job=$job_id"
done

job_name="in9_clip_lr_${MODE}"
if [[ "${FORCE_SUBMIT:-0}" != "1" ]] && \
   squeue -h -u "$USER" -n "$job_name" | grep -q .; then
  echo "clip_lr,$MODE,$RUN_TAG,ALREADY_QUEUED" >> "$record"
  echo "[SKIP] already queued: $job_name (set FORCE_SUBMIT=1 to override)"
else
  output="$(sbatch --parsable \
    "${SBATCH_OPTIONS[@]}" \
    --job-name="$job_name" \
    --export="ALL,RUN_TAG=${RUN_TAG},TARGET_COMPLETE_TRIALS=${TARGET},MAX_HOURS=${MAX_HOURS_VALUE}" \
    "$CLIP_LR_RUNNER")"
  job_id="${output%%;*}"
  echo "clip_lr,$MODE,$RUN_TAG,$job_id" >> "$record"
  echo "[SUBMITTED] method=clip_lr mode=$MODE run_tag=$RUN_TAG job=$job_id"
fi

job_name="in9_afr_${MODE}"
if [[ "${FORCE_SUBMIT:-0}" != "1" ]] && \
   squeue -h -u "$USER" -n "$job_name" | grep -q .; then
  echo "afr,$MODE,$RUN_TAG,ALREADY_QUEUED" >> "$record"
  echo "[SKIP] already queued: $job_name (set FORCE_SUBMIT=1 to override)"
else
  AFR_EXPORTS="ALL,RUN_TAG=${RUN_TAG},MAX_HOURS=${MAX_HOURS_VALUE}"
  if [[ "$MODE" == "smoke" ]]; then
    AFR_EXPORTS="${AFR_EXPORTS},AFR_SMOKE=1,STAGE1_EPOCHS=1,STAGE2_EPOCHS=5"
  fi
  output="$(sbatch --parsable \
    "${SBATCH_OPTIONS[@]}" \
    --job-name="$job_name" \
    --export="$AFR_EXPORTS" \
    "$AFR_RUNNER")"
  job_id="${output%%;*}"
  echo "afr,$MODE,$RUN_TAG,$job_id" >> "$record"
  echo "[SUBMITTED] method=afr mode=$MODE run_tag=$RUN_TAG job=$job_id"
fi

echo "[DONE] submission record: $record"
if [[ "$MODE" == "sweep" ]]; then
  echo "[RESUME] Re-run '$0 sweep'; stable RUN_TAG=main studies continue to 50 completed trials."
fi
