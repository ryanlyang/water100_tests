#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
CONFIG=experiments/fcv_vit_waterbirds100/configs/waterbirds100_vit_s16_first_study.yaml
SLURM=experiments/fcv_vit_waterbirds100/slurm

mkdir -p \
  "$OUTPUT/run_logs" \
  "$OUTPUT/online_runs" \
  "$OUTPUT/online_scores/fcv" \
  "$OUTPUT/online_scores/controls" \
  "$OUTPUT/online_scores/oracle" \
  "$OUTPUT/online_test_analysis_only" \
  "$OUTPUT/selection_results" \
  "$OUTPUT/preflight"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment: $ENV" >&2
  exit 1
fi
"$ENV/bin/python" -c "import torch, torchvision, timm, pandas, scipy, yaml, matplotlib, pyparsing"

for required in \
  "$OUTPUT/split_manifests/metadata_train.csv" \
  "$OUTPUT/split_manifests/metadata_val.csv" \
  "$OUTPUT/split_manifests/metadata_oracle_val_analysis_only.csv" \
  "$OUTPUT/split_manifests/metadata_test_analysis_only.csv" \
  "$OUTPUT/patch_masks/patch_masks_val.pt"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required frozen input: $required" >&2
    exit 1
  fi
done

active_jobs=$(squeue --noheader --user "$USER" --format='%i %j %T' | awk '$2 ~ /^fcv_/ {print}')
if [[ -n "$active_jobs" && "${ALLOW_DUPLICATE_SUBMISSION:-0}" != "1" ]]; then
  echo "Existing FCV jobs are active; refusing a duplicate campaign:" >&2
  echo "$active_jobs" >&2
  exit 1
fi

"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/inspect_config.py \
  --config "$CONFIG" --check-paths >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/validate_patch_masks.py \
  --config "$CONFIG" >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/check_storage_budget.py \
  --config "$CONFIG" --stage=submit_full_online_540_study >/dev/null
MPLBACKEND=Agg "$ENV/bin/python" \
  experiments/fcv_vit_waterbirds100/scripts/preflight_plotting.py \
  --config "$CONFIG" >/dev/null
"$ENV/bin/python" -m unittest \
  experiments.fcv_vit_waterbirds100.tests.test_online_study \
  experiments.fcv_vit_waterbirds100.tests.test_storage >/dev/null

submit_job() {
  local dependency=$1
  local script=$2
  local raw
  if [[ -n "$dependency" ]]; then
    raw=$(sbatch --parsable --dependency="$dependency" "$script")
  else
    raw=$(sbatch --parsable "$script")
  fi
  printf '%s\n' "${raw%%;*}"
}

cache_job=$(submit_job "" "$SLURM/cache_pretrained_model.sbatch")
smoke_job=$(submit_job "afterok:${cache_job}" "$SLURM/online_epoch_study_smoke.sbatch")
online_job=$(submit_job "afterok:${smoke_job}" "$SLURM/online_epoch_study_array.sbatch")
freeze_job=$(submit_job "afterok:${online_job}" "$SLURM/freeze_online_validation_selection.sbatch")
analysis_job=$(submit_job "afterok:${freeze_job}" "$SLURM/analyze_online_test_results.sbatch")

timestamp=$(date +%Y%m%d_%H%M%S)
record="$OUTPUT/run_logs/online540_submission_${timestamp}.txt"
commit=$(git rev-parse HEAD 2>/dev/null || printf UNKNOWN)
{
  printf 'submitted_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$commit"
  printf 'protocol=540 online candidates (27 runs x every epoch 1..20)\n'
  printf 'retention=max 3 unique primary-selector winners per run\n'
  printf 'post_freeze_retention=max 3 unique global primary-selector winners total\n'
  printf 'token_banks=node-local and deleted after every epoch\n'
  printf 'cache_job=%s\n' "$cache_job"
  printf 'real_online_interruption_resume_smoke_job=%s\n' "$smoke_job"
  printf 'smoke_gate=campaign-bound, restart-reusable, full-storage-and-runtime-projected\n'
  printf 'online_array_job=%s\n' "$online_job"
  printf 'freeze_validation_job=%s\n' "$freeze_job"
  printf 'posthoc_test_analysis_job=%s\n' "$analysis_job"
} > "$record"

cat "$record"
echo
echo "[DONE] Full online 540-candidate study queued."
echo "Final job: $analysis_job"
echo "Monitor: squeue --me -o '%.18i %.24j %.2t %.10M %R'"
