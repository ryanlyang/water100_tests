#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_full_campaign
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
CONFIG=experiments/fcv_vit_decoymnist/configs/decoymnist_vit_s16_fcv_full_online.yaml
SLURM=experiments/fcv_vit_decoymnist/slurm
DATA=/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png
TEACHER=/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist_openclip/val/prediction_cmap

mkdir -p "$OUTPUT/run_logs" "$OUTPUT/preflight" "$OUTPUT/selection_results"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment: $ENV" >&2
  exit 1
fi
for required in "$DATA/train" "$DATA/test" "$TEACHER"; do
  if [[ ! -d "$required" ]]; then
    echo "Missing required campaign input: $required" >&2
    exit 1
  fi
done

"$ENV/bin/python" -c "import torch, torchvision, timm, pandas, scipy, yaml, PIL"
"$ENV/bin/python" experiments/fcv_vit_decoymnist/scripts/inspect_full_campaign_config.py >/dev/null
"$ENV/bin/python" -m unittest \
  experiments.fcv_vit_decoymnist.tests.test_decoy_full_config \
  experiments.fcv_vit_decoymnist.tests.test_decoy_online_schema \
  experiments.fcv_vit_decoymnist.tests.test_decoy_online_study \
  experiments.fcv_vit_decoymnist.tests.test_decoy_selection_analysis >/dev/null

active_jobs=$(squeue --noheader --user "$USER" --format='%i %j %T' | awk '$2 ~ /^dcyfcv_/ {print}')
if [[ -n "$active_jobs" && "${ALLOW_DUPLICATE_SUBMISSION:-0}" != "1" ]]; then
  echo "Existing DecoyMNIST FCV jobs are active; refusing a duplicate campaign:" >&2
  echo "$active_jobs" >&2
  exit 1
fi

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

preflight_job=$(submit_job "" "$SLURM/full_campaign_preflight.sbatch")
smoke_job=$(submit_job "afterok:${preflight_job}" "$SLURM/full_campaign_online_smoke.sbatch")
array_job=$(submit_job "afterok:${smoke_job}" "$SLURM/full_campaign_online_array.sbatch")
freeze_job=$(submit_job "afterok:${array_job}" "$SLURM/full_campaign_freeze.sbatch")
report_job=$(submit_job "afterok:${freeze_job}" "$SLURM/full_campaign_analyze.sbatch")

timestamp=$(date +%Y%m%d_%H%M%S)
record="$OUTPUT/run_logs/submission_${timestamp}.txt"
commit=$(git rev-parse HEAD 2>/dev/null || printf UNKNOWN)
{
  printf 'submitted_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$commit"
  printf 'protocol=108 training runs x 10 online epochs = 1080 candidates\n'
  printf 'checkpoint_persistence=none\n'
  printf 'preflight_job=%s\n' "$preflight_job"
  printf 'one_epoch_smoke_job=%s\n' "$smoke_job"
  printf 'online_array_job=%s\n' "$array_job"
  printf 'selection_freeze_job=%s\n' "$freeze_job"
  printf 'posthoc_report_job=%s\n' "$report_job"
} > "$record"

cat "$record"
echo
echo "[DONE] Full DecoyMNIST FCV campaign queued."
echo "Final report job: $report_job"
echo "Monitor: squeue --me -o '%.18i %.24j %.2t %.10M %R'"
