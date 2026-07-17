#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
DATA=/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png
OUTPUT=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility
SLURM=experiments/fcv_vit_decoymnist/slurm

cd "$REPO"
mkdir -p "$OUTPUT/run_logs" "$OUTPUT/runs"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing Tigris FCV environment: $ENV" >&2
  exit 1
fi
if [[ ! -d "$DATA/train" || ! -d "$DATA/test" ]]; then
  echo "Missing DecoyMNIST train/test PNG folders under: $DATA" >&2
  exit 1
fi
"$ENV/bin/python" -c "import torch, torchvision, timm, pandas, numpy, PIL"

active=$(squeue --noheader --user "$USER" --format='%i %j %T' | awk '$2 ~ /^decoy_vit_(pre|sus|agg)$/ {print}')
if [[ -n "$active" && "${ALLOW_DUPLICATE_SUBMISSION:-0}" != "1" ]]; then
  echo "Existing DecoyMNIST susceptibility jobs are active:" >&2
  echo "$active" >&2
  exit 1
fi

preflight_raw=$(sbatch --parsable "$SLURM/preflight_susceptibility_pilot.sbatch")
preflight_job=${preflight_raw%%;*}
array_raw=$(sbatch --parsable --dependency="afterok:${preflight_job}" "$SLURM/run_susceptibility_pilot_array.sbatch")
array_job=${array_raw%%;*}
aggregate_raw=$(sbatch --parsable --dependency="afterok:${array_job}" "$SLURM/aggregate_susceptibility_pilot.sbatch")
aggregate_job=${aggregate_raw%%;*}

timestamp=$(date +%Y%m%d_%H%M%S)
record="$OUTPUT/run_logs/submission_${timestamp}.txt"
{
  printf 'submitted_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'protocol=unmodified DecoyMNIST ViT-S/16 susceptibility pilot\n'
  printf 'grid=3 learning rates x 3 seeds x 10 online epochs\n'
  printf 'diagnostics=biased validation and reversed test; original, digit-only, patch-only\n'
  printf 'checkpoint_retention=none\n'
  printf 'preflight_job=%s\n' "$preflight_job"
  printf 'pilot_array_job=%s\n' "$array_job"
  printf 'aggregate_job=%s\n' "$aggregate_job"
} > "$record"

cat "$record"
echo
echo "[DONE] DecoyMNIST ViT susceptibility pilot queued."
echo "Monitor: squeue --me -o '%.18i %.24j %.2t %.10M %R'"
echo "Final summary: $OUTPUT/pilot_summary.json"

