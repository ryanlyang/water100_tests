#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

mkdir -p "$OUTPUT/run_logs" "$OUTPUT/token_banks" "$OUTPUT/preflight"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi
if ! "$ENV/bin/python" -c "import torch, torchvision, timm"; then
  echo "Missing Step 6 dependencies in $ENV." >&2
  exit 1
fi
if [[ ! -f "$OUTPUT/split_manifests/metadata_val.csv" ]]; then
  echo "Missing Step 2 public validation manifest." >&2
  exit 1
fi
if [[ ! -f "$OUTPUT/patch_masks/patch_masks_val.pt" ]]; then
  echo "Missing Step 3 patch-mask artifact." >&2
  exit 1
fi

echo "Validating successful Step 3 status, mask bytes, preprocessing, and current maps..."
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/validate_patch_masks.py >/dev/null

echo "Validating the complete 540-candidate Step 4 pool..."
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/aggregate_candidate_metrics.py >/dev/null

preflight_job=$(sbatch --parsable \
  experiments/fcv_vit_waterbirds100/slurm/verify_reconstruction_preflight.sbatch)
array_job=$(sbatch --parsable --dependency="afterok:${preflight_job}" \
  experiments/fcv_vit_waterbirds100/slurm/build_token_bank_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterany:${array_job}" \
  experiments/fcv_vit_waterbirds100/slurm/aggregate_token_banks.sbatch)

echo "GH200 reconstruction preflight: $preflight_job"
echo "Token-bank array job: $array_job (27 tasks, 20 checkpoints per task)"
echo "Diagnostic aggregation job: $aggregate_job"
echo "Re-submission reuses candidate banks whose completed summaries still match."
