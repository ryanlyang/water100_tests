#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

mkdir -p "$OUTPUT/run_logs" "$OUTPUT/candidate_models"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi
if ! "$ENV/bin/python" -c "import torch, torchvision, timm"; then
  echo "Missing Step 4 dependencies in $ENV." >&2
  echo "Activate the environment and install timm: python -m pip install 'timm==1.0.28'" >&2
  exit 1
fi
if [[ ! -f "$OUTPUT/split_manifests/metadata_train.csv" || ! -f "$OUTPUT/split_manifests/metadata_val.csv" ]]; then
  echo "Missing Step 2 manifests under $OUTPUT/split_manifests." >&2
  echo "Run experiments/fcv_vit_waterbirds100/scripts/prepare_metadata.py first." >&2
  exit 1
fi
if [[ ! -f "$OUTPUT/patch_masks/patch_masks_val.pt" ]]; then
  echo "Missing Step 3 artifact: $OUTPUT/patch_masks/patch_masks_val.pt" >&2
  echo "Run experiments/fcv_vit_waterbirds100/scripts/prepare_patch_masks.py first." >&2
  exit 1
fi
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/validate_patch_masks.py >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/check_storage_budget.py \
  --stage submit_step4 >/dev/null

cache_job=$(sbatch --parsable \
  experiments/fcv_vit_waterbirds100/slurm/cache_pretrained_model.sbatch)
array_job=$(sbatch --parsable --dependency="afterok:${cache_job}" \
  experiments/fcv_vit_waterbirds100/slurm/train_candidate_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterany:${array_job}" \
  experiments/fcv_vit_waterbirds100/slurm/aggregate_candidate_metrics.sbatch)
finalize_job=$(sbatch --parsable --dependency="afterok:${array_job}:${aggregate_job}" \
  experiments/fcv_vit_waterbirds100/slurm/finalize_candidate_pool.sbatch)

echo "Pretrained-model cache job: $cache_job"
echo "Candidate array job: $array_job (starts after cache validation)"
echo "Diagnostic aggregation job: $aggregate_job (runs after the array exits)"
echo "Strict finalization/cleanup job: $finalize_job"
echo "All runs train 20 epochs, but only epochs 5, 10, and 20 form 81 candidates."
echo "The finalizer deletes optimizer-bearing resume states only after strict validation."
