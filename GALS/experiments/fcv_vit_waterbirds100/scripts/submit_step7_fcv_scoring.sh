#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

mkdir -p "$OUTPUT/run_logs" "$OUTPUT/fcv_scores"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi
if ! "$ENV/bin/python" -c "import torch, torchvision, timm, pandas"; then
  echo "Missing Step 7 dependencies in $ENV." >&2
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

echo "Validating all 540 candidates and 1,080 Step 6 banks..."
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/aggregate_candidate_metrics.py >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/aggregate_token_banks.py >/dev/null

plan_job=$(sbatch --parsable \
  experiments/fcv_vit_waterbirds100/slurm/prepare_opposite_donor_plan.sbatch)
score_job=$(sbatch --parsable --dependency="afterok:${plan_job}" \
  experiments/fcv_vit_waterbirds100/slurm/score_fcv_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterany:${score_job}" \
  experiments/fcv_vit_waterbirds100/slurm/aggregate_fcv_scores.sbatch)

echo "Shared donor-plan job: $plan_job"
echo "FCV scoring array job: $score_job (27 tasks, 20 checkpoints per task)"
echo "Diagnostic aggregation job: $aggregate_job"
echo "Re-submission reuses scores whose checkpoint/bank/plan provenance still matches."
