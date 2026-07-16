#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

mkdir -p "$OUTPUT/run_logs" "$OUTPUT/control_scores"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi
if ! "$ENV/bin/python" -c "import torch, torchvision, timm, pandas"; then
  echo "Missing Step 8 dependencies in $ENV." >&2
  exit 1
fi

echo "Validating the complete Step 6 and Step 7 candidate pools..."
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/aggregate_token_banks.py >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/aggregate_fcv_scores.py >/dev/null

plan_job=$(sbatch --parsable \
  experiments/fcv_vit_waterbirds100/slurm/prepare_control_plan.sbatch)
score_job=$(sbatch --parsable --dependency="afterok:${plan_job}" \
  experiments/fcv_vit_waterbirds100/slurm/score_fcv_controls_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterany:${score_job}" \
  experiments/fcv_vit_waterbirds100/slurm/aggregate_fcv_controls.sbatch)

echo "Shared control-plan job: $plan_job"
echo "Control scoring array job: $score_job (27 tasks, 20 checkpoints per task)"
echo "Diagnostic aggregation job: $aggregate_job"
echo "Re-submission reuses four-control outputs with matching provenance."
