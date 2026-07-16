#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

mkdir -p \
  "$OUTPUT/run_logs" \
  "$OUTPUT/token_banks/cleanup_receipts" \
  "$OUTPUT/fcv_scores" \
  "$OUTPUT/control_scores" \
  "$OUTPUT/preflight"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi
if ! "$ENV/bin/python" -c "import torch, torchvision, timm, pandas"; then
  echo "Missing streaming FCV dependencies in $ENV." >&2
  exit 1
fi

echo "Strictly validating the 81-candidate pool and the 35 GiB launch guard..."
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/aggregate_candidate_metrics.py >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/validate_patch_masks.py >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/check_storage_budget.py \
  --stage submit_step6_8 >/dev/null

preflight_job=$(sbatch --parsable \
  experiments/fcv_vit_waterbirds100/slurm/verify_reconstruction_preflight.sbatch)
plan_job=$(sbatch --parsable --dependency="afterok:${preflight_job}" \
  experiments/fcv_vit_waterbirds100/slurm/prepare_streaming_plans.sbatch)
stream_job=$(sbatch --parsable --dependency="afterok:${plan_job}" \
  experiments/fcv_vit_waterbirds100/slurm/stream_score_run_array.sbatch)
diagnostic_fcv_job=$(sbatch --parsable --dependency="afterany:${stream_job}" \
  experiments/fcv_vit_waterbirds100/slurm/aggregate_fcv_scores.sbatch)
diagnostic_control_job=$(sbatch --parsable --dependency="afterany:${stream_job}" \
  experiments/fcv_vit_waterbirds100/slurm/aggregate_fcv_controls.sbatch)
strict_job=$(sbatch --parsable \
  --dependency="afterok:${stream_job}:${diagnostic_fcv_job}:${diagnostic_control_job}" \
  experiments/fcv_vit_waterbirds100/slurm/aggregate_streaming_scores.sbatch)

echo "Reconstruction preflight: $preflight_job"
echo "Shared-plan preparation: $plan_job"
echo "Streaming array: $stream_job (27 tasks x 3 candidates, at most 4 concurrent)"
echo "Diagnostic FCV aggregation: $diagnostic_fcv_job"
echo "Diagnostic control aggregation: $diagnostic_control_job"
echo "Strict 81-candidate aggregation: $strict_job"
echo "Each candidate's two token banks are deleted only after FCV + controls validate."
