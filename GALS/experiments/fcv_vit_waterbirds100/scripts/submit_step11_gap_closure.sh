#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

mkdir -p \
  "$OUTPUT/run_logs" \
  "$OUTPUT/selection_results/candidate_pool_test_scores"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi
if ! "$ENV/bin/python" -c "import torch, torchvision, timm, pandas"; then
  echo "Missing Step 11 dependencies in $ENV." >&2
  exit 1
fi

echo "Validating the full candidate pool and frozen Step 10 results..."
"$ENV/bin/python" \
  experiments/fcv_vit_waterbirds100/scripts/aggregate_candidate_metrics.py >/dev/null
"$ENV/bin/python" \
  experiments/fcv_vit_waterbirds100/scripts/compute_gap_closure.py \
  --validate-step10-only >/dev/null

pool_job=$(sbatch --parsable \
  experiments/fcv_vit_waterbirds100/slurm/score_pool_test_array.sbatch)
diagnostic_job=$(sbatch --parsable --dependency="afterany:${pool_job}" \
  experiments/fcv_vit_waterbirds100/slurm/aggregate_pool_test_scores.sbatch)
gap_job=$(sbatch --parsable --dependency="afterok:${pool_job}" \
  experiments/fcv_vit_waterbirds100/slurm/compute_gap_closure.sbatch)

echo "Post-hoc pool test array: $pool_job (27 tasks, 3 checkpoints per task)"
echo "Diagnostic partial aggregation: $diagnostic_job"
echo "Strict gap-closure job: $gap_job"
echo "Pool test metrics are analysis-only and cannot change frozen selections."
echo "Output: $OUTPUT/selection_results/gap_closure_summary.csv"
