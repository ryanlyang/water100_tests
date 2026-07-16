#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

mkdir -p \
  "$OUTPUT/run_logs" \
  "$OUTPUT/selection_results/selector_scatter_plots"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi
if ! "$ENV/bin/python" -c "import pandas, scipy, matplotlib"; then
  echo "Missing Step 12 dependencies in $ENV." >&2
  exit 1
fi

echo "Strictly validating Step 11 and the frozen selection before submission..."
"$ENV/bin/python" \
  experiments/fcv_vit_waterbirds100/scripts/aggregate_pool_test_scores.py >/dev/null
"$ENV/bin/python" \
  experiments/fcv_vit_waterbirds100/scripts/compute_gap_closure.py >/dev/null

rank_job=$(sbatch --parsable \
  experiments/fcv_vit_waterbirds100/slurm/analyze_rank_quality.sbatch)

echo "Rank-analysis job: $rank_job"
echo "No GPU inference is repeated; Step 12 uses the complete Step 11 pool index."
echo "Output: $OUTPUT/selection_results/rank_correlation_results.csv"
