#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

mkdir -p "$OUTPUT/run_logs" "$OUTPUT/selection_results/final_test_scores"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi
if ! "$ENV/bin/python" -c "import torch, torchvision, timm, pandas"; then
  echo "Missing Step 10 dependencies in $ENV." >&2
  exit 1
fi

echo "Validating and freezing Step 9 selection without opening test data..."
"$ENV/bin/python" \
  experiments/fcv_vit_waterbirds100/scripts/evaluate_selected_checkpoints.py \
  --validate-selection-only >/dev/null

test_job=$(sbatch --parsable \
  experiments/fcv_vit_waterbirds100/slurm/evaluate_selected_checkpoints.sbatch)

echo "Final selected-checkpoint test job: $test_job"
echo "Unique checkpoints are evaluated once and expanded back to selector rows."
echo "Output: $OUTPUT/selection_results/final_test_results.csv"
