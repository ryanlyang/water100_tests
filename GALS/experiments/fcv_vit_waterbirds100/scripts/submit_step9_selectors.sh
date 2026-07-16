#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

mkdir -p "$OUTPUT/run_logs" "$OUTPUT/selection_results/oracle_scores"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi
if ! "$ENV/bin/python" -c "import torch, torchvision, timm, pandas"; then
  echo "Missing Step 9 dependencies in $ENV." >&2
  exit 1
fi

echo "Strictly validating all unprivileged selector inputs before submission..."
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/aggregate_candidate_metrics.py >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/aggregate_fcv_scores.py >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/aggregate_fcv_controls.py >/dev/null

oracle_job=$(sbatch --parsable \
  experiments/fcv_vit_waterbirds100/slurm/score_oracle_array.sbatch)
diagnostic_job=$(sbatch --parsable --dependency="afterany:${oracle_job}" \
  experiments/fcv_vit_waterbirds100/slurm/aggregate_oracle_metrics.sbatch)
selection_job=$(sbatch --parsable --dependency="afterok:${oracle_job}" \
  experiments/fcv_vit_waterbirds100/slurm/build_selection_table.sbatch)

echo "Oracle validation array: $oracle_job (27 tasks, 3 checkpoints per task)"
echo "Diagnostic partial aggregation: $diagnostic_job"
echo "Strict selector-table job: $selection_job"
echo "Re-submission reuses Oracle summaries with matching checkpoint/config/manifest hashes."
echo "Step 9 does not load or report test data; selected checkpoints pass to Step 10."
