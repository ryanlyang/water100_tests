#!/bin/bash -l
set -Eeuo pipefail

ROOT_DIR=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
VANILLA_JOB="$ROOT_DIR/GALS/run_waterbirds95_vanilla.sh"
GUIDED_SWEEP_JOB="$ROOT_DIR/run_guided_waterbird_sweep.sh"
INVERT_SWEEP_JOB="$ROOT_DIR/run_waterbird_invert_sweep.sh"

if [[ ! -f "$VANILLA_JOB" ]]; then
  echo "Missing vanilla job script: $VANILLA_JOB" >&2
  exit 1
fi
if [[ ! -f "$GUIDED_SWEEP_JOB" ]]; then
  echo "Missing guided sweep script: $GUIDED_SWEEP_JOB" >&2
  exit 1
fi
if [[ ! -f "$INVERT_SWEEP_JOB" ]]; then
  echo "Missing invert sweep script: $INVERT_SWEEP_JOB" >&2
  exit 1
fi

first_submit=$(BOOTSTRAP_ENV=1 RECREATE_ENV=1 sbatch "$VANILLA_JOB")
echo "$first_submit"
first_job_id=$(echo "$first_submit" | awk '{print $4}')
if [[ -z "$first_job_id" ]]; then
  echo "Failed to parse job id from sbatch output." >&2
  exit 1
fi

guided_submit=$(sbatch --dependency=afterok:"$first_job_id" "$GUIDED_SWEEP_JOB")
echo "$guided_submit"

invert_submit=$(sbatch --dependency=afterok:"$first_job_id" "$INVERT_SWEEP_JOB")
echo "$invert_submit"
