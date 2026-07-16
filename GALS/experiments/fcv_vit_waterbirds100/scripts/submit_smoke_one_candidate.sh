#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
SMOKE_TAG=${SMOKE_TAG:-$(date +%Y%m%d_%H%M%S)}

cd "$REPO"
mkdir -p "$OUTPUT/run_logs"
if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi

job=$(sbatch --parsable --export="ALL,SMOKE_TAG=$SMOKE_TAG" \
  experiments/fcv_vit_waterbirds100/slurm/smoke_one_candidate.sbatch)
echo "GH200 one-candidate end-to-end smoke job: $job"
echo "Smoke tag: $SMOKE_TAG"
echo "Artifacts: $OUTPUT/smoke/$SMOKE_TAG"
