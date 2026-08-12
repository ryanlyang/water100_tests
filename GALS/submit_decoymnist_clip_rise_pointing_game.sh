#!/bin/bash
set -Eeuo pipefail
unset SBATCH_GET_USER_ENV SBATCH_EXPORT SBATCH_EXPORT_FILE
export SLURM_EXPORT_ENV=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/run_decoymnist_clip_rise_pointing_game.sh"
for method in clip_zs clip_lr; do
  short_method="${method/_/}"
  job_name="pgr1_dec_${short_method}"
  queued="$(squeue -h -u "$USER" -n "$job_name" -o '%A' | paste -sd ';' -)"
  if [[ -n "$queued" ]]; then
    echo "[SKIP-QUEUED] method=$method jobs=$queued"
    continue
  fi
  job_id="$(sbatch --parsable --job-name="$job_name" \
    --export="ALL,METHOD=$method" "$WORKER")"
  echo "[SUBMITTED] method=$method job=$job_id partition=debug"
done
