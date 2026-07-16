#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_banks_disabled
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --time=00:05:00

echo "Disabled: use slurm/stream_score_run_array.sbatch via submit_step6_8_streaming.sh." >&2
exit 2
