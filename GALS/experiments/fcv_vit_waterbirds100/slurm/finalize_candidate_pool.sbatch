#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_finalize
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step4_finalize_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step4_finalize_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date)] Strictly validating 81 candidates before resume-state cleanup."
srun python -u experiments/fcv_vit_waterbirds100/scripts/prune_completed_resume_states.py
srun python -u experiments/fcv_vit_waterbirds100/scripts/check_storage_budget.py \
  --stage step4_finalized
