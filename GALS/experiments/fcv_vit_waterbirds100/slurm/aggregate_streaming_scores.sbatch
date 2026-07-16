#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_streamidx
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step6_8_aggregate_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step6_8_aggregate_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date)] Strictly aggregating all 81 streamed candidates."
srun python -u experiments/fcv_vit_waterbirds100/scripts/aggregate_fcv_scores.py
srun python -u experiments/fcv_vit_waterbirds100/scripts/aggregate_fcv_controls.py
srun python -u experiments/fcv_vit_waterbirds100/scripts/check_storage_budget.py \
  --stage step6_8_complete
