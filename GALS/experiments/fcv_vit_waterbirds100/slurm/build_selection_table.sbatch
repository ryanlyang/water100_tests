#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_select
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step9_selection_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step9_selection_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID}"

# Revalidate every prerequisite strictly before selecting a checkpoint.
srun python -u experiments/fcv_vit_waterbirds100/scripts/aggregate_candidate_metrics.py
srun python -u experiments/fcv_vit_waterbirds100/scripts/aggregate_fcv_scores.py
srun python -u experiments/fcv_vit_waterbirds100/scripts/aggregate_fcv_controls.py
srun python -u experiments/fcv_vit_waterbirds100/scripts/aggregate_oracle_metrics.py
srun python -u experiments/fcv_vit_waterbirds100/scripts/build_selection_table.py
