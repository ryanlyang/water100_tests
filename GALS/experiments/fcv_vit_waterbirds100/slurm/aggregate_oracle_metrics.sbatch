#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_orcidx
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step9_oracle_aggregate_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step9_oracle_aggregate_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/selection_results

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID}"
srun python -u experiments/fcv_vit_waterbirds100/scripts/aggregate_oracle_metrics.py \
  --allow-incomplete \
  --output-csv "$OUTPUT/candidate_oracle_scores_partial.csv" \
  --output-summary "$OUTPUT/candidate_oracle_scores_partial_summary.json"
