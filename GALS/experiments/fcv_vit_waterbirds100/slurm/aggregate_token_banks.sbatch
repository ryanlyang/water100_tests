#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_bankagg
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step6_aggregate_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step6_aggregate_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

python -u experiments/fcv_vit_waterbirds100/scripts/aggregate_token_banks.py \
  --allow-incomplete
