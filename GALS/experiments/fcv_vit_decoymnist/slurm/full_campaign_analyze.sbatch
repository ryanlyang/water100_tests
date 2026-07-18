#!/usr/bin/env bash
#SBATCH --job-name=dcyfcv_report
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_full_campaign/run_logs/report_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_full_campaign/run_logs/report_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

python -u experiments/fcv_vit_decoymnist/scripts/analyze_full_campaign_results.py
