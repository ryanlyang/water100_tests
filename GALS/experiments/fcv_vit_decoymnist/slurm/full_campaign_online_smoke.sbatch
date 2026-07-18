#!/usr/bin/env bash
#SBATCH --job-name=dcyfcv_smoke
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_full_campaign/run_logs/smoke_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_full_campaign/run_logs/smoke_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date)] host=$(hostname) real one-epoch production-path smoke"
srun python -u experiments/fcv_vit_decoymnist/scripts/run_full_campaign_online.py \
  --run-index 0 \
  --device cuda \
  --smoke-one-epoch \
  --restart-partial
python -u experiments/fcv_vit_decoymnist/scripts/finalize_full_campaign_smoke.py
