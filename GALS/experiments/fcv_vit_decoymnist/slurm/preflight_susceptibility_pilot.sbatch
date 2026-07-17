#!/usr/bin/env bash
#SBATCH --job-name=decoy_vit_pre
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility/run_logs/preflight_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility/run_logs/preflight_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date)] host=$(hostname) DecoyMNIST ViT susceptibility preflight"
python --version
python -c "import torch, torchvision, timm; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'timm', timm.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun python -u experiments/fcv_vit_decoymnist/scripts/preflight_susceptibility_pilot.py

