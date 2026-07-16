#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_test
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step10_final_test_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step10_final_test_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID}"
python --version
python -c "import torch, torchvision, timm; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'timm', timm.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun python -u experiments/fcv_vit_waterbirds100/scripts/evaluate_selected_checkpoints.py \
  --device cuda \
  --batch-size 128 \
  --workers 8
