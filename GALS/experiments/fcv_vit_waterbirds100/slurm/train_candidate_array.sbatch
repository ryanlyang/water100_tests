#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_pool
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --array=0-26%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step4_%A_%a.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step4_%A_%a.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
python --version
python -c "import torch, torchvision, timm; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'timm', timm.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun python -u experiments/fcv_vit_waterbirds100/scripts/train_candidate.py \
  --run-index "$SLURM_ARRAY_TASK_ID" \
  --device cuda
