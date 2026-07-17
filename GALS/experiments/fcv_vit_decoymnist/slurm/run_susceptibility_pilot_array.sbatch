#!/usr/bin/env bash
#SBATCH --job-name=decoy_vit_sus
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --array=0-8%8
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility/run_logs/pilot_%A_%a.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility/run_logs/pilot_%A_%a.err

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
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun python -u experiments/fcv_vit_decoymnist/scripts/run_susceptibility_pilot.py \
  --run-index "$SLURM_ARRAY_TASK_ID" \
  --device cuda

