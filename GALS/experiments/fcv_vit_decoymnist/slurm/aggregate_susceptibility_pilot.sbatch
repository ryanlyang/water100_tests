#!/usr/bin/env bash
#SBATCH --job-name=decoy_vit_agg
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility/run_logs/aggregate_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility/run_logs/aggregate_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

python -u experiments/fcv_vit_decoymnist/scripts/aggregate_susceptibility_pilot.py

