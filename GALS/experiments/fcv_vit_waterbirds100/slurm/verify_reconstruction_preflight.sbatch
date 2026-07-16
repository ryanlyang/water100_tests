#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_recon
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step6_reconstruction_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step6_reconstruction_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"
mkdir -p "$OUTPUT/preflight"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun python -u experiments/fcv_vit_waterbirds100/scripts/aggregate_candidate_metrics.py >/dev/null
FIRST_CHECKPOINT=$(python -c \
  'import pandas as pd,sys; print(pd.read_csv(sys.argv[1]).sort_values(["run_index","epoch"]).iloc[0]["checkpoint_path"])' \
  "$OUTPUT/candidate_models/candidate_metrics_biased_val.csv")

srun python -u experiments/fcv_vit_waterbirds100/scripts/verify_vit_patch_forward.py \
  --pretrained \
  --output-report "$OUTPUT/preflight/reconstruction_pretrained.json"
srun python -u experiments/fcv_vit_waterbirds100/scripts/verify_vit_patch_forward.py \
  --checkpoint "$FIRST_CHECKPOINT" \
  --output-report "$OUTPUT/preflight/reconstruction_candidate.json"

echo "[DONE] GH200 pretrained and real-candidate reconstruction gates passed."
