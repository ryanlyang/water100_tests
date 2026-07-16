#!/bin/bash -l
# One-epoch guided KL / R4RR MobileNetV2 smoke test for Waterbirds-95.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/guided95_mobilenetv2_smoke_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/guided95_mobilenetv2_smoke_%j.err

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=0
export GUIDED_NUM_WORKERS=0
export GUIDED_CAM_DIAGNOSTICS="${GUIDED_CAM_DIAGNOSTICS:-1}"
export GUIDED_CAM_DIAGNOSTICS_EVERY="${GUIDED_CAM_DIAGNOSTICS_EVERY:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
GALS_ROOT="$REPO_ROOT/GALS"
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2
GT_ROOT=${GT_ROOT:-/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$GALS_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] Missing DATA_ROOT: $DATA_ROOT" >&2
  exit 2
fi
if [[ ! -d "$GT_ROOT" ]]; then
  echo "[ERROR] Missing GT_ROOT: $GT_ROOT" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT"
echo "GT masks: $GT_ROOT"
echo "Backbone: mobilenet_v2 pretrained=1"
echo "Smoke: tensor CAM check + one guided epoch"
which python

python -u GALS/smoke_mobilenetv2_cam.py --device cuda:0

python -u run_guided_waterbird.py \
  "$DATA_ROOT" \
  "$GT_ROOT" \
  --model-name mobilenet_v2 \
  --pretrained \
  --num-epochs 1 \
  --attention_epoch 0 \
  --kl_lambda 10 \
  --kl_increment 0 \
  --base_lr 1e-4 \
  --classifier_lr 1e-3 \
  --lr2_mult 1.0 \
  --seed 0

echo "[DONE] Guided Waterbirds-95 MobileNetV2 smoke test completed."
