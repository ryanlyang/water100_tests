#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_attention_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_attention_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME=${ENV_NAME:-gals_a100}
BOOTSTRAP_ENV=${BOOTSTRAP_ENV:-0}
RECREATE_ENV=${RECREATE_ENV:-0}
REQ_FILE=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/requirements.txt

if [[ "$BOOTSTRAP_ENV" -eq 1 ]]; then
  if [[ "$RECREATE_ENV" -eq 1 ]]; then
    conda env remove -n "$ENV_NAME" -y || true
  fi
  if ! conda env list | grep -E "^${ENV_NAME}[[:space:]]" >/dev/null; then
    conda create -y -n "$ENV_NAME" python=3.8
    conda activate "$ENV_NAME"
    conda install -y pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -c nvidia -c conda-forge
    conda install -y -c conda-forge pycocotools
    REQ_TMP=/tmp/${ENV_NAME}_reqs_$$.txt
    grep -v -E '^(opencv-python|pycocotools|torch|torchvision|torchray)' "$REQ_FILE" > "$REQ_TMP"
    pip install -r "$REQ_TMP"
    rm -f "$REQ_TMP"
    pip install torchray==1.0.0.2 --no-deps
    pip install opencv-python==4.6.0.66
    conda deactivate
  fi
fi

conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds
DATA_DIR=waterbird_complete95_forest2water2

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT/$DATA_DIR"
which python

srun --unbuffered python -u extract_attention.py \
  --config configs/waterbirds_95_attention.yaml \
  DATA.ROOT="$DATA_ROOT" \
  DATA.WATERBIRDS_DIR="$DATA_DIR"
