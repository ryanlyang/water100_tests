#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_abn_paramgroups_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_abn_paramgroups_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/MakeMNIST}"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"

# CDEP-style optimizer defaults.
LR=0.001
WEIGHT_DECAY=1e-4

# Keep ABN-specific behavior.
ABN_CLS_WEIGHT=3.2547104257357056

mkdir -p "${LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"

echo "[$(date)] Host: $(hostname)"
which python

python -u mnist/run_decoy_abn_paramgroups_fixed.py \
  --png-root "${DATA_PATH}" \
  --epochs 19 \
  --n-seeds 5 \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --abn-cls-weight "${ABN_CLS_WEIGHT}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}"
