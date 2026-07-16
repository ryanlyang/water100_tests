#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/optuna_color_param_igrad_revkl_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/optuna_color_param_igrad_revkl_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
DATA_PATH="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/ColorMNIST_png"
GT_PATH="/home/ryreu/guided_cnn/MNIST_AGAIN/ColorGen/LearningToLook/code/WeCLIPPlus/results_mnist/val/prediction_cmap"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
STUDY_NAME="color_param_optuna_igrad_revkl_${RUN_TAG}"
DB_PATH="${REPO_ROOT}/models/ColorMNIST_test/${STUDY_NAME}.db"

mkdir -p "${LOG_DIR}"
mkdir -p "${REPO_ROOT}/models/ColorMNIST_test"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"

echo "[$(date)] Host: $(hostname)"
which python

python -u mnist/run_color_param_optuna.py \
  --png-root "${DATA_PATH}" \
  --gt-path "${GT_PATH}" \
  --val-saliency igrad \
  --val-metric rev_kl \
  --n-trials 50 \
  --n-seeds 5 \
  --study-name "${STUDY_NAME}" \
  --db-path "${DB_PATH}"
