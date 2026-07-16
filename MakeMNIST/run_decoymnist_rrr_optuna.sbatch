#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/optuna_decoymnist_rrr_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/optuna_decoymnist_rrr_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
DATA_PATH="${REPO_ROOT}/data/DecoyMNIST_png"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
STUDY_NAME="decoymnist_rrr_${RUN_TAG}"
DB_PATH="${REPO_ROOT}/models/DecoyMNIST/${STUDY_NAME}.db"

mkdir -p "${LOG_DIR}"
mkdir -p "${REPO_ROOT}/models/DecoyMNIST"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"

python -u mnist/run_decoymnist_rrr_optuna.py \
  --png-root "${DATA_PATH}" \
  --n-trials 50 \
  --n-seeds 5 \
  --study-name "${STUDY_NAME}" \
  --db-path "${DB_PATH}"
