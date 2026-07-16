#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/optuna_param_igrad_revkl_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/optuna_param_igrad_revkl_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
GT_PATH="/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
OPTUNA_SEED="$(date +%s)"
OBJECTIVE_SEED="$((OPTUNA_SEED % 100000 + 1))"
STUDY_NAME="decoy_param_optuna_igrad_revkl_${RUN_TAG}"
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

echo "[$(date)] Host: $(hostname)"
which python
echo "Optuna seed: ${OPTUNA_SEED}"
echo "Objective seed: ${OBJECTIVE_SEED}"

python -u mnist/run_decoy_param_optuna.py \
  --gt-path "${GT_PATH}" \
  --beta 10 \
  --val-saliency igrad \
  --val-metric fwd_kl \
  --checkpoint-selector val_acc \
  --selector-start attention \
  --n-trials 10 \
  --n-seeds 5 \
  --objective-seed "${OBJECTIVE_SEED}" \
  --optuna-seed "${OPTUNA_SEED}" \
  --study-name "${STUDY_NAME}" \
  --db-path "${DB_PATH}"
