#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_vanilla_paramgroups_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_vanilla_paramgroups_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
DATA_PATH="${REPO_ROOT}/data/DecoyMNIST_png"

# Fixed hyperparameters from your trial.
BASE_LR=0.031210590691245817
CLASSIFIER_LR=0.0008517287145349147
MOMENTUM=0.8914661939990524

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

python -u mnist/run_decoy_vanilla_paramgroups_fixed.py \
  --png-root "${DATA_PATH}" \
  --epochs 30 \
  --n-seeds 5 \
  --base-lr "${BASE_LR}" \
  --classifier-lr "${CLASSIFIER_LR}" \
  --momentum "${MOMENTUM}"
