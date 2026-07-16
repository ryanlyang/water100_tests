#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/mnist_train_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/mnist_train_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"

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

echo "Training ColorMNIST (vanilla, 30 epochs)"
srun --unbuffered python -u mnist/ColorMNIST/train.py \
  --regularizer_rate 0 \
  --epochs 30 \
  --seed 0 \
  --use-png

echo "Training DecoyMNIST (vanilla, 30 epochs)"
srun --unbuffered python -u mnist/DecoyMNIST/train_mnist_decoy.py \
  --regularizer_rate 0 \
  --epochs 30 \
  --seed 0 \
  --use-png
