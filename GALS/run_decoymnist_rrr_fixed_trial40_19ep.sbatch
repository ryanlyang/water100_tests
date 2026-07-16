#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_rrr_fixed_trial40_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_rrr_fixed_trial40_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
CONDA_ENV="${CONDA_ENV:-learntolook}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"

DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"
MASK_PATH="${MASK_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png/clip_rn50_attention_gradcam}"

# Keep DecoyMNIST optimizer defaults unchanged.
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# Fixed Trial-40 hyperparameters.
GRAD_WEIGHT="${GRAD_WEIGHT:-72503.48035960984}"
GRAD_CRITERION="${GRAD_CRITERION:-L1}"
EPOCHS="${EPOCHS:-19}"
N_SEEDS="${N_SEEDS:-5}"
SEED_START="${SEED_START:-0}"

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
echo "Repo: ${REPO_ROOT}"
echo "Data: ${DATA_PATH}"
echo "Masks: ${MASK_PATH}"
echo "epochs=${EPOCHS} lr=${LR} wd=${WEIGHT_DECAY} grad_weight=${GRAD_WEIGHT} grad_criterion=${GRAD_CRITERION}"

python -u run_decoymnist_rrr_fixed.py \
  --png-root "${DATA_PATH}" \
  --mask-root "${MASK_PATH}" \
  --epochs "${EPOCHS}" \
  --n-seeds "${N_SEEDS}" \
  --seed-start "${SEED_START}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --grad-weight "${GRAD_WEIGHT}" \
  --grad-criterion "${GRAD_CRITERION}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}"
