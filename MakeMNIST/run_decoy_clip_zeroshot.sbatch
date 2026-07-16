#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_clip_zeroshot_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_clip_zeroshot_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/MakeMNIST}"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"
CLIP_MODEL="${CLIP_MODEL:-RN50}"
SEEDS="${SEEDS:-0,1,2,3,4}"

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

auto_csv="${LOG_DIR}/decoy_clip_zeroshot_${SLURM_JOB_ID}.csv"

python -u run_decoy_clip_zeroshot.py \
  --png-root "${DATA_PATH}" \
  --split test \
  --clip-model "${CLIP_MODEL}" \
  --seeds "${SEEDS}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --output-csv "${auto_csv}"
