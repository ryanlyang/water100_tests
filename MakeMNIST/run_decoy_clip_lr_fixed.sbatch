#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_clip_lr_fixed_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_clip_lr_fixed_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/MakeMNIST}"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"
CLIP_REPO="${CLIP_REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"

# Fixed params from your Waterbirds-100 CLIP-LR best trial
CLIP_MODEL="${CLIP_MODEL:-RN50}"
C_VAL="${C_VAL:-0.2515000498909345}"
PENALTY="${PENALTY:-l2}"
SOLVER="${SOLVER:-lbfgs}"
FIT_INTERCEPT="${FIT_INTERCEPT:-1}"
SEEDS="${SEEDS:-0,1,2,3,4}"
VAL_FRAC="${VAL_FRAC:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"

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

OUT_CSV="${LOG_DIR}/decoy_clip_lr_fixed_${SLURM_JOB_ID}.csv"

fit_arg="--fit-intercept"
if [[ "${FIT_INTERCEPT}" == "0" ]]; then
  fit_arg="--no-fit-intercept"
fi

python -u run_decoy_clip_lr_fixed.py \
  --png-root "${DATA_PATH}" \
  --clip-repo "${CLIP_REPO}" \
  --clip-model "${CLIP_MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --C "${C_VAL}" \
  --penalty "${PENALTY}" \
  --solver "${SOLVER}" \
  "${fit_arg}" \
  --seeds "${SEEDS}" \
  --val-frac "${VAL_FRAC}" \
  --split-seed "${SPLIT_SEED}" \
  --num-workers "${NUM_WORKERS}" \
  --output-csv "${OUT_CSV}"
