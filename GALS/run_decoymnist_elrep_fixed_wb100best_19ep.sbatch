#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_elrep_fixed_wb100best_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_elrep_fixed_wb100best_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
CONDA_ENV="${CONDA_ENV:-gals_a100}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"

DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"

# Keep DecoyMNIST optimizer defaults unchanged.
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EPOCHS="${EPOCHS:-19}"
N_SEEDS="${N_SEEDS:-5}"
SEED_START="${SEED_START:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# ElRep-specific weights transferred from the best Waterbirds-100 ElRep trial.
THETA1="${THETA1:-4.1741182912970346e-05}"
THETA2="${THETA2:-2.5433427234421545e-06}"

OUT_CSV="${OUT_CSV:-${LOG_DIR}/decoy_elrep_fixed_wb100best_${SLURM_JOB_ID}.csv}"
SAVE_DIR="${SAVE_DIR:-}"

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
echo "Output CSV: ${OUT_CSV}"
echo "epochs=${EPOCHS} lr=${LR} wd=${WEIGHT_DECAY} theta1=${THETA1} theta2=${THETA2}"
echo "seeds=${SEED_START}..$((SEED_START + N_SEEDS - 1)) workers=${NUM_WORKERS}"
if [[ -n "${SAVE_DIR}" ]]; then
  echo "Checkpoint dir: ${SAVE_DIR}"
fi

ARGS=(
  --png-root "${DATA_PATH}"
  --epochs "${EPOCHS}"
  --n-seeds "${N_SEEDS}"
  --seed-start "${SEED_START}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --theta1 "${THETA1}"
  --theta2 "${THETA2}"
  --num-workers "${NUM_WORKERS}"
  --output-csv "${OUT_CSV}"
)

if [[ -n "${SAVE_DIR}" ]]; then
  ARGS+=(--save-dir "${SAVE_DIR}")
fi

python -u run_decoymnist_elrep_fixed.py "${ARGS[@]}"
