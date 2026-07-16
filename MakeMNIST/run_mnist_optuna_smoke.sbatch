#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/optuna_smoke_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/optuna_smoke_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
COLOR_DATA="${REPO_ROOT}/data/ColorMNIST_png"
DECOY_DATA="${REPO_ROOT}/data/DecoyMNIST_png"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"

mkdir -p "${LOG_DIR}"
mkdir -p "${REPO_ROOT}/models/ColorMNIST_test"
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

run_smoke () {
  local script_path="$1"
  local data_root="$2"
  local model_dir="$3"
  local name
  name="$(basename "${script_path}" .py)"
  local study="${name}_smoke_${RUN_TAG}"
  local db_path="${model_dir}/${study}.db"

  echo "============================================================"
  echo "Smoke test: ${name}"
  echo "Data: ${data_root}"
  echo "DB: ${db_path}"
  echo "============================================================"

  srun --unbuffered python -u "${script_path}" \
    --png-root "${data_root}" \
    --epochs 1 \
    --batch-size 64 \
    --test-batch-size 256 \
    --n-trials 1 \
    --n-seeds 1 \
    --num-workers 0 \
    --study-name "${study}" \
    --db-path "${db_path}"
}

# ColorMNIST smoke
run_smoke "mnist/run_colormnist_vanilla_optuna.py" "${COLOR_DATA}" "${REPO_ROOT}/models/ColorMNIST_test"
run_smoke "mnist/run_colormnist_cdep_optuna.py" "${COLOR_DATA}" "${REPO_ROOT}/models/ColorMNIST_test"
run_smoke "mnist/run_colormnist_rrr_optuna.py" "${COLOR_DATA}" "${REPO_ROOT}/models/ColorMNIST_test"
run_smoke "mnist/run_colormnist_eg_optuna.py" "${COLOR_DATA}" "${REPO_ROOT}/models/ColorMNIST_test"

# DecoyMNIST smoke
run_smoke "mnist/run_decoymnist_vanilla_optuna.py" "${DECOY_DATA}" "${REPO_ROOT}/models/DecoyMNIST"
run_smoke "mnist/run_decoymnist_cdep_optuna.py" "${DECOY_DATA}" "${REPO_ROOT}/models/DecoyMNIST"
run_smoke "mnist/run_decoymnist_rrr_optuna.py" "${DECOY_DATA}" "${REPO_ROOT}/models/DecoyMNIST"
run_smoke "mnist/run_decoymnist_eg_optuna.py" "${DECOY_DATA}" "${REPO_ROOT}/models/DecoyMNIST"

echo "All smoke tests completed."
