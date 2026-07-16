#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/optuna_decoy_selector_scatter_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/optuna_decoy_selector_scatter_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
DATA_PATH="${REPO_ROOT}/data/DecoyMNIST_png"
GT_PATH="/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
STUDY_NAME="decoy_param_selector_scatter_${RUN_TAG}"
DB_PATH="${REPO_ROOT}/models/DecoyMNIST/${STUDY_NAME}.db"
ARTIFACT_DIR="${REPO_ROOT}/models/DecoyMNIST/${STUDY_NAME}_artifacts"
SUMMARY_CSV="${ARTIFACT_DIR}/trial_summary.csv"
EVAL_CSV="${ARTIFACT_DIR}/trial_eval.csv"

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

echo "Running Decoy Optuna sweep with dual trial selectors..."
python -u mnist/run_decoy_param_optuna.py \
  --png-root "${DATA_PATH}" \
  --gt-path "${GT_PATH}" \
  --beta 10 \
  --lr-min 1e-5 \
  --lr-max 5e-2 \
  --lr2-min 1e-5 \
  --lr2-max 5e-2 \
  --lr2-mult-min 0.1 \
  --lr2-mult-max 3.0 \
  --attention-epoch-min 1 \
  --attention-epoch-max 29 \
  --kl-lambda-min 1.0 \
  --kl-lambda-max 500.0 \
  --kl-incr-fixed 0.0 \
  --ig-steps 16 \
  --val-saliency igrad \
  --val-metric fwd_kl \
  --selector-start attention \
  --save-trial-checkpoints \
  --trial-artifacts-dir "${ARTIFACT_DIR}" \
  --trial-summary-csv "${SUMMARY_CSV}" \
  --n-trials 50 \
  --n-seeds 5 \
  --skip-phase2 \
  --study-name "${STUDY_NAME}" \
  --db-path "${DB_PATH}"

echo "Evaluating saved per-trial checkpoints on test..."
python -u mnist/eval_decoy_trial_checkpoints.py \
  --summary-csv "${SUMMARY_CSV}" \
  --png-root "${DATA_PATH}" \
  --output-csv "${EVAL_CSV}"

echo "Plotting selector diagnostic scatters..."
python -u mnist/plot_decoy_trial_scatter.py \
  --eval-csv "${EVAL_CSV}" \
  --out-dir "${ARTIFACT_DIR}" \
  --prefix "${STUDY_NAME}"

echo "Done. Artifacts in: ${ARTIFACT_DIR}"
