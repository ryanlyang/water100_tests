#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/manual_decoy_selector_scatter_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/manual_decoy_selector_scatter_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
DATA_PATH="${REPO_ROOT}/data/DecoyMNIST_png"
GT_PATH="/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
STUDY_NAME="decoy_manual_kl_selector_scatter_${RUN_TAG}"
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

echo "Running manual Decoy KL sweep (no Optuna)..."
python -u mnist/run_decoy_param_optuna.py \
  --manual-kl-sweep \
  --png-root "${DATA_PATH}" \
  --gt-path "${GT_PATH}" \
  --beta 10 \
  --manual-attention-epoch 15 \
  --manual-lr 1e-3 \
  --manual-lr2 5e-3 \
  --manual-lr2-mult 0.1 \
  --manual-kl-min 0.01 \
  --manual-kl-max 100.0 \
  --manual-num-points 10 \
  --kl-incr-fixed 0.0 \
  --ig-steps 16 \
  --val-saliency igrad \
  --val-metric fwd_kl \
  --selector-start attention \
  --save-trial-checkpoints \
  --trial-artifacts-dir "${ARTIFACT_DIR}" \
  --trial-summary-csv "${SUMMARY_CSV}"

echo "Evaluating saved per-run checkpoints on test..."
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
