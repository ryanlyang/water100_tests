#!/bin/bash -l
# Fixed guided DecoyMNIST run:
# - same backbone/training style as cdepstyle vanilla script
# - single Adam LR (no base/classifier split)
# - guided stage enabled after attention epoch

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_fixed_guided_19ep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_fixed_guided_19ep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST"
CONDA_ENV="learntolook"
LOG_DIR="/home/ryreu/guided_cnn/logsMNIST"
DATA_PATH="${REPO_ROOT}/data/DecoyMNIST_png"
GT_PATH_WECLIP="/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap"
GT_PATH_OPENAI_XCIT="/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist_openai_xcit/val/prediction_cmap"
GT_PATH_OPENCLIP="/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist_openclip/val/prediction_cmap"

# Fixed run settings
EPOCHS=19
ATTN_SRC=73
SRC_EPOCHS=200
# attention epoch transfer by normalized epoch fraction: round((ATTN_SRC / SRC_EPOCHS) * EPOCHS)
ATTENTION_EPOCH=7
KL_LAMBDA=495.60509512105125
KL_INCR=0.0
LR=0.001

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
echo "Epoch mapping: ${ATTN_SRC}/${SRC_EPOCHS} -> attention_epoch=${ATTENTION_EPOCH} for epochs=${EPOCHS}"
echo "Data: ${DATA_PATH}"

GT_NAMES=("WeCLIP" "OpenAI_XCiT" "OpenCLIP")
GT_PATHS=("${GT_PATH_WECLIP}" "${GT_PATH_OPENAI_XCIT}" "${GT_PATH_OPENCLIP}")

for i in "${!GT_NAMES[@]}"; do
  echo
  echo "============================================================"
  echo "Running guided Decoy for GT source: ${GT_NAMES[$i]}"
  echo "Masks: ${GT_PATHS[$i]}"
  echo "============================================================"

  python -u mnist/run_decoy_guided_cdepstyle_valacc.py \
    --png-root "${DATA_PATH}" \
    --gt-path "${GT_PATHS[$i]}" \
    --epochs "${EPOCHS}" \
    --val-frac 0.1 \
    --batch-size 64 \
    --test-batch-size 1000 \
    --weight-decay 1e-4 \
    --lr "${LR}" \
    --attention-epoch "${ATTENTION_EPOCH}" \
    --kl-lambda "${KL_LAMBDA}" \
    --kl-incr "${KL_INCR}" \
    --n-seeds 5 \
    --split-seed 0
done
