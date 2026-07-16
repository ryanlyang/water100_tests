#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/run_colormnist_optuna.sbatch"

# 1. WeCLIP (original)
sbatch --export="ALL,GT_PATH=/home/ryreu/guided_cnn/MNIST_AGAIN/ColorGen/LearningToLook/code/WeCLIPPlus/results_mnist/val/prediction_cmap,STUDY_NAME=colormnist_weclip" \
  "${SBATCH_FILE}"

# 2. OpenAI XCiT
sbatch --export="ALL,GT_PATH=/home/ryreu/guided_cnn/MNIST_AGAIN/ColorGen/LearningToLook/code/WeCLIPPlus/results_mnist_openai_xcit/val/prediction_cmap,STUDY_NAME=colormnist_openai_xcit" \
  "${SBATCH_FILE}"

# 3. OpenCLIP
sbatch --export="ALL,GT_PATH=/home/ryreu/guided_cnn/MNIST_AGAIN/ColorGen/LearningToLook/code/WeCLIPPlus/results_mnist_openclip/val/prediction_cmap,STUDY_NAME=colormnist_openclip" \
  "${SBATCH_FILE}"

echo "Submitted 3 ColorMNIST Optuna jobs."
