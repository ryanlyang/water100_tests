#!/bin/bash -l
# Generate deterministic DecoyMNIST and clean-digit surrogate teacher maps.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=prep_decoy_surrogate
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/%x_%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyMNIST_surrogate_oracle_v1}"
MNIST_ROOT="${MNIST_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data}"
DATASET_SEED="${DATASET_SEED:-0}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUTPUT_ROOT}.tar.gz}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT" "$MNIST_ROOT"

# Conda activation hooks may read unset optional variables.
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$GALS_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$GALS_ROOT"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[RUN] deterministic CDEP-style DecoyMNIST + oracle surrogate teacher maps"
echo "[RUN] dataset_seed=$DATASET_SEED mask_threshold=$MASK_THRESHOLD"
echo "[RUN] output_root=$OUTPUT_ROOT"
echo "[RUN] dataset_root=$OUTPUT_ROOT/DecoyMNIST_png"
echo "[RUN] teacher_maps=$OUTPUT_ROOT/teacher_maps/prediction_cmap"
echo "[RUN] archive=$ARCHIVE_PATH"
which python

python -u prepare_decoymnist_surrogate_teacher_data.py \
  --output-root "$OUTPUT_ROOT" \
  --mnist-root "$MNIST_ROOT" \
  --dataset-seed "$DATASET_SEED" \
  --mask-threshold "$MASK_THRESHOLD" \
  --progress-every 1000 \
  --archive-path "$ARCHIVE_PATH"

echo "[DONE] dataset=$OUTPUT_ROOT/DecoyMNIST_png"
echo "[DONE] teacher_maps=$OUTPUT_ROOT/teacher_maps/prediction_cmap"
echo "[DONE] manifest=$OUTPUT_ROOT/metadata/completion_manifest.json"
echo "[DONE] archive=$ARCHIVE_PATH"
