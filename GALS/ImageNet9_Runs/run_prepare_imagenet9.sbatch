#!/usr/bin/env bash
#SBATCH --job-name=prepare_in9
#SBATCH --partition=debug
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/prepare_in9_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/prepare_in9_%j.err

set -euo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/shared/rc/datasets/imagenet2012}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-${DATA_ROOT}/metadata/imagenet_annotations}"
OFFICIAL_TEST_ROOT="${OFFICIAL_TEST_ROOT:-${DATA_ROOT}/official_test/bg_challenge}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
SEED="${SEED:-0}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-5045}"
VAL_PER_CLASS="${VAL_PER_CLASS:-450}"

mkdir -p /home/ryreu/guided_cnn/logsImageNet9
cd "$REPO"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO"
echo "ImageNet: $IMAGENET_ROOT"
echo "Annotations: $ANNOTATION_ROOT"
echo "Official test: $OFFICIAL_TEST_ROOT"
echo "Output: $DATA_ROOT"
echo "Protocol: $PROTOCOL_NAME"
echo "Counts: train=${TRAIN_PER_CLASS}/class val=${VAL_PER_CLASS}/class seed=${SEED}"

PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
"$PYTHON_BIN" --version

"$PYTHON_BIN" -u ImageNet9_Runs/prepare_imagenet9.py \
  --imagenet-root "$IMAGENET_ROOT" \
  --annotation-root "$ANNOTATION_ROOT" \
  --official-test-root "$OFFICIAL_TEST_ROOT" \
  --output-root "$DATA_ROOT" \
  --protocol-name "$PROTOCOL_NAME" \
  --seed "$SEED" \
  --train-per-class "$TRAIN_PER_CLASS" \
  --val-per-class "$VAL_PER_CLASS" \
  --materialize-links

echo "[$(date)] ImageNet-9 preparation complete."
