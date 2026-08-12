#!/usr/bin/env bash
#SBATCH --job-name=audit_in9
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/audit_in9_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/audit_in9_%j.err

set -euo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"

mkdir -p /home/ryreu/guided_cnn/logsImageNet9
cd "$REPO"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO"
echo "Data: $DATA_ROOT"
echo "Protocol: $PROTOCOL_NAME"
echo "Selection objective: Original validation macro class accuracy"
echo "Official variants are held out from tuning."

PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m unittest discover \
  -s tests/imagenet9 \
  -p 'test_imagenet9_data.py' \
  -v

PYTHONNOUSERSITE=1 "$PYTHON_BIN" -u ImageNet9_Runs/audit_imagenet9_loaders.py \
  --data-root "$DATA_ROOT" \
  --protocol-name "$PROTOCOL_NAME"

echo "[$(date)] ImageNet-9 loader audit complete."
