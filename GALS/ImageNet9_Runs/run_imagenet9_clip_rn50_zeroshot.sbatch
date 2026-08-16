#!/usr/bin/env bash
#SBATCH --job-name=in9_clip_rn50_zs
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_clip_rn50_zs_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_clip_rn50_zs_%j.err

set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LOG_ROOT}/clip_rn50_zeroshot_openai}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/home/ryreu/guided_cnn/models/openai_clip}"
CONDA_ENV="${CONDA_ENV:-r4rr-weclip}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/${CONDA_ENV}/bin/python}"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$DOWNLOAD_ROOT"
cd "$REPO"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[$(date)] Host: $(hostname)"
echo "Model: OpenCLIP RN50 with OpenAI weights"
echo "Conda environment: $CONDA_ENV"
echo "Data: $DATA_ROOT"
echo "Protocol: $PROTOCOL_NAME"
echo "Evaluation: all eight official Backgrounds Challenge variants"
echo "Validation/tuning data used: NO"
echo "Output root: $OUTPUT_ROOT"
"$PYTHON_BIN" -c 'import sys, torch; print(sys.version); print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

srun --unbuffered "$PYTHON_BIN" -u ImageNet9_Runs/evaluate_imagenet9_clip_vit_zeroshot.py \
  --data-root "$DATA_ROOT" \
  --protocol-name "$PROTOCOL_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --models RN50 \
  --batch-size "${BATCH_SIZE:-256}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --device cuda:0 \
  --seed 0 \
  --download-root "$DOWNLOAD_ROOT"

echo "[$(date)] ImageNet-9 CLIP RN50 zero-shot evaluation complete."
