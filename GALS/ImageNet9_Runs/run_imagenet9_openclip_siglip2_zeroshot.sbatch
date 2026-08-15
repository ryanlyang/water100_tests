#!/usr/bin/env bash
#SBATCH --job-name=in9_zs_laion_siglip2
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_zs_laion_siglip2_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_zs_laion_siglip2_%j.err

set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LOG_ROOT}/openclip_laion_siglip2_zeroshot}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/home/ryreu/guided_cnn/models/openclip}"
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
echo "Models: OpenCLIP ViT-B-32/laion2b_s34b_b79k and ViT-B-16-SigLIP2-256/webli"
echo "Conda environment: $CONDA_ENV"
echo "Data: $DATA_ROOT"
echo "Protocol: $PROTOCOL_NAME"
echo "Evaluation: all eight official Backgrounds Challenge variants"
echo "Validation/tuning data used: NO"
echo "Output root: $OUTPUT_ROOT"
"$PYTHON_BIN" - <<'PY'
import open_clip
import sys
import torch

required = {
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-16-SigLIP2-256", "webli"),
}
available = set(open_clip.list_pretrained())
missing = sorted(required - available)
print(sys.version)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("open_clip", getattr(open_clip, "__version__", "unknown"))
if missing:
    raise RuntimeError(f"Installed open_clip is missing requested model pairs: {missing}")
PY

srun --unbuffered "$PYTHON_BIN" -u \
  ImageNet9_Runs/evaluate_imagenet9_openclip_siglip2_zeroshot.py \
  --data-root "$DATA_ROOT" \
  --protocol-name "$PROTOCOL_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --models openclip_laion siglip2 \
  --batch-size "${BATCH_SIZE:-256}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --device cuda:0 \
  --seed 0 \
  --download-root "$DOWNLOAD_ROOT"

echo "[$(date)] OpenCLIP LAION and SigLIP2 zero-shot evaluation complete."
