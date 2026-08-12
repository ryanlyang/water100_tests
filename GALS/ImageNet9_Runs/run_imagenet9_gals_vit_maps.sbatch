#!/usr/bin/env bash
#SBATCH --job-name=in9_gals_vit_maps
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_gals_vit_maps_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_gals_vit_maps_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/manifest.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/gals_maps/clip_vit_b32_transformer_v1}"
MODE="${MODE:-diagnostic}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"

mkdir -p /home/ryreu/guided_cnn/logsImageNet9 "$OUTPUT_ROOT"
cd "$REPO"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gals_a100
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

BPE="$REPO/CLIP/clip/bpe_simple_vocab_16e6.txt.gz"
if [[ ! -f "$BPE" ]]; then
  echo "[INFO] Downloading OpenAI CLIP tokenizer vocabulary"
  curl -L --fail --retry 3 -o "$BPE" \
    https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
fi

ARGS=(
  --manifest "$MANIFEST"
  --output-root "$OUTPUT_ROOT"
  --device cuda:0
  --clip-checkpoint ViT-B/32
  --skip-existing
)
if [[ "$MODE" == "diagnostic" ]]; then
  ARGS+=(
    --selection diagnostic
    --diagnostic-per-class 20
    --diagnostic-seed 0
    --write-qa
  )
elif [[ "$MODE" == "full" ]]; then
  START_INDEX=$((TASK_ID * CHUNK_SIZE))
  END_INDEX=$((START_INDEX + CHUNK_SIZE))
  ARGS+=(
    --selection all
    --start-index "$START_INDEX"
    --end-index "$END_INDEX"
  )
else
  echo "[ERROR] MODE must be diagnostic or full, got: $MODE" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Mode: $MODE task=$TASK_ID chunk_size=$CHUNK_SIZE"
echo "Manifest: $MANIFEST"
echo "Output: $OUTPUT_ROOT"
echo "Teacher: OpenAI CLIP ViT-B/32 transformer relevance"
echo "Official variants generated: NO"

srun --unbuffered "$PYTHON_BIN" -u \
  ImageNet9_Runs/generate_imagenet9_gals_vit_maps.py "${ARGS[@]}"

echo "[$(date)] ImageNet-9 GALS ViT map invocation finished."
