#!/bin/bash -l
# Zero-shot eval on DecoyMNIST + Waterbirds95/100 + RedMeat using:
#   1) OpenAI CLIP ViT
#   2) OpenCLIP LAION
#   3) SigLIP2

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsZeroShot/zeroshot_openai_openclip_siglip2_all_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsZeroShot/zeroshot_openai_openclip_siglip2_all_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=${LOG_DIR:-/home/ryreu/guided_cnn/logsZeroShot}
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONNOUSERSITE=1

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}
WB95_PATH=${WB95_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}
WB100_PATH=${WB100_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}
REDMEAT_PATH=${REDMEAT_PATH:-/home/ryreu/guided_cnn/Food101/data/food-101-redmeat}
DECOY_PNG_ROOT=${DECOY_PNG_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}

# Requested model families.
OPENAI_MODEL=${OPENAI_MODEL:-ViT-B/32}
LAION_MODEL=${LAION_MODEL:-ViT-B-32}
LAION_PRETRAINED=${LAION_PRETRAINED:-laion2b_s34b_b79k}
SIGLIP2_MODEL=${SIGLIP2_MODEL:-ViT-B-16-SigLIP2-256}
SIGLIP2_PRETRAINED=${SIGLIP2_PRETRAINED:-webli}

SEEDS=${SEEDS:-0}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
VARIANTS=${VARIANTS:-openai_vit,openclip_laion,siglip2}

OUT_CSV=${OUT_CSV:-$LOG_DIR/zeroshot_openai_openclip_siglip2_all_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("open_clip") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"])
PY

mkdir -p CLIP/clip
if [[ ! -f CLIP/clip/bpe_simple_vocab_16e6.txt.gz ]]; then
  echo "[INFO] Downloading CLIP BPE vocab for OpenAI CLIP fallback..."
  curl -L -o CLIP/clip/bpe_simple_vocab_16e6.txt.gz \
    https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "WB95: $WB95_PATH"
echo "WB100: $WB100_PATH"
echo "RedMeat: $REDMEAT_PATH"
echo "Decoy: $DECOY_PNG_ROOT"
echo "Variants: $VARIANTS"
echo "OpenAI CLIP: model=$OPENAI_MODEL"
echo "LAION: model=$LAION_MODEL pretrained=$LAION_PRETRAINED"
echo "SigLIP2: model=$SIGLIP2_MODEL pretrained=$SIGLIP2_PRETRAINED"
echo "Seeds: $SEEDS"
echo "Output CSV: $OUT_CSV"
which python

srun --unbuffered python -u run_zeroshot_openai_openclip_siglip2_all.py \
  --wb95-path "$WB95_PATH" \
  --wb100-path "$WB100_PATH" \
  --redmeat-path "$REDMEAT_PATH" \
  --decoy-png-root "$DECOY_PNG_ROOT" \
  --openai-model "$OPENAI_MODEL" \
  --laion-model "$LAION_MODEL" \
  --laion-pretrained "$LAION_PRETRAINED" \
  --siglip2-model "$SIGLIP2_MODEL" \
  --siglip2-pretrained "$SIGLIP2_PRETRAINED" \
  --variants "$VARIANTS" \
  --seeds "$SEEDS" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --output-csv "$OUT_CSV"

echo
echo "[DONE] Zero-shot run completed."
echo "CSV: $OUT_CSV"
