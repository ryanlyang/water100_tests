#!/bin/bash -l
# CLIP ViT zero-shot evaluation on Waterbirds-95 and Waterbirds-100.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/clip_vit_zeroshot_wb95_wb100_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/clip_vit_zeroshot_wb95_wb100_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird
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

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
WB95_PATH=/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2
WB100_PATH=/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2

CLIP_MODEL=${CLIP_MODEL:-ViT-B/32}
SEEDS=${SEEDS:-0,1,2,3,4}
SPLITS=${SPLITS:-val,test}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda}

OUT_WB95=${OUT_WB95:-$LOG_DIR/clip_vit_zeroshot_wb95_${SLURM_JOB_ID}.csv}
OUT_WB100=${OUT_WB100:-$LOG_DIR/clip_vit_zeroshot_wb100_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$WB95_PATH" ]]; then
  echo "Missing WB95_PATH: $WB95_PATH" >&2
  exit 1
fi
if [[ ! -d "$WB100_PATH" ]]; then
  echo "Missing WB100_PATH: $WB100_PATH" >&2
  exit 1
fi

mkdir -p CLIP/clip
if [[ ! -f CLIP/clip/bpe_simple_vocab_16e6.txt.gz ]]; then
  echo "[INFO] Downloading CLIP BPE vocab..."
  curl -L -o CLIP/clip/bpe_simple_vocab_16e6.txt.gz \
    https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "CLIP model: $CLIP_MODEL"
echo "Seeds: $SEEDS"
echo "Splits: $SPLITS"
echo "WB95: $WB95_PATH"
echo "WB100: $WB100_PATH"
echo "Out WB95: $OUT_WB95"
echo "Out WB100: $OUT_WB100"
which python

echo
echo "===== [1/2] CLIP ViT Zero-shot: Waterbirds-95 ====="
srun --unbuffered python -u run_clip_zeroshot_waterbirds.py \
  "$WB95_PATH" \
  --clip-model "$CLIP_MODEL" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seeds "$SEEDS" \
  --splits "$SPLITS" \
  --output-csv "$OUT_WB95"

echo
echo "===== [2/2] CLIP ViT Zero-shot: Waterbirds-100 ====="
srun --unbuffered python -u run_clip_zeroshot_waterbirds.py \
  "$WB100_PATH" \
  --clip-model "$CLIP_MODEL" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seeds "$SEEDS" \
  --splits "$SPLITS" \
  --output-csv "$OUT_WB100"

echo
echo "[DONE] CLIP ViT zero-shot runs finished."
echo "WB95 CSV:  $OUT_WB95"
echo "WB100 CSV: $OUT_WB100"

