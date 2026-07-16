#!/bin/bash -l
# CLIP ViT zero-shot evaluation on DecoyMNIST.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_clip_vit_zeroshot_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_clip_vit_zeroshot_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}
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
PNG_ROOT=${PNG_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}

CLIP_MODEL=${CLIP_MODEL:-ViT-B/32}
SEEDS=${SEEDS:-0,1,2,3,4}
SPLITS=${SPLITS:-val,test}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
VAL_FRAC=${VAL_FRAC:-0.10}
SPLIT_SEED=${SPLIT_SEED:-0}
DEVICE=${DEVICE:-cuda}

OUT_CSV=${OUT_CSV:-$LOG_DIR/decoy_clip_vit_zeroshot_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$PNG_ROOT/train" || ! -d "$PNG_ROOT/test" ]]; then
  echo "[ERROR] Missing DecoyMNIST PNG folders under: $PNG_ROOT" >&2
  exit 2
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
echo "PNG root: $PNG_ROOT"
echo "CLIP model: $CLIP_MODEL"
echo "Seeds: $SEEDS"
echo "Splits: $SPLITS"
echo "Output CSV: $OUT_CSV"
which python

srun --unbuffered python -u run_decoymnist_clip_vit_zeroshot.py \
  --png-root "$PNG_ROOT" \
  --clip-model "$CLIP_MODEL" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --val-frac "$VAL_FRAC" \
  --split-seed "$SPLIT_SEED" \
  --seeds "$SEEDS" \
  --splits "$SPLITS" \
  --output-csv "$OUT_CSV"

echo
echo "[DONE] DecoyMNIST CLIP ViT zero-shot run finished."
echo "CSV: $OUT_CSV"
