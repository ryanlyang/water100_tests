#!/bin/bash -l
# CLIP ViT zero-shot evaluation on RedMeat.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_vit_zeroshot_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_vit_zeroshot_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV_CANDIDATES=(
  "${SCRIPT_DIR}/common_env.sh"
  "${SBATCH_SUBMIT_DIR:-}/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/Food101/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
)
COMMON_ENV=""
for candidate in "${COMMON_ENV_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    COMMON_ENV="$candidate"
    break
  fi
done
if [[ -z "$COMMON_ENV" ]]; then
  echo "[ERROR] Could not locate common_env.sh" >&2
  exit 2
fi
source "$COMMON_ENV"

redmeat_set_defaults
redmeat_activate_env
redmeat_prepare_food_layout "$DATA_ROOT" "$DATA_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONNOUSERSITE=1

REPO_ROOT="$GALS_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"

CLIP_MODEL=${CLIP_MODEL:-ViT-B/32}
SEEDS=${SEEDS:-0,1,2,3,4}
SPLITS=${SPLITS:-val,test}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda}

OUT_CSV=${OUT_CSV:-$LOG_DIR/redmeat_clip_vit_zeroshot_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing DATASET_ROOT: $DATASET_ROOT" >&2
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
echo "Data: $DATASET_ROOT"
echo "CLIP model: $CLIP_MODEL"
echo "Seeds: $SEEDS"
echo "Splits: $SPLITS"
echo "Output CSV: $OUT_CSV"
which python

srun --unbuffered python -u run_clip_zeroshot_redmeat.py \
  "$DATASET_ROOT" \
  --clip-model "$CLIP_MODEL" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seeds "$SEEDS" \
  --splits "$SPLITS" \
  --output-csv "$OUT_CSV"

echo
echo "[DONE] CLIP ViT zero-shot run finished."
echo "CSV: $OUT_CSV"
