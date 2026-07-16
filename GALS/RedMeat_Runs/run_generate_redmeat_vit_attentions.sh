#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_vit_attention_gen_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_vit_attention_gen_%j.err
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

REPO_ROOT="$GALS_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"
META_CSV="$DATASET_ROOT/all_images.csv"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[ERROR] Missing GALS root: $REPO_ROOT" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing dataset root: $DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -f "$META_CSV" ]]; then
  echo "[ERROR] Missing metadata CSV: $META_CSV" >&2
  exit 2
fi

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p CLIP/clip
if [[ ! -f CLIP/clip/bpe_simple_vocab_16e6.txt.gz ]]; then
  echo "[INFO] Downloading CLIP BPE vocab..."
  curl -L -o CLIP/clip/bpe_simple_vocab_16e6.txt.gz \
    https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

CHUNK_SIZE=${CHUNK_SIZE:-500}
TOTAL_IMAGES=${TOTAL_IMAGES:-$(( $(wc -l < "$META_CSV") - 1 ))}
if [[ "$TOTAL_IMAGES" -le 0 ]]; then
  echo "[ERROR] Could not infer TOTAL_IMAGES from $META_CSV" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data root: $DATA_ROOT"
echo "Dataset dir: $DATA_DIR"
echo "Dataset root: $DATASET_ROOT"
echo "Total images: $TOTAL_IMAGES"
echo "Chunk size: $CHUNK_SIZE"
which python

echo "[GEN] Generating CLIP ViT attentions for RedMeat..."
for ((start=0; start<TOTAL_IMAGES; start+=CHUNK_SIZE)); do
  end=$((start + CHUNK_SIZE))
  echo "[GEN] chunk START_IDX=$start END_IDX=$end"
  srun --unbuffered python -u extract_attention.py \
    --config RedMeat_Runs/configs/redmeat_attention_vit.yaml \
    DATA.ROOT="$DATA_ROOT" \
    DATA.FOOD_SUBSET_DIR="$DATA_DIR" \
    SAVE_FOLDER=clip_vit_attention \
    DISABLE_VIS=true \
    SKIP_EXISTING=true \
    START_IDX="$start" \
    END_IDX="$end"
done

echo "[GEN] Done."
