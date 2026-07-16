#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_rn50_attention_gen_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_rn50_attention_gen_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

# -------- Paths / env (override via --export=ALL,VAR=...) --------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${SBATCH_SUBMIT_DIR:-${PWD:-}}}"
REPO_ROOT_INPUT="${REPO_ROOT:-}"

# NOTE: Under Slurm, BASH_SOURCE may point to a spool copy path.
# Resolve repo root from submit dir first, then fall back.
REPO_ROOT_CANDIDATES=(
  "$REPO_ROOT_INPUT"
  "$SUBMIT_DIR/GALS"
  "$SUBMIT_DIR"
  "$SCRIPT_DIR"
)
REPO_ROOT=""
for candidate in "${REPO_ROOT_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate/run_generate_decoymnist_rn50_attentions.py" ]]; then
    REPO_ROOT="$(cd -- "$candidate" && pwd)"
    break
  fi
done
if [[ -z "$REPO_ROOT" ]]; then
  echo "[ERROR] Could not locate GALS repo root containing run_generate_decoymnist_rn50_attentions.py" >&2
  echo "Checked: ${REPO_ROOT_CANDIDATES[*]}" >&2
  exit 2
fi

PNG_ROOT="${PNG_ROOT:-$REPO_ROOT/../MakeMNIST/data/DecoyMNIST_png}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PNG_ROOT/clip_rn50_attention_gradcam}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
CONDA_ENV="${CONDA_ENV:-gals_a100}"
CLIP_DIR="${CLIP_DIR:-$REPO_ROOT/CLIP}"

SPLITS="${SPLITS:-train,test}"
CHUNK_SIZE="${CHUNK_SIZE:-0}"    # 0 => run all at once
START_IDX="${START_IDX:-0}"      # used when CHUNK_SIZE=0
END_IDX="${END_IDX:--1}"         # used when CHUNK_SIZE=0; -1 => all
SEED="${SEED:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DISABLE_VIS="${DISABLE_VIS:-1}"
USE_PROMPTS_PER_CLASS="${USE_PROMPTS_PER_CLASS:-1}"
TARGET_LAYER="${TARGET_LAYER:-layer4.2.relu}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-25}"

mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "$REPO_ROOT"

if [[ ! -d "$PNG_ROOT" ]]; then
  echo "[ERROR] PNG_ROOT not found: $PNG_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

# CLIP tokenizer file helper (same pattern as other GALS scripts).
mkdir -p "$CLIP_DIR/clip"
if [[ ! -f "$CLIP_DIR/clip/bpe_simple_vocab_16e6.txt.gz" ]]; then
  echo "[INFO] Downloading CLIP BPE vocab..."
  curl -L -o "$CLIP_DIR/clip/bpe_simple_vocab_16e6.txt.gz" \
    https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
echo "Repo: $REPO_ROOT"
echo "CLIP_DIR: $CLIP_DIR"
echo "PNG_ROOT: $PNG_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "SPLITS: $SPLITS"
echo "CHUNK_SIZE: $CHUNK_SIZE"
echo "CONDA_ENV: $CONDA_ENV"
which python

COMMON_ARGS=(
  --png-root "$PNG_ROOT"
  --output-root "$OUTPUT_ROOT"
  --splits "$SPLITS"
  --seed "$SEED"
  --target-layer "$TARGET_LAYER"
  --empty-cache-every "$EMPTY_CACHE_EVERY"
)
if [[ "$SKIP_EXISTING" == "1" ]]; then
  COMMON_ARGS+=(--skip-existing)
else
  COMMON_ARGS+=(--no-skip-existing)
fi
if [[ "$DISABLE_VIS" == "1" ]]; then
  COMMON_ARGS+=(--disable-vis)
else
  COMMON_ARGS+=(--enable-vis)
fi
if [[ "$USE_PROMPTS_PER_CLASS" == "1" ]]; then
  COMMON_ARGS+=(--use-prompts-per-class)
else
  COMMON_ARGS+=(--no-use-prompts-per-class)
fi

if [[ "$CHUNK_SIZE" -gt 0 ]]; then
  # Count files in requested splits for chunking.
  TOTAL=0
  IFS=',' read -r -a _splits <<< "$SPLITS"
  for sp in "${_splits[@]}"; do
    sp="$(echo "$sp" | xargs)"
    [[ -z "$sp" ]] && continue
    c=$(find "$PNG_ROOT/$sp" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
    TOTAL=$((TOTAL + c))
  done
  echo "[INFO] total images across splits: $TOTAL"
  for ((s=0; s<TOTAL; s+=CHUNK_SIZE)); do
    e=$((s + CHUNK_SIZE))
    echo "[RUN] chunk START_IDX=$s END_IDX=$e"
    srun --unbuffered python -u run_generate_decoymnist_rn50_attentions.py \
      "${COMMON_ARGS[@]}" \
      --start-idx "$s" \
      --end-idx "$e"
  done
else
  echo "[RUN] single pass START_IDX=$START_IDX END_IDX=$END_IDX"
  srun --unbuffered python -u run_generate_decoymnist_rn50_attentions.py \
    "${COMMON_ARGS[@]}" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX"
fi

echo "[DONE] DecoyMNIST RN50 attention generation complete."
