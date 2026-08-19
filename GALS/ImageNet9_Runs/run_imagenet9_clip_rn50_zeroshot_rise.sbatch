#!/usr/bin/env bash
# Deterministic OpenAI CLIP RN50 zero-shot RISE Pointing Game for one variant.
#SBATCH --job-name=in9pg_zs_rn50
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

VARIANT="${VARIANT:?Submit with original|mixed_same|mixed_rand|mixed_next}"
case "$VARIANT" in
  original|mixed_same|mixed_rand|mixed_next) ;;
  *) echo "[ERROR] Unsupported VARIANT=$VARIANT" >&2; exit 2 ;;
esac

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL="${PROTOCOL:-reconstructed_original_bbox1_v1}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUN_ROOT="${RUN_ROOT:-${LOG_ROOT}/pointing_game_rise_clip_rn50_zeroshot}"
TRAINED_RISE_ROOT="${TRAINED_RISE_ROOT:-${LOG_ROOT}/pointing_game_rise_wb95_transfer}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-${DATA_ROOT}/official_test/bg_challenge}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL}/official_test_manifest.csv}"
MASK_ROOT="${MASK_ROOT:-${OFFICIAL_ROOT}/fg_mask/val}"
SOURCE_CONTRACT="${SOURCE_CONTRACT:-${LOG_ROOT}/clip_rn50_zeroshot_openai/evaluation_contract.json}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/home/ryreu/guided_cnn/models/openai_clip}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/clip_zs_rn50/${VARIANT}/seed_0}"
CONDA_ENV="${CONDA_ENV:-r4rr-weclip}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/${CONDA_ENV}/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-0}"

RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-4}"
MAX_MASKED_BATCH="${MAX_MASKED_BATCH:-128}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-${TRAINED_RISE_ROOT}/rise_masks/in9_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_224x224.npy}"

mkdir -p "$LOG_ROOT" "$OUTPUT_DIR" "$DOWNLOAD_ROOT" "$(dirname "$RISE_MASKS_PATH")"
cd "$REPO"
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

for path in "$SOURCE_CONTRACT" "$OFFICIAL_MANIFEST"; do
  [[ -f "$path" ]] || { echo "[ERROR] Missing file: $path" >&2; exit 2; }
done
[[ -d "$OFFICIAL_ROOT" ]] || { echo "[ERROR] Missing official root: $OFFICIAL_ROOT" >&2; exit 2; }
[[ -d "$MASK_ROOT" ]] || { echo "[ERROR] Missing foreground masks: $MASK_ROOT" >&2; exit 2; }

echo "[$(date)] Host: $(hostname)"
echo "Experiment: OpenAI CLIP RN50 zero-shot RISE Pointing Game"
echo "Variant=$VARIANT target=ground-truth-label deterministic_seed=0"
echo "Source classification contract: $SOURCE_CONTRACT"
echo "Output: $OUTPUT_DIR"
echo "RISE: N=$RISE_NUM_MASKS grid=$RISE_GRID_SIZE p1=$RISE_P1 seed=$RISE_SEED"
echo "RISE mask bank shared with trained-method evaluation: $RISE_MASKS_PATH"
"$PYTHON_BIN" --version

srun --unbuffered "$PYTHON_BIN" -u \
  ImageNet9_Runs/evaluate_imagenet9_wb95_transfer_rise.py \
  --method clip_zs_rn50 \
  --seed 0 \
  --variant "$VARIANT" \
  --evaluation-json "$SOURCE_CONTRACT" \
  --official-manifest "$OFFICIAL_MANIFEST" \
  --official-test-root "$OFFICIAL_ROOT" \
  --mask-root "$MASK_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --target-mode label \
  --image-batch-size "$IMAGE_BATCH_SIZE" \
  --max-masked-batch "$MAX_MASKED_BATCH" \
  --num-workers "$NUM_WORKERS" \
  --rise-num-masks "$RISE_NUM_MASKS" \
  --rise-grid-size "$RISE_GRID_SIZE" \
  --rise-p1 "$RISE_P1" \
  --rise-seed "$RISE_SEED" \
  --rise-masks-path "$RISE_MASKS_PATH" \
  --download-root "$DOWNLOAD_ROOT" \
  --device cuda:0

echo "[$(date)] CLIP RN50 zero-shot RISE evaluation complete: $VARIANT"
