#!/usr/bin/env bash
# One resumable method/seed/variant ImageNet-9 RISE Pointing Game job.
#SBATCH --job-name=in9pg95_rise
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

METHOD="${METHOD:?Submit with METHOD=erm|upweight|abn|elrep|gals|afr|clip_lr|r4rr}"
SEED="${SEED:?Submit with SEED=0|1|2|3|4}"
VARIANT="${VARIANT:-original}"
case "$METHOD" in
  erm|upweight|abn|elrep|gals|afr|clip_lr|r4rr) ;;
  *) echo "[ERROR] Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac
case "$SEED" in 0|1|2|3|4) ;; *) echo "[ERROR] Unsupported SEED=$SEED" >&2; exit 2 ;; esac
case "$VARIANT" in
  original|mixed_same|mixed_rand|mixed_next) ;;
  *) echo "[ERROR] Unsupported VARIANT=$VARIANT" >&2; exit 2 ;;
esac

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL="${PROTOCOL:-reconstructed_original_bbox1_v1}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
SOURCE_ROOT="${SOURCE_ROOT:-${LOG_ROOT}/transfer/waterbirds95}"
RUN_ROOT="${RUN_ROOT:-${LOG_ROOT}/pointing_game_rise_wb95_transfer}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-${DATA_ROOT}/official_test/bg_challenge}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL}/official_test_manifest.csv}"
MASK_ROOT="${MASK_ROOT:-${OFFICIAL_ROOT}/fg_mask/val}"
EVALUATION_JSON="${EVALUATION_JSON:-${SOURCE_ROOT}/${METHOD}/main/seed_${SEED}/official_evaluation.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/${METHOD}/${VARIANT}/seed_${SEED}}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-0}"

RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-4}"
MAX_MASKED_BATCH="${MAX_MASKED_BATCH:-128}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-${RUN_ROOT}/rise_masks/in9_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_224x224.npy}"

mkdir -p "$LOG_ROOT" "$OUTPUT_DIR" "$(dirname "$RISE_MASKS_PATH")"
cd "$REPO"
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gals_a100
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

[[ -f "$EVALUATION_JSON" ]] || { echo "[ERROR] Missing source evaluation: $EVALUATION_JSON" >&2; exit 2; }
[[ -f "$OFFICIAL_MANIFEST" ]] || { echo "[ERROR] Missing official manifest: $OFFICIAL_MANIFEST" >&2; exit 2; }
[[ -d "$MASK_ROOT" ]] || { echo "[ERROR] Missing official masks: $MASK_ROOT" >&2; exit 2; }

echo "[$(date)] Host: $(hostname)"
echo "Experiment: WB95-transfer ImageNet-9 RISE Pointing Game"
echo "Method=$METHOD seed=$SEED variant=$VARIANT target=ground-truth-label"
echo "Source evaluation: $EVALUATION_JSON"
echo "Official masks: $MASK_ROOT"
echo "Output: $OUTPUT_DIR"
echo "RISE: N=$RISE_NUM_MASKS grid=$RISE_GRID_SIZE p1=$RISE_P1 seed=$RISE_SEED"
echo "Resume granularity: one completed image batch"
"$PYTHON_BIN" --version

srun --unbuffered "$PYTHON_BIN" -u \
  ImageNet9_Runs/evaluate_imagenet9_wb95_transfer_rise.py \
  --method "$METHOD" \
  --seed "$SEED" \
  --variant "$VARIANT" \
  --evaluation-json "$EVALUATION_JSON" \
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
  --device cuda:0

echo "[$(date)] RISE Pointing Game complete: $OUTPUT_DIR"
