#!/bin/bash -l
# Evaluate one deterministic CLIP baseline on one Waterbirds test set with RISE.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

METHOD="${METHOD:?Submit with METHOD=clip_zs|clip_lr}"
DATASET="${DATASET:?Submit with DATASET=95|100}"
case "$METHOD" in
  clip_zs|clip_lr) ;;
  *) echo "[ERROR] Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac
case "$DATASET" in
  95|100) ;;
  *) echo "[ERROR] Unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
AFR_ROOT="${AFR_ROOT:-$PROJECT_ROOT/afr}"
METHOD_DIR="$RUN_ROOT/waterbirds_${DATASET}/$METHOD"
SEED=0

SPLIT="${SPLIT:-test}"
TARGET_MODE="${TARGET_MODE:-label}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CLIP_MODEL="${CLIP_MODEL:-RN50}"
CLIP_FEATURE_BATCH_SIZE="${CLIP_FEATURE_BATCH_SIZE:-256}"

RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
RISE_IMAGE_BATCH_SIZE="${RISE_IMAGE_BATCH_SIZE:-4}"
RISE_MAX_MASKED_BATCH="${RISE_MAX_MASKED_BATCH:-128}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-$RUN_ROOT/rise_masks/waterbirds_gals_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_224x224.npy}"

if [[ "$DATASET" == "95" ]]; then
  DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}"
  CLIP_C="${CLIP_C:-30.481669053249504}"
else
  DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}"
  CLIP_C="${CLIP_C:-0.2515000498909345}"
fi
MASK_ROOT="${MASK_ROOT:-/home/ryreu/guided_cnn/waterbirds/CUB_200_2011/segmentations}"

PG_DIR="$METHOD_DIR/seed_0/pointing_game"
PG_SUMMARY="$PG_DIR/pointing_game_summary.csv"
mkdir -p "$LOG_DIR" "$PG_DIR" "$(dirname "$RISE_MASKS_PATH")"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$GALS_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$GALS_ROOT"

result_is_valid() {
  python - "$PG_SUMMARY" "$DATASET" "$METHOD" "$CLIP_MODEL" "$CLIP_C" \
    "$RISE_NUM_MASKS" "$RISE_GRID_SIZE" "$RISE_P1" "$RISE_SEED" <<'PY'
import csv, math, sys
path, dataset, method, model, c, n, grid, p1, rise_seed = sys.argv[1:]
try:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    row = rows[0] if len(rows) == 1 else None
    valid = (
        row is not None
        and row.get("dataset") == f"waterbirds_{dataset}"
        and row.get("method") == method
        and int(row.get("seed", -1)) == 0
        and row.get("split") == "test"
        and row.get("target_mode") == "label"
        and row.get("explainer") == "rise"
        and row.get("clip_model") == model
        and (method != "clip_lr" or math.isclose(float(row["clip_c"]), float(c)))
        and int(row.get("rise_num_masks", -1)) == int(n)
        and int(row.get("rise_grid_size", -1)) == int(grid)
        and math.isclose(float(row.get("rise_p1", "nan")), float(p1))
        and int(row.get("rise_seed", -1)) == int(rise_seed)
        and int(row.get("pg_total", 0)) > 0
        and int(row.get("errors", 1)) == 0
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

if result_is_valid; then
  echo "[RESUME] Valid deterministic result already exists: $PG_SUMMARY"
else
  echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
  echo "[RUN] dataset=$DATASET method=$METHOD model=$CLIP_MODEL seed=$SEED"
  if [[ "$METHOD" == "clip_lr" ]]; then
    echo "[RUN] CLIP-LR C=$CLIP_C penalty=l2 solver=lbfgs fit_intercept=true"
  else
    echo "[RUN] CLIP-ZS class prompts=8-template Waterbirds ensemble"
  fi
  echo "[RUN] RISE N=$RISE_NUM_MASKS grid=$RISE_GRID_SIZE p1=$RISE_P1 seed=$RISE_SEED"
  which python

  python -u waterbirds_rise_pointing_game_eval.py \
    --dataset-tag "$DATASET" \
    --data-path "$DATA_PATH" \
    --mask-root "$MASK_ROOT" \
    --afr-root "$AFR_ROOT" \
    --method "$METHOD" \
    --seed "$SEED" \
    --clip-model "$CLIP_MODEL" \
    --clip-c "$CLIP_C" \
    --clip-feature-batch-size "$CLIP_FEATURE_BATCH_SIZE" \
    --split "$SPLIT" \
    --target-mode "$TARGET_MODE" \
    --mask-threshold "$MASK_THRESHOLD" \
    --image-batch-size "$RISE_IMAGE_BATCH_SIZE" \
    --max-masked-batch "$RISE_MAX_MASKED_BATCH" \
    --num-workers "$NUM_WORKERS" \
    --max-samples "$MAX_SAMPLES" \
    --sample-seed "$SAMPLE_SEED" \
    --rise-num-masks "$RISE_NUM_MASKS" \
    --rise-grid-size "$RISE_GRID_SIZE" \
    --rise-p1 "$RISE_P1" \
    --rise-seed "$RISE_SEED" \
    --rise-masks-path "$RISE_MASKS_PATH" \
    --device cuda:0 \
    --output-dir "$PG_DIR" \
    2>&1 | tee "$METHOD_DIR/seed_0/pointing_game_rise.log"
fi

python -u summarize_waterbirds_rise_pointing_game_5seed.py \
  --method-dir "$METHOD_DIR" \
  --seeds 0

echo "[DONE] $METHOD_DIR/pointing_game_rise_5seed_summary.csv"
