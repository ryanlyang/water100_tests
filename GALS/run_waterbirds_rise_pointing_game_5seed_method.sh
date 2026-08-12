#!/bin/bash -l
# Evaluate one completed Waterbirds method (seeds 0-4) with RISE Pointing Game.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

METHOD="${METHOD:?Submit with METHOD=vanilla|elrep|upweight|abn|gals|afr|r4rr}"
DATASET="${DATASET:?Submit with DATASET=95|100}"
case "$METHOD" in
  vanilla|elrep|upweight|abn|gals|afr|r4rr) ;;
  *) echo "[ERROR] Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac
case "$DATASET" in
  95|100) ;;
  *) echo "[ERROR] Unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
CHECKPOINT_RUN_ROOT="${CHECKPOINT_RUN_ROOT:-$LOG_DIR/pointing_game_5seed_cam}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
SOURCE_METHOD_DIR="$CHECKPOINT_RUN_ROOT/waterbirds_${DATASET}/$METHOD"
METHOD_DIR="$RUN_ROOT/waterbirds_${DATASET}/$METHOD"
AFR_ROOT="${AFR_ROOT:-$PROJECT_ROOT/afr}"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
SPLIT="${SPLIT:-test}"
TARGET_MODE="${TARGET_MODE:-label}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
RISE_IMAGE_BATCH_SIZE="${RISE_IMAGE_BATCH_SIZE:-4}"
RISE_MAX_MASKED_BATCH="${RISE_MAX_MASKED_BATCH:-128}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-$RUN_ROOT/rise_masks/waterbirds_gals_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_224x224.npy}"

if [[ "$SPLIT" != "test" ]]; then
  echo "[ERROR] This reproducibility runner is fixed to the Waterbirds test split." >&2
  exit 2
fi

if [[ "$DATASET" == "95" ]]; then
  DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}"
else
  DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}"
fi
MASK_ROOT="${MASK_ROOT:-/home/ryreu/guided_cnn/waterbirds/CUB_200_2011/segmentations}"

mkdir -p "$LOG_DIR" "$METHOD_DIR" "$(dirname "$RISE_MASKS_PATH")"
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

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[RUN] Waterbirds RISE Pointing Game dataset=$DATASET method=$METHOD seeds=$SEEDS_CSV"
echo "[RUN] checkpoint_source=$SOURCE_METHOD_DIR"
echo "[RUN] output=$METHOD_DIR split=$SPLIT target_mode=$TARGET_MODE"
echo "[RUN] masks=$MASK_ROOT (CUB segmentation)"
echo "[RUN] RISE N=$RISE_NUM_MASKS grid=$RISE_GRID_SIZE p1=$RISE_P1 seed=$RISE_SEED"
echo "[RUN] shared_mask_bank=$RISE_MASKS_PATH"
which python

source_manifest_is_valid() {
  local path="$1"
  local seed="$2"
  python - "$path" "$DATASET" "$METHOD" "$seed" <<'PY'
import json, os, sys
path, dataset, method, seed = sys.argv[1:]
try:
    obj = json.load(open(path, "r", encoding="utf-8"))
    checkpoints = [obj.get("checkpoint", "")]
    if method == "afr":
        checkpoints.append(obj.get("stage1_checkpoint", ""))
    valid = (
        str(obj.get("dataset")) == dataset
        and obj.get("method") == method
        and int(obj.get("seed", -1)) == int(seed)
        and all(path and os.path.isfile(path) for path in checkpoints)
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

pointing_summary_is_valid() {
  local path="$1"
  local seed="$2"
  local checkpoint="$3"
  local stage1_checkpoint="$4"
  python - "$path" "$DATASET" "$METHOD" "$seed" "$SPLIT" "$TARGET_MODE" \
    "$MASK_THRESHOLD" "$MAX_SAMPLES" "$SAMPLE_SEED" \
    "$RISE_NUM_MASKS" "$RISE_GRID_SIZE" "$RISE_P1" "$RISE_SEED" \
    "$checkpoint" "$stage1_checkpoint" <<'PY'
import csv, math, sys
(
    path, dataset, method, seed, split, target_mode, mask_threshold,
    max_samples, sample_seed, num_masks, grid_size, p1, rise_seed,
    checkpoint, stage1_checkpoint,
) = sys.argv[1:]
try:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    row = rows[0] if len(rows) == 1 else None
    valid = (
        row is not None
        and row.get("dataset") == f"waterbirds_{dataset}"
        and row.get("method") == method
        and int(row.get("seed", -1)) == int(seed)
        and row.get("split") == split
        and row.get("target_mode") == target_mode
        and row.get("explainer") == "rise"
        and row.get("primary_pg_protocol") == "rise_pixel_argmax"
        and int(row.get("mask_protocol_version", -1)) == 1
        and row.get("mask_source") == "CUB_200_2011_segmentations"
        and int(row.get("mask_threshold", -1)) == int(mask_threshold)
        and int(row.get("max_samples", -1)) == int(max_samples)
        and int(row.get("sample_seed", -1)) == int(sample_seed)
        and int(row.get("rise_num_masks", -1)) == int(num_masks)
        and int(row.get("rise_grid_size", -1)) == int(grid_size)
        and math.isclose(float(row.get("rise_p1", "nan")), float(p1))
        and int(row.get("rise_seed", -1)) == int(rise_seed)
        and row.get("checkpoint") == checkpoint
        and row.get("afr_stage1_checkpoint", "") == stage1_checkpoint
        and int(row.get("pg_total", 0)) > 0
        and int(row.get("errors", 1)) == 0
        and int(row.get("missing_images", 1)) == 0
        and int(row.get("missing_masks", 1)) == 0
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for seed_raw in "${SEEDS[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  [[ -n "$seed" ]] || continue

  SOURCE_MANIFEST="$SOURCE_METHOD_DIR/seed_${seed}/training_manifest.json"
  if ! source_manifest_is_valid "$SOURCE_MANIFEST" "$seed"; then
    echo "[ERROR] Missing or invalid completed checkpoint manifest: $SOURCE_MANIFEST" >&2
    echo "[ERROR] Finish the existing five-seed training job before running RISE." >&2
    exit 2
  fi

  mapfile -t CHECKPOINTS < <(python - "$SOURCE_MANIFEST" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj["checkpoint"])
print(obj.get("stage1_checkpoint", ""))
PY
  )
  CHECKPOINT="${CHECKPOINTS[0]}"
  STAGE1_CHECKPOINT="${CHECKPOINTS[1]:-}"
  SEED_DIR="$METHOD_DIR/seed_${seed}"
  PG_DIR="$SEED_DIR/pointing_game"
  PG_SUMMARY="$PG_DIR/pointing_game_summary.csv"
  mkdir -p "$PG_DIR"

  if pointing_summary_is_valid "$PG_SUMMARY" "$seed" "$CHECKPOINT" "$STAGE1_CHECKPOINT"; then
    echo "[RESUME] dataset=$DATASET method=$METHOD seed=$seed valid RISE result exists."
    continue
  fi

  EVAL_ARGS=(
    --dataset-tag "$DATASET"
    --data-path "$DATA_PATH"
    --mask-root "$MASK_ROOT"
    --checkpoint "$CHECKPOINT"
    --afr-root "$AFR_ROOT"
    --method "$METHOD"
    --seed "$seed"
    --split "$SPLIT"
    --target-mode "$TARGET_MODE"
    --mask-threshold "$MASK_THRESHOLD"
    --image-batch-size "$RISE_IMAGE_BATCH_SIZE"
    --max-masked-batch "$RISE_MAX_MASKED_BATCH"
    --num-workers "$NUM_WORKERS"
    --max-samples "$MAX_SAMPLES"
    --sample-seed "$SAMPLE_SEED"
    --rise-num-masks "$RISE_NUM_MASKS"
    --rise-grid-size "$RISE_GRID_SIZE"
    --rise-p1 "$RISE_P1"
    --rise-seed "$RISE_SEED"
    --rise-masks-path "$RISE_MASKS_PATH"
    --device cuda:0
    --output-dir "$PG_DIR"
  )
  if [[ "$METHOD" == "afr" ]]; then
    EVAL_ARGS+=(--afr-stage1-checkpoint "$STAGE1_CHECKPOINT")
  fi

  echo "[POINTING-RISE] dataset=$DATASET method=$METHOD seed=$seed checkpoint=$CHECKPOINT"
  python -u waterbirds_rise_pointing_game_eval.py "${EVAL_ARGS[@]}" \
    2>&1 | tee "$SEED_DIR/pointing_game_rise.log"
done

python -u summarize_waterbirds_rise_pointing_game_5seed.py \
  --method-dir "$METHOD_DIR" \
  --seeds "$SEEDS_CSV"

echo "[DONE] $METHOD_DIR/pointing_game_rise_5seed_summary.csv"
