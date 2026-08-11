#!/bin/bash -l
# Evaluate one completed DecoyMNIST method for seeds 0-4 with RISE Pointing Game.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

METHOD="${METHOD:?Submit with METHOD=vanilla|elrep|upweight|abn|gals|afr|r4rr}"
case "$METHOD" in
  vanilla|elrep|upweight|abn|gals|afr|r4rr) ;;
  *) echo "[ERROR] Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
CHECKPOINT_RUN_ROOT="${CHECKPOINT_RUN_ROOT:-$LOG_DIR/pointing_game_5seed_gradcam}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
SOURCE_METHOD_DIR="$CHECKPOINT_RUN_ROOT/$METHOD"
METHOD_DIR="$RUN_ROOT/$METHOD"

PNG_ROOT="${PNG_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"
MNIST_ROOT="${MNIST_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data}"
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
RISE_IMAGE_BATCH_SIZE="${RISE_IMAGE_BATCH_SIZE:-16}"
RISE_MAX_MASKED_BATCH="${RISE_MAX_MASKED_BATCH:-8192}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-$RUN_ROOT/rise_masks/decoymnist_gals_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_28x28.npy}"

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
echo "[RUN] DecoyMNIST RISE Pointing Game method=$METHOD seeds=$SEEDS_CSV"
echo "[RUN] checkpoint_source=$SOURCE_METHOD_DIR"
echo "[RUN] output=$METHOD_DIR split=$SPLIT target_mode=$TARGET_MODE"
echo "[RUN] RISE N=$RISE_NUM_MASKS grid=$RISE_GRID_SIZE p1=$RISE_P1 seed=$RISE_SEED"
echo "[RUN] shared_mask_bank=$RISE_MASKS_PATH"
which python

source_manifest_is_valid() {
  local path="$1"
  local seed="$2"
  python - "$path" "$METHOD" "$seed" <<'PY'
import json, os, sys
path, method, seed = sys.argv[1:]
try:
    obj = json.load(open(path, "r", encoding="utf-8"))
    valid = (
        obj.get("dataset") == "decoymnist"
        and obj.get("method") == method
        and int(obj.get("seed", -1)) == int(seed)
        and os.path.isfile(obj.get("checkpoint", ""))
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
  python - "$path" "$METHOD" "$seed" "$SPLIT" "$TARGET_MODE" \
    "$MASK_THRESHOLD" "$MAX_SAMPLES" "$SAMPLE_SEED" \
    "$RISE_NUM_MASKS" "$RISE_GRID_SIZE" "$RISE_P1" "$RISE_SEED" "$checkpoint" <<'PY'
import csv, math, sys
(
    path, method, seed, split, target_mode, mask_threshold, max_samples,
    sample_seed, num_masks, grid_size, p1, rise_seed, checkpoint,
) = sys.argv[1:]
try:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    row = rows[0] if len(rows) == 1 else None
    valid = (
        row is not None
        and row.get("dataset") == "decoymnist"
        and row.get("method") == method
        and int(row.get("seed", -1)) == int(seed)
        and row.get("split") == split
        and row.get("target_mode") == target_mode
        and row.get("explainer") == "rise"
        and row.get("primary_pg_protocol") == "rise_pixel_argmax"
        and int(row.get("mask_protocol_version", -1)) == 1
        and int(row.get("mask_threshold", -1)) == int(mask_threshold)
        and int(row.get("max_samples", -1)) == int(max_samples)
        and int(row.get("sample_seed", -1)) == int(sample_seed)
        and int(row.get("rise_num_masks", -1)) == int(num_masks)
        and int(row.get("rise_grid_size", -1)) == int(grid_size)
        and math.isclose(float(row.get("rise_p1", "nan")), float(p1))
        and int(row.get("rise_seed", -1)) == int(rise_seed)
        and row.get("checkpoint") == checkpoint
        and int(row.get("pg_total", 0)) > 0
        and int(row.get("errors", 1)) == 0
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
    echo "[ERROR] Finish the existing DecoyMNIST training job before running this evaluator." >&2
    exit 2
  fi

  checkpoint="$(python - "$SOURCE_MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], "r", encoding="utf-8"))["checkpoint"])
PY
)"

  SEED_DIR="$METHOD_DIR/seed_${seed}"
  PG_DIR="$SEED_DIR/pointing_game"
  PG_SUMMARY="$PG_DIR/pointing_game_summary.csv"
  mkdir -p "$PG_DIR"

  if pointing_summary_is_valid "$PG_SUMMARY" "$seed" "$checkpoint"; then
    echo "[RESUME] method=$METHOD seed=$seed valid RISE result exists."
    continue
  fi

  echo "[POINTING-RISE] method=$METHOD seed=$seed checkpoint=$checkpoint"
  python -u decoymnist_rise_pointing_game_eval.py \
    --png-root "$PNG_ROOT" \
    --mnist-root "$MNIST_ROOT" \
    --checkpoint "$checkpoint" \
    --method "$METHOD" \
    --seed "$seed" \
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
    2>&1 | tee "$SEED_DIR/pointing_game.log"
done

python -u summarize_decoymnist_rise_pointing_game_5seed.py \
  --method-dir "$METHOD_DIR" \
  --seeds "$SEEDS_CSV"

echo "[DONE] $METHOD_DIR/pointing_game_5seed_summary.csv"
