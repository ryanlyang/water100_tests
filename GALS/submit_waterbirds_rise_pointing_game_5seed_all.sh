#!/bin/bash
# Submit 14 evaluation-only jobs: seven methods x two Waterbirds datasets.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/run_waterbirds_rise_pointing_game_5seed_method.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
CHECKPOINT_RUN_ROOT="${CHECKPOINT_RUN_ROOT:-$LOG_DIR/pointing_game_5seed_cam}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
SPLIT="test"
TARGET_MODE="label"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
RISE_IMAGE_BATCH_SIZE="${RISE_IMAGE_BATCH_SIZE:-4}"
RISE_MAX_MASKED_BATCH="${RISE_MAX_MASKED_BATCH:-128}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-$RUN_ROOT/rise_masks/waterbirds_gals_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_224x224.npy}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT"

METHODS=(vanilla elrep upweight abn gals afr r4rr)
DATASETS=(95 100)

if [[ "$DRY_RUN" != "1" ]]; then
  missing=0
  for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
      for seed in 0 1 2 3 4; do
        manifest="$CHECKPOINT_RUN_ROOT/waterbirds_${dataset}/$method/seed_${seed}/training_manifest.json"
        if ! python - "$manifest" "$dataset" "$method" "$seed" <<'PY'
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
        and all(value and os.path.isfile(value) for value in checkpoints)
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
        then
          echo "[MISSING] $manifest" >&2
          missing=1
        fi
      done
    done
  done
  if [[ "$missing" == "1" ]]; then
    echo "[ERROR] RISE submission aborted before queuing jobs: some source checkpoints are incomplete." >&2
    exit 2
  fi

  set +u
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV_NAME:-gals_a100}"
  set -u
  PYTHONNOUSERSITE=1 python "$SCRIPT_DIR/prepare_gals_rise_mask_bank.py" \
    --output "$RISE_MASKS_PATH" \
    --num-masks "$RISE_NUM_MASKS" \
    --grid-size "$RISE_GRID_SIZE" \
    --height 224 \
    --width 224 \
    --p1 "$RISE_P1" \
    --seed "$RISE_SEED"
fi

export LOG_DIR CHECKPOINT_RUN_ROOT RUN_ROOT SEEDS_CSV SPLIT TARGET_MODE
export MASK_THRESHOLD MAX_SAMPLES SAMPLE_SEED
export RISE_NUM_MASKS RISE_GRID_SIZE RISE_P1 RISE_SEED
export RISE_IMAGE_BATCH_SIZE RISE_MAX_MASKED_BATCH RISE_MASKS_PATH

JOB_FILE="$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
printf 'dataset,method,job_id\n' > "$JOB_FILE"

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    job_name="pgr5_wb${dataset}_${method}"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY RUN] sbatch --job-name=$job_name --export=ALL,DATASET=$dataset,METHOD=$method $WORKER"
      job_id="DRY_RUN"
    else
      job_id="$(sbatch --parsable \
        --job-name="$job_name" \
        --export="ALL,DATASET=$dataset,METHOD=$method" \
        "$WORKER")"
      echo "[SUBMITTED] dataset=$dataset method=$method job=$job_id"
    fi
    printf '%s,%s,%s\n' "$dataset" "$method" "$job_id" >> "$JOB_FILE"
  done
done

echo
echo "[DONE] submission record: $JOB_FILE"
echo "[DONE] RISE result root: $RUN_ROOT"
echo "[INFO] These jobs reuse completed checkpoints; they do not retrain models."
echo "[INFO] After all jobs finish:"
echo "  cd $SCRIPT_DIR"
echo "  python summarize_waterbirds_rise_pointing_game_5seed.py --run-root $RUN_ROOT --seeds $SEEDS_CSV"
