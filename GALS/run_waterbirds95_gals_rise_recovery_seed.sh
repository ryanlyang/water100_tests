#!/bin/bash
# Train one Waterbirds-95 GALS seed, evaluate it with RISE, then discard the
# temporary checkpoint only after the per-seed result has been validated.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

SEED="${SEED:?Submit with SEED=0|1|2|3|4}"
case "$SEED" in
  0|1|2|3|4) ;;
  *) echo "[ERROR] SEED must be one of 0,1,2,3,4 (got $SEED)" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"

# Keep temporary training artifacts separate from the original CAM campaign.
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$LOG_DIR/pointing_game_recovery_sources_wb95_gals}"
RISE_RUN_ROOT="${RISE_RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
SOURCE_SEED_DIR="$SOURCE_RUN_ROOT/waterbirds_95/gals/seed_${SEED}"
SOURCE_MANIFEST="$SOURCE_SEED_DIR/training_manifest.json"
RISE_SEED_DIR="$RISE_RUN_ROOT/waterbirds_95/gals/seed_${SEED}"
RISE_SUMMARY="$RISE_SEED_DIR/pointing_game/pointing_game_summary.csv"

RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-$RISE_RUN_ROOT/rise_masks/waterbirds_gals_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_224x224.npy}"
DELETE_CHECKPOINT_AFTER_RISE="${DELETE_CHECKPOINT_AFTER_RISE:-1}"

TRAIN_WORKER="$GALS_ROOT/run_waterbirds_pointing_game_5seed_method.sh"
RISE_WORKER="$GALS_ROOT/run_waterbirds_rise_pointing_game_5seed_method.sh"
[[ -f "$TRAIN_WORKER" ]] || { echo "[ERROR] Missing $TRAIN_WORKER" >&2; exit 2; }
[[ -f "$RISE_WORKER" ]] || { echo "[ERROR] Missing $RISE_WORKER" >&2; exit 2; }

mkdir -p "$LOG_DIR" "$SOURCE_SEED_DIR" "$RISE_SEED_DIR"

result_is_valid() {
  python - "$RISE_SUMMARY" "$SEED" <<'PY'
import csv
import sys

path, seed = sys.argv[1:]
try:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    row = rows[0] if len(rows) == 1 else None
    valid = (
        row is not None
        and row.get("dataset") == "waterbirds_95"
        and row.get("method") == "gals"
        and int(row.get("seed", -1)) == int(seed)
        and row.get("split") == "test"
        and row.get("explainer") == "rise"
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

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[RECOVERY] dataset=95 method=gals seed=$SEED"
echo "[RECOVERY] temporary_source=$SOURCE_SEED_DIR"
echo "[RECOVERY] rise_output=$RISE_SEED_DIR"

# A completed RISE CSV is sufficient even if its original checkpoint was
# intentionally removed. This makes resubmission safe after cleanup.
if result_is_valid; then
  echo "[RESUME] Valid RISE result already exists for seed=$SEED; nothing to do."
  exit 0
fi

DATASET=95 \
METHOD=gals \
SEEDS_CSV="$SEED" \
TRAIN_ONLY=1 \
RUN_ROOT="$SOURCE_RUN_ROOT" \
LOG_DIR="$LOG_DIR" \
bash "$TRAIN_WORKER"

if [[ ! -s "$SOURCE_MANIFEST" ]]; then
  echo "[ERROR] Training did not produce $SOURCE_MANIFEST" >&2
  exit 2
fi

DATASET=95 \
METHOD=gals \
SEEDS_CSV="$SEED" \
CHECKPOINT_RUN_ROOT="$SOURCE_RUN_ROOT" \
RUN_ROOT="$RISE_RUN_ROOT" \
LOG_DIR="$LOG_DIR" \
NUM_WORKERS=0 \
RISE_NUM_MASKS="$RISE_NUM_MASKS" \
RISE_GRID_SIZE="$RISE_GRID_SIZE" \
RISE_P1="$RISE_P1" \
RISE_SEED="$RISE_SEED" \
RISE_MASKS_PATH="$RISE_MASKS_PATH" \
bash "$RISE_WORKER"

if ! result_is_valid; then
  echo "[ERROR] RISE evaluation did not produce a valid result for seed=$SEED" >&2
  exit 2
fi

# Preserve the training metadata next to the permanent result before removing
# the large model file. The manifest intentionally becomes metadata-only.
cp "$SOURCE_MANIFEST" "$RISE_SEED_DIR/source_training_manifest.json"

if [[ "$DELETE_CHECKPOINT_AFTER_RISE" == "1" ]]; then
  checkpoint="$(python - "$SOURCE_MANIFEST" "$SEED" <<'PY'
import json
import sys

path, seed = sys.argv[1:]
obj = json.load(open(path, "r", encoding="utf-8"))
if obj.get("dataset") != "95" or obj.get("method") != "gals" or int(obj.get("seed", -1)) != int(seed):
    raise SystemExit("Refusing cleanup: manifest identity mismatch")
print(obj["checkpoint"])
PY
)"

  checkpoint_real="$(readlink -f "$checkpoint")"
  gals_real="$(readlink -f "$GALS_ROOT")"
  source_real="$(readlink -f "$SOURCE_RUN_ROOT")"
  case "$checkpoint_real" in
    "$gals_real"/*|"$source_real"/*) ;;
    *)
      echo "[ERROR] Refusing to remove checkpoint outside expected roots: $checkpoint_real" >&2
      exit 2
      ;;
  esac
  if [[ -f "$checkpoint_real" ]]; then
    rm -f "$checkpoint_real"
    echo "[CLEANUP] Removed temporary checkpoint: $checkpoint_real"
  else
    echo "[CLEANUP] Checkpoint was already absent: $checkpoint_real"
  fi
fi

echo "[DONE] seed=$SEED result=$RISE_SUMMARY"
