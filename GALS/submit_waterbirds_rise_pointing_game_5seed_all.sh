#!/bin/bash
# Submit 14 evaluation-only jobs: seven methods x two Waterbirds datasets.

set -Eeuo pipefail

# The worker activates its own Conda environment. Prevent inherited SBATCH
# settings from asking Slurm to reconstruct the submit host's login environment,
# which can leave jobs held with "user env retrieval failed" before bash starts.
unset SBATCH_GET_USER_ENV SBATCH_EXPORT SBATCH_EXPORT_FILE
export SLURM_EXPORT_ENV=ALL

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
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT"

METHODS=(vanilla elrep upweight abn gals afr r4rr)
DATASETS=(95 100)

source_pair_is_ready() {
  local dataset="$1"
  local method="$2"
  local report_missing="${3:-0}"
  local ready=1
  local seed manifest
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
      if [[ "$report_missing" == "1" ]]; then
        echo "[MISSING] $manifest" >&2
      fi
      ready=0
    fi
  done
  [[ "$ready" == "1" ]]
}

result_pair_is_complete() {
  local dataset="$1"
  local method="$2"
  python - "$RUN_ROOT" "$CHECKPOINT_RUN_ROOT" "$dataset" "$method" \
    "$TARGET_MODE" "$MASK_THRESHOLD" "$MAX_SAMPLES" "$SAMPLE_SEED" \
    "$RISE_NUM_MASKS" "$RISE_GRID_SIZE" "$RISE_P1" "$RISE_SEED" <<'PY'
import csv, json, math, os, sys
(
    run_root, checkpoint_root, dataset, method, target_mode, mask_threshold,
    max_samples, sample_seed, num_masks, grid_size, p1, rise_seed,
) = sys.argv[1:]
valid = True
for seed in range(5):
    result_path = os.path.join(
        run_root, f"waterbirds_{dataset}", method, f"seed_{seed}",
        "pointing_game", "pointing_game_summary.csv",
    )
    manifest_path = os.path.join(
        checkpoint_root, f"waterbirds_{dataset}", method, f"seed_{seed}",
        "training_manifest.json",
    )
    try:
        rows = list(csv.DictReader(open(result_path, newline="", encoding="utf-8")))
        row = rows[0] if len(rows) == 1 else None
        manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
        checkpoint = manifest["checkpoint"]
        stage1 = manifest.get("stage1_checkpoint", "") if method == "afr" else ""
        valid = valid and (
            row is not None
            and row.get("dataset") == f"waterbirds_{dataset}"
            and row.get("method") == method
            and int(row.get("seed", -1)) == seed
            and row.get("split") == "test"
            and row.get("target_mode") == target_mode
            and row.get("explainer") == "rise"
            and row.get("mask_source") == "CUB_200_2011_segmentations"
            and int(row.get("mask_threshold", -1)) == int(mask_threshold)
            and int(row.get("max_samples", -1)) == int(max_samples)
            and int(row.get("sample_seed", -1)) == int(sample_seed)
            and int(row.get("rise_num_masks", -1)) == int(num_masks)
            and int(row.get("rise_grid_size", -1)) == int(grid_size)
            and math.isclose(float(row.get("rise_p1", "nan")), float(p1))
            and int(row.get("rise_seed", -1)) == int(rise_seed)
            and row.get("checkpoint") == checkpoint
            and row.get("afr_stage1_checkpoint", "") == stage1
            and int(row.get("pg_total", 0)) > 0
            and int(row.get("errors", 1)) == 0
        )
    except Exception:
        valid = False
    if not valid:
        break
raise SystemExit(0 if valid else 1)
PY
}

if [[ "$DRY_RUN" != "1" ]]; then
  incomplete_pairs=0
  ready_pairs=0
  for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
      if source_pair_is_ready "$dataset" "$method" 1; then
        ready_pairs=$((ready_pairs + 1))
      else
        incomplete_pairs=$((incomplete_pairs + 1))
      fi
    done
  done
  if [[ "$incomplete_pairs" -gt 0 && "$ALLOW_PARTIAL" != "1" ]]; then
    echo "[ERROR] RISE submission aborted before queuing jobs: some source checkpoints are incomplete." >&2
    echo "[INFO] Use ALLOW_PARTIAL=1 to queue only complete dataset-method pairs." >&2
    exit 2
  fi
  if [[ "$ready_pairs" == "0" ]]; then
    echo "[ERROR] No dataset-method pair has all five source checkpoints yet." >&2
    exit 2
  fi
  if [[ "$incomplete_pairs" -gt 0 ]]; then
    echo "[INFO] Partial mode: $ready_pairs pairs ready; $incomplete_pairs pairs will be skipped."
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
printf 'dataset,method,status,job_id\n' > "$JOB_FILE"

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    job_name="pgr5_wb${dataset}_${method}"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY RUN] sbatch --job-name=$job_name --export=ALL,DATASET=$dataset,METHOD=$method $WORKER"
      status="DRY_RUN"
      job_id="DRY_RUN"
    elif ! source_pair_is_ready "$dataset" "$method" 0; then
      echo "[SKIP-INCOMPLETE] dataset=$dataset method=$method"
      status="INCOMPLETE"
      job_id=""
    elif result_pair_is_complete "$dataset" "$method"; then
      echo "[SKIP-COMPLETE] dataset=$dataset method=$method"
      status="COMPLETE"
      job_id=""
    else
      queued_job_ids="$(squeue -h -u "$USER" -n "$job_name" -o '%A' | paste -sd ';' -)"
      if [[ -n "$queued_job_ids" ]]; then
        echo "[SKIP-QUEUED] dataset=$dataset method=$method jobs=$queued_job_ids"
        status="QUEUED"
        job_id="$queued_job_ids"
      else
        job_id="$(sbatch --parsable \
          --job-name="$job_name" \
          --export="ALL,DATASET=$dataset,METHOD=$method" \
          "$WORKER")"
        echo "[SUBMITTED] dataset=$dataset method=$method job=$job_id"
        status="SUBMITTED"
      fi
    fi
    printf '%s,%s,%s,%s\n' "$dataset" "$method" "$status" "$job_id" >> "$JOB_FILE"
  done
done

echo
echo "[DONE] submission record: $JOB_FILE"
echo "[DONE] RISE result root: $RUN_ROOT"
echo "[INFO] These jobs reuse completed checkpoints; they do not retrain models."
echo "[INFO] After all jobs finish:"
echo "  cd $SCRIPT_DIR"
echo "  python summarize_waterbirds_rise_pointing_game_5seed.py --run-root $RUN_ROOT --seeds $SEEDS_CSV"
