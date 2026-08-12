#!/bin/bash
# Submit evaluation-only DecoyMNIST RISE Pointing Game jobs.

set -Eeuo pipefail

# The worker activates its own Conda environment. Prevent inherited SBATCH
# settings from asking Slurm to reconstruct the submit host's login environment,
# which can leave jobs held with "user env retrieval failed" before bash starts.
unset SBATCH_GET_USER_ENV SBATCH_EXPORT SBATCH_EXPORT_FILE
export SLURM_EXPORT_ENV=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/run_decoymnist_rise_pointing_game_5seed_method.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
CHECKPOINT_RUN_ROOT="${CHECKPOINT_RUN_ROOT:-$LOG_DIR/pointing_game_5seed_gradcam}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
METHODS_CSV="${METHODS_CSV:-vanilla,elrep,upweight,abn,gals,afr,r4rr}"
SPLIT="${SPLIT:-test}"
TARGET_MODE="${TARGET_MODE:-label}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
RISE_IMAGE_BATCH_SIZE="${RISE_IMAGE_BATCH_SIZE:-16}"
RISE_MAX_MASKED_BATCH="${RISE_MAX_MASKED_BATCH:-8192}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-$RUN_ROOT/rise_masks/decoymnist_gals_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_28x28.npy}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT"

export LOG_DIR CHECKPOINT_RUN_ROOT RUN_ROOT SEEDS_CSV SPLIT TARGET_MODE
export MASK_THRESHOLD MAX_SAMPLES SAMPLE_SEED RISE_NUM_MASKS RISE_GRID_SIZE
export RISE_P1 RISE_SEED RISE_IMAGE_BATCH_SIZE RISE_MAX_MASKED_BATCH RISE_MASKS_PATH
export PNG_ROOT MNIST_ROOT ENV_NAME PROJECT_ROOT GALS_ROOT

IFS=',' read -r -a METHODS <<< "$METHODS_CSV"
JOB_FILE="$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
printf 'dataset,explainer,method,status,job_id\n' > "$JOB_FILE"

result_method_is_complete() {
  local method="$1"
  python - "$RUN_ROOT" "$CHECKPOINT_RUN_ROOT" "$method" "$SPLIT" "$TARGET_MODE" \
    "$MASK_THRESHOLD" "$MAX_SAMPLES" "$SAMPLE_SEED" \
    "$RISE_NUM_MASKS" "$RISE_GRID_SIZE" "$RISE_P1" "$RISE_SEED" <<'PY'
import csv, json, math, os, sys
(
    run_root, checkpoint_root, method, split, target_mode, mask_threshold,
    max_samples, sample_seed, num_masks, grid_size, p1, rise_seed,
) = sys.argv[1:]
valid = True
for seed in range(5):
    result_path = os.path.join(
        run_root, method, f"seed_{seed}", "pointing_game", "pointing_game_summary.csv"
    )
    manifest_path = os.path.join(
        checkpoint_root, method, f"seed_{seed}", "training_manifest.json"
    )
    try:
        rows = list(csv.DictReader(open(result_path, newline="", encoding="utf-8")))
        row = rows[0] if len(rows) == 1 else None
        checkpoint = json.load(open(manifest_path, "r", encoding="utf-8"))["checkpoint"]
        valid = valid and (
            row is not None
            and row.get("dataset") == "decoymnist"
            and row.get("method") == method
            and int(row.get("seed", -1)) == seed
            and row.get("split") == split
            and row.get("target_mode") == target_mode
            and row.get("explainer") == "rise"
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
    if not valid:
        break
raise SystemExit(0 if valid else 1)
PY
}

for method_raw in "${METHODS[@]}"; do
  method="${method_raw//[[:space:]]/}"
  [[ -n "$method" ]] || continue
  case "$method" in
    vanilla|elrep|upweight|abn|gals|afr|r4rr) ;;
    *) echo "[ERROR] Unsupported method in METHODS_CSV: $method" >&2; exit 2 ;;
  esac
  job_name="pgr5_decoy_${method}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] sbatch --job-name=$job_name --export=ALL,METHOD=$method $WORKER"
    status="DRY_RUN"
    job_id="DRY_RUN"
  elif result_method_is_complete "$method"; then
    echo "[SKIP-COMPLETE] method=$method"
    status="COMPLETE"
    job_id=""
  else
    queued_job_ids="$(squeue -h -u "$USER" -n "$job_name" -o '%A' | paste -sd ';' -)"
    if [[ -n "$queued_job_ids" ]]; then
      echo "[SKIP-QUEUED] method=$method jobs=$queued_job_ids"
      status="QUEUED"
      job_id="$queued_job_ids"
    else
      job_id="$(sbatch --parsable \
        --job-name="$job_name" \
        --export="ALL,METHOD=$method" \
        "$WORKER")"
      echo "[SUBMITTED] method=$method explainer=RISE job=$job_id"
      status="SUBMITTED"
    fi
  fi
  printf 'decoymnist,rise,%s,%s,%s\n' "$method" "$status" "$job_id" >> "$JOB_FILE"
done

echo
echo "[DONE] submission record: $JOB_FILE"
echo "[DONE] result root: $RUN_ROOT"
echo "[INFO] Existing checkpoints are read from: $CHECKPOINT_RUN_ROOT"
echo "[INFO] No training is performed by these jobs."
echo "[INFO] After all selected jobs finish:"
echo "  cd $SCRIPT_DIR"
echo "  python summarize_decoymnist_rise_pointing_game_5seed.py --run-root $RUN_ROOT --seeds $SEEDS_CSV"
