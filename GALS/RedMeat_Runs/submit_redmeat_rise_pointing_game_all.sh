#!/bin/bash
# Submit one resumable RedMeat RISE Pointing Game job per method.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/run_redmeat_rise_pointing_game_5seed_method.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsRedMeat}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
CLIP_SEEDS_CSV="${CLIP_SEEDS_CSV:-0}"
DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/Food101/data/food-101-redmeat}"
MASK_ROOT="${MASK_ROOT:-$DATA_PATH/redmeat_pointing_masks}"
TEACHER_MAPS="${TEACHER_MAPS:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_laion_dinovit/val/prediction_cmap}"
GALS_MAPS="${GALS_MAPS:-$DATA_PATH/clip_rn50_attention_gradcam}"
AFR_ROOT="${AFR_ROOT:-}"
EXISTING_CHECKPOINT_CSV="${EXISTING_CHECKPOINT_CSV:-}"
TARGET_MODE="${TARGET_MODE:-label}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
RISE_BANK="${RISE_BANK:-$RUN_ROOT/shared/rise_n${RISE_NUM_MASKS}_g${RISE_GRID_SIZE}_p${RISE_P1}_s${RISE_SEED}_224.npy}"
CLIP_MODEL="${CLIP_MODEL:-RN50}"
CLIP_C="${CLIP_C:-1.329346323656201}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT"

export LOG_DIR RUN_ROOT SEEDS_CSV CLIP_SEEDS_CSV DATA_PATH MASK_ROOT
export TEACHER_MAPS GALS_MAPS AFR_ROOT EXISTING_CHECKPOINT_CSV
export TARGET_MODE MASK_THRESHOLD MAX_SAMPLES SAMPLE_SEED
export RISE_NUM_MASKS RISE_GRID_SIZE RISE_P1 RISE_SEED RISE_BANK
export CLIP_MODEL CLIP_C

METHODS=(vanilla elrep upweight abn gals afr r4rr clip_lr clip_zs)
JOB_FILE="$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
printf 'dataset,method,seeds,job_id\n' > "$JOB_FILE"

for method in "${METHODS[@]}"; do
  job_name="pg5_rm_${method}"
  method_seeds="$SEEDS_CSV"
  if [[ "$method" == "clip_lr" || "$method" == "clip_zs" ]]; then
    method_seeds="$CLIP_SEEDS_CSV"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] sbatch --job-name=$job_name --export=ALL,METHOD=$method $WORKER"
    job_id="DRY_RUN"
  else
    job_id="$(sbatch --parsable \
      --job-name="$job_name" \
      --export="ALL,METHOD=$method" \
      "$WORKER")"
    echo "[SUBMITTED] method=$method seeds=$method_seeds job=$job_id"
  fi
  printf 'redmeat,%s,%s,%s\n' "$method" "$method_seeds" "$job_id" >> "$JOB_FILE"
done

echo
echo "[DONE] submission record: $JOB_FILE"
echo "[DONE] stable result root: $RUN_ROOT"
echo "[INFO] Seven trained methods use seeds $SEEDS_CSV; deterministic CLIP methods use $CLIP_SEEDS_CSV."
echo "[INFO] Resubmitting resumes valid training manifests and per-seed evaluations."
echo "[INFO] After all jobs finish:"
echo "  cd $(dirname "$SCRIPT_DIR")"
echo "  python RedMeat_Runs/summarize_redmeat_rise_pointing_game.py --run-root $RUN_ROOT --seeds $SEEDS_CSV --clip-seeds $CLIP_SEEDS_CSV"
