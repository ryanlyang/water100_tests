#!/bin/bash
# Submit five independent WB100 vanilla train -> RISE -> cleanup recovery jobs.

set -Eeuo pipefail

unset SBATCH_GET_USER_ENV SBATCH_EXPORT SBATCH_EXPORT_FILE
export SLURM_EXPORT_ENV=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/run_waterbirds_rise_recovery_seed.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
RISE_RUN_ROOT="${RISE_RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
PARTITION="${PARTITION:-debug}"
WALLTIME="${WALLTIME:-1-00:00:00}"
DRY_RUN="${DRY_RUN:-0}"

RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-$RISE_RUN_ROOT/rise_masks/waterbirds_gals_N${RISE_NUM_MASKS}_s${RISE_GRID_SIZE}_p${RISE_P1}_seed${RISE_SEED}_224x224.npy}"

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RISE_RUN_ROOT" "$(dirname "$RISE_MASKS_PATH")"

if [[ "$DRY_RUN" != "1" ]]; then
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

export LOG_DIR RISE_RUN_ROOT RISE_NUM_MASKS RISE_GRID_SIZE RISE_P1 RISE_SEED
export RISE_MASKS_PATH DELETE_CHECKPOINT_AFTER_RISE

record="$RISE_RUN_ROOT/submitted_wb100_vanilla_recovery_$(date +%Y%m%d_%H%M%S).csv"
printf 'dataset,method,seed,job_id\n' > "$record"
job_ids=()

for seed in 0 1 2 3 4; do
  job_name="pgr100v_s${seed}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] seed=$seed partition=$PARTITION time=$WALLTIME"
    job_id="DRY_RUN"
  else
    job_id="$(sbatch --parsable \
      --partition="$PARTITION" \
      --time="$WALLTIME" \
      --job-name="$job_name" \
      --export="ALL,SEED=$seed,RECOVERY_DATASET=100,RECOVERY_METHOD=vanilla" \
      "$WORKER")"
    job_ids+=("$job_id")
    echo "[SUBMITTED] seed=$seed job=$job_id"
  fi
  printf '100,vanilla,%s,%s\n' "$seed" "$job_id" >> "$record"
done

echo
echo "[DONE] submission record: $record"
if [[ ${#job_ids[@]} -gt 0 ]]; then
  echo "[INFO] jobs: ${job_ids[*]}"
fi
echo "[INFO] Each job validates its RISE CSV before deleting its temporary checkpoint."
echo "[INFO] After all five jobs complete, rebuild the combined table with:"
echo "  cd $SCRIPT_DIR"
echo "  python summarize_waterbirds_rise_pointing_game_5seed.py --run-root $RISE_RUN_ROOT --seeds 0,1,2,3,4"
