#!/bin/bash
# Submit one RedMeat CLIP-LR light-unfreezing job per seed.

set -Eeuo pipefail

unset SBATCH_GET_USER_ENV SBATCH_EXPORT SBATCH_EXPORT_FILE
export SLURM_EXPORT_ENV=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/run_redmeat_clip_lr_light_unfreeze_seed.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsRedMeat}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/clip_lr_rn50_full_visual_finetune}"
PARTITION="${PARTITION:-debug}"
WALLTIME="${WALLTIME:-1-00:00:00}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT"
export LOG_DIR RUN_ROOT

record="$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
printf 'seed,job_id\n' > "$record"
job_ids=()

for seed in 0 1 2 3 4; do
  job_name="cliplrft_s${seed}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] seed=$seed partition=$PARTITION time=$WALLTIME"
    job_id="DRY_RUN"
  else
    job_id="$(sbatch --parsable \
      --partition="$PARTITION" \
      --time="$WALLTIME" \
      --job-name="$job_name" \
      --export="ALL,SEED=$seed" \
      "$WORKER")"
    job_ids+=("$job_id")
    echo "[SUBMITTED] seed=$seed job=$job_id"
  fi
  printf '%s,%s\n' "$seed" "$job_id" >> "$record"
done

echo
echo "[DONE] submission record: $record"
echo "[INFO] jobs: ${job_ids[*]:-NONE}"
echo "[INFO] After all jobs complete:"
echo "  cd $(dirname "$SCRIPT_DIR")"
echo "  python RedMeat_Runs/summarize_clip_lr_light_unfreeze_study.py --run-root $RUN_ROOT --seeds 0,1,2,3,4"
