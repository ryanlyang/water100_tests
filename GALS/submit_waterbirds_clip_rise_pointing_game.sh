#!/bin/bash
# Submit deterministic CLIP-ZS and CLIP-LR RISE jobs for WB95 and WB100.

set -Eeuo pipefail

unset SBATCH_GET_USER_ENV SBATCH_EXPORT SBATCH_EXPORT_FILE
export SLURM_EXPORT_ENV=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${WORKER:-$SCRIPT_DIR/run_waterbirds_clip_rise_pointing_game.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$WORKER" ]] || { echo "[ERROR] Missing worker: $WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT"

for dataset in 95 100; do
  for method in clip_zs clip_lr; do
    short_method="${method/_/}"
    job_name="pgr1_wb${dataset}_${short_method}"
    summary="$RUN_ROOT/waterbirds_${dataset}/${method}/seed_0/pointing_game/pointing_game_summary.csv"

    if [[ -s "$summary" ]]; then
      echo "[SKIP-EXISTS] dataset=$dataset method=$method summary=$summary"
      continue
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY RUN] sbatch --job-name=$job_name --export=ALL,DATASET=$dataset,METHOD=$method $WORKER"
      continue
    fi
    queued="$(squeue -h -u "$USER" -n "$job_name" -o '%A' | paste -sd ';' -)"
    if [[ -n "$queued" ]]; then
      echo "[SKIP-QUEUED] dataset=$dataset method=$method jobs=$queued"
      continue
    fi
    job_id="$(sbatch --parsable \
      --job-name="$job_name" \
      --export="ALL,DATASET=$dataset,METHOD=$method" \
      "$WORKER")"
    echo "[SUBMITTED] dataset=$dataset method=$method job=$job_id partition=debug"
  done
done

echo "[DONE] Deterministic CLIP RISE jobs write under: $RUN_ROOT"
