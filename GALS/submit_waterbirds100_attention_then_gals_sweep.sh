#!/usr/bin/env bash
set -Eeuo pipefail

# Submits Stage 1 (extract attention) and Stage 2 (GALS sweep) with a dependency.
#
# Defaults:
# - If attention dir already exists, Stage 1 is skipped (set FORCE_EXTRACT=1 to always rerun).
# - Any env vars like N_TRIALS, SAMPLER, KEEP, MAX_HOURS, TUNE_WEIGHT_DECAY, ENV_NAME, BOOTSTRAP_ENV
#   are forwarded to the sbatch jobs via --export=ALL.

DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds}
DATA_DIR=${DATA_DIR:-waterbird_1.0_forest2water2}
ATT_DIR=${ATT_DIR:-clip_rn50_attention_gradcam}
FORCE_EXTRACT=${FORCE_EXTRACT:-0}

ATT_PATH="$DATA_ROOT/$DATA_DIR/$ATT_DIR"

if [[ "$FORCE_EXTRACT" -eq 1 || ! -d "$ATT_PATH" ]]; then
  echo "[SUBMIT] Stage 1: extract attention"
  ATT_JOB_ID=$(sbatch --parsable --export=ALL run_waterbirds100_extract_attention.sh)
  echo "[SUBMIT] Attention job id: $ATT_JOB_ID"
  echo "[SUBMIT] Stage 2: GALS sweep (afterok:$ATT_JOB_ID)"
  SWEEP_JOB_ID=$(sbatch --parsable --dependency=afterok:"$ATT_JOB_ID" --export=ALL run_waterbirds100_gals_sweep.sh)
  echo "[SUBMIT] Sweep job id: $SWEEP_JOB_ID"
else
  echo "[SUBMIT] Attention dir exists; skipping Stage 1: $ATT_PATH"
  echo "[SUBMIT] Stage 2: GALS sweep"
  SWEEP_JOB_ID=$(sbatch --parsable --export=ALL run_waterbirds100_gals_sweep.sh)
  echo "[SUBMIT] Sweep job id: $SWEEP_JOB_ID"
fi

