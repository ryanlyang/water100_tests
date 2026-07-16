#!/usr/bin/env bash
set -Eeuo pipefail

# Submits Stage 1 (extract CLIP ViT attention) and Stage 2 (GALS sweep) with a dependency.
#
# Defaults:
# - If attention dir already exists, Stage 1 is skipped (set FORCE_EXTRACT=1 to always rerun).
# - Any env vars like N_TRIALS, SAMPLER, KEEP, MAX_HOURS, TUNE_WEIGHT_DECAY, POST_SEEDS, ENV_NAME, BOOTSTRAP_ENV
#   are forwarded to the sbatch jobs via --export=ALL.

DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds}
DATA_DIR=${DATA_DIR:-waterbird_complete95_forest2water2}
ATT_DIR=${ATT_DIR:-clip_vit_attention}
FORCE_EXTRACT=${FORCE_EXTRACT:-0}

ATT_PATH="$DATA_ROOT/$DATA_DIR/$ATT_DIR"

if [[ "$FORCE_EXTRACT" -eq 1 || ! -d "$ATT_PATH" ]]; then
  echo "[SUBMIT] Stage 1: extract ViT attention"
  ATT_JOB_ID=$(sbatch --parsable --export=ALL run_waterbirds95_extract_attention_vit.sh)
  echo "[SUBMIT] Attention job id: $ATT_JOB_ID"
  echo "[SUBMIT] Stage 2: GALS ViT sweep (afterok:$ATT_JOB_ID)"
  SWEEP_JOB_ID=$(sbatch --parsable --dependency=afterok:"$ATT_JOB_ID" --export=ALL run_waterbirds95_gals_sweep_vit.sh)
  echo "[SUBMIT] Sweep job id: $SWEEP_JOB_ID"
else
  echo "[SUBMIT] Attention dir exists; skipping Stage 1: $ATT_PATH"
  echo "[SUBMIT] Stage 2: GALS ViT sweep"
  SWEEP_JOB_ID=$(sbatch --parsable --export=ALL run_waterbirds95_gals_sweep_vit.sh)
  echo "[SUBMIT] Sweep job id: $SWEEP_JOB_ID"
fi

