#!/usr/bin/env bash
set -Eeuo pipefail

# Submits ONE generator job to build CLIP ViT attentions for Waterbirds-95 + Waterbirds-100,
# then submits four dependent jobs that require those attentions:
#   1) Guided strategy sweep (WB95) using GALS ViT attentions
#   2) Guided strategy sweep (WB100) using GALS ViT attentions
#   3) GALS sweep (WB95) using ViT attentions
#   4) GALS sweep (WB100) using ViT attentions
#
# All env vars are forwarded via --export=ALL so you can override:
#   - N_TRIALS, SWEEP_SEED, TRAIN_SEED, POST_SEEDS, etc.
#   - For the generator: WB95_DIR, WB100_DIR
#
# Run from the GALS folder on RC:
#   bash submit_vit_attention_then_run_all_vit_sweeps.sh

GEN_SCRIPT=${GEN_SCRIPT:-run_generate_waterbirds_vit_attentions_95_100.sh}

echo "[SUBMIT] Generator: $GEN_SCRIPT"
GEN_JOB_ID=$(sbatch --parsable --export=ALL "$GEN_SCRIPT")
echo "[SUBMIT] Generator job id: $GEN_JOB_ID"

dep=(--dependency=afterok:"$GEN_JOB_ID" --export=ALL)

echo "[SUBMIT] Dependent: guided (WB95) using GALS ViT attentions"
GUIDED_95_JOB_ID=$(sbatch --parsable "${dep[@]}" ../run_guided_waterbird_gals_vitatt_sweep.sh)
echo "[SUBMIT] guided95 job id: $GUIDED_95_JOB_ID"

echo "[SUBMIT] Dependent: guided (WB100) using GALS ViT attentions"
GUIDED_100_JOB_ID=$(sbatch --parsable "${dep[@]}" ../run_guided_100_galsvit_sweep.sh)
echo "[SUBMIT] guided100 job id: $GUIDED_100_JOB_ID"

echo "[SUBMIT] Dependent: GALS (WB95) sweep using ViT attentions"
GALS_95_JOB_ID=$(sbatch --parsable "${dep[@]}" run_waterbirds95_gals_sweep_vit.sh)
echo "[SUBMIT] gals95 job id: $GALS_95_JOB_ID"

echo "[SUBMIT] Dependent: GALS (WB100) sweep using ViT attentions"
GALS_100_JOB_ID=$(sbatch --parsable "${dep[@]}" run_waterbirds100_gals_sweep_vit.sh)
echo "[SUBMIT] gals100 job id: $GALS_100_JOB_ID"

echo "[SUBMIT] Done."

