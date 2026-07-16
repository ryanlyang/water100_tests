#!/usr/bin/env bash
set -Eeuo pipefail

# Submit *all* Waterbirds jobs we set up, with correct Slurm dependencies.
#
# What it submits:
# - Sweeps (optionally with dependencies if you provide generator job ids):
#   - GALS (RN50) sweeps (95/100) can depend on the corresponding RN50 attention job
#   - GALS (ViT), GALS/GradCAM (ViT), GALS/ABN (ViT) sweeps can depend on the ViT attention job
#   - Guided strategy sweeps:
#     - GT-mask guided (95/100)
#     - GALS ViT attention guided (95/100; can depend on ViT attention job)
#     - GALS RN50 attention guided (95/100; can depend on RN50 attention jobs)
# - Independent sweeps:
#   - GALS "ourmasks" sweeps (95/100) [depends on MASK_DIRs existing, but no job dependency]
#   - UpWeight sweeps (95/100)
#   - ABN baseline sweeps (95/100) (requires ABN weight file to exist on RC)
#   - CLIP + Logistic Regression sweeps (95/100)
#   - Vanilla CNN sweeps (95/100)
#
# Run from the `GALS/` folder on RC:
#   bash submit_all_waterbirds_experiments.sh
#
# Use existing generator job ids (recommended):
#   VIT_JOB_ID=21038942 bash submit_all_waterbirds_experiments.sh
#   RN50_95_JOB_ID=... RN50_100_JOB_ID=... bash submit_all_waterbirds_experiments.sh
#   (This script will NOT submit attention generation jobs.)
#
# Skips:
#   SKIP_GUIDED=1
#   SKIP_VIT_METHODS=1 (RRR/GradCAM/ABN ViT sweeps)
#   SKIP_RN50_SWEEPS=1
#   SKIP_OURMASKS=1
#   SKIP_BASELINES=1  (UpWeight + ABN baseline)
#   SKIP_CLIP_LR=1
#   SKIP_VANILLA_CNN=1
#
# Global sweep knobs applied to all submitted runs:
#   N_TRIALS=50
#   POST_SEEDS=5
#   POST_SEED_START=0
# Global Slurm walltime override for every submitted job:
#   SBATCH_TIME=15-00:00:00
#
# Note: This script only *submits* jobs; it doesn't run anything locally.

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1" >&2; exit 2; }
}

need_cmd sbatch

DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds}
WB95_DIR=${WB95_DIR:-waterbird_complete95_forest2water2}
WB100_DIR=${WB100_DIR:-waterbird_1.0_forest2water2}
VIT_ATT_DIR=${VIT_ATT_DIR:-clip_vit_attention}
RN50_ATT_DIR=${RN50_ATT_DIR:-clip_rn50_attention_gradcam}

MASK95_DIR=${MASK95_DIR:-/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
MASK100_DIR=${MASK100_DIR:-/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
N_TRIALS=${N_TRIALS:-50}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
SBATCH_TIME=${SBATCH_TIME:-15-00:00:00}
EXPORT_SWEEP="ALL,N_TRIALS=${N_TRIALS},POST_SEEDS=${POST_SEEDS},POST_SEED_START=${POST_SEED_START}"

echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] WB95_DIR=$WB95_DIR"
echo "[INFO] WB100_DIR=$WB100_DIR"
echo "[INFO] N_TRIALS=$N_TRIALS"
echo "[INFO] POST_SEEDS=$POST_SEEDS"
echo "[INFO] POST_SEED_START=$POST_SEED_START"
echo "[INFO] SBATCH_TIME=$SBATCH_TIME"

submit_parsable() {
  # shellcheck disable=SC2068
  sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" $@
}

dep_afterok() {
  # Usage: dep_afterok <jid> -> prints "--dependency=afterok:<jid>"
  echo "--dependency=afterok:$1"
}

maybe_dep() {
  # Usage: maybe_dep <jid or empty> -> prints dependency args if jid set
  if [[ -n "${1:-}" ]]; then
    echo "--dependency=afterok:$1"
  fi
}

VIT_JOB_ID=${VIT_JOB_ID:-}
RN50_95_JOB_ID=${RN50_95_JOB_ID:-}
RN50_100_JOB_ID=${RN50_100_JOB_ID:-}
if [[ -n "$VIT_JOB_ID" ]]; then
  echo "[SUBMIT] Using existing VIT_JOB_ID=$VIT_JOB_ID (for dependencies)"
else
  if [[ ! -d "${DATA_ROOT}/${WB95_DIR}/${VIT_ATT_DIR}" || ! -d "${DATA_ROOT}/${WB100_DIR}/${VIT_ATT_DIR}" ]]; then
    echo "[WARN] ViT attention dirs not found under datasets; ViT-dependent sweeps may fail." >&2
    echo "       Expected: ${DATA_ROOT}/${WB95_DIR}/${VIT_ATT_DIR} and ${DATA_ROOT}/${WB100_DIR}/${VIT_ATT_DIR}" >&2
    echo "       Provide VIT_JOB_ID=... to add dependencies, or generate attentions first." >&2
  fi
fi

if [[ -n "$RN50_95_JOB_ID" ]]; then
  echo "[SUBMIT] Using existing RN50_95_JOB_ID=$RN50_95_JOB_ID (for dependencies)"
else
  if [[ ! -d "${DATA_ROOT}/${WB95_DIR}/${RN50_ATT_DIR}" ]]; then
    echo "[WARN] RN50 attention dir not found for WB95; RN50-dependent sweeps may fail." >&2
    echo "       Expected: ${DATA_ROOT}/${WB95_DIR}/${RN50_ATT_DIR}" >&2
    echo "       Provide RN50_95_JOB_ID=... to add dependencies, or generate attentions first." >&2
  fi
fi

if [[ -n "$RN50_100_JOB_ID" ]]; then
  echo "[SUBMIT] Using existing RN50_100_JOB_ID=$RN50_100_JOB_ID (for dependencies)"
else
  if [[ ! -d "${DATA_ROOT}/${WB100_DIR}/${RN50_ATT_DIR}" ]]; then
    echo "[WARN] RN50 attention dir not found for WB100; RN50-dependent sweeps may fail." >&2
    echo "       Expected: ${DATA_ROOT}/${WB100_DIR}/${RN50_ATT_DIR}" >&2
    echo "       Provide RN50_100_JOB_ID=... to add dependencies, or generate attentions first." >&2
  fi
fi

echo "===================="
echo "[SUBMIT] Stage 2: dependent sweeps"
echo "===================="

if [[ "${SKIP_RN50_SWEEPS:-0}" -ne 1 ]]; then
  echo "[SUBMIT] GALS (RN50) sweeps..."
  dep95=()
  dep100=()
  if [[ -n "$RN50_95_JOB_ID" ]]; then dep95=("$(dep_afterok "$RN50_95_JOB_ID")"); fi
  if [[ -n "$RN50_100_JOB_ID" ]]; then dep100=("$(dep_afterok "$RN50_100_JOB_ID")"); fi
  j_gals95_rn50="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${dep95[@]:-} run_waterbirds95_gals_sweep.sh)"
  j_gals100_rn50="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${dep100[@]:-} run_waterbirds100_gals_sweep.sh)"
  echo "  gals95_rn50:  $j_gals95_rn50"
  echo "  gals100_rn50: $j_gals100_rn50"
else
  echo "[SUBMIT] SKIP_RN50_SWEEPS=1"
fi

if [[ "${SKIP_VIT_METHODS:-0}" -ne 1 ]]; then
  echo "[SUBMIT] GALS ViT-method sweeps (GALS/GradCAM/ABN)..."
  depvit=()
  if [[ -n "$VIT_JOB_ID" ]]; then depvit=("$(dep_afterok "$VIT_JOB_ID")"); fi
  j_gals95_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_waterbirds95_gals_sweep_vit.sh)"
  j_gals100_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_waterbirds100_gals_sweep_vit.sh)"
  j_gc95_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_waterbirds95_gradcam_sweep_vit.sh)"
  j_gc100_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_waterbirds100_gradcam_sweep_vit.sh)"
  j_abn95_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_waterbirds95_abn_sweep_vit.sh)"
  j_abn100_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_waterbirds100_abn_sweep_vit.sh)"
  echo "  gals95_vit:    $j_gals95_vit"
  echo "  gals100_vit:   $j_gals100_vit"
  echo "  gradcam95_vit: $j_gc95_vit"
  echo "  gradcam100_vit:$j_gc100_vit"
  echo "  abn95_vit:     $j_abn95_vit"
  echo "  abn100_vit:    $j_abn100_vit"
else
  echo "[SUBMIT] SKIP_VIT_METHODS=1"
fi

if [[ "${SKIP_GUIDED:-0}" -ne 1 ]]; then
  echo "[SUBMIT] Guided strategy sweeps..."
  j_guided95_gt="$(submit_parsable ../run_guided_waterbird_sweep.sh)"
  j_guided100_gt="$(submit_parsable ../run_guided_100_sweep.sh)"
  echo "  guided95:          $j_guided95_gt"
  echo "  guided100:         $j_guided100_gt"

  echo "  guided (GALS ViT attentions as GT):"
  depvit=()
  if [[ -n "$VIT_JOB_ID" ]]; then depvit=("$(dep_afterok "$VIT_JOB_ID")"); fi
  j_guided95_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} ../run_guided_waterbird_gals_vitatt_sweep.sh)"
  j_guided100_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} ../run_guided_100_galsvit_sweep.sh)"
  echo "    guided95_galsvit:  $j_guided95_vit"
  echo "    guided100_galsvit: $j_guided100_vit"

  echo "  guided (GALS RN50 attentions as GT):"
  dep95=()
  dep100=()
  if [[ -n "$RN50_95_JOB_ID" ]]; then dep95=("$(dep_afterok "$RN50_95_JOB_ID")"); fi
  if [[ -n "$RN50_100_JOB_ID" ]]; then dep100=("$(dep_afterok "$RN50_100_JOB_ID")"); fi
  j_guided95_rn50="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${dep95[@]:-} ../run_guided_waterbird_gals_rn50att_sweep.sh)"
  j_guided100_rn50="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${dep100[@]:-} ../run_guided_100_galsrn50_sweep.sh)"
  echo "    guided95_galsrn50:  $j_guided95_rn50"
  echo "    guided100_galsrn50: $j_guided100_rn50"
else
  echo "[SUBMIT] SKIP_GUIDED=1"
fi

echo "===================="
echo "[SUBMIT] Stage 3: independent sweeps"
echo "===================="

if [[ "${SKIP_OURMASKS:-0}" -ne 1 ]]; then
  echo "[SUBMIT] GALS ourmasks sweeps (no attention generation dependency)..."
  if [[ -d "$MASK95_DIR" ]]; then
    j_our95="$(sbatch --parsable --time="$SBATCH_TIME" --export="${EXPORT_SWEEP},MASK_DIR=$MASK95_DIR" run_waterbirds95_gals_ourmasks_sweep.sh)"
    echo "  ourmasks95: $j_our95"
  else
    echo "  [WARN] MASK95_DIR missing: $MASK95_DIR (skipping WB95 ourmasks)"
  fi
  if [[ -d "$MASK100_DIR" ]]; then
    j_our100="$(sbatch --parsable --time="$SBATCH_TIME" --export="${EXPORT_SWEEP},MASK_DIR=$MASK100_DIR" run_waterbirds100_gals_ourmasks_sweep.sh)"
    echo "  ourmasks100: $j_our100"
  else
    echo "  [WARN] MASK100_DIR missing: $MASK100_DIR (skipping WB100 ourmasks)"
  fi
else
  echo "[SUBMIT] SKIP_OURMASKS=1"
fi

if [[ "${SKIP_BASELINES:-0}" -ne 1 ]]; then
  echo "[SUBMIT] UpWeight + ABN baselines..."
  j_up95="$(submit_parsable run_waterbirds95_upweight_sweep.sh)"
  j_up100="$(submit_parsable run_waterbirds100_upweight_sweep.sh)"
  echo "  upweight95:  $j_up95"
  echo "  upweight100: $j_up100"

  # ABN baseline requires the ABN pretrained weights file to exist on RC.
  echo "  [NOTE] ABN needs: /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/weights/resnet50_abn_imagenet.pth.tar"
  j_abn95_base="$(submit_parsable run_waterbirds95_abn_sweep.sh)"
  j_abn100_base="$(submit_parsable run_waterbirds100_abn_sweep.sh)"
  echo "  abn95_baseline:  $j_abn95_base"
  echo "  abn100_baseline: $j_abn100_base"
else
  echo "[SUBMIT] SKIP_BASELINES=1"
fi

if [[ "${SKIP_CLIP_LR:-0}" -ne 1 ]]; then
  echo "[SUBMIT] CLIP + Logistic Regression sweeps..."
  j_clip_lr95="$(submit_parsable run_waterbirds95_clip_lr_sweep.sh)"
  j_clip_lr100="$(submit_parsable run_waterbirds100_clip_lr_sweep.sh)"
  echo "  clip_lr95:  $j_clip_lr95"
  echo "  clip_lr100: $j_clip_lr100"
else
  echo "[SUBMIT] SKIP_CLIP_LR=1"
fi

if [[ "${SKIP_VANILLA_CNN:-0}" -ne 1 ]]; then
  echo "[SUBMIT] Vanilla CNN sweeps..."
  j_van95="$(submit_parsable run_waterbirds95_vanilla_cnn_sweep.sh)"
  j_van100="$(submit_parsable run_waterbirds100_vanilla_cnn_sweep.sh)"
  echo "  vanilla95:  $j_van95"
  echo "  vanilla100: $j_van100"
else
  echo "[SUBMIT] SKIP_VANILLA_CNN=1"
fi

echo "[SUBMIT] Done."
