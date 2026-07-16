#!/usr/bin/env bash
set -Eeuo pipefail

# Submit all RedMeat experiments (Waterbirds matrix adapted to Food101 RedMeat).
#
# This submits:
# - Attention-map dependent sweeps:
#   - GALS (RN50 attentions)
#   - GALS (gradient_outside / RRR-style)
#   - GradCAM
#   - ABN supervision (ABN_SUPERVISION)
#   - Guided strategy sweep using ViT .pth maps as direct guidance
#   - Guided strategy sweep using RN50 .pth maps as direct guidance
# - Independent sweeps:
#   - Guided strategy sweep using external PNG masks (+ post-best reruns on extra GT roots)
#   - GALS with external PNG masks (ourmasks)
#   - UpWeight baseline
#   - ABN baseline
#   - CLIP + Logistic Regression
#   - Vanilla CNN
#
# Run from anywhere:
#   bash GALS/RedMeat_Runs/submit_all_redmeat_experiment.sh
#
# Optional dependency on an already-submitted ViT generation job:
#   VIT_JOB_ID=<jobid> bash GALS/RedMeat_Runs/submit_all_redmeat_experiment.sh
# Optional dependency on an already-submitted RN50 generation job:
#   RN50_JOB_ID=<jobid> bash GALS/RedMeat_Runs/submit_all_redmeat_experiment.sh
#
# Global knobs (export before calling):
#   N_TRIALS=50
#   POST_SEEDS=5
#   POST_SEED_START=0
#   SBATCH_TIME=15-00:00:00
#
# Skips:
#   SKIP_VIT_METHODS=1
#   SKIP_GALS_RN50=1
#   SKIP_GUIDED_GALSVIT=1
#   SKIP_GUIDED_GALSRN50=1
#   SKIP_GUIDED=1
#   SKIP_OURMASKS=1
#   SKIP_BASELINES=1
#   SKIP_CLIP_LR=1
#   SKIP_VANILLA_CNN=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
COMMON_ENV_CANDIDATES=(
  "${SCRIPT_DIR}/common_env.sh"
  "${SBATCH_SUBMIT_DIR:-}/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/Food101/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
)
COMMON_ENV=""
for candidate in "${COMMON_ENV_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    COMMON_ENV="$candidate"
    break
  fi
done
if [[ -z "$COMMON_ENV" ]]; then
  echo "[ERROR] Could not locate common_env.sh" >&2
  exit 2
fi
source "$COMMON_ENV"
redmeat_set_defaults

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1" >&2; exit 2; }
}
need_cmd sbatch

VIT_ATT_DIR=${VIT_ATT_DIR:-clip_vit_attention}
RN50_ATT_DIR=${RN50_ATT_DIR:-clip_rn50_attention_gradcam}
MASK_DIR=${MASK_DIR:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_dinovit/val/prediction_cmap/}

N_TRIALS=${N_TRIALS:-50}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
SBATCH_TIME=${SBATCH_TIME:-15-00:00:00}

EXPORT_SWEEP="ALL,N_TRIALS=${N_TRIALS},POST_SEEDS=${POST_SEEDS},POST_SEED_START=${POST_SEED_START},PROJECT_ROOT=${PROJECT_ROOT},GALS_ROOT=${GALS_ROOT},DATA_ROOT=${DATA_ROOT},DATA_DIR=${DATA_DIR},LOG_DIR=${LOG_DIR},MASK_DIR=${MASK_DIR}"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] GALS_ROOT=$GALS_ROOT"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] N_TRIALS=$N_TRIALS"
echo "[INFO] POST_SEEDS=$POST_SEEDS"
echo "[INFO] POST_SEED_START=$POST_SEED_START"
echo "[INFO] SBATCH_TIME=$SBATCH_TIME"

submit_parsable() {
  # shellcheck disable=SC2068
  sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" $@
}

dep_afterok() {
  echo "--dependency=afterok:$1"
}

VIT_JOB_ID=${VIT_JOB_ID:-}
if [[ -n "$VIT_JOB_ID" ]]; then
  echo "[SUBMIT] Using existing VIT_JOB_ID=$VIT_JOB_ID (for dependencies)"
else
  if [[ ! -d "${DATA_ROOT}/${DATA_DIR}/${VIT_ATT_DIR}" ]]; then
    echo "[WARN] ViT attention dir not found; ViT-dependent sweeps may fail." >&2
    echo "       Expected: ${DATA_ROOT}/${DATA_DIR}/${VIT_ATT_DIR}" >&2
    echo "       Provide VIT_JOB_ID=... or run: sbatch RedMeat_Runs/run_generate_redmeat_vit_attentions.sh" >&2
  fi
fi

RN50_JOB_ID=${RN50_JOB_ID:-}
if [[ -n "$RN50_JOB_ID" ]]; then
  echo "[SUBMIT] Using existing RN50_JOB_ID=$RN50_JOB_ID (for dependencies)"
else
  if [[ ! -d "${DATA_ROOT}/${DATA_DIR}/${RN50_ATT_DIR}" ]]; then
    echo "[WARN] RN50 attention dir not found; RN50-dependent sweeps may fail." >&2
    echo "       Expected: ${DATA_ROOT}/${DATA_DIR}/${RN50_ATT_DIR}" >&2
    echo "       Provide RN50_JOB_ID=... or run: sbatch RedMeat_Runs/run_generate_redmeat_rn50_attentions.sh" >&2
  fi
fi

echo "===================="
echo "[SUBMIT] Stage 1: ViT-attention dependent sweeps"
echo "===================="

depvit=()
if [[ -n "$VIT_JOB_ID" ]]; then depvit=("$(dep_afterok "$VIT_JOB_ID")"); fi

if [[ "${SKIP_VIT_METHODS:-0}" -ne 1 ]]; then
  echo "[SUBMIT] GALS/ViT method sweeps..."
  j_gals_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_redmeat_gals_vit_sweep_optuna.sh)"
  j_gc_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_redmeat_gradcam_vit_sweep_optuna.sh)"
  j_abn_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_redmeat_abn_vit_sweep_optuna.sh)"
  echo "  gals_vit:    $j_gals_vit"
  echo "  gradcam_vit: $j_gc_vit"
  echo "  abn_vit:     $j_abn_vit"
else
  echo "[SUBMIT] SKIP_VIT_METHODS=1"
fi

deprn50=()
if [[ -n "$RN50_JOB_ID" ]]; then deprn50=("$(dep_afterok "$RN50_JOB_ID")"); fi

if [[ "${SKIP_GALS_RN50:-0}" -ne 1 ]]; then
  echo "[SUBMIT] GALS sweep using RN50 attentions..."
  j_gals_rn50="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${deprn50[@]:-} run_redmeat_gals_rn50_sweep_optuna.sh)"
  echo "  gals_rn50: $j_gals_rn50"
else
  echo "[SUBMIT] SKIP_GALS_RN50=1"
fi

if [[ "${SKIP_GUIDED_GALSVIT:-0}" -ne 1 ]]; then
  echo "[SUBMIT] Guided strategy sweep using GALS ViT .pth attentions as guidance..."
  j_guided_galsvit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_guided_redmeat_gals_vitatt_sweep.sh)"
  echo "  guided_redmeat_galsvit: $j_guided_galsvit"
else
  echo "[SUBMIT] SKIP_GUIDED_GALSVIT=1"
fi

if [[ "${SKIP_GUIDED_GALSRN50:-0}" -ne 1 ]]; then
  echo "[SUBMIT] Guided strategy sweep using GALS RN50 .pth attentions as guidance..."
  j_guided_galsrn50="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${deprn50[@]:-} run_guided_redmeat_gals_rn50att_sweep.sh)"
  echo "  guided_redmeat_galsrn50: $j_guided_galsrn50"
else
  echo "[SUBMIT] SKIP_GUIDED_GALSRN50=1"
fi

if [[ "${SKIP_GUIDED:-0}" -ne 1 ]]; then
  echo "[SUBMIT] Guided strategy sweep using external PNG masks..."
  j_guided="$(submit_parsable run_guided_redmeat_sweep.sh)"
  echo "  guided_redmeat: $j_guided"
else
  echo "[SUBMIT] SKIP_GUIDED=1"
fi

echo "===================="
echo "[SUBMIT] Stage 2: independent sweeps"
echo "===================="

if [[ "${SKIP_OURMASKS:-0}" -ne 1 ]]; then
  if [[ -d "$MASK_DIR" ]]; then
    j_ourmasks="$(submit_parsable run_redmeat_gals_ourmasks_sweep_optuna.sh)"
    echo "  gals_ourmasks: $j_ourmasks"
  else
    echo "  [WARN] MASK_DIR missing: $MASK_DIR (skipping gals_ourmasks)"
  fi
else
  echo "[SUBMIT] SKIP_OURMASKS=1"
fi

if [[ "${SKIP_BASELINES:-0}" -ne 1 ]]; then
  echo "[SUBMIT] UpWeight + ABN baselines..."
  j_up="$(submit_parsable run_redmeat_upweight_sweep_optuna.sh)"
  echo "  upweight: $j_up"
  echo "  [NOTE] ABN needs: ${GALS_ROOT}/weights/resnet50_abn_imagenet.pth.tar"
  j_abn_base="$(submit_parsable run_redmeat_abn_sweep_optuna.sh)"
  echo "  abn_baseline: $j_abn_base"
else
  echo "[SUBMIT] SKIP_BASELINES=1"
fi

if [[ "${SKIP_CLIP_LR:-0}" -ne 1 ]]; then
  j_clip_lr="$(submit_parsable run_redmeat_clip_lr_sweep_optuna.sh)"
  echo "  clip_lr: $j_clip_lr"
else
  echo "[SUBMIT] SKIP_CLIP_LR=1"
fi

if [[ "${SKIP_VANILLA_CNN:-0}" -ne 1 ]]; then
  j_vanilla="$(submit_parsable run_redmeat_vanilla_cnn_sweep_optuna.sh)"
  echo "  vanilla_cnn: $j_vanilla"
else
  echo "[SUBMIT] SKIP_VANILLA_CNN=1"
fi

echo "[SUBMIT] Done."
