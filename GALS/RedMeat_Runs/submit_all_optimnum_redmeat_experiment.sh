#!/usr/bin/env bash
set -Eeuo pipefail

# Submit the separate "optimnum" RedMeat experiment stack.
#
# Objective used by these new sweeps:
#   log_optim_num = log(val_acc) - beta * ig_fwd_kl
#
# This script intentionally submits ONLY the new optimnum variants.
# Existing baseline submit scripts are untouched.

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
OPTIM_BETA=${OPTIM_BETA:-1}
SBATCH_TIME=${SBATCH_TIME:-15-00:00:00}

EXPORT_SWEEP="ALL,N_TRIALS=${N_TRIALS},POST_SEEDS=${POST_SEEDS},POST_SEED_START=${POST_SEED_START},OPTIM_BETA=${OPTIM_BETA},PROJECT_ROOT=${PROJECT_ROOT},GALS_ROOT=${GALS_ROOT},DATA_ROOT=${DATA_ROOT},DATA_DIR=${DATA_DIR},LOG_DIR=${LOG_DIR},MASK_DIR=${MASK_DIR}"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] GALS_ROOT=$GALS_ROOT"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] N_TRIALS=$N_TRIALS"
echo "[INFO] POST_SEEDS=$POST_SEEDS"
echo "[INFO] POST_SEED_START=$POST_SEED_START"
echo "[INFO] OPTIM_BETA=$OPTIM_BETA"
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
    echo "[WARN] ViT attention dir not found; ViT-dependent optimnum sweeps may fail." >&2
    echo "       Expected: ${DATA_ROOT}/${DATA_DIR}/${VIT_ATT_DIR}" >&2
  fi
fi

RN50_JOB_ID=${RN50_JOB_ID:-}
if [[ -n "$RN50_JOB_ID" ]]; then
  echo "[SUBMIT] Using existing RN50_JOB_ID=$RN50_JOB_ID (for dependencies)"
else
  if [[ ! -d "${DATA_ROOT}/${DATA_DIR}/${RN50_ATT_DIR}" ]]; then
    echo "[WARN] RN50 attention dir not found; RN50-dependent optimnum sweeps may fail." >&2
    echo "       Expected: ${DATA_ROOT}/${DATA_DIR}/${RN50_ATT_DIR}" >&2
  fi
fi

echo "===================="
echo "[SUBMIT] Stage 1: optimnum sweeps (attention-dependent)"
echo "===================="

depvit=()
if [[ -n "$VIT_JOB_ID" ]]; then depvit=("$(dep_afterok "$VIT_JOB_ID")"); fi

deprn50=()
if [[ -n "$RN50_JOB_ID" ]]; then deprn50=("$(dep_afterok "$RN50_JOB_ID")"); fi

if [[ "${SKIP_VIT_METHODS:-0}" -ne 1 ]]; then
  j_gals_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_redmeat_gals_vit_sweep_optuna_optimnum.sh)"
  j_gc_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_redmeat_gradcam_vit_sweep_optuna_optimnum.sh)"
  j_abn_vit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_redmeat_abn_vit_sweep_optuna_optimnum.sh)"
  echo "  gals_vit_optimnum:    $j_gals_vit"
  echo "  gradcam_vit_optimnum: $j_gc_vit"
  echo "  abn_vit_optimnum:     $j_abn_vit"
else
  echo "[SUBMIT] SKIP_VIT_METHODS=1"
fi

if [[ "${SKIP_GALS_RN50:-0}" -ne 1 ]]; then
  j_gals_rn50="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${deprn50[@]:-} run_redmeat_gals_rn50_sweep_optuna_optimnum.sh)"
  echo "  gals_rn50_optimnum: $j_gals_rn50"
else
  echo "[SUBMIT] SKIP_GALS_RN50=1"
fi

if [[ "${SKIP_GUIDED_GALSVIT:-0}" -ne 1 ]]; then
  j_guided_galsvit="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${depvit[@]:-} run_guided_redmeat_gals_vitatt_sweep_optimnum.sh)"
  echo "  guided_redmeat_galsvit_optimnum: $j_guided_galsvit"
else
  echo "[SUBMIT] SKIP_GUIDED_GALSVIT=1"
fi

if [[ "${SKIP_GUIDED_GALSRN50:-0}" -ne 1 ]]; then
  j_guided_galsrn50="$(sbatch --parsable --time="$SBATCH_TIME" --export="$EXPORT_SWEEP" ${deprn50[@]:-} run_guided_redmeat_gals_rn50att_sweep_optimnum.sh)"
  echo "  guided_redmeat_galsrn50_optimnum: $j_guided_galsrn50"
else
  echo "[SUBMIT] SKIP_GUIDED_GALSRN50=1"
fi

if [[ "${SKIP_GUIDED:-0}" -ne 1 ]]; then
  j_guided="$(submit_parsable run_guided_redmeat_sweep_optimnum.sh)"
  echo "  guided_redmeat_optimnum: $j_guided"
else
  echo "[SUBMIT] SKIP_GUIDED=1"
fi

echo "===================="
echo "[SUBMIT] Stage 2: optimnum sweeps (independent)"
echo "===================="

if [[ "${SKIP_OURMASKS:-0}" -ne 1 ]]; then
  if [[ -d "$MASK_DIR" ]]; then
    j_ourmasks="$(submit_parsable run_redmeat_gals_ourmasks_sweep_optuna_optimnum.sh)"
    echo "  gals_ourmasks_optimnum: $j_ourmasks"
  else
    echo "  [WARN] MASK_DIR missing: $MASK_DIR (skipping gals_ourmasks_optimnum)"
  fi
else
  echo "[SUBMIT] SKIP_OURMASKS=1"
fi

echo "[SUBMIT] Done."
