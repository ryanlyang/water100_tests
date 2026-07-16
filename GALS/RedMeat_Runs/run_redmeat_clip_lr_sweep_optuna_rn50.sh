#!/bin/bash -l
# Explicit RN50 CLIP+LR sweep for RedMeat.
# This is a thin wrapper around the main CLIP+LR sweep script with CLIP_MODEL forced to RN50.
#
# Usage:
#   sbatch RedMeat_Runs/run_redmeat_clip_lr_sweep_optuna_rn50.sh

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_lr_rn50_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_lr_rn50_sweep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${SBATCH_SUBMIT_DIR:-${PWD:-}}}"

# Force RN50 backbone for CLIP feature extraction.
export CLIP_MODEL=RN50

# Use RN50-specific default output names unless caller overrides.
export OUT_CSV="${OUT_CSV:-/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_lr_rn50_sweep_${SLURM_JOB_ID}.csv}"
export POST_OUT_CSV="${POST_OUT_CSV:-/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_lr_rn50_best5_${SLURM_JOB_ID}.csv}"

BASE_SCRIPT_CANDIDATES=(
  "${SCRIPT_DIR}/run_redmeat_clip_lr_sweep_optuna.sh"
  "${SUBMIT_DIR}/RedMeat_Runs/run_redmeat_clip_lr_sweep_optuna.sh"
  "/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/RedMeat_Runs/run_redmeat_clip_lr_sweep_optuna.sh"
  "/home/ryreu/guided_cnn/Food101/Waterbird_Runs/GALS/RedMeat_Runs/run_redmeat_clip_lr_sweep_optuna.sh"
)
BASE_SCRIPT=""
for candidate in "${BASE_SCRIPT_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    BASE_SCRIPT="$candidate"
    break
  fi
done

if [[ -z "$BASE_SCRIPT" ]]; then
  echo "[ERROR] Could not locate run_redmeat_clip_lr_sweep_optuna.sh" >&2
  echo "Checked: ${BASE_SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi

echo "[RN50 wrapper] Base script: $BASE_SCRIPT"
exec bash "$BASE_SCRIPT"
