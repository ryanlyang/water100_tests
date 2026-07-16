#!/bin/bash -l
# WB100 local sensitivity: lr2_mult*1.25 (from 0.123 -> 0.15375), others fixed.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/guided100_sens_lr2x2_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/guided100_sens_lr2x2_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

export LR2_MULT=0.15375
export SUMMARY_CSV="${SUMMARY_CSV:-/home/ryreu/guided_cnn/logsWaterbird/guided100_sens_lr2125_${SLURM_JOB_ID}.csv}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${SBATCH_SUBMIT_DIR:-${PWD:-}}}"

BASE_SCRIPT_CANDIDATES=(
  "${SUBMIT_DIR}/run_guided_waterbirds100_openclip_laion_5seeds.sh"
  "${SCRIPT_DIR}/run_guided_waterbirds100_openclip_laion_5seeds.sh"
  "/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/run_guided_waterbirds100_openclip_laion_5seeds.sh"
)
BASE_SCRIPT=""
for candidate in "${BASE_SCRIPT_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    BASE_SCRIPT="$candidate"
    break
  fi
done

if [[ -z "$BASE_SCRIPT" ]]; then
  echo "[ERROR] Could not locate run_guided_waterbirds100_openclip_laion_5seeds.sh" >&2
  echo "Checked: ${BASE_SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi

echo "[WRAPPER] Base script: $BASE_SCRIPT"
exec bash "$BASE_SCRIPT"
