#!/bin/bash -l
# AFR RedMeat rerun with 5 seeds, keeping the same training/grid setup.
#
# Usage:
#   sbatch run_afr_redmeat_repro_tier3_5seeds.sh
#
# Optional overrides:
#   sbatch --export=ALL,SEEDS=0,1,2,3,4,FULL_PAPER_GRID=1 run_afr_redmeat_repro_tier3_5seeds.sh

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/afr_redmeat_repro_tier3_5seeds_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/afr_redmeat_repro_tier3_5seeds_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Default to 5 seeds (can still be overridden via --export=ALL,SEEDS=...).
export SEEDS="${SEEDS:-0,1,2,3,4}"

# Keep outputs separated from 3-seed runs.
export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/guided_cnn/logsRedMeat/afr_repro_redmeat_full_5seeds_${SLURM_JOB_ID}}"
export LOGS_ROOT="${LOGS_ROOT:-/home/ryreu/guided_cnn/logsRedMeat/afr_repro_redmeat_full_5seeds_logs_${SLURM_JOB_ID}}"

echo "[5SEEDS] SEEDS=${SEEDS}"
echo "[5SEEDS] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[5SEEDS] LOGS_ROOT=${LOGS_ROOT}"

exec bash "${SCRIPT_DIR}/run_afr_redmeat_repro_tier3.sh"

