#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/afr_redmeat_repro_tier3_constant_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/afr_redmeat_repro_tier3_constant_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

SCRIPT_PATH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Force stage-1 schedule for this variant.
export STAGE1_SCHEDULER=constant_lr_scheduler

exec "${SCRIPT_PATH_DIR}/run_afr_redmeat_repro_tier3.sh" "$@"
