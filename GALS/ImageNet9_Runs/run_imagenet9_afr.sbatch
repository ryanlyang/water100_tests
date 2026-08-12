#!/usr/bin/env bash
#SBATCH --job-name=in9_afr_sweep
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_afr_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_afr_%j.err

set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/manifest.csv}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUN_TAG="${RUN_TAG:-main}"
RUN_ROOT="${RUN_ROOT:-${LOG_ROOT}/sweeps/afr/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
MAX_HOURS="${MAX_HOURS:-94}"

mkdir -p "$LOG_ROOT" "$RUN_ROOT"
cd "$REPO"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gals_a100
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ARGS=(
  --manifest "$MANIFEST"
  --run-root "$RUN_ROOT"
  --seed "${TRAIN_SEED:-0}"
  --split-seed 21
  --stage1-prop 0.8
  --stage1-epochs "${STAGE1_EPOCHS:-20}"
  --stage2-epochs "${STAGE2_EPOCHS:-500}"
  --batch-size "${BATCH_SIZE:-96}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE:-256}"
  --num-workers "${SLURM_CPUS_PER_TASK:-8}"
  --max-hours "$MAX_HOURS"
  --device cuda:0
)
if [[ "${AFR_SMOKE:-0}" == "1" ]]; then
  ARGS+=(--smoke)
fi

echo "[$(date)] Host: $(hostname)"
echo "Method: AFR native two-stage grid"
echo "Manifest: $MANIFEST"
echo "Run root: $RUN_ROOT"
echo "Objective: Original validation macro class accuracy"
echo "Official variants used for tuning: NO"

srun --unbuffered "$PYTHON_BIN" -u ImageNet9_Runs/run_imagenet9_afr.py "${ARGS[@]}"

echo "[$(date)] ImageNet-9 AFR invocation finished."
