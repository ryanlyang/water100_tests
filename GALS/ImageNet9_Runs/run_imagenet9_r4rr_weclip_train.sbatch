#!/usr/bin/env bash
#SBATCH --job-name=in9_r4rr_weclip
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_r4rr_weclip_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_r4rr_weclip_%j.err

set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1/voc_workspace}"
WECLIP_ROOT="${WECLIP_ROOT:-${REPO}/RightForTheRightRegions/WeCLIPPlus}"
MODE="${MODE:-full}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "[ERROR] MODE must be smoke or full, got: $MODE" >&2
  exit 2
fi
RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1/training/${MODE}}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SMOKE_ITERS="${SMOKE_ITERS:-2}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
CONDA_ENV="${CONDA_ENV:-r4rr-weclip}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/${CONDA_ENV}/bin/python}"

mkdir -p /home/ryreu/guided_cnn/logsImageNet9 "$RUN_ROOT"
cd "$REPO"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$WECLIP_ROOT:$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DISABLED=true
export XFORMERS_DISABLED=1

echo "[$(date)] Host: $(hostname)"
echo "Mode: $MODE"
echo "Repo: $REPO"
echo "VOC workspace: $WORKSPACE_ROOT"
echo "Run root: $RUN_ROOT"
echo "Conda environment: $CONDA_ENV"
echo "Teacher: OpenCLIP ViT-B/16 OpenAI + DINOv2 ViT-L/14-register"
echo "Batch size: $BATCH_SIZE workers=$NUM_WORKERS seed=$SEED"
echo "Checkpoint interval: $CHECKPOINT_INTERVAL"
"$PYTHON_BIN" --version

srun --unbuffered "$PYTHON_BIN" -u ImageNet9_Runs/train_imagenet9_r4rr_weclip.py \
  --workspace-root "$WORKSPACE_ROOT" \
  --weclip-root "$WECLIP_ROOT" \
  --run-root "$RUN_ROOT" \
  --mode "$MODE" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$BATCH_SIZE" \
  --smoke-iters "$SMOKE_ITERS" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL"

echo "[$(date)] ImageNet-9 WeCLIP+ $MODE training invocation finished."
