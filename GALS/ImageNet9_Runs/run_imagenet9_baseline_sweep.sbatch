#!/usr/bin/env bash
#SBATCH --job-name=in9_sweep
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_sweep_%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_sweep_%x_%j.err

set -Eeuo pipefail

METHOD="${METHOD:?Submit with --export=ALL,METHOD=erm|upweight|abn|elrep}"
case "$METHOD" in
  erm|upweight|abn|elrep) ;;
  *) echo "[ERROR] Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/manifest.csv}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUN_TAG="${RUN_TAG:-main}"
RUN_ROOT="${RUN_ROOT:-${LOG_ROOT}/sweeps/${METHOD}/${RUN_TAG}}"
TARGET_COMPLETE_TRIALS="${TARGET_COMPLETE_TRIALS:-50}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-96}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
SWEEP_SEED="${SWEEP_SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-0}"
MAX_HOURS="${MAX_HOURS:-94}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"

mkdir -p "$LOG_ROOT" "$RUN_ROOT/trials"
cd "$REPO"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gals_a100
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DISABLED=true

ABN_CHECKPOINT="${ABN_CHECKPOINT:-}"
if [[ "$METHOD" == "abn" && -z "$ABN_CHECKPOINT" ]]; then
  candidates=(
    "$REPO/weights/resnet50_abn_imagenet.pth.tar"
    "/home/ryreu/guided_cnn/waterbirds/old_Waterbird_Runs/GALS/weights/resnet50_abn_imagenet.pth.tar"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      ABN_CHECKPOINT="$candidate"
      break
    fi
  done
fi
if [[ "$METHOD" == "abn" && ! -f "$ABN_CHECKPOINT" ]]; then
  echo "[ERROR] ABN checkpoint not found: ${ABN_CHECKPOINT:-NONE}" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Method: $METHOD"
echo "Manifest: $MANIFEST"
echo "Run root: $RUN_ROOT"
echo "Study: imagenet9_${METHOD}_${RUN_TAG}"
echo "Target completed trials: $TARGET_COMPLETE_TRIALS"
echo "Epochs: $EPOCHS batch_size=$BATCH_SIZE workers=$NUM_WORKERS"
echo "Maximum driver runtime: $MAX_HOURS hours"
echo "Objective: Original validation macro class accuracy"
echo "Official variants used for tuning: NO"
"$PYTHON_BIN" --version

ARGS=(
  --method "$METHOD"
  --manifest "$MANIFEST"
  --study-db "$RUN_ROOT/optuna.sqlite3"
  --study-name "imagenet9_${METHOD}_${RUN_TAG}"
  --output-csv "$RUN_ROOT/trials.csv"
  --summary-json "$RUN_ROOT/summary.json"
  --trial-logs "$RUN_ROOT/trials"
  --target-complete-trials "$TARGET_COMPLETE_TRIALS"
  --sweep-seed "$SWEEP_SEED"
  --train-seed "$TRAIN_SEED"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --weight-decay 1e-5
  --fixed-momentum 0.9
  --max-hours "$MAX_HOURS"
  --device cuda:0
  --python "$PYTHON_BIN"
)
if [[ "$METHOD" == "abn" ]]; then
  ARGS+=(--abn-checkpoint "$ABN_CHECKPOINT")
fi

srun --unbuffered "$PYTHON_BIN" -u ImageNet9_Runs/sweep_imagenet9_baseline.py "${ARGS[@]}"

echo "[$(date)] ImageNet-9 $METHOD sweep invocation finished."

