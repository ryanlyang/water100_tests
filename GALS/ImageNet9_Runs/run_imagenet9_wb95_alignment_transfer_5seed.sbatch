#!/usr/bin/env bash
#SBATCH --job-name=in9x95_align
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_transfer_%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_transfer_%x_%j.err

set -Eeuo pipefail

ALIGNMENT_LOSS="${ALIGNMENT_LOSS:?Set reverse_kl|jensen_shannon|squared_l2|cosine}"
case "$ALIGNMENT_LOSS" in
  reverse_kl|jensen_shannon|squared_l2|cosine) ;;
  *) echo "[ERROR] Unsupported ALIGNMENT_LOSS=$ALIGNMENT_LOSS" >&2; exit 2 ;;
esac

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/manifest.csv}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/official_test_manifest.csv}"
OFFICIAL_TEST_ROOT="${OFFICIAL_TEST_ROOT:-${DATA_ROOT}/official_test/bg_challenge}"
TEACHER_MAP_ROOT="${TEACHER_MAP_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1/inference/full/val/prediction_cmap}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUN_ROOT="${RUN_ROOT:-${LOG_ROOT}/transfer/waterbirds95_alignment/${ALIGNMENT_LOSS}/main}"
SEEDS="${SEEDS:-0,1,2,3,4}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

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
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[$(date)] Host: $(hostname)"
echo "Experiment: WB95 alignment-loss hyperparameter transfer to ImageNet-9"
echo "Alignment loss: $ALIGNMENT_LOSS"
echo "Sweep source: $SWEEP_LOG_DIR"
echo "Seeds: $SEEDS"
echo "Target schedule: 21 epochs, exposure-scaled attention epoch, kl_increment=0"
echo "Run root: $RUN_ROOT"
echo "ImageNet-9 validation use: checkpoint selection only"
"$PYTHON_BIN" --version

ARGS=(
  --alignment-loss "$ALIGNMENT_LOSS"
  --sweep-log-dir "$SWEEP_LOG_DIR"
  --min-sweep-trials 50
  --manifest "$MANIFEST"
  --teacher-map-root "$TEACHER_MAP_ROOT"
  --official-manifest "$OFFICIAL_MANIFEST"
  --official-test-root "$OFFICIAL_TEST_ROOT"
  --run-root "$RUN_ROOT"
  --seeds "$SEEDS"
  --batch-size 96
  --eval-batch-size 256
  --num-workers "$NUM_WORKERS"
  --device cuda:0
  --python "$PYTHON_BIN"
)
if [[ -n "${SWEEP_CSV:-}" ]]; then
  ARGS+=(--sweep-csv "$SWEEP_CSV")
fi

srun --unbuffered "$PYTHON_BIN" -u \
  ImageNet9_Runs/run_imagenet9_wb95_alignment_transfer_5seed.py "${ARGS[@]}"

echo "[$(date)] WB95 $ALIGNMENT_LOSS transfer complete."
