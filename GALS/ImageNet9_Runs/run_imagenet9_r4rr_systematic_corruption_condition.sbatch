#!/usr/bin/env bash
# One resumable five-seed ImageNet-9 R4RR corruption condition.
#SBATCH --job-name=in9c_r4rr
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

CONDITION="${CONDITION:?Submit with class_<name> or random_matched}"
case "$CONDITION" in
  class_dog|class_bird|class_vehicle|class_reptile|class_carnivore|class_insect|class_instrument|class_primate|class_fish|random_matched) ;;
  *) echo "[ERROR] Unsupported CONDITION=$CONDITION" >&2; exit 2 ;;
esac

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
PROTOCOL="${PROTOCOL:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL}/manifest.csv}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL}/official_test_manifest.csv}"
OFFICIAL_TEST_ROOT="${OFFICIAL_TEST_ROOT:-${DATA_ROOT}/official_test/bg_challenge}"
TEACHER_MAP_ROOT="${TEACHER_MAP_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1/inference/full/val/prediction_cmap}"
SWEEP_SUMMARY="${SWEEP_SUMMARY:-${LOG_ROOT}/sweeps/r4rr/main/summary.json}"
STUDY_ROOT="${STUDY_ROOT:-${LOG_ROOT}/r4rr_systematic_teacher_corruption/imagenet9}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${STUDY_ROOT}/corruption_manifests}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STUDY_ROOT}/${CONDITION}}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
SEEDS="${SEEDS:-0,1,2,3,4}"
CORRUPTION_SEED="${CORRUPTION_SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$MANIFEST_ROOT"
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
echo "Study: ImageNet-9 class-conditional R4RR teacher corruption"
echo "Condition: $CONDITION"
echo "Seeds: $SEEDS corruption_seed=$CORRUPTION_SEED"
echo "Teacher maps: $TEACHER_MAP_ROOT"
echo "Sweep summary: $SWEEP_SUMMARY"
echo "Output: $OUTPUT_ROOT"
echo "[LOCKED] 5,045/45,405 maps; 20 epochs; forward KL; kl_increment=0"
echo "[LOCKED] best ImageNet-9 validation-selected R4RR hyperparameters; no retuning"
"$PYTHON_BIN" --version

for path in "$MANIFEST" "$OFFICIAL_MANIFEST" "$SWEEP_SUMMARY"; do
  [[ -f "$path" ]] || { echo "[ERROR] Missing file: $path" >&2; exit 2; }
done
[[ -d "$OFFICIAL_TEST_ROOT" ]] || { echo "[ERROR] Missing official test root: $OFFICIAL_TEST_ROOT" >&2; exit 2; }
[[ -d "$TEACHER_MAP_ROOT" ]] || { echo "[ERROR] Missing teacher maps: $TEACHER_MAP_ROOT" >&2; exit 2; }

srun --unbuffered "$PYTHON_BIN" -u \
  ImageNet9_Runs/run_imagenet9_r4rr_systematic_corruption.py \
  --condition "$CONDITION" \
  --corruption-seed "$CORRUPTION_SEED" \
  --corruption-manifest-root "$MANIFEST_ROOT" \
  --sweep-summary "$SWEEP_SUMMARY" \
  --manifest "$MANIFEST" \
  --teacher-map-root "$TEACHER_MAP_ROOT" \
  --official-manifest "$OFFICIAL_MANIFEST" \
  --official-test-root "$OFFICIAL_TEST_ROOT" \
  --run-root "$OUTPUT_ROOT" \
  --seeds "$SEEDS" \
  --epochs 20 \
  --batch-size 96 \
  --eval-batch-size 256 \
  --num-workers "$NUM_WORKERS" \
  --kl-increment 0 \
  --device cuda:0 \
  --python "$PYTHON_BIN"

echo "[$(date)] Condition complete: $CONDITION"

