#!/usr/bin/env bash
#SBATCH --job-name=in9_r4rr_maps
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-45
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_r4rr_maps_%A_%a.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_r4rr_maps_%A_%a.err

set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
TEACHER_ROOT="${TEACHER_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${TEACHER_ROOT}/voc_workspace}"
TRAINING_RESULT="${TRAINING_RESULT:-${TEACHER_ROOT}/training/full/training_result.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TEACHER_ROOT}/inference/full/val}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
TOTAL_IMAGES="${TOTAL_IMAGES:-45405}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
START_INDEX="${START_INDEX:-$((TASK_INDEX * CHUNK_SIZE))}"
END_INDEX="${END_INDEX:-$((START_INDEX + CHUNK_SIZE))}"
if (( END_INDEX > TOTAL_IMAGES )); then END_INDEX="$TOTAL_IMAGES"; fi
CONDA_ENV="${CONDA_ENV:-r4rr-weclip}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/${CONDA_ENV}/bin/python}"

if (( START_INDEX >= TOTAL_IMAGES )); then
  echo "[SKIP] Empty shard start=$START_INDEX total=$TOTAL_IMAGES"
  exit 0
fi

mkdir -p /home/ryreu/guided_cnn/logsImageNet9 "$OUTPUT_ROOT"
cd "$REPO"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/RightForTheRightRegions/WeCLIPPlus:$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export CLIP_TEXT_DATASET=imagenet9
export CLIP_BACKEND=openclip
export CLIP_MODEL_NAME=ViT-B-16
export CLIP_PRETRAINED=openai
export WECLIP_EVAL_NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XFORMERS_DISABLED=1

echo "[$(date)] Host: $(hostname)"
echo "Checkpoint result: $TRAINING_RESULT"
echo "Workspace: $WORKSPACE_ROOT"
echo "Output: $OUTPUT_ROOT/prediction_cmap"
echo "Shard: [$START_INDEX,$END_INDEX)"
echo "Protocol: scales=1.0,1.5 flip-average + DenseCRF"
"$PYTHON_BIN" --version

srun --unbuffered "$PYTHON_BIN" -u ImageNet9_Runs/generate_imagenet9_r4rr_weclip_maps.py \
  --workspace-root "$WORKSPACE_ROOT" \
  --training-result "$TRAINING_RESULT" \
  --output-root "$OUTPUT_ROOT" \
  --start-index "$START_INDEX" \
  --end-index "$END_INDEX" \
  --num-workers "${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}" \
  --device cuda:0 \
  --test-scales 1.0 1.5

echo "[$(date)] ImageNet-9 R4RR map shard complete."
