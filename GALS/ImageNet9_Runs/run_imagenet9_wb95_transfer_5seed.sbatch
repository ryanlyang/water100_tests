#!/usr/bin/env bash
#SBATCH --job-name=in9x95_method
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_transfer_%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_transfer_%x_%j.err

set -Eeuo pipefail

METHOD="${METHOD:?Submit with METHOD=erm|upweight|abn|elrep|gals|afr|clip_lr|r4rr}"
case "$METHOD" in
  erm|upweight|abn|elrep|gals|afr|clip_lr|r4rr) ;;
  *) echo "[ERROR] Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/manifest.csv}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/official_test_manifest.csv}"
OFFICIAL_TEST_ROOT="${OFFICIAL_TEST_ROOT:-${DATA_ROOT}/official_test/bg_challenge}"
CONFIG="${CONFIG:-${REPO}/ImageNet9_Runs/configs/waterbirds95_hparam_transfer.yaml}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUN_ROOT="${RUN_ROOT:-${LOG_ROOT}/transfer/waterbirds95/${METHOD}/main}"
SEEDS="${SEEDS:-0,1,2,3,4}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
GALS_MAP_ROOT="${GALS_MAP_ROOT:-${DATA_ROOT}/gals_maps/clip_vit_b32_transformer_v1}"
R4RR_MAP_ROOT="${R4RR_MAP_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1/inference/full/val/prediction_cmap}"
CLIP_FEATURE_CACHE="${CLIP_FEATURE_CACHE:-${LOG_ROOT}/sweeps/clip_lr/main/openai_clip_rn50_features.npz}"
ABN_CHECKPOINT="${ABN_CHECKPOINT:-}"

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

if [[ "$METHOD" == "abn" && -z "$ABN_CHECKPOINT" ]]; then
  for candidate in \
    "$REPO/weights/resnet50_abn_imagenet.pth.tar" \
    "/home/ryreu/guided_cnn/waterbirds/old_Waterbird_Runs/GALS/weights/resnet50_abn_imagenet.pth.tar"; do
    if [[ -f "$candidate" ]]; then ABN_CHECKPOINT="$candidate"; break; fi
  done
fi

echo "[$(date)] Host: $(hostname)"
echo "Experiment: Waterbirds-95 hyperparameter transfer to ImageNet-9"
echo "Method: $METHOD"
echo "Seeds: $SEEDS"
echo "Config: $CONFIG"
echo "Run root: $RUN_ROOT"
echo "Standard exposure match: 200 x 4795 -> 21 x 45405 images"
echo "ImageNet-9 validation use: checkpoint selection only"
echo "Official variants used for hyperparameter selection: NO"
"$PYTHON_BIN" --version

ARGS=(
  --method "$METHOD"
  --config "$CONFIG"
  --manifest "$MANIFEST"
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

case "$METHOD" in
  abn)
    [[ -f "$ABN_CHECKPOINT" ]] || { echo "[ERROR] ABN checkpoint not found" >&2; exit 2; }
    ARGS+=(--abn-checkpoint "$ABN_CHECKPOINT")
    ;;
  gals)
    [[ -d "$GALS_MAP_ROOT" ]] || { echo "[ERROR] GALS maps not found: $GALS_MAP_ROOT" >&2; exit 2; }
    ARGS+=(--gals-map-root "$GALS_MAP_ROOT")
    ;;
  r4rr)
    [[ -d "$R4RR_MAP_ROOT" ]] || { echo "[ERROR] R4RR maps not found: $R4RR_MAP_ROOT" >&2; exit 2; }
    ARGS+=(--r4rr-map-root "$R4RR_MAP_ROOT")
    ;;
  clip_lr)
    [[ -f "$CLIP_FEATURE_CACHE" ]] || { echo "[ERROR] CLIP feature cache not found: $CLIP_FEATURE_CACHE" >&2; exit 2; }
    ARGS+=(--clip-feature-cache "$CLIP_FEATURE_CACHE")
    ;;
esac

srun --unbuffered "$PYTHON_BIN" -u \
  ImageNet9_Runs/run_imagenet9_wb95_transfer_5seed.py "${ARGS[@]}"

echo "[$(date)] ImageNet-9 WB95 transfer $METHOD complete."
