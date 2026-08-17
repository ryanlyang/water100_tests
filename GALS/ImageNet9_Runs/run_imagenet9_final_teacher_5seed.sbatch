#!/usr/bin/env bash
#SBATCH --job-name=in9ft_teacher
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_final_%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_final_%x_%j.err

set -Eeuo pipefail

VARIANT="${VARIANT:?Submit with a registered GALS or R4RR final variant}"
case "$VARIANT" in
  gals|gals_gradcam|gals_abn|gals_rn50|r4rr|r4rr_trial13|r4rr_reverse_kl|r4rr_jensen_shannon|r4rr_squared_l2|r4rr_cosine) ;;
  *) echo "[ERROR] Unsupported VARIANT=$VARIANT" >&2; exit 2 ;;
esac

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/manifest.csv}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/official_test_manifest.csv}"
OFFICIAL_TEST_ROOT="${OFFICIAL_TEST_ROOT:-${DATA_ROOT}/official_test/bg_challenge}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
SEEDS="${SEEDS:-0,1,2,3,4}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
ABN_CHECKPOINT="${ABN_CHECKPOINT:-}"
R4RR_MAP_ROOT="${R4RR_MAP_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1/inference/full/val/prediction_cmap}"
VIT_MAP_ROOT="${VIT_MAP_ROOT:-${DATA_ROOT}/gals_maps/clip_vit_b32_transformer_v1}"
RN50_MAP_ROOT="${RN50_MAP_ROOT:-${DATA_ROOT}/gals_maps/clip_rn50_gradcam_v1}"

case "$VARIANT" in
  gals)
    METHOD="gals"
    SWEEP_DIR="gals"
    MAP_ROOT="$VIT_MAP_ROOT"
    ;;
  gals_gradcam)
    METHOD="gals_gradcam"
    SWEEP_DIR="gals_gradcam"
    MAP_ROOT="$VIT_MAP_ROOT"
    ;;
  gals_abn)
    METHOD="gals_abn"
    SWEEP_DIR="gals_abn"
    MAP_ROOT="$VIT_MAP_ROOT"
    ;;
  gals_rn50)
    METHOD="gals"
    SWEEP_DIR="gals_rn50"
    MAP_ROOT="$RN50_MAP_ROOT"
    ;;
  r4rr)
    METHOD="r4rr"
    SWEEP_DIR="r4rr"
    ALIGNMENT_LOSS="forward_kl"
    ;;
  r4rr_trial13)
    METHOD="r4rr"
    SWEEP_DIR="r4rr"
    ALIGNMENT_LOSS="forward_kl"
    TRIAL_NUMBER=13
    ;;
  r4rr_reverse_kl)
    METHOD="r4rr"
    SWEEP_DIR="r4rr_reverse_kl"
    ALIGNMENT_LOSS="reverse_kl"
    ;;
  r4rr_jensen_shannon)
    METHOD="r4rr"
    SWEEP_DIR="r4rr_jensen_shannon"
    ALIGNMENT_LOSS="jensen_shannon"
    ;;
  r4rr_squared_l2)
    METHOD="r4rr"
    SWEEP_DIR="r4rr_squared_l2"
    ALIGNMENT_LOSS="squared_l2"
    ;;
  r4rr_cosine)
    METHOD="r4rr"
    SWEEP_DIR="r4rr_cosine"
    ALIGNMENT_LOSS="cosine"
    ;;
esac

SWEEP_ROOT="${SWEEP_ROOT:-${LOG_ROOT}/sweeps/${SWEEP_DIR}/main}"
FINAL_ROOT="${FINAL_ROOT:-${LOG_ROOT}/final/${VARIANT}/main}"
mkdir -p "$LOG_ROOT" "$FINAL_ROOT"
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
echo "Variant: $VARIANT"
echo "Seeds: $SEEDS"
echo "Sweep root: $SWEEP_ROOT"
echo "Final root: $FINAL_ROOT"
echo "Selection: Original validation macro class accuracy"
echo "Official variants used for selection: NO"
echo "Official evaluation: all eight Backgrounds Challenge variants"
"$PYTHON_BIN" --version

COMMON=(
  --sweep-summary "$SWEEP_ROOT/summary.json"
  --manifest "$MANIFEST"
  --official-manifest "$OFFICIAL_MANIFEST"
  --official-test-root "$OFFICIAL_TEST_ROOT"
  --run-root "$FINAL_ROOT"
  --seeds "$SEEDS"
  --epochs 20
  --batch-size 96
  --eval-batch-size 256
  --num-workers "$NUM_WORKERS"
  --device cuda:0
  --python "$PYTHON_BIN"
)

if [[ "$METHOD" == "r4rr" ]]; then
  R4RR_EXTRA=()
  if [[ -n "${TRIAL_NUMBER:-}" ]]; then
    R4RR_EXTRA+=(--trial-number "$TRIAL_NUMBER")
  fi
  srun --unbuffered "$PYTHON_BIN" -u \
    ImageNet9_Runs/run_imagenet9_final_r4rr.py \
    --alignment-loss "$ALIGNMENT_LOSS" \
    --teacher-map-root "$R4RR_MAP_ROOT" \
    "${R4RR_EXTRA[@]}" \
    "${COMMON[@]}"
else
  EXTRA=(--method "$METHOD" --gals-map-root "$MAP_ROOT")
  if [[ "$METHOD" == "gals_abn" ]]; then
    candidates=(
      "$REPO/weights/resnet50_abn_imagenet.pth.tar"
      "/home/ryreu/guided_cnn/waterbirds/old_Waterbird_Runs/GALS/weights/resnet50_abn_imagenet.pth.tar"
    )
    if [[ -z "$ABN_CHECKPOINT" ]]; then
      for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then ABN_CHECKPOINT="$candidate"; break; fi
      done
    fi
    if [[ ! -f "$ABN_CHECKPOINT" ]]; then
      echo "[ERROR] ABN checkpoint not found: ${ABN_CHECKPOINT:-NONE}" >&2
      exit 2
    fi
    EXTRA+=(--abn-checkpoint "$ABN_CHECKPOINT")
  fi
  srun --unbuffered "$PYTHON_BIN" -u \
    ImageNet9_Runs/run_imagenet9_final_baseline.py \
    "${EXTRA[@]}" \
    "${COMMON[@]}"
fi

echo "[$(date)] ImageNet-9 final $VARIANT five-seed job complete."
