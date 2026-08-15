#!/usr/bin/env bash
#SBATCH --job-name=in9_final
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/in9_final_%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/in9_final_%x_%j.err

set -Eeuo pipefail

METHOD="${METHOD:?Submit with METHOD=erm|upweight|abn|elrep|afr|clip_lr}"
case "$METHOD" in
  erm|upweight|abn|elrep|afr|clip_lr) ;;
  *) echo "[ERROR] Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL_NAME="${PROTOCOL_NAME:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/manifest.csv}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL_NAME}/official_test_manifest.csv}"
OFFICIAL_TEST_ROOT="${OFFICIAL_TEST_ROOT:-${DATA_ROOT}/official_test/bg_challenge}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
SWEEP_ROOT="${SWEEP_ROOT:-${LOG_ROOT}/sweeps/${METHOD}/main}"
FINAL_ROOT="${FINAL_ROOT:-${LOG_ROOT}/final/${METHOD}/main}"
SEEDS="${SEEDS:-0,1,2,3,4}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

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

echo "[$(date)] Host: $(hostname)"
echo "Method: $METHOD"
echo "Seeds: $SEEDS"
echo "Sweep root: $SWEEP_ROOT"
echo "Final root: $FINAL_ROOT"
echo "Selection: Original validation macro class accuracy"
echo "Official variants used for selection: NO"
echo "Official evaluation: all eight Backgrounds Challenge variants"
"$PYTHON_BIN" --version

COMMON=(
  --official-manifest "$OFFICIAL_MANIFEST"
  --official-test-root "$OFFICIAL_TEST_ROOT"
  --run-root "$FINAL_ROOT"
  --seeds "$SEEDS"
  --num-workers "$NUM_WORKERS"
  --device cuda:0
)

if [[ "$METHOD" =~ ^(erm|upweight|abn|elrep)$ ]]; then
  ARGS=(
    --method "$METHOD"
    --sweep-summary "$SWEEP_ROOT/summary.json"
    --manifest "$MANIFEST"
    "${COMMON[@]}"
    --epochs 20
    --batch-size 96
    --eval-batch-size 256
    --python "$PYTHON_BIN"
  )
  if [[ "$METHOD" == "abn" ]]; then
    ABN_CHECKPOINT="${ABN_CHECKPOINT:-}"
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
    ARGS+=(--abn-checkpoint "$ABN_CHECKPOINT")
  fi
  srun --unbuffered "$PYTHON_BIN" -u \
    ImageNet9_Runs/run_imagenet9_final_baseline.py "${ARGS[@]}"
elif [[ "$METHOD" == "afr" ]]; then
  srun --unbuffered "$PYTHON_BIN" -u \
    ImageNet9_Runs/run_imagenet9_final_afr.py \
    --source-summary "$SWEEP_ROOT/summary.json" \
    --source-contract "$SWEEP_ROOT/contract.json" \
    --manifest "$MANIFEST" \
    "${COMMON[@]}" \
    --batch-size 96 \
    --embedding-batch-size 256 \
    --eval-batch-size 256 \
    --max-hours-per-seed 70 \
    --python "$PYTHON_BIN"
else
  srun --unbuffered "$PYTHON_BIN" -u \
    ImageNet9_Runs/run_imagenet9_final_clip_lr.py \
    --sweep-summary "$SWEEP_ROOT/summary.json" \
    --train-feature-cache "$SWEEP_ROOT/openai_clip_rn50_features.npz" \
    "${COMMON[@]}" \
    --batch-size 256
fi

echo "[$(date)] ImageNet-9 final $METHOD five-seed job complete."
