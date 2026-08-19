#!/bin/bash -l
# Run one DecoyMNIST R4RR teacher-corruption condition over seeds 0-4.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=r4c_decoy
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

CONDITION="${CONDITION:?Submit with CONDITION=digit_0|...|digit_9|random_10pct}"
case "$CONDITION" in
  digit_[0-9]|random_10pct) ;;
  *) echo "[ERROR] Unsupported CONDITION=$CONDITION" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/r4rr_round2_systematic_teacher_corruption/decoymnist}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$RUN_ROOT/corruption_manifests}"
PNG_ROOT="${PNG_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"
TEACHER_MAP_PATH="${TEACHER_MAP_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap}"
HPARAMS_CONFIG="${HPARAMS_CONFIG:-$GALS_ROOT/RightForTheRightRegions/configs/r4rr_optimized_hparams.yaml}"
PYTHON_RUNNER="${PYTHON_RUNNER:-$GALS_ROOT/RightForTheRightRegions/repro_runs/r4rr/ablations/r4rr_decoy_systematic_teacher_corruption.py}"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
CORRUPTION_SEED="${CORRUPTION_SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"

mkdir -p "$LOG_DIR" "$RUN_ROOT" "$MANIFEST_ROOT" "$RUN_ROOT/$CONDITION"

# Conda activation hooks may read unset optional variables.
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$GALS_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled

cd "$GALS_ROOT"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[RUN] dataset=decoymnist condition=$CONDITION seeds=$SEEDS_CSV"
echo "[RUN] corruption_seed=$CORRUPTION_SEED random_fraction=0.10"
echo "[RUN] png_root=$PNG_ROOT"
echo "[RUN] teacher_maps=$TEACHER_MAP_PATH"
echo "[RUN] output=$RUN_ROOT/$CONDITION"
echo "[LOCKED] epochs=19 Adam lr=0.001 weight_decay=0.0001 attention_epoch=7 kl_lambda=495.61 kl_increment=0"
which python

[[ -d "$PNG_ROOT/train" && -d "$PNG_ROOT/test" ]] || {
  echo "[ERROR] Missing DecoyMNIST PNG train/test directories under $PNG_ROOT" >&2
  exit 2
}
[[ -d "$TEACHER_MAP_PATH" ]] || {
  echo "[ERROR] Missing R4RR teacher maps: $TEACHER_MAP_PATH" >&2
  exit 2
}
[[ -f "$HPARAMS_CONFIG" ]] || {
  echo "[ERROR] Missing optimized hyperparameter config: $HPARAMS_CONFIG" >&2
  exit 2
}
[[ -f "$PYTHON_RUNNER" ]] || {
  echo "[ERROR] Missing corruption runner: $PYTHON_RUNNER" >&2
  exit 2
}

command=(
  python -u "$PYTHON_RUNNER"
  --condition "$CONDITION"
  --png-root "$PNG_ROOT"
  --teacher-map-path "$TEACHER_MAP_PATH"
  --output-root "$RUN_ROOT"
  --manifest-root "$MANIFEST_ROOT"
  --hparams-config "$HPARAMS_CONFIG"
  --seeds "$SEEDS_CSV"
  --corruption-seed "$CORRUPTION_SEED"
  --random-fraction 0.10
  --split-seed 0
  --val-fraction 0.10
  --batch-size 64
  --test-batch-size 1000
  --num-workers "$NUM_WORKERS"
  --print-every 5
)
if [[ "$SAVE_CHECKPOINTS" == "1" ]]; then
  command+=(--save-checkpoints)
fi

"${command[@]}" 2>&1 | tee "$RUN_ROOT/$CONDITION/job_${SLURM_JOB_ID:-local}.log"

echo "[DONE] condition=$CONDITION"
echo "[DONE] summary=$RUN_ROOT/$CONDITION/summary.csv"

