#!/bin/bash -l
# Run one RedMeat R4RR teacher-corruption condition over seeds 0-4.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=r4c_meat
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

CONDITION="${CONDITION:?Submit with CONDITION=class_<redmeat_class>|random_20pct}"
case "$CONDITION" in
  class_baby_back_ribs|class_filet_mignon|class_pork_chop|class_prime_rib|class_steak|random_20pct) ;;
  *) echo "[ERROR] Unsupported CONDITION=$CONDITION" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsRedMeat}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/r4rr_round2_systematic_teacher_corruption/redmeat}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$RUN_ROOT/corruption_manifests}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/Food101/data/food-101-redmeat}"
TEACHER_MAP_PATH="${TEACHER_MAP_PATH:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_laion_dinovit/val/prediction_cmap}"
HPARAMS_CONFIG="${HPARAMS_CONFIG:-$GALS_ROOT/RightForTheRightRegions/configs/r4rr_optimized_hparams.yaml}"
PYTHON_RUNNER="${PYTHON_RUNNER:-$GALS_ROOT/RightForTheRightRegions/repro_runs/r4rr/ablations/r4rr_redmeat_systematic_teacher_corruption.py}"

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
echo "[RUN] dataset=redmeat condition=$CONDITION seeds=$SEEDS_CSV"
echo "[RUN] corruption_seed=$CORRUPTION_SEED exact_random_count=500"
echo "[RUN] data_root=$DATA_ROOT"
echo "[RUN] teacher_maps=$TEACHER_MAP_PATH"
echo "[RUN] output=$RUN_ROOT/$CONDITION"
echo "[LOCKED] ResNet50 epochs=150 batch=96 SGD momentum=0.9 weight_decay=1e-5"
echo "[LOCKED] base_lr=0.0024 classifier_lr=0.000233 attention_epoch=2 kl_lambda=11.44 lr2_mult=1.567 kl_increment=0"
which python

[[ -f "$DATA_ROOT/all_images.csv" ]] || {
  echo "[ERROR] Missing RedMeat metadata: $DATA_ROOT/all_images.csv" >&2
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
  --data-root "$DATA_ROOT"
  --teacher-map-path "$TEACHER_MAP_PATH"
  --output-root "$RUN_ROOT"
  --manifest-root "$MANIFEST_ROOT"
  --hparams-config "$HPARAMS_CONFIG"
  --seeds "$SEEDS_CSV"
  --corruption-seed "$CORRUPTION_SEED"
  --random-count 500
  --batch-size 96
  --num-workers "$NUM_WORKERS"
)
if [[ "$SAVE_CHECKPOINTS" == "1" ]]; then
  command+=(--save-checkpoints)
fi

"${command[@]}" 2>&1 | tee "$RUN_ROOT/$CONDITION/job_${SLURM_JOB_ID:-local}.log"

echo "[DONE] condition=$CONDITION"
echo "[DONE] summary=$RUN_ROOT/$CONDITION/summary.csv"

