#!/bin/bash -l
# Five-seed evaluation of one completed R4RR alignment-loss sweep.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=r4rr_align_best5
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.err
#SBATCH --signal=TERM@120

set -Eeo pipefail

DATASET=${DATASET:?Set DATASET to wb95, wb100, or redmeat}
ALIGNMENT_LOSS=${ALIGNMENT_LOSS:?Set ALIGNMENT_LOSS}
case "$DATASET" in wb95|wb100|redmeat) ;; *) echo "[ERROR] Invalid DATASET=$DATASET" >&2; exit 2;; esac
case "$ALIGNMENT_LOSS" in reverse_kl|jensen_shannon|squared_l2|cosine) ;; *) echo "[ERROR] Invalid ALIGNMENT_LOSS=$ALIGNMENT_LOSS" >&2; exit 2;; esac

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"
set -u

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}
export GUIDED_NUM_WORKERS=${GUIDED_NUM_WORKERS:-4}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}
GALS_ROOT=${GALS_ROOT:-$REPO_ROOT/GALS}
SEEDS=${SEEDS:-0,1,2,3,4}
MIN_SWEEP_TRIALS=${MIN_SWEEP_TRIALS:-50}
RUN_ROOT=${RUN_ROOT:-/home/ryreu/guided_cnn/logsWaterbird/r4rr_alignment_best5}

case "$DATASET" in
  wb95)
    DATA_PATH=${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}
    TEACHER_MAP_PATH=${TEACHER_MAP_PATH:-/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
    SWEEP_LOG_DIR=${SWEEP_LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}
    ;;
  wb100)
    DATA_PATH=${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}
    TEACHER_MAP_PATH=${TEACHER_MAP_PATH:-/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
    SWEEP_LOG_DIR=${SWEEP_LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}
    ;;
  redmeat)
    DATA_PATH=${DATA_PATH:-/home/ryreu/guided_cnn/Food101/data/food-101-redmeat}
    TEACHER_MAP_PATH=${TEACHER_MAP_PATH:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_dinovit/val/prediction_cmap}
    SWEEP_LOG_DIR=${SWEEP_LOG_DIR:-/home/ryreu/guided_cnn/logsRedMeat}
    ;;
esac

PYTHON_RUNNER="$GALS_ROOT/run_r4rr_alignment_best5.py"
OUTPUT_CSV=${OUTPUT_CSV:-$RUN_ROOT/${DATASET}_${ALIGNMENT_LOSS}_best5.csv}
SUMMARY_CSV=${SUMMARY_CSV:-$RUN_ROOT/${DATASET}_${ALIGNMENT_LOSS}_best5_summary.csv}
mkdir -p "$RUN_ROOT"

for path in "$GALS_ROOT" "$DATA_PATH" "$TEACHER_MAP_PATH" "$SWEEP_LOG_DIR" "$PYTHON_RUNNER"; do
  [[ -e "$path" ]] || { echo "[ERROR] Missing required path: $path" >&2; exit 1; }
done

cd "$GALS_ROOT"
export PYTHONPATH="$REPO_ROOT:$GALS_ROOT:${PYTHONPATH:-}"

cmd=(
  python -u "$PYTHON_RUNNER"
  --dataset "$DATASET"
  --alignment-loss "$ALIGNMENT_LOSS"
  --log-dir "$SWEEP_LOG_DIR"
  --min-sweep-trials "$MIN_SWEEP_TRIALS"
  --data-path "$DATA_PATH"
  --teacher-map-path "$TEACHER_MAP_PATH"
  --seeds "$SEEDS"
  --output-csv "$OUTPUT_CSV"
  --summary-csv "$SUMMARY_CSV"
)
if [[ -n "${SWEEP_CSV:-}" ]]; then
  cmd+=(--sweep-csv "$SWEEP_CSV")
fi

echo "[$(date)] Host: $(hostname)"
echo "Dataset: $DATASET"
echo "Alignment loss: $ALIGNMENT_LOSS"
echo "Seeds: $SEEDS"
echo "Sweep CSV: ${SWEEP_CSV:-auto-resolve completed sweep from $SWEEP_LOG_DIR}"
echo "Per-seed CSV: $OUTPUT_CSV"
echo "Summary CSV: $SUMMARY_CSV"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
which python

srun --unbuffered "${cmd[@]}"
