#!/bin/bash -l
# Five-seed DecoyMNIST transfer of one best Waterbirds-100 alignment-loss trial.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=d5_align
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/%x_%j.err
#SBATCH --signal=TERM@120

set -Eeo pipefail

ALIGNMENT_LOSS=${ALIGNMENT_LOSS:?Set ALIGNMENT_LOSS}
case "$ALIGNMENT_LOSS" in reverse_kl|jensen_shannon|squared_l2|cosine) ;; *) echo "[ERROR] Invalid ALIGNMENT_LOSS=$ALIGNMENT_LOSS" >&2; exit 2;; esac

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"
set -u

export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PROJECT_ROOT=${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}
GALS_ROOT=${GALS_ROOT:-$PROJECT_ROOT/GALS}
PNG_ROOT=${PNG_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}
TEACHER_MAP_PATH=${TEACHER_MAP_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap}
SWEEP_LOG_DIR=${SWEEP_LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}
RUN_ROOT=${RUN_ROOT:-/home/ryreu/guided_cnn/logsMNIST/r4rr_alignment_transfer_best5}
SEEDS=${SEEDS:-0,1,2,3,4}
MIN_SWEEP_TRIALS=${MIN_SWEEP_TRIALS:-50}
NUM_WORKERS=${NUM_WORKERS:-4}
OUTPUT_CSV=${OUTPUT_CSV:-$RUN_ROOT/decoy_${ALIGNMENT_LOSS}_wb100_transfer_best5.csv}
SUMMARY_CSV=${SUMMARY_CSV:-$RUN_ROOT/decoy_${ALIGNMENT_LOSS}_wb100_transfer_best5_summary.csv}
PYTHON_RUNNER="$GALS_ROOT/run_decoymnist_r4rr_alignment_transfer.py"

mkdir -p "$RUN_ROOT"
for path in "$GALS_ROOT" "$PNG_ROOT" "$TEACHER_MAP_PATH" "$SWEEP_LOG_DIR" "$PYTHON_RUNNER"; do
  [[ -e "$path" ]] || { echo "[ERROR] Missing required path: $path" >&2; exit 1; }
done

cd "$GALS_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$GALS_ROOT:${PYTHONPATH:-}"

cmd=(
  python -u "$PYTHON_RUNNER"
  --alignment-loss "$ALIGNMENT_LOSS"
  --sweep-log-dir "$SWEEP_LOG_DIR"
  --min-sweep-trials "$MIN_SWEEP_TRIALS"
  --png-root "$PNG_ROOT"
  --teacher-map-path "$TEACHER_MAP_PATH"
  --seeds "$SEEDS"
  --num-workers "$NUM_WORKERS"
  --output-csv "$OUTPUT_CSV"
  --summary-csv "$SUMMARY_CSV"
)
if [[ -n "${SWEEP_CSV:-}" ]]; then
  cmd+=(--sweep-csv "$SWEEP_CSV")
fi

echo "[$(date)] Host: $(hostname)"
echo "Alignment loss: $ALIGNMENT_LOSS"
echo "WB100 sweep: ${SWEEP_CSV:-auto-resolve completed sweep from $SWEEP_LOG_DIR}"
echo "Fixed Decoy setup: epochs=19 lr=1e-3 weight_decay=1e-4 seeds=$SEEDS"
echo "Output CSV: $OUTPUT_CSV"
echo "Summary CSV: $SUMMARY_CSV"
which python

srun --unbuffered "${cmd[@]}"
