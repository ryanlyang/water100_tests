#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/guided_sweep_95_resume_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/guided_sweep_95_resume_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}
LOG_DIR=${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}
DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}
SWEEP_GT_ROOT=${SWEEP_GT_ROOT:-/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}

# Default targets failed WB95 guided sweep job 21056086.
RESUME_CSV=${RESUME_CSV:-/home/ryreu/guided_cnn/logsWaterbird/guided_waterbird_sweep_21056086.csv}
N_TRIALS=${N_TRIALS:-50}

# Resume in-place by default.
SWEEP_OUT=${SWEEP_OUT:-$RESUME_CSV}

mkdir -p "$LOG_DIR"

if [[ ! -f "$RESUME_CSV" ]]; then
  echo "[ERROR] RESUME_CSV does not exist: $RESUME_CSV" >&2
  exit 1
fi
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[ERROR] Missing REPO_ROOT: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/run_guided_waterbird_sweep.py" ]]; then
  echo "[ERROR] Missing sweep runner: $REPO_ROOT/run_guided_waterbird_sweep.py" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] Missing DATA_ROOT: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$SWEEP_GT_ROOT" ]]; then
  echo "[ERROR] Missing SWEEP_GT_ROOT: $SWEEP_GT_ROOT" >&2
  exit 1
fi

# Most stable for transient image decode failures.
export GUIDED_NUM_WORKERS=${GUIDED_NUM_WORKERS:-0}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=0

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT"
echo "Sweep masks: $SWEEP_GT_ROOT"
echo "Resume CSV: $RESUME_CSV"
echo "Output CSV: $SWEEP_OUT"
echo "Target n_trials: $N_TRIALS"
echo "GUIDED_NUM_WORKERS=$GUIDED_NUM_WORKERS"
which python

# Guard against inherited client-side SLURM var mismatch at srun step launch.
unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK

srun --unbuffered python -u run_guided_waterbird_sweep.py \
  "$DATA_ROOT" \
  "$SWEEP_GT_ROOT" \
  --n-trials "$N_TRIALS" \
  --resume-csv "$RESUME_CSV" \
  --output-csv "$SWEEP_OUT"
