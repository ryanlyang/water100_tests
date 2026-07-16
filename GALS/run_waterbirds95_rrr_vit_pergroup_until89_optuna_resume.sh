#!/bin/bash -l
# Resume-like continuation for WB95 RRR ViT per_group sweep from prior CSV trials.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/wb95_rrr_vit_pergroup_until89_resume_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/wb95_rrr_vit_pergroup_until89_resume_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds
DATA_DIR=waterbird_complete95_forest2water2
ATT_DIR=clip_vit_attention

N_TRIALS=${N_TRIALS:-100000}
SWEEP_SEED=${SWEEP_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
KEEP=${KEEP:-best}
MAX_HOURS=${MAX_HOURS:-167.5}
BASE_LR_MIN=${BASE_LR_MIN:-1e-5}
BASE_LR_MAX=${BASE_LR_MAX:-5e-2}
CLS_LR_MIN=${CLS_LR_MIN:-1e-5}
CLS_LR_MAX=${CLS_LR_MAX:-5e-2}
OBJECTIVE=${OBJECTIVE:-per_group}
STOP_THRESHOLD=${STOP_THRESHOLD:-89.0}

RESUME_CSV=${RESUME_CSV:-}
if [[ -z "$RESUME_CSV" ]]; then
  RESUME_CSV="$(ls -t "$LOG_DIR"/wb95_rrr_vit_pergroup_until89_*.csv 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$RESUME_CSV" ]]; then
  echo "[ERROR] Could not auto-find prior CSV. Set RESUME_CSV=/path/to/csv" >&2
  exit 2
fi

# Append new rows to the same CSV by default.
OUT_CSV=${OUT_CSV:-$RESUME_CSV}
TRIAL_LOGS=${TRIAL_LOGS:-$LOG_DIR/wb95_rrr_vit_pergroup_until89_resume_logs_${SLURM_JOB_ID}}
RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-wb95_rrr_vit_pergroup_until89_resume_${SLURM_JOB_ID}}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
if [[ ! -d "$DATA_ROOT/$DATA_DIR/$ATT_DIR" ]]; then
  echo "[ERROR] Missing attention maps at: $DATA_ROOT/$DATA_DIR/$ATT_DIR" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Resume CSV: $RESUME_CSV"
echo "Output CSV: $OUT_CSV"
echo "Objective: $OBJECTIVE | stop_threshold: $STOP_THRESHOLD"
which python

srun --unbuffered python -u run_gals_sweep.py \
  --method rrr \
  --config configs/waterbirds_95_gals_vit.yaml \
  --data-root "$DATA_ROOT" \
  --waterbirds-dir "$DATA_DIR" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --train-seed "$TRAIN_SEED" \
  --sampler "$SAMPLER" \
  --keep "$KEEP" \
  --output-csv "$OUT_CSV" \
  --logs-dir "$TRIAL_LOGS" \
  --base-lr-min "$BASE_LR_MIN" \
  --base-lr-max "$BASE_LR_MAX" \
  --cls-lr-min "$CLS_LR_MIN" \
  --cls-lr-max "$CLS_LR_MAX" \
  --objective "$OBJECTIVE" \
  --stop-threshold "$STOP_THRESHOLD" \
  --max-hours "$MAX_HOURS" \
  --run-name-prefix "$RUN_NAME_PREFIX" \
  --resume-csv "$RESUME_CSV"
