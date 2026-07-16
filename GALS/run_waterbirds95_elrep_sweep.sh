#!/bin/bash -l
# ElRep ERM sweep for Waterbirds-95 using our ResNet-50 student.
# Defaults: 50 Optuna trials, then rerun best hyperparams on seeds 0-4.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_elrep_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_elrep_sweep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME=${ENV_NAME:-gals_a100}
conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
DATA_PATH=/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
NUM_EPOCHS=${NUM_EPOCHS:-200}
NUM_WORKERS=${NUM_WORKERS:-4}

BASE_LR_MIN=${BASE_LR_MIN:-1e-5}
BASE_LR_MAX=${BASE_LR_MAX:-5e-2}
CLS_LR_MIN=${CLS_LR_MIN:-1e-5}
CLS_LR_MAX=${CLS_LR_MAX:-5e-2}
THETA1_MIN=${THETA1_MIN:-1e-5}
THETA1_MAX=${THETA1_MAX:-1e-2}
THETA2_MIN=${THETA2_MIN:-1e-6}
THETA2_MAX=${THETA2_MAX:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
MOMENTUM=${MOMENTUM:-0.9}
NESTEROV=${NESTEROV:-0}

OUT_CSV=${OUT_CSV:-$LOG_DIR/elrep95_sweep_${SLURM_JOB_ID}.csv}
POST_OUT_CSV=${POST_OUT_CSV:-$LOG_DIR/elrep95_best5_${SLURM_JOB_ID}.csv}
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/ElRep_Checkpoints}

cd "$REPO_ROOT/GALS"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "[ERROR] Missing DATA_PATH: $DATA_PATH" >&2
  exit 2
fi

python -c "import optuna" 2>/dev/null || { echo "[INFO] Installing optuna..."; pip install -q optuna; }

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "Method: ERM+ElRep ResNet-50"
echo "Trials: $N_TRIALS (sampler=$SAMPLER sweep_seed=$SWEEP_SEED train_seed=$TRAIN_SEED)"
echo "Epochs: $NUM_EPOCHS workers=$NUM_WORKERS"
echo "Ranges: base_lr=[$BASE_LR_MIN,$BASE_LR_MAX] cls_lr=[$CLS_LR_MIN,$CLS_LR_MAX] theta1=[$THETA1_MIN,$THETA1_MAX] theta2=[$THETA2_MIN,$THETA2_MAX]"
echo "Fixed: momentum=$MOMENTUM weight_decay=$WEIGHT_DECAY nesterov=$NESTEROV"
echo "Output CSV: $OUT_CSV"
echo "Post CSV: $POST_OUT_CSV"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
which python

ARGS=(
  "$DATA_PATH"
  --n-trials "$N_TRIALS"
  --seed "$SWEEP_SEED"
  --train-seed "$TRAIN_SEED"
  --sampler "$SAMPLER"
  --model resnet50
  --tune-mode full
  --pretrained
  --num-epochs "$NUM_EPOCHS"
  --num-workers "$NUM_WORKERS"
  --base-lr-min "$BASE_LR_MIN" --base-lr-max "$BASE_LR_MAX"
  --cls-lr-min "$CLS_LR_MIN" --cls-lr-max "$CLS_LR_MAX"
  --theta1-min "$THETA1_MIN" --theta1-max "$THETA1_MAX"
  --theta2-min "$THETA2_MIN" --theta2-max "$THETA2_MAX"
  --weight-decay "$WEIGHT_DECAY"
  --momentum "$MOMENTUM"
  --output-csv "$OUT_CSV"
  --post-seeds "$POST_SEEDS"
  --post-seed-start "$POST_SEED_START"
  --post-output-csv "$POST_OUT_CSV"
  --checkpoint-dir "$CKPT_DIR"
)

if [[ "$NESTEROV" -eq 1 ]]; then
  ARGS+=(--nesterov)
fi

srun --unbuffered python -u run_elrep_waterbird_sweep.py "${ARGS[@]}"
