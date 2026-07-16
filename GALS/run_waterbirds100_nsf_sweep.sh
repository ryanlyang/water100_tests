#!/bin/bash -l
# NSF sweep (Optuna) for Waterbirds-100.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds100_nsf_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds100_nsf_sweep_%j.err
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
DATA_PATH=/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
NUM_EPOCHS=${NUM_EPOCHS:-200}

BASE_LR_MIN=${BASE_LR_MIN:-1e-5}
BASE_LR_MAX=${BASE_LR_MAX:-5e-2}
CLS_LR_MIN=${CLS_LR_MIN:-1e-5}
CLS_LR_MAX=${CLS_LR_MAX:-5e-2}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
MOMENTUM=${MOMENTUM:-0.9}
NESTEROV=${NESTEROV:-0}

TRANSFORM_LR_MIN=${TRANSFORM_LR_MIN:-1e-4}
TRANSFORM_LR_MAX=${TRANSFORM_LR_MAX:-1e-2}
TRANSFORM_STEPS_CHOICES=${TRANSFORM_STEPS_CHOICES:-5,10,25,50,100}
CLASSIFIER_STEPS_CHOICES=${CLASSIFIER_STEPS_CHOICES:-100,250,500,1000,2000}
NSF_CLASSIFIER_LR=${NSF_CLASSIFIER_LR:-1e-3}
BETA_REG_WEIGHT=${BETA_REG_WEIGHT:-10.0}

OUT_CSV=${OUT_CSV:-$LOG_DIR/nsf100_sweep_${SLURM_JOB_ID}.csv}
POST_OUT_CSV=${POST_OUT_CSV:-$LOG_DIR/nsf100_best5_${SLURM_JOB_ID}.csv}
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/NSF_Checkpoints}

cd "$REPO_ROOT/GALS"
export PYTHONPATH="$REPO_ROOT/GALS:$REPO_ROOT:${PYTHONPATH:-}"

python -c "import optuna" 2>/dev/null || {
  echo "[INFO] Installing optuna..."
  pip install -q optuna
}

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "Trials: $N_TRIALS (sampler=$SAMPLER sweep_seed=$SWEEP_SEED train_seed=$TRAIN_SEED)"
echo "Epochs: $NUM_EPOCHS"
echo "Ranges: base_lr=[$BASE_LR_MIN,$BASE_LR_MAX] cls_lr=[$CLS_LR_MIN,$CLS_LR_MAX] transform_lr=[$TRANSFORM_LR_MIN,$TRANSFORM_LR_MAX]"
echo "Choices: transform_steps={$TRANSFORM_STEPS_CHOICES} classifier_steps={$CLASSIFIER_STEPS_CHOICES}"
echo "Fixed: momentum=$MOMENTUM weight_decay=$WEIGHT_DECAY nsf_classifier_lr=$NSF_CLASSIFIER_LR beta_reg_weight=$BETA_REG_WEIGHT"
echo "Objective: nsf_val_balanced_group"
echo "Output CSV: $OUT_CSV"
echo "Post CSV: $POST_OUT_CSV"
which python

ARGS=(
  "$DATA_PATH"
  --n-trials "$N_TRIALS"
  --seed "$SWEEP_SEED"
  --train-seed "$TRAIN_SEED"
  --sampler "$SAMPLER"
  --num-epochs "$NUM_EPOCHS"
  --base-lr-min "$BASE_LR_MIN" --base-lr-max "$BASE_LR_MAX"
  --cls-lr-min "$CLS_LR_MIN" --cls-lr-max "$CLS_LR_MAX"
  --weight-decay "$WEIGHT_DECAY"
  --momentum "$MOMENTUM"
  --transform-lr-min "$TRANSFORM_LR_MIN" --transform-lr-max "$TRANSFORM_LR_MAX"
  --transform-steps-choices "$TRANSFORM_STEPS_CHOICES"
  --classifier-steps-choices "$CLASSIFIER_STEPS_CHOICES"
  --nsf-classifier-lr "$NSF_CLASSIFIER_LR"
  --beta-reg-weight "$BETA_REG_WEIGHT"
  --output-csv "$OUT_CSV"
  --post-seeds "$POST_SEEDS"
  --post-seed-start "$POST_SEED_START"
  --post-output-csv "$POST_OUT_CSV"
  --checkpoint-dir "$CKPT_DIR"
)

if [[ "$NESTEROV" -eq 1 ]]; then
  ARGS+=(--nesterov)
fi

srun --unbuffered python -u run_nsf_waterbird_sweep.py "${ARGS[@]}"
