#!/bin/bash -l
# Vanilla CNN sweep (Optuna) for Waterbirds-95.
# Defaults:
# - 100 trials (TPE)
# - rerun best hyperparams for 5 seeds

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_vanilla_cnn_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_vanilla_cnn_sweep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME=${ENV_NAME:-gals_a100}
BOOTSTRAP_ENV=${BOOTSTRAP_ENV:-0}
RECREATE_ENV=${RECREATE_ENV:-0}
REQ_FILE=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/requirements.txt

if [[ "$BOOTSTRAP_ENV" -eq 1 ]]; then
  if [[ "$RECREATE_ENV" -eq 1 ]]; then
    conda env remove -n "$ENV_NAME" -y || true
  fi
  if ! conda env list | grep -E "^${ENV_NAME}[[:space:]]" >/dev/null; then
    conda create -y -n "$ENV_NAME" python=3.8
    conda activate "$ENV_NAME"
    conda install -y pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -c nvidia -c conda-forge
    conda install -y -c conda-forge pycocotools
    REQ_TMP=/tmp/${ENV_NAME}_reqs_$$.txt
    grep -v -E '^(opencv-python|pycocotools|torch|torchvision|torchray)' "$REQ_FILE" > "$REQ_TMP"
    pip install -r "$REQ_TMP"
    rm -f "$REQ_TMP"
    pip install torchray==1.0.0.2 --no-deps
    pip install opencv-python==4.6.0.66
    conda deactivate
  fi
fi

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

BASE_LR_MIN=${BASE_LR_MIN:-1e-5}
BASE_LR_MAX=${BASE_LR_MAX:-5e-2}
CLS_LR_MIN=${CLS_LR_MIN:-1e-5}
CLS_LR_MAX=${CLS_LR_MAX:-5e-2}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
MOMENTUM_MIN=${MOMENTUM_MIN:-0.85}
MOMENTUM_MAX=${MOMENTUM_MAX:-0.95}
NESTEROV=${NESTEROV:-0}

OUT_CSV=${OUT_CSV:-$LOG_DIR/vanilla95_sweep_${SLURM_JOB_ID}.csv}
POST_OUT_CSV=${POST_OUT_CSV:-$LOG_DIR/vanilla95_best5_${SLURM_JOB_ID}.csv}
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/Vanilla_Checkpoints}

cd "$REPO_ROOT/GALS"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Missing DATA_PATH: $DATA_PATH" >&2
  exit 1
fi

python -c "import optuna" 2>/dev/null || {
  echo "[INFO] Installing optuna..."
  pip install -q optuna
}

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "Trials: $N_TRIALS (sampler=$SAMPLER sweep_seed=$SWEEP_SEED train_seed=$TRAIN_SEED)"
echo "Epochs: $NUM_EPOCHS"
echo "Ranges: base_lr=[$BASE_LR_MIN,$BASE_LR_MAX] cls_lr=[$CLS_LR_MIN,$CLS_LR_MAX] momentum=[$MOMENTUM_MIN,$MOMENTUM_MAX]"
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
  --num-epochs "$NUM_EPOCHS"
  --base-lr-min "$BASE_LR_MIN" --base-lr-max "$BASE_LR_MAX"
  --cls-lr-min "$CLS_LR_MIN" --cls-lr-max "$CLS_LR_MAX"
  --weight-decay "$WEIGHT_DECAY"
  --momentum-min "$MOMENTUM_MIN"
  --momentum-max "$MOMENTUM_MAX"
  --output-csv "$OUT_CSV"
  --post-seeds "$POST_SEEDS"
  --post-seed-start "$POST_SEED_START"
  --post-output-csv "$POST_OUT_CSV"
  --checkpoint-dir "$CKPT_DIR"
)

if [[ "$NESTEROV" -eq 1 ]]; then
  ARGS+=(--nesterov)
fi

srun --unbuffered python -u run_vanilla_waterbird_sweep.py "${ARGS[@]}"
