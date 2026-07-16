#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_gals_ourmasks_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_gals_ourmasks_sweep_%j.err
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds
DATA_DIR=waterbird_complete95_forest2water2

# Your binary masks (black/red) directory:
MASK_DIR=${MASK_DIR:-/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
# Be robust if MASK_DIR accidentally includes surrounding quotes (common with --export usage).
MASK_DIR="${MASK_DIR%\"}"
MASK_DIR="${MASK_DIR#\"}"
MASK_DIR="${MASK_DIR%\'}"
MASK_DIR="${MASK_DIR#\'}"
ALT_MASK1_DIR=${ALT_MASK1_DIR:-/home/ryreu/guided_cnn/waterbirds/New_Teach/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
ALT_MASK2_DIR=${ALT_MASK2_DIR:-/home/ryreu/guided_cnn/waterbirds/newCLIP/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
ALT_MASK3_DIR=${ALT_MASK3_DIR:-/home/ryreu/guided_cnn/waterbirds/newCLIP/LearningToLook/code/WeCLIPPlus/results_siglip2/val/prediction_cmap}
POST_MASK_DIRS=${POST_MASK_DIRS:-$ALT_MASK1_DIR,$ALT_MASK2_DIR,$ALT_MASK3_DIR}
POST_MASK_LABELS=${POST_MASK_LABELS:-new_teach,newclip,siglip2}

N_TRIALS=${N_TRIALS:-100}
SWEEP_SEED=${SWEEP_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
KEEP=${KEEP:-best}
MAX_HOURS=${MAX_HOURS:-}
TUNE_WEIGHT_DECAY=${TUNE_WEIGHT_DECAY:-0}
BASE_LR_MIN=${BASE_LR_MIN:-1e-5}
BASE_LR_MAX=${BASE_LR_MAX:-5e-2}
CLS_LR_MIN=${CLS_LR_MIN:-1e-5}
CLS_LR_MAX=${CLS_LR_MAX:-5e-2}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
POST_KEEP=${POST_KEEP:-all}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT/$DATA_DIR"
echo "MASK_DIR: $MASK_DIR"
echo "POST_MASK_DIRS: $POST_MASK_DIRS"
echo "POST_MASK_LABELS: $POST_MASK_LABELS"
echo "Trials: $N_TRIALS (sampler=$SAMPLER sweep_seed=$SWEEP_SEED train_seed=$TRAIN_SEED keep=$KEEP)"
which python

if [[ ! -d "$MASK_DIR" ]]; then
  echo "[ERROR] MASK_DIR does not exist: $MASK_DIR" >&2
  exit 2
fi
if [[ -n "${POST_MASK_DIRS:-}" ]]; then
  IFS=',' read -r -a _post_mask_arr <<< "$POST_MASK_DIRS"
  for _d in "${_post_mask_arr[@]}"; do
    if [[ -z "$_d" ]]; then
      continue
    fi
    if [[ ! -d "$_d" ]]; then
      echo "[ERROR] POST_MASK_DIR does not exist: $_d" >&2
      exit 2
    fi
  done
fi

python -c "import optuna" 2>/dev/null || {
  echo "[INFO] Installing optuna..."
  pip install -q optuna
}

OUT_CSV="$LOG_DIR/gals95_ourmasks_sweep_${SLURM_JOB_ID}.csv"
TRIAL_LOGS="$LOG_DIR/gals95_ourmasks_sweep_logs_${SLURM_JOB_ID}"

ARGS=(--config configs/waterbirds_95_gals_ourmasks.yaml
  --data-root "$DATA_ROOT"
  --waterbirds-dir "$DATA_DIR"
  --n-trials "$N_TRIALS"
  --seed "$SWEEP_SEED"
  --train-seed "$TRAIN_SEED"
  --sampler "$SAMPLER"
  --keep "$KEEP"
  --output-csv "$OUT_CSV"
  --logs-dir "$TRIAL_LOGS"
  --base-lr-min "$BASE_LR_MIN"
  --base-lr-max "$BASE_LR_MAX"
  --cls-lr-min "$CLS_LR_MIN"
  --cls-lr-max "$CLS_LR_MAX"
  --post-seeds "$POST_SEEDS"
  --post-seed-start "$POST_SEED_START"
  --post-keep "$POST_KEEP"
  --post-segmentation-dirs "$POST_MASK_DIRS"
  --post-segmentation-labels "$POST_MASK_LABELS"
)

if [[ "$TUNE_WEIGHT_DECAY" -eq 1 ]]; then
  ARGS+=(--tune-weight-decay)
fi

if [[ -n "${MAX_HOURS:-}" ]]; then
  ARGS+=(--max-hours "$MAX_HOURS")
fi

srun --unbuffered python -u run_gals_sweep.py "${ARGS[@]}" \
  DATA.SEGMENTATION_DIR="$MASK_DIR"
