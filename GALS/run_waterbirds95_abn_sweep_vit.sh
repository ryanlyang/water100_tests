#!/bin/bash -l
# ABN sweep for Waterbirds-95 using CLIP ViT attention maps (ABN_SUPERVISION against clip_vit).

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_abn_vit_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_abn_vit_sweep_%j.err
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
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds
DATA_DIR=waterbird_complete95_forest2water2
ATT_DIR=clip_vit_attention

N_TRIALS=${N_TRIALS:-100}
SWEEP_SEED=${SWEEP_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
KEEP=${KEEP:-best}
MAX_HOURS=${MAX_HOURS:-}
TUNE_WEIGHT_DECAY=${TUNE_WEIGHT_DECAY:-0}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
POST_KEEP=${POST_KEEP:-all}

ABN_ATT_WEIGHT_MIN=${ABN_ATT_WEIGHT_MIN:-1e-2}
ABN_ATT_WEIGHT_MAX=${ABN_ATT_WEIGHT_MAX:-1e2}
BASE_LR_MIN=${BASE_LR_MIN:-1e-3}
BASE_LR_MAX=${BASE_LR_MAX:-1e-1}
CLS_LR_MIN=${CLS_LR_MIN:-1e-4}
CLS_LR_MAX=${CLS_LR_MAX:-1e-2}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

if [[ ! -d "$DATA_ROOT/$DATA_DIR/$ATT_DIR" ]]; then
  echo "[ERROR] Missing attention maps at: $DATA_ROOT/$DATA_DIR/$ATT_DIR" >&2
  echo "Run: sbatch run_waterbirds95_extract_attention_vit.sh" >&2
  exit 2
fi

python -c "import optuna" 2>/dev/null || { pip install -q optuna; }

OUT_CSV="/home/ryreu/guided_cnn/logsWaterbird/abn95_vit_sweep_${SLURM_JOB_ID}.csv"
TRIAL_LOGS="/home/ryreu/guided_cnn/logsWaterbird/abn95_vit_sweep_logs_${SLURM_JOB_ID}"

ARGS=(--method abn_att
  --config configs/waterbirds_95_abn_vit.yaml
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
  --abn-att-weight-min "$ABN_ATT_WEIGHT_MIN"
  --abn-att-weight-max "$ABN_ATT_WEIGHT_MAX"
  --post-seeds "$POST_SEEDS"
  --post-seed-start "$POST_SEED_START"
  --post-keep "$POST_KEEP"
)
if [[ "$TUNE_WEIGHT_DECAY" -eq 1 ]]; then
  ARGS+=(--tune-weight-decay)
fi
if [[ -n "${MAX_HOURS:-}" ]]; then
  ARGS+=(--max-hours "$MAX_HOURS")
fi

srun --unbuffered python -u run_gals_sweep.py "${ARGS[@]}"
