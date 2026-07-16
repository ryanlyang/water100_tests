#!/bin/bash -l
# Retrain/load the Waterbirds-100 ElRep seed-4 checkpoint and run Pointing Game.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/elrep100_seed4_pointing_game_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/elrep100_seed4_pointing_game_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}"
MASK_ROOT="${MASK_ROOT:-/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}"
ENV_NAME="${ENV_NAME:-gals_a100}"

mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd "$REPO_ROOT/GALS"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Fixed Waterbirds-100 ElRep best hyperparameters from elrep100_best5_21360854.csv.
ELREP_SEED="${ELREP_SEED:-4}"
BASE_LR="${BASE_LR:-0.019255584330594423}"
CLASSIFIER_LR="${CLASSIFIER_LR:-4.9279546242498994e-05}"
THETA1="${THETA1:-4.1741182912970346e-05}"
THETA2="${THETA2:-2.5433427234421545e-06}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
MOMENTUM="${MOMENTUM:-0.9}"
NESTEROV="${NESTEROV:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"

SPLIT="${SPLIT:-val}"
TARGET_MODE="${TARGET_MODE:-label}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/ElRep_Checkpoints/wb100_seed4_pointing_game}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/elrep100_seed4_pointing_game_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}}"
TRAIN_LOG="${OUTPUT_DIR}/elrep_seed${ELREP_SEED}_train.log"
ELREP_CKPT="${ELREP_CKPT:-}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"

mkdir -p "$CKPT_DIR" "$OUTPUT_DIR"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "Mask root: $MASK_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Seed: $ELREP_SEED"
echo "ElRep hparams: base_lr=$BASE_LR classifier_lr=$CLASSIFIER_LR theta1=$THETA1 theta2=$THETA2"
echo "Train: epochs=$NUM_EPOCHS workers=$NUM_WORKERS wd=$WEIGHT_DECAY momentum=$MOMENTUM nesterov=$NESTEROV"
echo "Pointing Game: split=$SPLIT target_mode=$TARGET_MODE max_samples=$MAX_SAMPLES"
which python

if [[ -z "$ELREP_CKPT" && "$REUSE_EXISTING" == "1" ]]; then
  mapfile -t _existing < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "elrep_resnet50_*_seed${ELREP_SEED}_*.pth" -printf "%T@ %p\n" | sort -nr | awk 'NR==1 {print $2}')
  ELREP_CKPT="${_existing[0]:-}"
  if [[ -n "$ELREP_CKPT" ]]; then
    echo "[INFO] Reusing existing ElRep checkpoint: $ELREP_CKPT"
  fi
fi

if [[ -z "$ELREP_CKPT" ]]; then
  echo "[INFO] No ElRep checkpoint provided/found. Training exact seed-$ELREP_SEED run with checkpoint saving enabled."
  TRAIN_ARGS=(
    "$DATA_PATH"
    --seed "$ELREP_SEED"
    --model resnet50
    --tune-mode full
    --pretrained
    --num-epochs "$NUM_EPOCHS"
    --num-workers "$NUM_WORKERS"
    --base-lr "$BASE_LR"
    --classifier-lr "$CLASSIFIER_LR"
    --theta1 "$THETA1"
    --theta2 "$THETA2"
    --weight-decay "$WEIGHT_DECAY"
    --momentum "$MOMENTUM"
    --checkpoint-dir "$CKPT_DIR"
  )
  if [[ "$NESTEROV" -eq 1 ]]; then
    TRAIN_ARGS+=(--nesterov)
  fi

  srun --unbuffered python -u run_elrep_waterbird.py "${TRAIN_ARGS[@]}" | tee "$TRAIN_LOG"
  mapfile -t _made < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "elrep_resnet50_*_seed${ELREP_SEED}_*.pth" -printf "%T@ %p\n" | sort -nr | awk 'NR==1 {print $2}')
  ELREP_CKPT="${_made[0]:-}"
fi

if [[ -z "$ELREP_CKPT" || ! -f "$ELREP_CKPT" ]]; then
  echo "[ERROR] Could not resolve ElRep checkpoint. ELREP_CKPT=$ELREP_CKPT" >&2
  exit 2
fi

echo "[INFO] Running Pointing Game with ElRep checkpoint: $ELREP_CKPT"

PG_ARGS=(
  --datasets 100
  --split "$SPLIT"
  --target-mode "$TARGET_MODE"
  --max-samples "$MAX_SAMPLES"
  --sample-seed "$SAMPLE_SEED"
  --seed 0
  --methods elrep
  --wb100-data-path "$DATA_PATH"
  --wb100-mask-root "$MASK_ROOT"
  --elrep100-ckpt "$ELREP_CKPT"
  --output-dir "$OUTPUT_DIR"
)

srun --unbuffered python -u waterbirds_pointing_game_eval.py "${PG_ARGS[@]}"

echo "[DONE] ElRep checkpoint: $ELREP_CKPT"
echo "[DONE] Summary: $OUTPUT_DIR/pointing_game_summary.csv"
echo "[DONE] Per image: $OUTPUT_DIR/pointing_game_per_image.csv"
