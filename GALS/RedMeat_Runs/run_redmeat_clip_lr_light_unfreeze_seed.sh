#!/bin/bash
# One seed of the RedMeat CLIP RN50 light-unfreezing study.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

SEED="${SEED:?Submit with SEED=0|1|2|3|4}"
case "$SEED" in
  0|1|2|3|4) ;;
  *) echo "[ERROR] SEED must be one of 0,1,2,3,4" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GALS_ROOT="${GALS_ROOT:-$(dirname "$SCRIPT_DIR")}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/Food101/data/food-101-redmeat}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsRedMeat}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/clip_lr_rn50_full_visual_finetune}"
OUTPUT_DIR="$RUN_ROOT/seed_${SEED}"

EVAL_EPOCHS="${EVAL_EPOCHS:-0,1,2,4,8,16}"
UNFREEZE_SCOPE="${UNFREEZE_SCOPE:-full_visual}"
ENCODER_LR="${ENCODER_LR:-1e-5}"
HEAD_LR="${HEAD_LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASELINE_C="${BASELINE_C:-1.329346323656201}"
C_MIN="${C_MIN:-1e-2}"
C_MAX="${C_MAX:-1e2}"
C_TRIALS="${C_TRIALS:-25}"
C_SWEEP_SEED="${C_SWEEP_SEED:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$GALS_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$GALS_ROOT"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} seed=$SEED"
echo "[RUN] data=$DATA_ROOT output=$OUTPUT_DIR"
echo "[RUN] epochs=$EVAL_EPOCHS scope=$UNFREEZE_SCOPE encoder_lr=$ENCODER_LR head_lr=$HEAD_LR"
echo "[RUN] fixed_C=$BASELINE_C retuned_C=Optuna[$C_MIN,$C_MAX] trials=$C_TRIALS"
which python

python -u RedMeat_Runs/run_clip_lr_light_unfreeze_study.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --clip-model RN50 \
  --eval-epochs "$EVAL_EPOCHS" \
  --unfreeze-scope "$UNFREEZE_SCOPE" \
  --encoder-lr "$ENCODER_LR" \
  --head-lr "$HEAD_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --batch-size "$BATCH_SIZE" \
  --feature-batch-size "$FEATURE_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --baseline-c "$BASELINE_C" \
  --c-min "$C_MIN" \
  --c-max "$C_MAX" \
  --c-trials "$C_TRIALS" \
  --c-sweep-seed "$C_SWEEP_SEED" \
  --device cuda:0
