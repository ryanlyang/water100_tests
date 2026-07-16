#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbirdSeeds/invert_100_L100_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbirdSeeds/invert_100_L100_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbirdSeeds
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gals_a100

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
GT_ROOT=/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap

ATTENTION_EPOCH=38
KL_LAMBDA=1.3848179138470018
BASE_LR=0.002864673705012986
CLASSIFIER_LR=0.000010252393978291112

SEEDS=${SEEDS:-"0 1 2 3 4"}
SUMMARY_CSV=${SUMMARY_CSV:-$LOG_DIR/invert_100_L100_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing DATA_ROOT: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$GT_ROOT" ]]; then
  echo "Missing GT_ROOT: $GT_ROOT" >&2
  exit 1
fi

echo "seed,attention_epoch,kl_lambda,base_lr,classifier_lr,best_balanced_val_acc,test_acc,per_group,worst_group,log_path" > "$SUMMARY_CSV"

for SEED in $SEEDS; do
  RUN_LOG="$LOG_DIR/invert_100_L100_seed${SEED}_${SLURM_JOB_ID}.log"
  echo "=== SEED ${SEED} ==="
  srun --unbuffered python -u run_waterbird_invert.py \
    "$DATA_ROOT" \
    "$GT_ROOT" \
    --seed "$SEED" \
    --attention_epoch "$ATTENTION_EPOCH" \
    --kl_lambda "$KL_LAMBDA" \
    --base_lr "$BASE_LR" \
    --classifier_lr "$CLASSIFIER_LR" | tee "$RUN_LOG"

  BEST_VAL=$(grep -F "[VAL] Best Balanced Acc" "$RUN_LOG" | tail -n 1 | awk '{print $5}')
  TEST_ACC=$(grep -F "[TEST] Loss" "$RUN_LOG" | tail -n 1 | awk '{print $5}' | tr -d '%')
  PER_GROUP=$(grep -F "[TEST] Per Group" "$RUN_LOG" | tail -n 1 | awk '{print $4}' | tr -d '%')
  WORST_GROUP=$(grep -F "[TEST] Per Group" "$RUN_LOG" | tail -n 1 | awk '{print $7}' | tr -d '%')
  echo "${SEED},${ATTENTION_EPOCH},${KL_LAMBDA},${BASE_LR},${CLASSIFIER_LR},${BEST_VAL},${TEST_ACC},${PER_GROUP},${WORST_GROUP},${RUN_LOG}" >> "$SUMMARY_CSV"
done
