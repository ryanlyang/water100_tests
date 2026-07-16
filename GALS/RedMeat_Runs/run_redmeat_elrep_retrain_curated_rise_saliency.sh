#!/bin/bash -l
# Retrain RedMeat ElRep with the locked best hyperparameters, then generate curated RISE saliency maps.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_elrep_curated_rise_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_elrep_curated_rise_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsRedMeat}"
REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/Food101/data/food-101-redmeat}"
MASK_ROOT="${MASK_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_laion_dinovit/val/prediction_cmap}"
ENV_NAME="${ENV_NAME:-gals_a100}"

# Locked RedMeat ElRep best hyperparameters from redmeat_elrep_best5_21360855.csv.
# Default seed is 2 because it was the strongest of the five RedMeat ElRep seeds on worst-class accuracy.
ELREP_SEED="${ELREP_SEED:-2}"
BASE_LR="${BASE_LR:-0.002381123051117795}"
CLASSIFIER_LR="${CLASSIFIER_LR:-1.0449032698523769e-05}"
THETA1="${THETA1:-9.521567541537433e-05}"
THETA2="${THETA2:-1.4275756893065589e-05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
MOMENTUM="${MOMENTUM:-0.9}"
NESTEROV="${NESTEROV:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-150}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TARGET_CLASS="${TARGET_CLASS:-label}"
SPLIT="${SPLIT:-all}"
RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_GPU_BATCH="${RISE_GPU_BATCH:-16}"
RISE_SEED="${RISE_SEED:-0}"

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LOG_DIR/redmeat_elrep_curated_rise_${RUN_ID}}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/ElRep_Checkpoints/redmeat_curated_saliency_seed${ELREP_SEED}}"
TRAIN_LOG="${OUTPUT_ROOT}/elrep_redmeat_seed${ELREP_SEED}_train.log"
ELREP_CKPT="${ELREP_CKPT:-}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT" "$CKPT_DIR"

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

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "Mask root: $MASK_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Checkpoint dir: $CKPT_DIR"
echo "Seed: $ELREP_SEED"
echo "ElRep hparams: base_lr=$BASE_LR classifier_lr=$CLASSIFIER_LR theta1=$THETA1 theta2=$THETA2"
echo "Train: epochs=$NUM_EPOCHS workers=$NUM_WORKERS wd=$WEIGHT_DECAY momentum=$MOMENTUM nesterov=$NESTEROV"
echo "RISE: num_masks=$RISE_NUM_MASKS grid=$RISE_GRID_SIZE p1=$RISE_P1 gpu_batch=$RISE_GPU_BATCH seed=$RISE_SEED"
which python

if [[ -z "$ELREP_CKPT" && "$REUSE_EXISTING" == "1" ]]; then
  mapfile -t _existing < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "elrep_redmeat_resnet50_*_seed${ELREP_SEED}_*.pth" -printf "%T@ %p\n" | sort -nr | awk 'NR==1 {print $2}')
  ELREP_CKPT="${_existing[0]:-}"
  if [[ -n "$ELREP_CKPT" ]]; then
    echo "[INFO] Reusing existing ElRep checkpoint: $ELREP_CKPT"
  fi
fi

if [[ -z "$ELREP_CKPT" ]]; then
  echo "[INFO] Training RedMeat ElRep checkpoint for saliency."
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
  srun --unbuffered python -u RedMeat_Runs/run_elrep_redmeat.py "${TRAIN_ARGS[@]}" | tee "$TRAIN_LOG"
  mapfile -t _made < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "elrep_redmeat_resnet50_*_seed${ELREP_SEED}_*.pth" -printf "%T@ %p\n" | sort -nr | awk 'NR==1 {print $2}')
  ELREP_CKPT="${_made[0]:-}"
fi

if [[ -z "$ELREP_CKPT" || ! -f "$ELREP_CKPT" ]]; then
  echo "[ERROR] Could not resolve RedMeat ElRep checkpoint. ELREP_CKPT=$ELREP_CKPT" >&2
  exit 2
fi

echo "[INFO] Generating RedMeat ElRep RISE saliency with checkpoint: $ELREP_CKPT"

srun --unbuffered python -u run_elrep_curated_rise_saliency.py \
  --dataset redmeat \
  --data-path "$DATA_PATH" \
  --mask-root "$MASK_ROOT" \
  --elrep-ckpt "$ELREP_CKPT" \
  --split "$SPLIT" \
  --target-class "$TARGET_CLASS" \
  --device cuda:0 \
  --allow-missing \
  --rise-num-masks "$RISE_NUM_MASKS" \
  --rise-grid-size "$RISE_GRID_SIZE" \
  --rise-p1 "$RISE_P1" \
  --rise-gpu-batch "$RISE_GPU_BATCH" \
  --rise-seed "$RISE_SEED" \
  --output-dir "$OUTPUT_ROOT"

echo "[DONE] RedMeat ElRep checkpoint: $ELREP_CKPT"
echo "[DONE] RedMeat ElRep curated RISE saliency output: $OUTPUT_ROOT"
