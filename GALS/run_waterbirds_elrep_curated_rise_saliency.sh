#!/bin/bash -l
# Generate curated RISE saliency maps for saved Waterbirds-95/100 ElRep checkpoints.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_elrep_curated_rise_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_elrep_curated_rise_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
ENV_NAME="${ENV_NAME:-gals_a100}"

WB95_DATA="${WB95_DATA:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}"
WB100_DATA="${WB100_DATA:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}"

WB95_MASK_ROOT="${WB95_MASK_ROOT:-/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}"
WB100_MASK_ROOT="${WB100_MASK_ROOT:-/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}"

WB95_ELREP_CKPT="${WB95_ELREP_CKPT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/ElRep_Checkpoints/wb95_seed4_pointing_game/elrep_resnet50_theta13.33258e-05_theta21.81017e-05_seed4_20260627_005345.pth}"
WB100_ELREP_CKPT="${WB100_ELREP_CKPT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/ElRep_Checkpoints/wb100_seed4_pointing_game/elrep_resnet50_theta14.17412e-05_theta22.54334e-06_seed4_20260627_005607.pth}"

TARGET_CLASS="${TARGET_CLASS:-label}"
SPLIT="${SPLIT:-val}"
RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_GPU_BATCH="${RISE_GPU_BATCH:-16}"
RISE_SEED="${RISE_SEED:-0}"

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LOG_DIR/waterbirds_elrep_curated_rise_${RUN_ID}}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd "$REPO_ROOT/GALS"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "WB95 data: $WB95_DATA"
echo "WB95 ckpt: $WB95_ELREP_CKPT"
echo "WB100 data: $WB100_DATA"
echo "WB100 ckpt: $WB100_ELREP_CKPT"
echo "RISE: num_masks=$RISE_NUM_MASKS grid=$RISE_GRID_SIZE p1=$RISE_P1 gpu_batch=$RISE_GPU_BATCH seed=$RISE_SEED"
which python

srun --unbuffered python -u run_elrep_curated_rise_saliency.py \
  --dataset wb95 \
  --data-path "$WB95_DATA" \
  --mask-root "$WB95_MASK_ROOT" \
  --elrep-ckpt "$WB95_ELREP_CKPT" \
  --split "$SPLIT" \
  --target-class "$TARGET_CLASS" \
  --device cuda:0 \
  --rise-num-masks "$RISE_NUM_MASKS" \
  --rise-grid-size "$RISE_GRID_SIZE" \
  --rise-p1 "$RISE_P1" \
  --rise-gpu-batch "$RISE_GPU_BATCH" \
  --rise-seed "$RISE_SEED" \
  --output-dir "$OUTPUT_ROOT/wb95"

srun --unbuffered python -u run_elrep_curated_rise_saliency.py \
  --dataset wb100 \
  --data-path "$WB100_DATA" \
  --mask-root "$WB100_MASK_ROOT" \
  --elrep-ckpt "$WB100_ELREP_CKPT" \
  --split "$SPLIT" \
  --target-class "$TARGET_CLASS" \
  --device cuda:0 \
  --allow-missing \
  --rise-num-masks "$RISE_NUM_MASKS" \
  --rise-grid-size "$RISE_GRID_SIZE" \
  --rise-p1 "$RISE_P1" \
  --rise-gpu-batch "$RISE_GPU_BATCH" \
  --rise-seed "$RISE_SEED" \
  --output-dir "$OUTPUT_ROOT/wb100"

echo "[DONE] Waterbirds ElRep curated RISE saliency outputs:"
echo "  WB95:  $OUTPUT_ROOT/wb95"
echo "  WB100: $OUTPUT_ROOT/wb100"
