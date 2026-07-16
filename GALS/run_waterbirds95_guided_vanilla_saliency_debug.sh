#!/bin/bash -l
# Train guided + vanilla + GALS-ViT WB95 models, then generate saliency visualizations.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/wb95_guided_vanilla_gals_saliency_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/wb95_guided_vanilla_gals_saliency_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_ROOT=${LOG_ROOT:-/home/ryreu/guided_cnn/logsWaterbird}
mkdir -p "$LOG_ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}
DATA_PATH=${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}
GUIDED_GT_ROOT=${GUIDED_GT_ROOT:-/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}

OUT_DIR_DEFAULT="${LOG_ROOT}/wb95_guided_vanilla_gals_saliency_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR=${OUT_DIR:-$OUT_DIR_DEFAULT}

NUM_VAL_SAMPLES=${NUM_VAL_SAMPLES:-150}
SAMPLE_SEED=${SAMPLE_SEED:-0}
SAMPLE_STRATEGY=${SAMPLE_STRATEGY:-balanced}
TARGET_CLASS=${TARGET_CLASS:-label}
SALIENCY_METHOD=${SALIENCY_METHOD:-rise}
RISE_NUM_MASKS=${RISE_NUM_MASKS:-2000}
RISE_GRID_SIZE=${RISE_GRID_SIZE:-8}
RISE_P1=${RISE_P1:-0.1}
RISE_GPU_BATCH=${RISE_GPU_BATCH:-16}
RISE_SEED=${RISE_SEED:-0}
RISE_MASKS_PATH=${RISE_MASKS_PATH:-}

# Guided fixed params (from run_guided_waterbirds95_bestfixed_multigt_debug.sh)
GUIDED_SEED=${GUIDED_SEED:-0}
GUIDED_ATTENTION_EPOCH=${GUIDED_ATTENTION_EPOCH:-109}
GUIDED_KL_LAMBDA=${GUIDED_KL_LAMBDA:-295.3017825649997}
GUIDED_KL_INCR=${GUIDED_KL_INCR:-0.0}
GUIDED_BASE_LR=${GUIDED_BASE_LR:-4.81872261305513e-05}
GUIDED_CLASSIFIER_LR=${GUIDED_CLASSIFIER_LR:-0.002932587987450572}
GUIDED_LR2_MULT=${GUIDED_LR2_MULT:-0.40940723404327417}
GUIDED_NUM_WORKERS=${GUIDED_NUM_WORKERS:-0}

# Vanilla defaults (same as WB100 unless overridden)
VANILLA_SEED=${VANILLA_SEED:-0}
VANILLA_BASE_LR=${VANILLA_BASE_LR:-0.031210590691245817}
VANILLA_CLASSIFIER_LR=${VANILLA_CLASSIFIER_LR:-0.0008517287145349147}
VANILLA_MOMENTUM=${VANILLA_MOMENTUM:-0.8914661939990524}
VANILLA_WEIGHT_DECAY=${VANILLA_WEIGHT_DECAY:-1e-5}
VANILLA_NUM_WORKERS=${VANILLA_NUM_WORKERS:-0}
VANILLA_NESTEROV=${VANILLA_NESTEROV:-0}

# GALS-ViT fixed params (from user-provided WB95 Trial 11)
RUN_GALS=${RUN_GALS:-1}
GALS_SEED=${GALS_SEED:-0}
GALS_CONFIG=${GALS_CONFIG:-configs/waterbirds_95_gals_vit.yaml}
GALS_DATA_ROOT=${GALS_DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds}
GALS_WATERBIRDS_DIR=${GALS_WATERBIRDS_DIR:-waterbird_complete95_forest2water2}
GALS_ATTENTION_DIR=${GALS_ATTENTION_DIR:-clip_vit_attention}
GALS_BASE_LR=${GALS_BASE_LR:-0.009095914443979123}
GALS_CLASSIFIER_LR=${GALS_CLASSIFIER_LR:-0.000988718410062346}
GALS_GRAD_WEIGHT=${GALS_GRAD_WEIGHT:-11213.40683824834}
GALS_GRAD_CRITERION=${GALS_GRAD_CRITERION:-L1}
GALS_WEIGHT_DECAY=${GALS_WEIGHT_DECAY:-1e-5}
GALS_MOMENTUM=${GALS_MOMENTUM:-0.9}
GALS_NUM_WORKERS=${GALS_NUM_WORKERS:-0}

GUIDED_CKPT=${GUIDED_CKPT:-}
VANILLA_CKPT=${VANILLA_CKPT:-}
GALS_CKPT=${GALS_CKPT:-}

cd "$REPO_ROOT/GALS"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "[ERROR] Missing DATA_PATH: $DATA_PATH" >&2
  exit 1
fi
if [[ ! -d "$GUIDED_GT_ROOT" ]]; then
  echo "[ERROR] Missing GUIDED_GT_ROOT: $GUIDED_GT_ROOT" >&2
  exit 1
fi
if [[ "$RUN_GALS" == "1" && -z "$GALS_CKPT" ]]; then
  GALS_ATT_PATH="$GALS_DATA_ROOT/$GALS_WATERBIRDS_DIR/$GALS_ATTENTION_DIR"
  if [[ ! -d "$GALS_ATT_PATH" ]]; then
    echo "[ERROR] Missing GALS attention dir: $GALS_ATT_PATH" >&2
    exit 1
  fi
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "Guided GT root: $GUIDED_GT_ROOT"
echo "Output dir: $OUT_DIR"
echo "Num val samples: $NUM_VAL_SAMPLES (seed=$SAMPLE_SEED strategy=$SAMPLE_STRATEGY target_class=$TARGET_CLASS)"
echo "Saliency: method=$SALIENCY_METHOD rise_num_masks=$RISE_NUM_MASKS rise_grid_size=$RISE_GRID_SIZE rise_p1=$RISE_P1 rise_gpu_batch=$RISE_GPU_BATCH rise_seed=$RISE_SEED rise_masks_path=${RISE_MASKS_PATH:-AUTO}"
echo "Guided params: attn=$GUIDED_ATTENTION_EPOCH kl=$GUIDED_KL_LAMBDA kl_incr=$GUIDED_KL_INCR base_lr=$GUIDED_BASE_LR cls_lr=$GUIDED_CLASSIFIER_LR lr2_mult=$GUIDED_LR2_MULT seed=$GUIDED_SEED workers=$GUIDED_NUM_WORKERS"
echo "Vanilla params: base_lr=$VANILLA_BASE_LR cls_lr=$VANILLA_CLASSIFIER_LR momentum=$VANILLA_MOMENTUM wd=$VANILLA_WEIGHT_DECAY seed=$VANILLA_SEED workers=$VANILLA_NUM_WORKERS nesterov=$VANILLA_NESTEROV"
echo "GALS-ViT params: run=$RUN_GALS cfg=$GALS_CONFIG data_root=$GALS_DATA_ROOT wb_dir=$GALS_WATERBIRDS_DIR att_dir=$GALS_ATTENTION_DIR base_lr=$GALS_BASE_LR cls_lr=$GALS_CLASSIFIER_LR grad_weight=$GALS_GRAD_WEIGHT grad_criterion=$GALS_GRAD_CRITERION wd=$GALS_WEIGHT_DECAY momentum=$GALS_MOMENTUM seed=$GALS_SEED workers=$GALS_NUM_WORKERS"
which python

CMD=(
  python -u waterbirds100_guided_vanilla_saliency.py
  --data-path "$DATA_PATH"
  --guided-gt-root "$GUIDED_GT_ROOT"
  --output-dir "$OUT_DIR"
  --num-val-samples "$NUM_VAL_SAMPLES"
  --sample-seed "$SAMPLE_SEED"
  --sample-strategy "$SAMPLE_STRATEGY"
  --target-class "$TARGET_CLASS"
  --saliency-method "$SALIENCY_METHOD"
  --rise-num-masks "$RISE_NUM_MASKS"
  --rise-grid-size "$RISE_GRID_SIZE"
  --rise-p1 "$RISE_P1"
  --rise-gpu-batch "$RISE_GPU_BATCH"
  --rise-seed "$RISE_SEED"
  --guided-seed "$GUIDED_SEED"
  --guided-attention-epoch "$GUIDED_ATTENTION_EPOCH"
  --guided-kl-lambda "$GUIDED_KL_LAMBDA"
  --guided-kl-incr "$GUIDED_KL_INCR"
  --guided-base-lr "$GUIDED_BASE_LR"
  --guided-classifier-lr "$GUIDED_CLASSIFIER_LR"
  --guided-lr2-mult "$GUIDED_LR2_MULT"
  --guided-num-workers "$GUIDED_NUM_WORKERS"
  --vanilla-seed "$VANILLA_SEED"
  --vanilla-base-lr "$VANILLA_BASE_LR"
  --vanilla-classifier-lr "$VANILLA_CLASSIFIER_LR"
  --vanilla-momentum "$VANILLA_MOMENTUM"
  --vanilla-weight-decay "$VANILLA_WEIGHT_DECAY"
  --vanilla-num-workers "$VANILLA_NUM_WORKERS"
  --gals-seed "$GALS_SEED"
  --gals-config "$GALS_CONFIG"
  --gals-data-root "$GALS_DATA_ROOT"
  --gals-waterbirds-dir "$GALS_WATERBIRDS_DIR"
  --gals-attention-dir "$GALS_ATTENTION_DIR"
  --gals-base-lr "$GALS_BASE_LR"
  --gals-classifier-lr "$GALS_CLASSIFIER_LR"
  --gals-grad-weight "$GALS_GRAD_WEIGHT"
  --gals-grad-criterion "$GALS_GRAD_CRITERION"
  --gals-weight-decay "$GALS_WEIGHT_DECAY"
  --gals-momentum "$GALS_MOMENTUM"
  --gals-num-workers "$GALS_NUM_WORKERS"
)

if [[ "$VANILLA_NESTEROV" == "1" ]]; then
  CMD+=(--vanilla-nesterov)
fi
if [[ "$RUN_GALS" != "1" ]]; then
  CMD+=(--no-gals)
fi
if [[ -n "$GUIDED_CKPT" ]]; then
  CMD+=(--guided-ckpt "$GUIDED_CKPT")
fi
if [[ -n "$VANILLA_CKPT" ]]; then
  CMD+=(--vanilla-ckpt "$VANILLA_CKPT")
fi
if [[ -n "$GALS_CKPT" ]]; then
  CMD+=(--gals-ckpt "$GALS_CKPT")
fi
if [[ -n "$RISE_MASKS_PATH" ]]; then
  CMD+=(--rise-masks-path "$RISE_MASKS_PATH")
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --unbuffered "${CMD[@]}"
else
  "${CMD[@]}"
fi
