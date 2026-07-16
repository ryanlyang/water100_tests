#!/bin/bash -l
# Generate ElRep Grad-CAM saliency maps for the exact DecoyMNIST samples used previously.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_elrep_saliency_from_index_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_elrep_saliency_from_index_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
ENV_NAME="${ENV_NAME:-gals_a100}"

SOURCE_RUN="${SOURCE_RUN:-/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist_openclip/val/decoy_multimodel_saliency_nonzero_20260304_022211}"
PNG_ROOT="${PNG_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"

ELREP_SEED="${ELREP_SEED:-0}"
ELREP_CKPT="${ELREP_CKPT:-}"
CKPT_DIR="${CKPT_DIR:-$LOG_DIR/ckpts/elrep}"
TRAIN_CSV="${TRAIN_CSV:-$LOG_DIR/decoy_elrep_for_saliency_${SLURM_JOB_ID:-local}.csv}"

THETA1="${THETA1:-4.1741182912970346e-05}"
THETA2="${THETA2:-2.5433427234421545e-06}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EPOCHS="${EPOCHS:-19}"
VAL_FRAC="${VAL_FRAC:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"

TARGET_CLASS="${TARGET_CLASS:-label}"
WRITE_INTO_SOURCE="${WRITE_INTO_SOURCE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_DIR/decoy_elrep_saliency_from_index_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}}"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$OUTPUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$REPO_ROOT/GALS"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Source run: $SOURCE_RUN"
echo "PNG root: $PNG_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "ElRep seed: $ELREP_SEED"
echo "ElRep checkpoint input: ${ELREP_CKPT:-AUTO_TRAIN_OR_REUSE}"
echo "ElRep hparams: lr=$LR wd=$WEIGHT_DECAY theta1=$THETA1 theta2=$THETA2 epochs=$EPOCHS"
echo "Write into source: $WRITE_INTO_SOURCE"
which python

if [[ -z "$ELREP_CKPT" ]]; then
  mapfile -t _existing < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "decoy_elrep_seed${ELREP_SEED}_*.pth" -printf "%T@ %p\n" | sort -nr | awk 'NR==1 {print $2}')
  ELREP_CKPT="${_existing[0]:-}"
  if [[ -n "$ELREP_CKPT" ]]; then
    echo "[INFO] Reusing existing ElRep checkpoint: $ELREP_CKPT"
  fi
fi

if [[ -z "$ELREP_CKPT" ]]; then
  echo "[INFO] No ElRep checkpoint found/provided. Training one seed for saliency."
  srun --unbuffered python -u run_decoymnist_elrep_fixed.py \
    --png-root "$PNG_ROOT" \
    --epochs "$EPOCHS" \
    --val-frac "$VAL_FRAC" \
    --split-seed "$SPLIT_SEED" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --theta1 "$THETA1" \
    --theta2 "$THETA2" \
    --n-seeds 1 \
    --seed-start "$ELREP_SEED" \
    --num-workers "$NUM_WORKERS" \
    --save-dir "$CKPT_DIR" \
    --output-csv "$TRAIN_CSV"

  mapfile -t _made < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "decoy_elrep_seed${ELREP_SEED}_*.pth" -printf "%T@ %p\n" | sort -nr | awk 'NR==1 {print $2}')
  ELREP_CKPT="${_made[0]:-}"
fi

if [[ -z "$ELREP_CKPT" || ! -f "$ELREP_CKPT" ]]; then
  echo "[ERROR] Could not resolve ElRep checkpoint. ELREP_CKPT=$ELREP_CKPT" >&2
  exit 2
fi

SAL_ARGS=(
  --source-run "$SOURCE_RUN"
  --elrep-ckpt "$ELREP_CKPT"
  --png-root "$PNG_ROOT"
  --output-dir "$OUTPUT_DIR"
  --target-class "$TARGET_CLASS"
  --device cuda:0
)
if [[ "$WRITE_INTO_SOURCE" == "1" ]]; then
  SAL_ARGS+=(--write-into-source)
fi

srun --unbuffered python -u run_decoymnist_elrep_saliency_from_index.py "${SAL_ARGS[@]}"

echo "[DONE] ElRep checkpoint: $ELREP_CKPT"
echo "[DONE] ElRep DecoyMNIST saliency output: $OUTPUT_DIR"
if [[ "$WRITE_INTO_SOURCE" == "1" ]]; then
  echo "[DONE] Also wrote elrep_saliency_* into: $SOURCE_RUN/samples"
fi
