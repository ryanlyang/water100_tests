#!/bin/bash -l
# Generate CLIP RN50 Grad-CAM attention maps for Waterbirds-95 and Waterbirds-100.
# Debug-partition, chunked, resumable (SKIP_EXISTING=true).

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_rn50_attention_gen_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_rn50_attention_gen_%j.err
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds
WB95_DIR=${WB95_DIR:-waterbird_complete95_forest2water2}
WB100_DIR=${WB100_DIR:-waterbird_1.0_forest2water2}

RUN_WB95=${RUN_WB95:-1}
RUN_WB100=${RUN_WB100:-1}
CHUNK_SIZE=${CHUNK_SIZE:-1000}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

if [[ "$RUN_WB95" -eq 1 && ! -d "$DATA_ROOT/$WB95_DIR" ]]; then
  echo "[ERROR] Missing Waterbirds-95 dataset dir: $DATA_ROOT/$WB95_DIR" >&2
  exit 2
fi
if [[ "$RUN_WB100" -eq 1 && ! -d "$DATA_ROOT/$WB100_DIR" ]]; then
  echo "[ERROR] Missing Waterbirds-100 dataset dir: $DATA_ROOT/$WB100_DIR" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data root: $DATA_ROOT"
echo "WB95 dir: $WB95_DIR (RUN_WB95=$RUN_WB95)"
echo "WB100 dir: $WB100_DIR (RUN_WB100=$RUN_WB100)"
echo "Chunk size: $CHUNK_SIZE"
which python

if [[ "$RUN_WB95" -eq 1 ]]; then
  WB95_META="$DATA_ROOT/$WB95_DIR/metadata.csv"
  if [[ ! -f "$WB95_META" ]]; then
    echo "[ERROR] Missing WB95 metadata: $WB95_META" >&2
    exit 2
  fi
  WB95_N=${WB95_N:-$(( $(wc -l < "$WB95_META") - 1 ))}
  echo "[GEN] WB95 RN50 attentions (N=$WB95_N)..."
  for ((start=0; start<WB95_N; start+=CHUNK_SIZE)); do
    end=$((start + CHUNK_SIZE))
    echo "[GEN] WB95 chunk: START_IDX=$start END_IDX=$end"
    srun --unbuffered python -u extract_attention.py \
      --config configs/waterbirds_95_attention.yaml \
      DATA.ROOT="$DATA_ROOT" \
      DATA.WATERBIRDS_DIR="$WB95_DIR" \
      DISABLE_VIS=true \
      SKIP_EXISTING=true \
      START_IDX="$start" \
      END_IDX="$end"
  done
fi

if [[ "$RUN_WB100" -eq 1 ]]; then
  WB100_META="$DATA_ROOT/$WB100_DIR/metadata.csv"
  if [[ ! -f "$WB100_META" ]]; then
    echo "[ERROR] Missing WB100 metadata: $WB100_META" >&2
    exit 2
  fi
  WB100_N=${WB100_N:-$(( $(wc -l < "$WB100_META") - 1 ))}
  echo "[GEN] WB100 RN50 attentions (N=$WB100_N)..."
  for ((start=0; start<WB100_N; start+=CHUNK_SIZE)); do
    end=$((start + CHUNK_SIZE))
    echo "[GEN] WB100 chunk: START_IDX=$start END_IDX=$end"
    srun --unbuffered python -u extract_attention.py \
      --config configs/waterbirds_100_attention.yaml \
      DATA.ROOT="$DATA_ROOT" \
      DATA.WATERBIRDS_DIR="$WB100_DIR" \
      DISABLE_VIS=true \
      SKIP_EXISTING=true \
      START_IDX="$start" \
      END_IDX="$end"
  done
fi

echo "[GEN] Done."

