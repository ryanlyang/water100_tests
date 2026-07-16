#!/bin/bash -l
# CLIP + Logistic Regression sweep for Waterbirds-95.
# Uses run_clip_lr_sweep.py (Optuna TPE by default).

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_clip_lr_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_clip_lr_sweep_%j.err
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
DATA_PATH=/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2

N_TRIALS=${N_TRIALS:-100}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
CLIP_MODEL=${CLIP_MODEL:-RN50}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
OBJECTIVE=${OBJECTIVE:-val_avg_group_acc}
C_MIN=${C_MIN:-1e-2}
C_MAX=${C_MAX:-1e2}
MAX_ITER=${MAX_ITER:-5000}
PENALTY_SOLVERS=${PENALTY_SOLVERS:-l2:lbfgs}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}

OUT_CSV=${OUT_CSV:-$LOG_DIR/clip_lr95_sweep_${SLURM_JOB_ID}.csv}
POST_OUT_CSV=${POST_OUT_CSV:-$LOG_DIR/clip_lr95_best5_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Missing DATA_PATH: $DATA_PATH" >&2
  exit 1
fi

mkdir -p CLIP/clip
if [[ ! -f CLIP/clip/bpe_simple_vocab_16e6.txt.gz ]]; then
  echo "[INFO] Downloading CLIP BPE vocab..."
  curl -L -o CLIP/clip/bpe_simple_vocab_16e6.txt.gz \
    https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
fi

python -c "import optuna" 2>/dev/null || {
  echo "[INFO] Installing optuna..."
  pip install -q optuna
}
python -c "import sklearn" 2>/dev/null || {
  echo "[INFO] Installing scikit-learn..."
  pip install -q scikit-learn
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "CLIP model: $CLIP_MODEL"
echo "Trials: $N_TRIALS (sampler=$SAMPLER seed=$SWEEP_SEED objective=$OBJECTIVE)"
echo "Penalty/solvers: $PENALTY_SOLVERS"
echo "Output CSV: $OUT_CSV"
echo "Post seeds: $POST_SEEDS (start=$POST_SEED_START)"
echo "Post output CSV: $POST_OUT_CSV"
which python

srun --unbuffered python -u run_clip_lr_sweep.py \
  "$DATA_PATH" \
  --clip-model "$CLIP_MODEL" \
  --device cuda \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --output-csv "$OUT_CSV" \
  --sampler "$SAMPLER" \
  --C-min "$C_MIN" \
  --C-max "$C_MAX" \
  --max-iter "$MAX_ITER" \
  --penalty-solvers "$PENALTY_SOLVERS" \
  --objective "$OBJECTIVE" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$POST_OUT_CSV"
