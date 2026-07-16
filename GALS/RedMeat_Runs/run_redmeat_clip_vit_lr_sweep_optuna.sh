#!/bin/bash -l
# CLIP ViT + Logistic Regression sweep on RedMeat.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_vit_lr_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_vit_lr_sweep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV_CANDIDATES=(
  "${SCRIPT_DIR}/common_env.sh"
  "${SBATCH_SUBMIT_DIR:-}/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/Food101/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
)
COMMON_ENV=""
for candidate in "${COMMON_ENV_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    COMMON_ENV="$candidate"
    break
  fi
done
if [[ -z "$COMMON_ENV" ]]; then
  echo "[ERROR] Could not locate common_env.sh" >&2
  exit 2
fi
source "$COMMON_ENV"

redmeat_set_defaults
redmeat_activate_env
redmeat_prepare_food_layout "$DATA_ROOT" "$DATA_DIR"

# Keep LR solver stack single-threaded to avoid sporadic BLAS/OpenMP crashes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

REPO_ROOT="$GALS_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"

N_TRIALS=${N_TRIALS:-100}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
CLIP_MODEL=${CLIP_MODEL:-ViT-B/32}
BATCH_SIZE=${BATCH_SIZE:-256}
# Keep 0 by default to avoid fork-related instability after CUDA feature extraction.
NUM_WORKERS=${NUM_WORKERS:-0}
OBJECTIVE=${OBJECTIVE:-val_avg_group_acc}
C_MIN=${C_MIN:-1e-2}
C_MAX=${C_MAX:-1e2}
MAX_ITER=${MAX_ITER:-5000}
TOL_MIN=${TOL_MIN:-1e-4}
TOL_MAX=${TOL_MAX:-1e-4}
PENALTY_SOLVERS=${PENALTY_SOLVERS:-l2:lbfgs}
FEATURE_MODES=${FEATURE_MODES:-l2}
CLASS_WEIGHT_OPTIONS=${CLASS_WEIGHT_OPTIONS:-none}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}

OUT_CSV=${OUT_CSV:-$LOG_DIR/redmeat_clip_vit_lr_sweep_${SLURM_JOB_ID}.csv}
POST_OUT_CSV=${POST_OUT_CSV:-$LOG_DIR/redmeat_clip_vit_lr_best5_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing DATASET_ROOT: $DATASET_ROOT" >&2
  exit 2
fi

mkdir -p CLIP/clip
if [[ ! -f CLIP/clip/bpe_simple_vocab_16e6.txt.gz ]]; then
  echo "[INFO] Downloading CLIP BPE vocab..."
  curl -L -o CLIP/clip/bpe_simple_vocab_16e6.txt.gz \
    https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
fi

python -c "import sklearn" 2>/dev/null || { echo "[INFO] Installing scikit-learn..."; pip install -q scikit-learn; }

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "CLIP model: $CLIP_MODEL"
echo "Trials: $N_TRIALS (sampler=$SAMPLER seed=$SWEEP_SEED objective=$OBJECTIVE)"
echo "Penalty/solvers: $PENALTY_SOLVERS"
echo "Feature modes: $FEATURE_MODES | class weights: $CLASS_WEIGHT_OPTIONS | tol=[$TOL_MIN,$TOL_MAX]"
echo "Output CSV: $OUT_CSV"
echo "Post seeds: $POST_SEEDS (start=$POST_SEED_START)"
echo "Post output CSV: $POST_OUT_CSV"
which python

srun --unbuffered python -u RedMeat_Runs/run_clip_lr_sweep_redmeat.py \
  "$DATASET_ROOT" \
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
  --tol-min "$TOL_MIN" \
  --tol-max "$TOL_MAX" \
  --max-iter "$MAX_ITER" \
  --penalty-solvers "$PENALTY_SOLVERS" \
  --feature-modes "$FEATURE_MODES" \
  --class-weight-options "$CLASS_WEIGHT_OPTIONS" \
  --objective "$OBJECTIVE" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$POST_OUT_CSV"
