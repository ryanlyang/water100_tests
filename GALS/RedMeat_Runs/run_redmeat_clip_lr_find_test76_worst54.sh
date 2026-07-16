#!/bin/bash -l
# Grid scan C for CLIP RN50 + LR on RedMeat until test_acc is 76.x and worst<=54.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_lr_find_test76_worst54_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_lr_find_test76_worst54_%j.err
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

REPO_ROOT="$GALS_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"

CLIP_MODEL=${CLIP_MODEL:-RN50}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
SEED=${SEED:-0}
FEATURE_MODE=${FEATURE_MODE:-l2}
FIT_INTERCEPT=${FIT_INTERCEPT:-1}
CLASS_WEIGHT=${CLASS_WEIGHT:-none}
TOL=${TOL:-1e-4}
MAX_ITER=${MAX_ITER:-5000}

C_MIN=${C_MIN:-1e-6}
C_MAX=${C_MAX:-1e6}
N_C_VALUES=${N_C_VALUES:-600}
C_VALUES=${C_VALUES:-}

TEST_ACC_MIN=${TEST_ACC_MIN:-76.0}
TEST_ACC_MAX=${TEST_ACC_MAX:-77.0}
TEST_WORST_CLASS_MAX=${TEST_WORST_CLASS_MAX:-54.0}
VAL_ACC_MIN=${VAL_ACC_MIN:-73.0}
REFINE_ROUNDS=${REFINE_ROUNDS:-2}
REFINE_TOP_K=${REFINE_TOP_K:-6}
REFINE_SPAN=${REFINE_SPAN:-3.0}
REFINE_N_VALUES=${REFINE_N_VALUES:-120}
MAX_MATCHES=${MAX_MATCHES:-1}
STOP_ON_MATCH=${STOP_ON_MATCH:-1}

OUT_CSV=${OUT_CSV:-$LOG_DIR/redmeat_clip_lr_find_test76_worst54_${SLURM_JOB_ID}.csv}

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
python -c "import threadpoolctl" 2>/dev/null || { echo "[INFO] Installing threadpoolctl..."; pip install -q threadpoolctl; }

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "CLIP model: $CLIP_MODEL"
echo "C scan: min=$C_MIN max=$C_MAX n=$N_C_VALUES fit_intercept=$FIT_INTERCEPT"
echo "Targets: val_acc > $VAL_ACC_MIN, test_acc in [$TEST_ACC_MIN,$TEST_ACC_MAX), test_worst_class <= $TEST_WORST_CLASS_MAX"
echo "Refine: rounds=$REFINE_ROUNDS top_k=$REFINE_TOP_K span=$REFINE_SPAN n_values=$REFINE_N_VALUES"
echo "Output CSV: $OUT_CSV"
which python

STOP_FLAG=--no-stop-on-match
if [[ "$STOP_ON_MATCH" == "1" ]]; then
  STOP_FLAG=--stop-on-match
fi

CMD=(
  python -u RedMeat_Runs/run_clip_lr_find_test76_worst54_redmeat.py
  "$DATASET_ROOT"
  --clip-model "$CLIP_MODEL"
  --device cuda
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --output-csv "$OUT_CSV"
  --c-min "$C_MIN"
  --c-max "$C_MAX"
  --n-c-values "$N_C_VALUES"
  --fit-intercept "$FIT_INTERCEPT"
  --feature-mode "$FEATURE_MODE"
  --class-weight "$CLASS_WEIGHT"
  --tol "$TOL"
  --max-iter "$MAX_ITER"
  --test-acc-min "$TEST_ACC_MIN"
  --test-acc-max "$TEST_ACC_MAX"
  --test-worst-class-max "$TEST_WORST_CLASS_MAX"
  --val-acc-min "$VAL_ACC_MIN"
  --refine-rounds "$REFINE_ROUNDS"
  --refine-top-k "$REFINE_TOP_K"
  --refine-span "$REFINE_SPAN"
  --refine-n-values "$REFINE_N_VALUES"
  --max-matches "$MAX_MATCHES"
  "$STOP_FLAG"
)

if [[ -n "$C_VALUES" ]]; then
  CMD+=(--c-values "$C_VALUES")
fi

srun --unbuffered "${CMD[@]}"

echo "[DONE] Search complete."
echo "CSV: $OUT_CSV"
