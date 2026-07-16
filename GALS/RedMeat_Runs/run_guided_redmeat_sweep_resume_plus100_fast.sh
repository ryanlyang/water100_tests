#!/bin/bash -l
# Resume guided RedMeat sweep from an existing CSV and run +100 new trials
# beyond the highest completed trial id.
#
# Defaults target your existing file:
#   /home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_sweep_21065307.csv
#
# Usage:
#   sbatch RedMeat_Runs/run_guided_redmeat_sweep_resume_plus100_fast.sh
# Optional overrides:
#   sbatch --export=ALL,RESUME_CSV=/path/to/file.csv,ADDITIONAL_TRIALS=100 \
#     RedMeat_Runs/run_guided_redmeat_sweep_resume_plus100_fast.sh

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_sweep_resume_plus100_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_sweep_resume_plus100_%j.err
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

REPO_ROOT="$PROJECT_ROOT"
GALS_REPO="$GALS_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"

# Keep behavior aligned with current guided sweep defaults (NEWCLIP primary).
PRIMARY_GT_ROOT=${PRIMARY_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_dinovit/val/prediction_cmap/}
ALT_GT_ROOT_1=${ALT_GT_ROOT_1:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_dinovit/val/prediction_cmap/}
ALT_GT_ROOT_2=${ALT_GT_ROOT_2:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_xcit/val/prediction_cmap/}
ALT_GT_ROOT_3=${ALT_GT_ROOT_3:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_siglip2_dinovit/val/prediction_cmap/}

RESUME_CSV=${RESUME_CSV:-$LOG_DIR/guided_redmeat_sweep_21065307.csv}
ADDITIONAL_TRIALS=${ADDITIONAL_TRIALS:-100}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
NUM_EPOCHS=${NUM_EPOCHS:-150}
GUIDED_MODEL_NAME=${GUIDED_MODEL_NAME:-resnet50}
GUIDED_CLIP_MODEL=${GUIDED_CLIP_MODEL:-RN50}
GUIDED_PRETRAINED=${GUIDED_PRETRAINED:-1}

# Speed-oriented default: skip post-seed reruns during this continuation job.
POST_SEEDS=${POST_SEEDS:-0}
POST_SEED_START=${POST_SEED_START:-0}

# Resume in-place by default.
SWEEP_OUT=${SWEEP_OUT:-$RESUME_CSV}
POST_OUT=${POST_OUT:-$LOG_DIR/guided_redmeat_sweep_best5_resume_plus100_${SLURM_JOB_ID}.csv}
POST_SUMMARY_OUT=${POST_SUMMARY_OUT:-$LOG_DIR/guided_redmeat_sweep_best5_resume_plus100_${SLURM_JOB_ID}_summary.csv}

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}

cd "$GALS_REPO"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -f "$RESUME_CSV" ]]; then
  echo "[ERROR] RESUME_CSV does not exist: $RESUME_CSV" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing DATASET_ROOT: $DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -d "$PRIMARY_GT_ROOT" ]]; then
  echo "[ERROR] Missing PRIMARY_GT_ROOT: $PRIMARY_GT_ROOT" >&2
  exit 2
fi

python -c "import optuna" 2>/dev/null || { echo "[INFO] Installing optuna..."; pip install -q optuna; }

# Compute current max completed trial from CSV, then add ADDITIONAL_TRIALS.
CURRENT_MAX_TRIAL=$(
python - "$RESUME_CSV" <<'PY'
import csv
import sys
path = sys.argv[1]
mx = -1
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        t = r.get("trial")
        if t is None:
            continue
        try:
            tv = int(float(str(t).strip()))
        except Exception:
            continue
        if tv > mx:
            mx = tv
print(mx)
PY
)

if [[ -z "$CURRENT_MAX_TRIAL" ]]; then
  echo "[ERROR] Could not parse CURRENT_MAX_TRIAL from $RESUME_CSV" >&2
  exit 2
fi
if (( CURRENT_MAX_TRIAL < 0 )); then
  NEXT_TRIAL=0
else
  NEXT_TRIAL=$((CURRENT_MAX_TRIAL + 1))
fi
N_TRIALS=$((NEXT_TRIAL + ADDITIONAL_TRIALS))

ALT_ARGS=()
for p in "$ALT_GT_ROOT_1" "$ALT_GT_ROOT_2" "$ALT_GT_ROOT_3"; do
  if [[ -n "$p" && -d "$p" ]]; then
    ALT_ARGS+=(--alt-gt-path "$p")
  else
    echo "[INFO] Skipping missing alt GT root: $p"
  fi
done

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "Primary GT masks: $PRIMARY_GT_ROOT"
echo "Resume CSV: $RESUME_CSV"
echo "Current max trial: $CURRENT_MAX_TRIAL"
echo "Next trial id: $NEXT_TRIAL"
echo "Additional trials requested: $ADDITIONAL_TRIALS"
echo "Total n-trials target: $N_TRIALS"
echo "Sampler: $SAMPLER (seed=$SWEEP_SEED)"
echo "Guided backbone: $GUIDED_MODEL_NAME (clip_model=$GUIDED_CLIP_MODEL pretrained=$GUIDED_PRETRAINED)"
echo "Epochs per trial: $NUM_EPOCHS"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-4}  Mem: 32G"
echo "Output CSV: $SWEEP_OUT"
echo "Post seeds: $POST_SEEDS (start=$POST_SEED_START)"
echo "Post output CSV: $POST_OUT"
echo "Post summary CSV: $POST_SUMMARY_OUT"
which python

MODEL_ARGS=(--model-name "$GUIDED_MODEL_NAME" --clip-model "$GUIDED_CLIP_MODEL")
if [[ "$GUIDED_PRETRAINED" -eq 1 ]]; then
  MODEL_ARGS+=(--pretrained)
else
  MODEL_ARGS+=(--no-pretrained)
fi

srun --unbuffered python -u RedMeat_Runs/run_guided_redmeat_sweep.py \
  "$DATASET_ROOT" \
  "$PRIMARY_GT_ROOT" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --sampler "$SAMPLER" \
  --num-epochs "$NUM_EPOCHS" \
  --resume-csv "$RESUME_CSV" \
  --output-csv "$SWEEP_OUT" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$POST_OUT" \
  --post-summary-csv "$POST_SUMMARY_OUT" \
  "${MODEL_ARGS[@]}" \
  "${ALT_ARGS[@]}"
