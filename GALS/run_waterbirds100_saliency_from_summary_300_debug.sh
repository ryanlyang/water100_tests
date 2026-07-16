#!/bin/bash -l
# Reuse existing WB100 guided/vanilla/GALS checkpoints from run_summary.json
# and generate a larger saliency set (default: 300 val samples).

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/wb100_saliency_from_summary_300_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/wb100_saliency_from_summary_300_%j.err
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
DATA_PATH=${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}
GUIDED_GT_ROOT=${GUIDED_GT_ROOT:-/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}

# Existing WB100 saliency run summary with checkpoint paths.
SUMMARY_JSON=${SUMMARY_JSON:-/home/ryreu/guided_cnn/logsWaterbird/wb100_guided_vanilla_gals_saliency_21064644/run_summary.json}

NUM_VAL_SAMPLES=${NUM_VAL_SAMPLES:-300}
SAMPLE_SEED=${SAMPLE_SEED:-0}
SAMPLE_STRATEGY=${SAMPLE_STRATEGY:-balanced}
TARGET_CLASS=${TARGET_CLASS:-label}

OUT_DIR_DEFAULT="${LOG_ROOT}/wb100_guided_vanilla_gals_saliency_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}_n${NUM_VAL_SAMPLES}"
OUT_DIR=${OUT_DIR:-$OUT_DIR_DEFAULT}

if [[ ! -d "$DATA_PATH" ]]; then
  echo "[ERROR] Missing DATA_PATH: $DATA_PATH" >&2
  exit 1
fi
if [[ ! -d "$GUIDED_GT_ROOT" ]]; then
  echo "[ERROR] Missing GUIDED_GT_ROOT: $GUIDED_GT_ROOT" >&2
  exit 1
fi
if [[ ! -f "$SUMMARY_JSON" ]]; then
  echo "[ERROR] Missing SUMMARY_JSON: $SUMMARY_JSON" >&2
  exit 1
fi

mapfile -t CKPTS < <(
python - "$SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1]).expanduser().resolve()
with p.open("r", encoding="utf-8") as f:
    obj = json.load(f)
print(obj.get("guided_checkpoint", ""))
print(obj.get("vanilla_checkpoint", ""))
print(obj.get("gals_vit_checkpoint", ""))
PY
)

GUIDED_CKPT="${GUIDED_CKPT:-${CKPTS[0]:-}}"
VANILLA_CKPT="${VANILLA_CKPT:-${CKPTS[1]:-}}"
GALS_CKPT="${GALS_CKPT:-${CKPTS[2]:-}}"

if [[ -z "$GUIDED_CKPT" || ! -f "$GUIDED_CKPT" ]]; then
  echo "[ERROR] Guided checkpoint missing/invalid: $GUIDED_CKPT" >&2
  exit 1
fi
if [[ -z "$VANILLA_CKPT" || ! -f "$VANILLA_CKPT" ]]; then
  echo "[ERROR] Vanilla checkpoint missing/invalid: $VANILLA_CKPT" >&2
  exit 1
fi

RUN_GALS=${RUN_GALS:-1}
if [[ -z "$GALS_CKPT" || ! -f "$GALS_CKPT" ]]; then
  echo "[WARN] GALS checkpoint missing in summary; running without GALS."
  RUN_GALS=0
fi

cd "$REPO_ROOT/GALS"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "Guided GT root: $GUIDED_GT_ROOT"
echo "Summary JSON: $SUMMARY_JSON"
echo "Output dir: $OUT_DIR"
echo "Num val samples: $NUM_VAL_SAMPLES (seed=$SAMPLE_SEED strategy=$SAMPLE_STRATEGY target_class=$TARGET_CLASS)"
echo "Guided ckpt: $GUIDED_CKPT"
echo "Vanilla ckpt: $VANILLA_CKPT"
echo "GALS ckpt: ${GALS_CKPT:-<none>}"
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
  --guided-ckpt "$GUIDED_CKPT"
  --vanilla-ckpt "$VANILLA_CKPT"
)

if [[ "$RUN_GALS" == "1" ]]; then
  CMD+=(--gals-ckpt "$GALS_CKPT")
else
  CMD+=(--no-gals)
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --unbuffered "${CMD[@]}"
else
  "${CMD[@]}"
fi

