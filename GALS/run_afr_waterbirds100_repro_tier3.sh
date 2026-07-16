#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/afr_waterbirds100_repro_tier3_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/afr_waterbirds100_repro_tier3_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1

SCRIPT_PATH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${SBATCH_SUBMIT_DIR:-${PWD:-}}}"

REPO_ROOT_CANDIDATES=(
  "${REPO_ROOT:-}"
  "${SUBMIT_DIR}/GALS"
  "${SUBMIT_DIR}"
  "${SCRIPT_PATH_DIR}"
)
REPO_ROOT=""
for candidate in "${REPO_ROOT_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate/run_afr_waterbirds_repro.py" ]]; then
    REPO_ROOT="$(cd -- "$candidate" && pwd)"
    break
  fi
done
if [[ -z "$REPO_ROOT" ]]; then
  echo "[ERROR] Could not locate GALS repo root containing run_afr_waterbirds_repro.py" >&2
  echo "Checked: ${REPO_ROOT_CANDIDATES[*]}" >&2
  exit 2
fi

AFR_ROOT_CANDIDATES=(
  "${AFR_ROOT:-}"
  "${REPO_ROOT}/../afr"
  "${REPO_ROOT}/afr"
  "${SUBMIT_DIR}/afr"
)
AFR_ROOT=""
for candidate in "${AFR_ROOT_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -d "$candidate" && -f "$candidate/train_supervised.py" ]]; then
    AFR_ROOT="$(cd -- "$candidate" && pwd)"
    break
  fi
done
if [[ -z "$AFR_ROOT" ]]; then
  echo "[ERROR] Could not locate AFR root containing train_supervised.py" >&2
  echo "Checked: ${AFR_ROOT_CANDIDATES[*]}" >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/guided_cnn/logsWaterbird/afr_repro_wb100_full_${SLURM_JOB_ID}}"
LOGS_ROOT="${LOGS_ROOT:-/home/ryreu/guided_cnn/logsWaterbird/afr_repro_wb100_full_logs_${SLURM_JOB_ID}}"

# Paper-faithful defaults:
# Stage-1: 50 epochs SGD+constant, LR=0.003, WD=1e-4.
# Stage-2: 500 epochs, LR=0.01, gamma=linspace(4,20,33), reg={0,0.1,0.2,0.3,0.4}.
SEEDS="${SEEDS:-0,1,2}"
FULL_PAPER_GRID="${FULL_PAPER_GRID:-1}"
GAMMAS="${GAMMAS:-4,6,8,10,12,14,16,18,20}"
REG_COEFFS="${REG_COEFFS:-0,0.1,0.2,0.3,0.4}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE1_EVAL_FREQ="${STAGE1_EVAL_FREQ:-10}"
STAGE1_SAVE_FREQ="${STAGE1_SAVE_FREQ:-10}"
STAGE1_SCHEDULER="${STAGE1_SCHEDULER:-constant_lr_scheduler}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-500}"
STAGE2_LR="${STAGE2_LR:-0.01}"
AUTO_INSTALL_AFR_DEPS="${AUTO_INSTALL_AFR_DEPS:-1}"

cd "${REPO_ROOT}"

echo "[$(date)] Host: $(hostname)"
echo "SCRIPT_PATH_DIR=${SCRIPT_PATH_DIR}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "SUBMIT_DIR_USED=${SUBMIT_DIR:-<unset>}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "AFR_ROOT=${AFR_ROOT}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "LOGS_ROOT=${LOGS_ROOT}"
echo "SEEDS=${SEEDS}"
echo "FULL_PAPER_GRID=${FULL_PAPER_GRID}"
echo "STAGE1_EPOCHS=${STAGE1_EPOCHS}"
echo "STAGE1_EVAL_FREQ=${STAGE1_EVAL_FREQ}"
echo "STAGE1_SAVE_FREQ=${STAGE1_SAVE_FREQ}"
echo "STAGE1_SCHEDULER=${STAGE1_SCHEDULER}"
echo "STAGE2_EPOCHS=${STAGE2_EPOCHS}"
echo "STAGE2_LR=${STAGE2_LR}"
echo "GAMMAS=${GAMMAS}"
echo "REG_COEFFS=${REG_COEFFS}"
which python

eval "$(
python - <<'PY'
import importlib.util
import sys
mods = [
    "torch",
    "torchvision",
    "numpy",
    "timm",
    "pandas",
    "wandb",
    "wilds",
    "einops",
    "transformers",
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
hard = [m for m in missing if m in ("torch", "torchvision", "numpy")]
soft = [m for m in missing if m not in ("torch", "torchvision", "numpy")]
print('HARD_MISSING="' + " ".join(hard) + '"')
print('SOFT_MISSING="' + " ".join(soft) + '"')
PY
)"

if [[ -n "${HARD_MISSING:-}" ]]; then
  echo "[ERROR] Missing required core packages: ${HARD_MISSING}" >&2
  echo "Install/activate the correct env first." >&2
  exit 2
fi

if [[ -n "${SOFT_MISSING:-}" ]]; then
  if [[ "${AUTO_INSTALL_AFR_DEPS}" == "1" ]]; then
    echo "[INFO] Installing missing AFR deps: ${SOFT_MISSING}"
    python -m pip install --no-cache-dir ${SOFT_MISSING}
  else
    echo "[ERROR] Missing Python packages: ${SOFT_MISSING}" >&2
    echo "Set AUTO_INSTALL_AFR_DEPS=1 or install manually." >&2
    exit 2
  fi
fi

ARGS=(
  --afr-root "${AFR_ROOT}"
  --data-dir "${DATA_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --logs-root "${LOGS_ROOT}"
  --python-exe "$(which python)"
  --seeds "${SEEDS}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage1-eval-freq "${STAGE1_EVAL_FREQ}"
  --stage1-save-freq "${STAGE1_SAVE_FREQ}"
  --stage1-scheduler "${STAGE1_SCHEDULER}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage2-lr "${STAGE2_LR}"
)

if [[ "${FULL_PAPER_GRID}" == "1" ]]; then
  ARGS+=(--full-paper-grid)
else
  ARGS+=(--gammas "${GAMMAS}" --reg-coeffs "${REG_COEFFS}")
fi

srun --unbuffered python -u run_afr_waterbirds_repro.py "${ARGS[@]}"
