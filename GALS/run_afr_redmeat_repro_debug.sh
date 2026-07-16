#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/afr_redmeat_repro_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/afr_redmeat_repro_debug_%j.err
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
  if [[ -n "$candidate" && -f "$candidate/run_afr_redmeat_repro.py" ]]; then
    REPO_ROOT="$(cd -- "$candidate" && pwd)"
    break
  fi
done
if [[ -z "$REPO_ROOT" ]]; then
  echo "[ERROR] Could not locate GALS repo root containing run_afr_redmeat_repro.py" >&2
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

DATA_DIR="${DATA_DIR:-/home/ryreu/guided_cnn/Food101/data/food-101-redmeat}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/guided_cnn/logsRedMeat/afr_repro_redmeat_${SLURM_JOB_ID}}"
LOGS_ROOT="${LOGS_ROOT:-/home/ryreu/guided_cnn/logsRedMeat/afr_repro_redmeat_logs_${SLURM_JOB_ID}}"

# Debug defaults
SEEDS="${SEEDS:-0}"
FULL_PAPER_GRID="${FULL_PAPER_GRID:-0}"
GAMMAS="${GAMMAS:-4,10,16}"
REG_COEFFS="${REG_COEFFS:-0,0.2,0.4}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
STAGE1_EVAL_FREQ="${STAGE1_EVAL_FREQ:-5}"
STAGE1_SAVE_FREQ="${STAGE1_SAVE_FREQ:-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
STAGE2_LR="${STAGE2_LR:-0.01}"

CLASSES="${CLASSES:-prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon}"
PLACE_MODE="${PLACE_MODE:-auto}"
PLACE_COL="${PLACE_COL:-}"
PLACE_CANDIDATES="${PLACE_CANDIDATES:-place,spurious,group,confounder,background,bg,environment,env,smoke,context}"
CHECK_IMAGES="${CHECK_IMAGES:-0}"

AUTO_INSTALL_AFR_DEPS="${AUTO_INSTALL_AFR_DEPS:-1}"

cd "${REPO_ROOT}"

# AFR train_embeddings.py writes plots to ./logs/ relative to AFR root.
mkdir -p "${AFR_ROOT}/logs"

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
echo "STAGE2_EPOCHS=${STAGE2_EPOCHS}"
echo "STAGE2_LR=${STAGE2_LR}"
echo "GAMMAS=${GAMMAS}"
echo "REG_COEFFS=${REG_COEFFS}"
echo "CLASSES=${CLASSES}"
echo "PLACE_MODE=${PLACE_MODE}"
echo "PLACE_COL=${PLACE_COL}"
echo "PLACE_CANDIDATES=${PLACE_CANDIDATES}"
echo "CHECK_IMAGES=${CHECK_IMAGES}"
which python

eval "$(
python - <<'PY'
import importlib.util
mods = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "timm",
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
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage2-lr "${STAGE2_LR}"
  --classes "${CLASSES}"
  --place-mode "${PLACE_MODE}"
  --place-candidates "${PLACE_CANDIDATES}"
)

if [[ -n "${PLACE_COL}" ]]; then
  ARGS+=(--place-col "${PLACE_COL}")
fi
if [[ "${CHECK_IMAGES}" == "1" ]]; then
  ARGS+=(--check-images)
fi

if [[ "${FULL_PAPER_GRID}" == "1" ]]; then
  ARGS+=(--full-paper-grid)
else
  ARGS+=(--gammas "${GAMMAS}" --reg-coeffs "${REG_COEFFS}")
fi

srun --unbuffered python -u run_afr_redmeat_repro.py "${ARGS[@]}"

