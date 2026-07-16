#!/bin/bash -l
# Pointing Game eval for Waterbirds 95/100 across vanilla, guided, GALS, AFR.
# Uses the best-known checkpoints from prior runs (auto-discovery), unless
# explicit checkpoint paths are provided via environment variables.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_pointing_game_eval_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_pointing_game_eval_debug_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

contains_token() {
  local csv="$1"
  local token="$2"
  [[ ",${csv}," == *",${token},"* ]]
}

latest_existing_file() {
  # Usage: latest_existing_file "glob1" "glob2" ...
  local out=""
  local cand=""
  for pat in "$@"; do
    cand="$(ls -1dt ${pat} 2>/dev/null | head -n1 || true)"
    if [[ -n "$cand" ]]; then
      if [[ -z "$out" || "$cand" -nt "$out" ]]; then
        out="$cand"
      fi
    fi
  done
  echo "$out"
}

resolve_ckpt_from_summary() {
  local summary="$1"
  local key="$2"
  python - "$summary" "$key" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1]).expanduser().resolve()
key = sys.argv[2]
if not summary.is_file():
    print("")
    raise SystemExit(0)
with summary.open("r", encoding="utf-8") as f:
    obj = json.load(f)
v = obj.get(key, "")
if isinstance(v, str):
    print(v)
else:
    print("")
PY
}

resolve_afr_best_ckpts() {
  # Prints 2 lines: stage1_ckpt then stage2_last_layer_ckpt
  local best_csv="$1"
  python - "$best_csv" <<'PY'
import sys
from pathlib import Path
import pandas as pd

csv_path = Path(sys.argv[1]).expanduser().resolve()
if not csv_path.is_file():
    print("")
    print("")
    raise SystemExit(0)

df = pd.read_csv(csv_path)
if df.empty:
    print("")
    print("")
    raise SystemExit(0)

score_col = "best_val_wga" if "best_val_wga" in df.columns else "best_test_at_val"
idx = df[score_col].astype(float).idxmax()
row = df.loc[idx]

stage1_dir = Path(str(row["stage1_dir"])).expanduser().resolve()
stage2_dir = Path(str(row["stage2_dir"])).expanduser().resolve()

stage1_ckpt = stage1_dir / "final_checkpoint.pt"
# AFR stage2 stores only the last-layer model state_dict in final_checkpoint.pt
# under the stage2 run directory.
stage2_last_layer_ckpt = stage2_dir / "final_checkpoint.pt"

print(stage1_ckpt)
print(stage2_last_layer_ckpt)
PY
}

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
  if [[ -n "$candidate" && -f "$candidate/waterbirds_pointing_game_eval.py" ]]; then
    REPO_ROOT="$(cd -- "$candidate" && pwd)"
    break
  fi
done
if [[ -z "$REPO_ROOT" ]]; then
  echo "[ERROR] Could not locate repo root containing waterbirds_pointing_game_eval.py" >&2
  echo "Checked: ${REPO_ROOT_CANDIDATES[*]}" >&2
  exit 2
fi

LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsWaterbird}"
mkdir -p "$LOG_ROOT"

DATASETS_RAW="${DATASETS:-95,100}"
METHODS_RAW="${METHODS:-guided,vanilla,gals,afr}"
DATASETS="$(echo "$DATASETS_RAW" | tr -d '[:space:]')"
METHODS="$(echo "$METHODS_RAW" | tr -d '[:space:]')"

SPLIT="${SPLIT:-test}"
TARGET_MODE="${TARGET_MODE:-label}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"

WB95_DATA_PATH="${WB95_DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}"
WB100_DATA_PATH="${WB100_DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}"

CUB_MASK_ROOT="${CUB_MASK_ROOT:-/home/ryreu/guided_cnn/waterbirds/CUB_200_2011/segmentations}"
WB95_MASK_ROOT="${WB95_MASK_ROOT:-$CUB_MASK_ROOT}"
WB100_MASK_ROOT="${WB100_MASK_ROOT:-$CUB_MASK_ROOT}"

OUTPUT_DIR_DEFAULT="${LOG_ROOT}/waterbirds_pointing_game_eval_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"

# Optional explicit summaries (auto-discovered if empty).
WB95_SALIENCY_SUMMARY="${WB95_SALIENCY_SUMMARY:-}"
WB100_SALIENCY_SUMMARY="${WB100_SALIENCY_SUMMARY:-}"

if contains_token "$METHODS" "guided" || contains_token "$METHODS" "vanilla" || contains_token "$METHODS" "gals"; then
  if contains_token "$DATASETS" "95" && [[ -z "$WB95_SALIENCY_SUMMARY" ]]; then
    WB95_SALIENCY_SUMMARY="$(latest_existing_file \
      "${LOG_ROOT}/wb95_guided_vanilla_gals_saliency_"*/run_summary.json \
      "${LOG_ROOT}/wb_saliency_guided_vanilla_gals_waterbird_complete95_forest2water2_"*/run_summary.json \
    )"
  fi
  if contains_token "$DATASETS" "100" && [[ -z "$WB100_SALIENCY_SUMMARY" ]]; then
    WB100_SALIENCY_SUMMARY="$(latest_existing_file \
      "${LOG_ROOT}/wb100_guided_vanilla_gals_saliency_"*/run_summary.json \
      "${LOG_ROOT}/wb_saliency_guided_vanilla_gals_waterbird_1_0_forest2water2_"*/run_summary.json \
    )"
  fi
fi

# Optional explicit checkpoints (override summary lookup when provided).
GUIDED95_CKPT="${GUIDED95_CKPT:-}"
GUIDED100_CKPT="${GUIDED100_CKPT:-}"
VANILLA95_CKPT="${VANILLA95_CKPT:-}"
VANILLA100_CKPT="${VANILLA100_CKPT:-}"
GALS95_CKPT="${GALS95_CKPT:-}"
GALS100_CKPT="${GALS100_CKPT:-}"

if contains_token "$DATASETS" "95"; then
  if contains_token "$METHODS" "guided" && [[ -z "$GUIDED95_CKPT" ]]; then
    [[ -n "$WB95_SALIENCY_SUMMARY" ]] || { echo "[ERROR] Missing WB95 saliency summary for guided ckpt lookup." >&2; exit 2; }
    GUIDED95_CKPT="$(resolve_ckpt_from_summary "$WB95_SALIENCY_SUMMARY" guided_checkpoint)"
  fi
  if contains_token "$METHODS" "vanilla" && [[ -z "$VANILLA95_CKPT" ]]; then
    [[ -n "$WB95_SALIENCY_SUMMARY" ]] || { echo "[ERROR] Missing WB95 saliency summary for vanilla ckpt lookup." >&2; exit 2; }
    VANILLA95_CKPT="$(resolve_ckpt_from_summary "$WB95_SALIENCY_SUMMARY" vanilla_checkpoint)"
  fi
  if contains_token "$METHODS" "gals" && [[ -z "$GALS95_CKPT" ]]; then
    [[ -n "$WB95_SALIENCY_SUMMARY" ]] || { echo "[ERROR] Missing WB95 saliency summary for GALS ckpt lookup." >&2; exit 2; }
    GALS95_CKPT="$(resolve_ckpt_from_summary "$WB95_SALIENCY_SUMMARY" gals_vit_checkpoint)"
  fi
fi

if contains_token "$DATASETS" "100"; then
  if contains_token "$METHODS" "guided" && [[ -z "$GUIDED100_CKPT" ]]; then
    [[ -n "$WB100_SALIENCY_SUMMARY" ]] || { echo "[ERROR] Missing WB100 saliency summary for guided ckpt lookup." >&2; exit 2; }
    GUIDED100_CKPT="$(resolve_ckpt_from_summary "$WB100_SALIENCY_SUMMARY" guided_checkpoint)"
  fi
  if contains_token "$METHODS" "vanilla" && [[ -z "$VANILLA100_CKPT" ]]; then
    [[ -n "$WB100_SALIENCY_SUMMARY" ]] || { echo "[ERROR] Missing WB100 saliency summary for vanilla ckpt lookup." >&2; exit 2; }
    VANILLA100_CKPT="$(resolve_ckpt_from_summary "$WB100_SALIENCY_SUMMARY" vanilla_checkpoint)"
  fi
  if contains_token "$METHODS" "gals" && [[ -z "$GALS100_CKPT" ]]; then
    [[ -n "$WB100_SALIENCY_SUMMARY" ]] || { echo "[ERROR] Missing WB100 saliency summary for GALS ckpt lookup." >&2; exit 2; }
    GALS100_CKPT="$(resolve_ckpt_from_summary "$WB100_SALIENCY_SUMMARY" gals_vit_checkpoint)"
  fi
fi

# AFR checkpoint resolution.
AFR_ROOT="${AFR_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/afr}"
AFR95_BEST_CSV="${AFR95_BEST_CSV:-}"
AFR100_BEST_CSV="${AFR100_BEST_CSV:-}"
AFR95_STAGE1_CKPT="${AFR95_STAGE1_CKPT:-}"
AFR95_LAST_LAYER_CKPT="${AFR95_LAST_LAYER_CKPT:-}"
AFR100_STAGE1_CKPT="${AFR100_STAGE1_CKPT:-}"
AFR100_LAST_LAYER_CKPT="${AFR100_LAST_LAYER_CKPT:-}"

if contains_token "$METHODS" "afr"; then
  if contains_token "$DATASETS" "95" && [[ -z "$AFR95_STAGE1_CKPT" ]]; then
    if [[ -z "$AFR95_BEST_CSV" ]]; then
      AFR95_BEST_CSV="$(latest_existing_file "${LOG_ROOT}/afr_repro_"*/afr_waterbirds_best_by_seed.csv)"
    fi
    [[ -n "$AFR95_BEST_CSV" ]] || { echo "[ERROR] Could not auto-find AFR95 best CSV." >&2; exit 2; }
    mapfile -t _afr95 < <(resolve_afr_best_ckpts "$AFR95_BEST_CSV")
    AFR95_STAGE1_CKPT="${_afr95[0]:-}"
    if [[ -z "$AFR95_LAST_LAYER_CKPT" ]]; then
      AFR95_LAST_LAYER_CKPT="${_afr95[1]:-}"
    fi
  fi

  if contains_token "$DATASETS" "100" && [[ -z "$AFR100_STAGE1_CKPT" ]]; then
    if [[ -z "$AFR100_BEST_CSV" ]]; then
      AFR100_BEST_CSV="$(latest_existing_file "${LOG_ROOT}/afr_repro_wb100_"*/afr_waterbirds_best_by_seed.csv)"
    fi
    [[ -n "$AFR100_BEST_CSV" ]] || { echo "[ERROR] Could not auto-find AFR100 best CSV." >&2; exit 2; }
    mapfile -t _afr100 < <(resolve_afr_best_ckpts "$AFR100_BEST_CSV")
    AFR100_STAGE1_CKPT="${_afr100[0]:-}"
    if [[ -z "$AFR100_LAST_LAYER_CKPT" ]]; then
      AFR100_LAST_LAYER_CKPT="${_afr100[1]:-}"
    fi
  fi
fi

require_file_if_used() {
  local label="$1"
  local path="$2"
  if [[ -n "$path" && ! -f "$path" ]]; then
    echo "[ERROR] Missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_dir_if_used() {
  local label="$1"
  local path="$2"
  if [[ -n "$path" && ! -d "$path" ]]; then
    echo "[ERROR] Missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_dir_if_used "REPO_ROOT" "$REPO_ROOT"
if contains_token "$DATASETS" "95"; then
  require_dir_if_used "WB95_DATA_PATH" "$WB95_DATA_PATH"
  require_dir_if_used "WB95_MASK_ROOT" "$WB95_MASK_ROOT"
fi
if contains_token "$DATASETS" "100"; then
  require_dir_if_used "WB100_DATA_PATH" "$WB100_DATA_PATH"
  require_dir_if_used "WB100_MASK_ROOT" "$WB100_MASK_ROOT"
fi
if contains_token "$METHODS" "afr"; then
  require_dir_if_used "AFR_ROOT" "$AFR_ROOT"
fi

require_file_if_used "WB95_SALIENCY_SUMMARY" "$WB95_SALIENCY_SUMMARY"
require_file_if_used "WB100_SALIENCY_SUMMARY" "$WB100_SALIENCY_SUMMARY"

require_file_if_used "GUIDED95_CKPT" "$GUIDED95_CKPT"
require_file_if_used "GUIDED100_CKPT" "$GUIDED100_CKPT"
require_file_if_used "VANILLA95_CKPT" "$VANILLA95_CKPT"
require_file_if_used "VANILLA100_CKPT" "$VANILLA100_CKPT"
require_file_if_used "GALS95_CKPT" "$GALS95_CKPT"
require_file_if_used "GALS100_CKPT" "$GALS100_CKPT"
require_file_if_used "AFR95_STAGE1_CKPT" "$AFR95_STAGE1_CKPT"
require_file_if_used "AFR95_LAST_LAYER_CKPT" "$AFR95_LAST_LAYER_CKPT"
require_file_if_used "AFR100_STAGE1_CKPT" "$AFR100_STAGE1_CKPT"
require_file_if_used "AFR100_LAST_LAYER_CKPT" "$AFR100_LAST_LAYER_CKPT"

if contains_token "$DATASETS" "95"; then
  if contains_token "$METHODS" "guided" && [[ -z "$GUIDED95_CKPT" ]]; then
    echo "[ERROR] Guided WB95 checkpoint could not be resolved." >&2
    exit 2
  fi
  if contains_token "$METHODS" "vanilla" && [[ -z "$VANILLA95_CKPT" ]]; then
    echo "[ERROR] Vanilla WB95 checkpoint could not be resolved." >&2
    exit 2
  fi
  if contains_token "$METHODS" "gals" && [[ -z "$GALS95_CKPT" ]]; then
    echo "[ERROR] GALS WB95 checkpoint could not be resolved." >&2
    exit 2
  fi
  if contains_token "$METHODS" "afr" && { [[ -z "$AFR95_STAGE1_CKPT" ]] || [[ -z "$AFR95_LAST_LAYER_CKPT" ]]; }; then
    echo "[ERROR] AFR WB95 checkpoints (stage1/stage2) could not be resolved." >&2
    exit 2
  fi
fi

if contains_token "$DATASETS" "100"; then
  if contains_token "$METHODS" "guided" && [[ -z "$GUIDED100_CKPT" ]]; then
    echo "[ERROR] Guided WB100 checkpoint could not be resolved." >&2
    exit 2
  fi
  if contains_token "$METHODS" "vanilla" && [[ -z "$VANILLA100_CKPT" ]]; then
    echo "[ERROR] Vanilla WB100 checkpoint could not be resolved." >&2
    exit 2
  fi
  if contains_token "$METHODS" "gals" && [[ -z "$GALS100_CKPT" ]]; then
    echo "[ERROR] GALS WB100 checkpoint could not be resolved." >&2
    exit 2
  fi
  if contains_token "$METHODS" "afr" && { [[ -z "$AFR100_STAGE1_CKPT" ]] || [[ -z "$AFR100_LAST_LAYER_CKPT" ]]; }; then
    echo "[ERROR] AFR WB100 checkpoints (stage1/stage2) could not be resolved." >&2
    exit 2
  fi
fi

cd "$REPO_ROOT"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Datasets: $DATASETS (split=$SPLIT)"
echo "Methods: $METHODS"
echo "WB95 data: $WB95_DATA_PATH"
echo "WB100 data: $WB100_DATA_PATH"
echo "WB95 masks: $WB95_MASK_ROOT"
echo "WB100 masks: $WB100_MASK_ROOT"
echo "WB95 summary: ${WB95_SALIENCY_SUMMARY:-<none>}"
echo "WB100 summary: ${WB100_SALIENCY_SUMMARY:-<none>}"
echo "Guided ckpts: 95=${GUIDED95_CKPT:-<none>} 100=${GUIDED100_CKPT:-<none>}"
echo "Vanilla ckpts: 95=${VANILLA95_CKPT:-<none>} 100=${VANILLA100_CKPT:-<none>}"
echo "GALS ckpts: 95=${GALS95_CKPT:-<none>} 100=${GALS100_CKPT:-<none>}"
echo "AFR root: $AFR_ROOT"
echo "AFR ckpts: 95 stage1=${AFR95_STAGE1_CKPT:-<none>} stage2=${AFR95_LAST_LAYER_CKPT:-<none>}"
echo "AFR ckpts: 100 stage1=${AFR100_STAGE1_CKPT:-<none>} stage2=${AFR100_LAST_LAYER_CKPT:-<none>}"
echo "Output dir: $OUTPUT_DIR"
which python

CMD=(
  python -u waterbirds_pointing_game_eval.py
  --datasets "$DATASETS"
  --split "$SPLIT"
  --target-mode "$TARGET_MODE"
  --max-samples "$MAX_SAMPLES"
  --sample-seed "$SAMPLE_SEED"
  --seed "$GLOBAL_SEED"
  --methods "$METHODS"
  --wb95-data-path "$WB95_DATA_PATH"
  --wb100-data-path "$WB100_DATA_PATH"
  --wb95-mask-root "$WB95_MASK_ROOT"
  --wb100-mask-root "$WB100_MASK_ROOT"
  --afr-root "$AFR_ROOT"
  --output-dir "$OUTPUT_DIR"
)

[[ -n "$GUIDED95_CKPT" ]] && CMD+=(--guided95-ckpt "$GUIDED95_CKPT")
[[ -n "$GUIDED100_CKPT" ]] && CMD+=(--guided100-ckpt "$GUIDED100_CKPT")
[[ -n "$VANILLA95_CKPT" ]] && CMD+=(--vanilla95-ckpt "$VANILLA95_CKPT")
[[ -n "$VANILLA100_CKPT" ]] && CMD+=(--vanilla100-ckpt "$VANILLA100_CKPT")
[[ -n "$GALS95_CKPT" ]] && CMD+=(--gals95-ckpt "$GALS95_CKPT")
[[ -n "$GALS100_CKPT" ]] && CMD+=(--gals100-ckpt "$GALS100_CKPT")
[[ -n "$AFR95_STAGE1_CKPT" ]] && CMD+=(--afr95-stage1-ckpt "$AFR95_STAGE1_CKPT")
[[ -n "$AFR95_LAST_LAYER_CKPT" ]] && CMD+=(--afr95-last-layer-ckpt "$AFR95_LAST_LAYER_CKPT")
[[ -n "$AFR100_STAGE1_CKPT" ]] && CMD+=(--afr100-stage1-ckpt "$AFR100_STAGE1_CKPT")
[[ -n "$AFR100_LAST_LAYER_CKPT" ]] && CMD+=(--afr100-last-layer-ckpt "$AFR100_LAST_LAYER_CKPT")

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --unbuffered "${CMD[@]}"
else
  "${CMD[@]}"
fi
