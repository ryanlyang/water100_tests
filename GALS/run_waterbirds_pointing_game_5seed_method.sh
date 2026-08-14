#!/bin/bash -l
# Train one method on one Waterbirds dataset for seeds 0-4, then run Pointing Game.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

METHOD="${METHOD:?Submit with METHOD=vanilla|elrep|upweight|abn|gals|afr|r4rr}"
DATASET="${DATASET:?Submit with DATASET=95|100}"
case "$METHOD" in
  vanilla|elrep|upweight|abn|gals|afr|r4rr) ;;
  *) echo "[ERROR] Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac
case "$DATASET" in
  95|100) ;;
  *) echo "[ERROR] Unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac

LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_cam}"
METHOD_DIR="$RUN_ROOT/waterbirds_${DATASET}/$METHOD"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
SPLIT="${SPLIT:-val}"
TARGET_MODE="${TARGET_MODE:-label}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
EXISTING_CHECKPOINT_CSV="${EXISTING_CHECKPOINT_CSV:-}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
AFR_ROOT="${AFR_ROOT:-$PROJECT_ROOT/afr}"
MASK_PROTOCOL="${MASK_PROTOCOL:-legacy}"
case "$MASK_PROTOCOL" in
  legacy|cub) ;;
  *) echo "[ERROR] MASK_PROTOCOL must be legacy or cub (got $MASK_PROTOCOL)" >&2; exit 2 ;;
esac

if [[ "$DATASET" == "95" ]]; then
  DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}"
  TEACHER_MAPS="${TEACHER_MAPS:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap}"
  if [[ "$MASK_PROTOCOL" == "cub" ]]; then
    MASK_ROOT="${WB95_MASK_ROOT:-/home/ryreu/guided_cnn/waterbirds/CUB_200_2011/segmentations}"
  else
    MASK_ROOT="${WB95_MASK_ROOT:-/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}"
  fi
else
  DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}"
  TEACHER_MAPS="${TEACHER_MAPS:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap}"
  if [[ "$MASK_PROTOCOL" == "cub" ]]; then
    MASK_ROOT="${WB100_MASK_ROOT:-/home/ryreu/guided_cnn/waterbirds/CUB_200_2011/segmentations}"
  else
    MASK_ROOT="${WB100_MASK_ROOT:-/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}"
  fi
fi

if [[ "$METHOD" == "abn" ]]; then
  ABN_WEIGHT="$GALS_ROOT/weights/resnet50_abn_imagenet.pth.tar"
  if [[ ! -f "$ABN_WEIGHT" ]]; then
    echo "[ERROR] Missing pretrained ABN checkpoint: $ABN_WEIGHT" >&2
    echo "Download the ABN authors' model_best.pth.tar and save it at that path." >&2
    exit 2
  fi
fi

mkdir -p "$LOG_DIR" "$METHOD_DIR"
# Some Conda packages install activation hooks that read optional variables.
# Temporarily disable nounset so those hooks can initialize the environment.
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"
set -u

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export SAVE_CHECKPOINTS=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="$PROJECT_ROOT:$GALS_ROOT:${PYTHONPATH:-}"

cd "$GALS_ROOT"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[RUN] dataset=$DATASET method=$METHOD seeds=$SEEDS_CSV"
echo "[RUN] data=$DATA_PATH"
echo "[RUN] masks=$MASK_ROOT protocol=$MASK_PROTOCOL split=$SPLIT target_mode=$TARGET_MODE"
echo "[RUN] output=$METHOD_DIR"
echo "[RUN] existing_checkpoint_csv=${EXISTING_CHECKPOINT_CSV:-NONE}"
echo "[RUN] train_only=$TRAIN_ONLY"
which python

manifest_is_valid() {
  local manifest="$1"
  local seed="$2"
  python - "$manifest" "$DATASET" "$METHOD" "$seed" <<'PY'
import json, os, sys
p, dataset, method, seed = sys.argv[1:]
if not os.path.isfile(p):
    raise SystemExit(1)
try:
    obj = json.load(open(p, "r", encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if (str(obj.get("dataset")) != dataset or
        str(obj.get("method")) != method or
        str(obj.get("seed")) != seed):
    raise SystemExit(1)
paths = [obj.get("checkpoint", "")]
if obj.get("method") == "afr":
    paths.append(obj.get("stage1_checkpoint", ""))
raise SystemExit(0 if all(x and os.path.isfile(x) for x in paths) else 1)
PY
}

pointing_summary_is_valid() {
  local path="$1"
  python - "$path" "$DATASET" "$METHOD" "$SPLIT" <<'PY'
import csv, os, sys
path, dataset, method, split = sys.argv[1:]
if not os.path.isfile(path):
    raise SystemExit(1)
try:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    row = rows[0] if len(rows) == 1 else None
    valid = (row is not None and row.get("dataset") == f"waterbirds_{dataset}" and
             row.get("method") == method and row.get("split") == split and
             int(row.get("pg_total", "0")) > 0 and int(row.get("errors", "1")) == 0)
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

lookup_existing_checkpoint() {
  local seed="$1"
  [[ -n "$EXISTING_CHECKPOINT_CSV" && -f "$EXISTING_CHECKPOINT_CSV" ]] || return 0
  python - "$EXISTING_CHECKPOINT_CSV" "$DATASET" "$METHOD" "$seed" <<'PY'
import csv, sys
path, dataset, method, seed = sys.argv[1:]
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if (str(row.get("dataset", "")).strip() == dataset and
                str(row.get("method", "")).strip().lower() == method and
                str(row.get("seed", "")).strip() == seed):
            print(f"{row.get('checkpoint', '').strip()}|{row.get('stage1_checkpoint', '').strip()}")
            break
PY
}

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for seed_raw in "${SEEDS[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  [[ -n "$seed" ]] || continue
  SEED_DIR="$METHOD_DIR/seed_${seed}"
  TRAIN_MANIFEST="$SEED_DIR/training_manifest.json"
  PG_DIR="$SEED_DIR/pointing_game"
  PG_SUMMARY="$PG_DIR/pointing_game_summary.csv"
  mkdir -p "$SEED_DIR" "$PG_DIR"

  if manifest_is_valid "$TRAIN_MANIFEST" "$seed"; then
    echo "[RESUME] seed=$seed valid training manifest exists; skipping training."
  else
    TRAIN_ARGS=(
      --dataset "$DATASET"
      --method "$METHOD"
      --seed "$seed"
      --data-path "$DATA_PATH"
      --teacher-maps "$TEACHER_MAPS"
      --afr-root "$AFR_ROOT"
      --output-dir "$SEED_DIR"
      --result-json "$TRAIN_MANIFEST"
      --num-epochs 200
      --batch-size 96
      --num-workers "${SLURM_CPUS_PER_TASK:-4}"
    )
    existing_record="$(lookup_existing_checkpoint "$seed")"
    if [[ -n "$existing_record" ]]; then
      IFS='|' read -r existing_ckpt existing_stage1 <<< "$existing_record"
      TRAIN_ARGS+=(--existing-checkpoint "$existing_ckpt")
      if [[ -n "$existing_stage1" ]]; then
        TRAIN_ARGS+=(--existing-stage1-checkpoint "$existing_stage1")
      fi
      echo "[IMPORT] seed=$seed checkpoint=$existing_ckpt"
    else
      echo "[TRAIN] seed=$seed"
    fi
    python -u train_waterbirds_pointing_checkpoint.py "${TRAIN_ARGS[@]}" \
      2>&1 | tee "$SEED_DIR/training.log"
  fi

  if [[ "$TRAIN_ONLY" == "1" ]]; then
    echo "[TRAIN-ONLY] seed=$seed manifest ready; skipping CAM Pointing Game."
    continue
  fi

  if pointing_summary_is_valid "$PG_SUMMARY"; then
    echo "[RESUME] seed=$seed Pointing Game summary exists; skipping evaluation."
    continue
  fi

  mapfile -t CHECKPOINTS < <(python - "$TRAIN_MANIFEST" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj["checkpoint"])
print(obj.get("stage1_checkpoint", ""))
PY
  )
  CKPT="${CHECKPOINTS[0]}"
  STAGE1_CKPT="${CHECKPOINTS[1]:-}"

  PG_ARGS=(
    --datasets "$DATASET"
    --methods "$METHOD"
    --split "$SPLIT"
    --target-mode "$TARGET_MODE"
    --max-samples "$MAX_SAMPLES"
    --sample-seed "$SAMPLE_SEED"
    --seed "$seed"
    --afr-root "$AFR_ROOT"
    --output-dir "$PG_DIR"
  )
  if [[ "$DATASET" == "95" ]]; then
    PG_ARGS+=(--wb95-data-path "$DATA_PATH" --wb95-mask-root "$MASK_ROOT")
    case "$METHOD" in
      r4rr) PG_ARGS+=(--r4rr95-ckpt "$CKPT") ;;
      vanilla) PG_ARGS+=(--vanilla95-ckpt "$CKPT") ;;
      elrep) PG_ARGS+=(--elrep95-ckpt "$CKPT") ;;
      gals) PG_ARGS+=(--gals95-ckpt "$CKPT") ;;
      upweight) PG_ARGS+=(--upweight95-ckpt "$CKPT") ;;
      abn) PG_ARGS+=(--abn95-ckpt "$CKPT") ;;
      afr) PG_ARGS+=(--afr95-stage1-ckpt "$STAGE1_CKPT" --afr95-last-layer-ckpt "$CKPT") ;;
    esac
  else
    PG_ARGS+=(--wb100-data-path "$DATA_PATH" --wb100-mask-root "$MASK_ROOT")
    case "$METHOD" in
      r4rr) PG_ARGS+=(--r4rr100-ckpt "$CKPT") ;;
      vanilla) PG_ARGS+=(--vanilla100-ckpt "$CKPT") ;;
      elrep) PG_ARGS+=(--elrep100-ckpt "$CKPT") ;;
      gals) PG_ARGS+=(--gals100-ckpt "$CKPT") ;;
      upweight) PG_ARGS+=(--upweight100-ckpt "$CKPT") ;;
      abn) PG_ARGS+=(--abn100-ckpt "$CKPT") ;;
      afr) PG_ARGS+=(--afr100-stage1-ckpt "$STAGE1_CKPT" --afr100-last-layer-ckpt "$CKPT") ;;
    esac
  fi

  echo "[POINTING] seed=$seed"
  python -u waterbirds_pointing_game_eval.py "${PG_ARGS[@]}" \
    2>&1 | tee "$SEED_DIR/pointing_game.log"
done

if [[ "$TRAIN_ONLY" == "1" ]]; then
  echo "[DONE] Training manifests are ready for requested seeds: $SEEDS_CSV"
  exit 0
fi

python -u summarize_waterbirds_pointing_game_5seed.py \
  --method-dir "$METHOD_DIR" \
  --seeds "$SEEDS_CSV"

echo "[DONE] $METHOD_DIR/pointing_game_5seed_summary.csv"
