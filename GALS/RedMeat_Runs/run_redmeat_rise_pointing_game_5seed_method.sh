#!/bin/bash -l
# Train one RedMeat method for seeds 0-4, then run shared RISE Pointing Game.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

METHOD="${METHOD:?Submit with METHOD=vanilla|elrep|upweight|abn|gals|afr|r4rr|clip_lr|clip_zs}"
case "$METHOD" in
  vanilla|elrep|upweight|abn|gals|afr|r4rr|clip_lr|clip_zs) ;;
  *) echo "[ERROR] Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsRedMeat}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_rise}"
METHOD_DIR="$RUN_ROOT/$METHOD"
DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/Food101/data/food-101-redmeat}"
MASK_ROOT="${MASK_ROOT:-$DATA_PATH/redmeat_pointing_masks}"
TEACHER_MAPS="${TEACHER_MAPS:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_laion_dinovit/val/prediction_cmap}"
GALS_MAPS="${GALS_MAPS:-$DATA_PATH/clip_rn50_attention_gradcam}"
AFR_ROOT="${AFR_ROOT:-$PROJECT_ROOT/afr}"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
if [[ "$METHOD" == "clip_lr" || "$METHOD" == "clip_zs" ]]; then
  EFFECTIVE_SEEDS_CSV="${CLIP_SEEDS_CSV:-0}"
else
  EFFECTIVE_SEEDS_CSV="$SEEDS_CSV"
fi
TARGET_MODE="${TARGET_MODE:-label}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
SKIP_SOURCE_CHECKSUM="${SKIP_SOURCE_CHECKSUM:-0}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-4}"
MAX_MASKED_BATCH="${MAX_MASKED_BATCH:-128}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
PIN_MEMORY="${PIN_MEMORY:-0}"

RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"
RISE_BANK="${RISE_BANK:-$RUN_ROOT/shared/rise_n${RISE_NUM_MASKS}_g${RISE_GRID_SIZE}_p${RISE_P1}_s${RISE_SEED}_224.npy}"

CLIP_MODEL="${CLIP_MODEL:-RN50}"
CLIP_C="${CLIP_C:-1.329346323656201}"
CLIP_FEATURE_BATCH_SIZE="${CLIP_FEATURE_BATCH_SIZE:-256}"
EXISTING_CHECKPOINT_CSV="${EXISTING_CHECKPOINT_CSV:-}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"

mkdir -p "$LOG_DIR" "$METHOD_DIR" "$(dirname "$RISE_BANK")"
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$GALS_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export SAVE_CHECKPOINTS=1

cd "$GALS_ROOT"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[RUN] dataset=redmeat method=$METHOD seeds=$EFFECTIVE_SEEDS_CSV"
echo "[RUN] data=$DATA_PATH"
echo "[RUN] masks=$MASK_ROOT target_mode=$TARGET_MODE threshold=$MASK_THRESHOLD"
echo "[RUN] rise_bank=$RISE_BANK N=$RISE_NUM_MASKS grid=$RISE_GRID_SIZE p1=$RISE_P1 seed=$RISE_SEED"
echo "[RUN] output=$METHOD_DIR"
echo "[RUN] existing_checkpoint_csv=${EXISTING_CHECKPOINT_CSV:-NONE} train_only=$TRAIN_ONLY"
which python

[[ -f "$DATA_PATH/all_images.csv" ]] || {
  echo "[ERROR] Missing RedMeat metadata: $DATA_PATH/all_images.csv" >&2
  exit 2
}
[[ -f "$MASK_ROOT/manifest.csv" && -f "$MASK_ROOT/package_metadata.json" ]] || {
  echo "[ERROR] Missing canonical RedMeat mask package under $MASK_ROOT" >&2
  exit 2
}
if [[ "$METHOD" == "gals" ]]; then
  [[ -d "$GALS_MAPS" ]] || {
    echo "[ERROR] Missing GALS maps: $GALS_MAPS" >&2
    exit 2
  }
  if ! find "$GALS_MAPS" -type f -name '*.pth' -print -quit | grep -q .; then
    echo "[ERROR] GALS map directory has no .pth maps: $GALS_MAPS" >&2
    exit 2
  fi
fi
if [[ "$METHOD" == "r4rr" ]]; then
  [[ -d "$TEACHER_MAPS" ]] || {
    echo "[ERROR] Missing R4RR teacher maps: $TEACHER_MAPS" >&2
    exit 2
  }
fi
if [[ "$METHOD" == "abn" ]]; then
  ABN_WEIGHT="$GALS_ROOT/weights/resnet50_abn_imagenet.pth.tar"
  [[ -f "$ABN_WEIGHT" ]] || {
    echo "[ERROR] Missing ABN ImageNet initialization: $ABN_WEIGHT" >&2
    exit 2
  }
fi

manifest_is_valid() {
  local manifest="$1"
  local seed="$2"
  python - "$manifest" "$METHOD" "$seed" <<'PY'
import json, os, sys
path, method, seed = sys.argv[1:]
try:
    obj = json.load(open(path, "r", encoding="utf-8"))
    paths = [obj.get("checkpoint", "")]
    if method == "afr":
        paths.append(obj.get("stage1_checkpoint", ""))
    valid = (
        obj.get("dataset") == "redmeat"
        and obj.get("method") == method
        and int(obj.get("seed", -1)) == int(seed)
        and all(value and os.path.isfile(value) for value in paths)
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

pointing_summary_is_valid() {
  local summary="$1"
  local seed="$2"
  local checkpoint="$3"
  python - "$summary" "$METHOD" "$seed" "$checkpoint" \
    "$TARGET_MODE" "$MASK_THRESHOLD" "$MAX_SAMPLES" "$SAMPLE_SEED" \
    "$RISE_NUM_MASKS" "$RISE_GRID_SIZE" "$RISE_P1" "$RISE_SEED" <<'PY'
import csv, math, os, sys
(
    path, method, seed, checkpoint, target_mode, mask_threshold,
    max_samples, sample_seed, rise_n, rise_grid, rise_p1, rise_seed,
) = sys.argv[1:]
try:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    row = rows[0] if len(rows) == 1 else None
    expected_total = int(max_samples) if int(max_samples) > 0 else 1250
    recorded_checkpoint = row.get("checkpoint", "") if row else ""
    checkpoint_matches = (
        (not checkpoint and not recorded_checkpoint)
        or (
            checkpoint
            and recorded_checkpoint
            and os.path.realpath(recorded_checkpoint) == os.path.realpath(checkpoint)
        )
    )
    valid = (
        row is not None
        and row.get("dataset") == "redmeat"
        and row.get("method") == method
        and int(row.get("seed", -1)) == int(seed)
        and row.get("split") == "test"
        and row.get("target_mode") == target_mode
        and row.get("primary_pg_protocol") == "rise_pixel_argmax"
        and int(row.get("mask_protocol_version", -1)) == 1
        and int(row.get("mask_threshold", -1)) == int(mask_threshold)
        and int(row.get("max_samples", -1)) == int(max_samples)
        and int(row.get("sample_seed", -1)) == int(sample_seed)
        and int(row.get("rise_num_masks", -1)) == int(rise_n)
        and int(row.get("rise_grid_size", -1)) == int(rise_grid)
        and math.isclose(float(row.get("rise_p1", "nan")), float(rise_p1))
        and int(row.get("rise_seed", -1)) == int(rise_seed)
        and checkpoint_matches
        and int(row.get("pg_total", 0)) == expected_total
        and int(row.get("errors", 1)) == 0
        and row.get("mask_manifest_sha256", "")
        and row.get("rise_masks_sha256", "")
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

lookup_existing_checkpoint() {
  local seed="$1"
  [[ -n "$EXISTING_CHECKPOINT_CSV" && -f "$EXISTING_CHECKPOINT_CSV" ]] || return 0
  python - "$EXISTING_CHECKPOINT_CSV" "$METHOD" "$seed" <<'PY'
import csv, sys
path, method, seed = sys.argv[1:]
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if (
            str(row.get("dataset", "")).strip().lower() == "redmeat"
            and str(row.get("method", "")).strip().lower() == method
            and str(row.get("seed", "")).strip() == seed
        ):
            print(
                f"{row.get('checkpoint', '').strip()}|"
                f"{row.get('stage1_checkpoint', '').strip()}"
            )
            break
PY
}

IFS=',' read -r -a SEEDS <<< "$EFFECTIVE_SEEDS_CSV"
for seed_raw in "${SEEDS[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  [[ -n "$seed" ]] || continue
  SEED_DIR="$METHOD_DIR/seed_${seed}"
  TRAIN_MANIFEST="$SEED_DIR/training_manifest.json"
  PG_DIR="$SEED_DIR/pointing_game"
  PG_SUMMARY="$PG_DIR/pointing_game_summary.csv"
  mkdir -p "$SEED_DIR" "$PG_DIR"

  CKPT=""
  STAGE1_CKPT=""
  if [[ "$METHOD" != "clip_lr" && "$METHOD" != "clip_zs" ]]; then
    if manifest_is_valid "$TRAIN_MANIFEST" "$seed"; then
      echo "[RESUME] seed=$seed valid training manifest exists; skipping training."
    else
      TRAIN_ARGS=(
        --method "$METHOD"
        --seed "$seed"
        --output-dir "$SEED_DIR"
        --result-json "$TRAIN_MANIFEST"
        --data-path "$DATA_PATH"
        --teacher-maps "$TEACHER_MAPS"
        --afr-root "$AFR_ROOT"
        --num-epochs 150
        --batch-size 96
        --num-workers "$NUM_WORKERS"
      )
      existing_record="$(lookup_existing_checkpoint "$seed")"
      if [[ -n "$existing_record" ]]; then
        IFS='|' read -r existing_checkpoint existing_stage1 <<< "$existing_record"
        TRAIN_ARGS+=(--existing-checkpoint "$existing_checkpoint")
        if [[ -n "$existing_stage1" ]]; then
          TRAIN_ARGS+=(--existing-stage1-checkpoint "$existing_stage1")
        fi
        echo "[IMPORT] seed=$seed checkpoint=$existing_checkpoint"
      else
        echo "[TRAIN] method=$METHOD seed=$seed"
      fi
      python -u RedMeat_Runs/train_redmeat_pointing_checkpoint.py "${TRAIN_ARGS[@]}" \
        2>&1 | tee "$SEED_DIR/training.log"
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
  fi

  if [[ "$TRAIN_ONLY" == "1" ]]; then
    echo "[TRAIN-ONLY] seed=$seed training manifest ready; skipping RISE evaluation."
    continue
  fi
  if pointing_summary_is_valid "$PG_SUMMARY" "$seed" "$CKPT"; then
    echo "[RESUME] seed=$seed valid RISE Pointing Game result exists; skipping."
    continue
  fi

  EVAL_ARGS=(
    --data-root "$DATA_PATH"
    --mask-root "$MASK_ROOT"
    --method "$METHOD"
    --seed "$seed"
    --target-mode "$TARGET_MODE"
    --mask-threshold "$MASK_THRESHOLD"
    --max-samples "$MAX_SAMPLES"
    --sample-seed "$SAMPLE_SEED"
    --rise-num-masks "$RISE_NUM_MASKS"
    --rise-grid-size "$RISE_GRID_SIZE"
    --rise-p1 "$RISE_P1"
    --rise-seed "$RISE_SEED"
    --rise-masks-path "$RISE_BANK"
    --image-batch-size "$IMAGE_BATCH_SIZE"
    --max-masked-batch "$MAX_MASKED_BATCH"
    --num-workers "$NUM_WORKERS"
    --clip-model "$CLIP_MODEL"
    --clip-c "$CLIP_C"
    --clip-feature-batch-size "$CLIP_FEATURE_BATCH_SIZE"
    --device cuda:0
    --output-dir "$PG_DIR"
  )
  if [[ -n "$CKPT" ]]; then
    EVAL_ARGS+=(--checkpoint "$CKPT")
  fi
  if [[ -n "$STAGE1_CKPT" ]]; then
    EVAL_ARGS+=(--afr-stage1-checkpoint "$STAGE1_CKPT")
  fi
  if [[ "$SKIP_SOURCE_CHECKSUM" == "1" ]]; then
    EVAL_ARGS+=(--skip-source-checksum)
  fi
  if [[ "$PIN_MEMORY" == "1" ]]; then
    EVAL_ARGS+=(--pin-memory)
  fi

  echo "[POINTING] method=$METHOD seed=$seed"
  python -u RedMeat_Runs/redmeat_rise_pointing_game_eval.py "${EVAL_ARGS[@]}" \
    2>&1 | tee "$SEED_DIR/pointing_game.log"
done

if [[ "$TRAIN_ONLY" == "1" ]]; then
  echo "[DONE] Training manifests are ready for seeds: $EFFECTIVE_SEEDS_CSV"
  exit 0
fi

python -u RedMeat_Runs/summarize_redmeat_rise_pointing_game.py \
  --method-dir "$METHOD_DIR" \
  --seeds "$EFFECTIVE_SEEDS_CSV" \
  --clip-seeds "$EFFECTIVE_SEEDS_CSV"

echo "[DONE] $METHOD_DIR/pointing_game_seed_summary.csv"
