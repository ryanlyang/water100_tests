#!/bin/bash -l
# Train one DecoyMNIST method for seeds 0-4 and evaluate clean-digit Pointing Game.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/%x_%j.err
#SBATCH --signal=TERM@180

set -Eeuo pipefail

METHOD="${METHOD:?Submit with METHOD=vanilla|elrep|upweight|abn|gals|afr|r4rr}"
case "$METHOD" in
  vanilla|elrep|upweight|abn|gals|afr|r4rr) ;;
  *) echo "[ERROR] Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}"
GALS_ROOT="${GALS_ROOT:-$PROJECT_ROOT/GALS}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/pointing_game_5seed_gradcam}"
METHOD_DIR="$RUN_ROOT/$METHOD"

PNG_ROOT="${PNG_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"
MNIST_ROOT="${MNIST_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data}"
GALS_MAPS="${GALS_MAPS:-$PNG_ROOT/clip_vit_attention}"
R4RR_MAPS="${R4RR_MAPS:-/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap}"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
SPLIT="${SPLIT:-test}"
TARGET_MODE="${TARGET_MODE:-label}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

TRAIN_DIR="$GALS_ROOT/RightForTheRightRegions/repro_runs/other_models/decoymnist/train"
R4RR_TRAIN="$GALS_ROOT/RightForTheRightRegions/repro_runs/r4rr/train/r4rr_decoy_fixed.py"

mkdir -p "$LOG_DIR" "$METHOD_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$GALS_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled

cd "$GALS_ROOT"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[RUN] method=$METHOD seeds=$SEEDS_CSV split=$SPLIT target_mode=$TARGET_MODE"
echo "[RUN] png_root=$PNG_ROOT"
echo "[RUN] clean_mnist_root=$MNIST_ROOT mask_threshold=$MASK_THRESHOLD"
echo "[RUN] output=$METHOD_DIR"
which python

[[ -d "$PNG_ROOT/train" && -d "$PNG_ROOT/test" ]] || {
  echo "[ERROR] Missing DecoyMNIST PNG train/test directories under $PNG_ROOT" >&2
  exit 2
}
[[ -d "$MNIST_ROOT/MNIST" ]] || {
  echo "[ERROR] Missing raw torchvision MNIST under $MNIST_ROOT/MNIST" >&2
  exit 2
}
if [[ "$METHOD" == "vanilla" || "$METHOD" == "gals" ]]; then
  [[ -d "$GALS_MAPS" ]] || { echo "[ERROR] Missing GALS maps: $GALS_MAPS" >&2; exit 2; }
fi
if [[ "$METHOD" == "r4rr" ]]; then
  [[ -d "$R4RR_MAPS" ]] || { echo "[ERROR] Missing R4RR maps: $R4RR_MAPS" >&2; exit 2; }
fi

manifest_is_valid() {
  local manifest="$1"
  local seed="$2"
  python - "$manifest" "$METHOD" "$seed" <<'PY'
import json, os, sys
path, method, seed = sys.argv[1:]
try:
    obj = json.load(open(path, "r", encoding="utf-8"))
    valid = (
        obj.get("dataset") == "decoymnist"
        and obj.get("method") == method
        and int(obj.get("seed", -1)) == int(seed)
        and os.path.isfile(obj.get("checkpoint", ""))
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

pointing_summary_is_valid() {
  local summary="$1"
  local seed="$2"
  python - "$summary" "$METHOD" "$seed" "$SPLIT" <<'PY'
import csv, os, sys
path, method, seed, split = sys.argv[1:]
try:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    row = rows[0] if len(rows) == 1 else None
    valid = (
        row is not None
        and row.get("dataset") == "decoymnist"
        and row.get("method") == method
        and int(row.get("seed", -1)) == int(seed)
        and row.get("split") == split
        and int(row.get("pg_total", 0)) > 0
        and int(row.get("errors", 1)) == 0
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

write_training_manifest() {
  local checkpoint_dir="$1"
  local manifest="$2"
  local seed="$3"
  python - "$checkpoint_dir" "$manifest" "$METHOD" "$seed" <<'PY'
import json, os, sys
from pathlib import Path

import torch

checkpoint_dir, manifest, method, seed_text = sys.argv[1:]
seed = int(seed_text)
paths = sorted(Path(checkpoint_dir).glob("*.pth"), key=lambda p: p.stat().st_mtime_ns)
if not paths:
    raise SystemExit(f"No checkpoint produced under {checkpoint_dir}")
checkpoint = paths[-1].resolve()
try:
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(str(checkpoint), map_location="cpu")
if not isinstance(payload, dict):
    raise SystemExit(f"Checkpoint payload is not a dictionary: {checkpoint}")
observed_seed = payload.get("seed", seed)
if int(observed_seed) != seed:
    raise SystemExit(f"Checkpoint seed mismatch: expected {seed}, observed {observed_seed}")
state = None
for key in ("model_state_dict", "state_dict", "model"):
    if isinstance(payload.get(key), dict):
        state = payload[key]
        break
if state is None:
    raise SystemExit(f"Checkpoint has no model state_dict: {checkpoint}")

obj = {
    "dataset": "decoymnist",
    "method": method,
    "seed": seed,
    "checkpoint": str(checkpoint),
    "checkpoint_keys": len(state),
}
for key in (
    "best_epoch",
    "best_val_acc",
    "best_val_loss",
    "test_acc",
    "test_balanced_class_acc",
    "test_worst_class_acc",
):
    if key in payload:
        value = payload[key]
        obj[key] = value.item() if hasattr(value, "item") else value
path = Path(manifest)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(str(temporary), str(path))
print(f"[MANIFEST] checkpoint={checkpoint}")
PY
}

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for seed_raw in "${SEEDS[@]}"; do
  seed="${seed_raw//[[:space:]]/}"
  [[ -n "$seed" ]] || continue

  SEED_DIR="$METHOD_DIR/seed_${seed}"
  CHECKPOINT_DIR="$SEED_DIR/checkpoints"
  TRAIN_MANIFEST="$SEED_DIR/training_manifest.json"
  TRAIN_LOG="$SEED_DIR/training.log"
  PG_DIR="$SEED_DIR/pointing_game"
  PG_SUMMARY="$PG_DIR/pointing_game_summary.csv"
  mkdir -p "$CHECKPOINT_DIR" "$PG_DIR"

  if manifest_is_valid "$TRAIN_MANIFEST" "$seed"; then
    echo "[RESUME] seed=$seed valid checkpoint manifest exists; skipping training."
  else
    echo "[TRAIN] method=$METHOD seed=$seed"
    case "$METHOD" in
      vanilla)
        python -u "$TRAIN_DIR/gals_decoy_fixed.py" \
          --png-root "$PNG_ROOT" \
          --mask-root "$GALS_MAPS" \
          --loss-mode rrr \
          --grad-weight 0.0 \
          --grad-criterion L1 \
          --epochs 19 \
          --lr 0.001 \
          --weight-decay 0.0001 \
          --n-seeds 1 \
          --seed-start "$seed" \
          --num-workers "$NUM_WORKERS" \
          --print-every 5 \
          --no-progress-bar \
          --checkpoint-dir "$CHECKPOINT_DIR" \
          2>&1 | tee "$TRAIN_LOG"
        ;;
      elrep)
        python -u "$TRAIN_DIR/elrep_decoy_fixed.py" \
          --png-root "$PNG_ROOT" \
          --epochs 19 \
          --lr 0.001 \
          --weight-decay 0.0001 \
          --theta1 4.1741182912970346e-05 \
          --theta2 2.5433427234421545e-06 \
          --n-seeds 1 \
          --seed-start "$seed" \
          --num-workers "$NUM_WORKERS" \
          --print-every 5 \
          --save-dir "$CHECKPOINT_DIR" \
          --output-csv "$SEED_DIR/training_metrics.csv" \
          2>&1 | tee "$TRAIN_LOG"
        ;;
      upweight)
        python -u "$TRAIN_DIR/upweight_decoy_fixed.py" \
          --png-root "$PNG_ROOT" \
          --epochs 19 \
          --lr 0.001 \
          --weight-decay 0.0001 \
          --n-seeds 1 \
          --seed-start "$seed" \
          --num-workers "$NUM_WORKERS" \
          --print-every 5 \
          --save-dir "$CHECKPOINT_DIR" \
          2>&1 | tee "$TRAIN_LOG"
        ;;
      abn)
        python -u "$TRAIN_DIR/abn_decoy_fixed.py" \
          --png-root "$PNG_ROOT" \
          --epochs 19 \
          --lr 0.001 \
          --weight-decay 0.0001 \
          --abn-cls-weight 3.2547104257357056 \
          --n-seeds 1 \
          --seed-start "$seed" \
          --num-workers "$NUM_WORKERS" \
          --print-every 5 \
          --save-dir "$CHECKPOINT_DIR" \
          2>&1 | tee "$TRAIN_LOG"
        ;;
      gals)
        python -u "$TRAIN_DIR/gals_decoy_fixed.py" \
          --png-root "$PNG_ROOT" \
          --mask-root "$GALS_MAPS" \
          --loss-mode rrr \
          --grad-weight 97631.97904483072 \
          --grad-criterion L1 \
          --epochs 19 \
          --lr 0.001 \
          --weight-decay 0.0001 \
          --n-seeds 1 \
          --seed-start "$seed" \
          --num-workers "$NUM_WORKERS" \
          --print-every 5 \
          --no-progress-bar \
          --checkpoint-dir "$CHECKPOINT_DIR" \
          2>&1 | tee "$TRAIN_LOG"
        ;;
      afr)
        python -u "$TRAIN_DIR/afr_decoy_fixed.py" \
          --png-root "$PNG_ROOT" \
          --seeds "$seed" \
          --stage1-epochs 19 \
          --stage2-epochs 500 \
          --lr 0.001 \
          --weight-decay 0.0001 \
          --stage2-lr 0.001 \
          --stage2-weight-decay 0.0 \
          --gamma 4.0 \
          --reg-coeff 0.0 \
          --num-workers "$NUM_WORKERS" \
          --print-every 50 \
          --save-dir "$CHECKPOINT_DIR" \
          --output-csv "$SEED_DIR/training_metrics.csv" \
          2>&1 | tee "$TRAIN_LOG"
        ;;
      r4rr)
        python -u "$R4RR_TRAIN" \
          --png-root "$PNG_ROOT" \
          --teacher-map-path "$R4RR_MAPS" \
          --epochs 19 \
          --lr 0.001 \
          --weight-decay 0.0001 \
          --attention-epoch 7 \
          --kl-lambda 495.61 \
          --kl-incr 0.0 \
          --n-seeds 1 \
          --seed-start "$seed" \
          --split-seed 0 \
          --num-workers "$NUM_WORKERS" \
          --print-every 5 \
          --save-dir "$CHECKPOINT_DIR" \
          2>&1 | tee "$TRAIN_LOG"
        ;;
    esac
    write_training_manifest "$CHECKPOINT_DIR" "$TRAIN_MANIFEST" "$seed"
  fi

  if pointing_summary_is_valid "$PG_SUMMARY" "$seed"; then
    echo "[RESUME] seed=$seed valid Pointing Game result exists; skipping evaluation."
    continue
  fi

  checkpoint="$(python - "$TRAIN_MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], "r", encoding="utf-8"))["checkpoint"])
PY
)"
  echo "[POINTING] method=$METHOD seed=$seed checkpoint=$checkpoint"
  python -u decoymnist_pointing_game_eval.py \
    --png-root "$PNG_ROOT" \
    --mnist-root "$MNIST_ROOT" \
    --checkpoint "$checkpoint" \
    --method "$METHOD" \
    --seed "$seed" \
    --split "$SPLIT" \
    --target-mode "$TARGET_MODE" \
    --mask-threshold "$MASK_THRESHOLD" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-samples "$MAX_SAMPLES" \
    --sample-seed "$SAMPLE_SEED" \
    --device cuda:0 \
    --output-dir "$PG_DIR" \
    2>&1 | tee "$SEED_DIR/pointing_game.log"
done

python -u summarize_decoymnist_pointing_game_5seed.py \
  --method-dir "$METHOD_DIR" \
  --seeds "$SEEDS_CSV"

echo "[DONE] $METHOD_DIR/pointing_game_5seed_summary.csv"
