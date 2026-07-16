#!/bin/bash -l
# Guided KL / R4RR MobileNetV2 sweep for Waterbirds-95.
# Defaults: 50 Optuna trials, then rerun the best setting on seeds 0-4.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/guided95_mobilenetv2_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/guided95_mobilenetv2_sweep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME=${ENV_NAME:-gals_a100}
conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
export GUIDED_NUM_WORKERS="${GUIDED_NUM_WORKERS:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2
SWEEP_GT_ROOT=${SWEEP_GT_ROOT:-/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
SEED_LIST=${SEED_LIST:-"0 1 2 3 4"}
RESUME_CSV=${RESUME_CSV:-}

if [[ -n "$RESUME_CSV" && -z "${SWEEP_OUT:-}" ]]; then
  SWEEP_OUT="$RESUME_CSV"
else
  SWEEP_OUT=${SWEEP_OUT:-$LOG_DIR/guided95_mobilenetv2_sweep_${SLURM_JOB_ID}.csv}
fi
SEED_SWEEP_OUT=${SEED_SWEEP_OUT:-$LOG_DIR/guided95_mobilenetv2_best5_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] Missing DATA_ROOT: $DATA_ROOT" >&2
  exit 2
fi
if [[ ! -d "$SWEEP_GT_ROOT" ]]; then
  echo "[ERROR] Missing SWEEP_GT_ROOT: $SWEEP_GT_ROOT" >&2
  exit 2
fi
RESUME_ARGS=()
if [[ -n "$RESUME_CSV" ]]; then
  if [[ ! -f "$RESUME_CSV" ]]; then
    echo "[ERROR] Missing RESUME_CSV: $RESUME_CSV" >&2
    exit 2
  fi
  RESUME_ARGS=(--resume-csv "$RESUME_CSV")
fi

python -c "import optuna" 2>/dev/null || { echo "[INFO] Installing optuna..."; pip install -q optuna; }

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT"
echo "GT masks: $SWEEP_GT_ROOT"
echo "Backbone: mobilenet_v2 pretrained=1"
echo "Trials: $N_TRIALS (sampler=$SAMPLER sweep_seed=$SWEEP_SEED)"
if [[ -n "$RESUME_CSV" ]]; then echo "Resume CSV: $RESUME_CSV"; fi
echo "Output CSV: $SWEEP_OUT"
echo "Best5 CSV: $SEED_SWEEP_OUT"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS GUIDED_NUM_WORKERS=$GUIDED_NUM_WORKERS"
which python

srun --unbuffered python -u run_guided_waterbird_sweep.py \
  "$DATA_ROOT" \
  "$SWEEP_GT_ROOT" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --sampler "$SAMPLER" \
  --model-name mobilenet_v2 \
  --pretrained \
  "${RESUME_ARGS[@]}" \
  --output-csv "$SWEEP_OUT"

if [[ ! -f "$SWEEP_OUT" ]]; then
  echo "[ERROR] Sweep CSV not found: $SWEEP_OUT" >&2
  exit 2
fi

eval "$(
python - "$SWEEP_OUT" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("No sweep rows found")
best = max(rows, key=lambda r: float(r["best_balanced_val_acc"]))
mapping = {
    "BEST_TRIAL": "trial",
    "BEST_VAL": "best_balanced_val_acc",
    "BEST_ATTN": "attention_epoch",
    "BEST_KL": "kl_lambda",
    "BEST_KL_INCR": "kl_incr",
    "BEST_BASE_LR": "base_lr",
    "BEST_CLS_LR": "classifier_lr",
    "BEST_LR2_MULT": "lr2_mult",
}
for out_key, in_key in mapping.items():
    print(f'{out_key}="{best[in_key]}"')
PY
)"

echo "[BEST] trial=$BEST_TRIAL val=$BEST_VAL attn=$BEST_ATTN kl=$BEST_KL kl_incr=$BEST_KL_INCR base_lr=$BEST_BASE_LR cls_lr=$BEST_CLS_LR lr2_mult=$BEST_LR2_MULT"

for seed in $SEED_LIST; do
  echo "[SEED-RERUN] seed=$seed"
  python - "$DATA_ROOT" "$SWEEP_GT_ROOT" "$seed" "$SEED_SWEEP_OUT" \
    "$BEST_ATTN" "$BEST_KL" "$BEST_KL_INCR" "$BEST_BASE_LR" "$BEST_CLS_LR" "$BEST_LR2_MULT" <<'PY'
import csv
import os
import sys
from types import SimpleNamespace

import run_guided_waterbird as rgw

data_root = sys.argv[1]
gt_root = sys.argv[2]
seed = int(sys.argv[3])
out_csv = sys.argv[4]
attn_epoch = int(float(sys.argv[5]))
kl_lambda = float(sys.argv[6])
kl_incr = float(sys.argv[7])
base_lr = float(sys.argv[8])
classifier_lr = float(sys.argv[9])
lr2_mult = float(sys.argv[10])

rgw.SEED = seed
rgw.base_lr = base_lr
rgw.classifier_lr = classifier_lr
rgw.lr2_mult = lr2_mult
args = SimpleNamespace(
    data_path=data_root,
    gt_path=gt_root,
    model_name="mobilenet_v2",
    pretrained=True,
)
best_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(args, attn_epoch, kl_lambda, kl_incr)

header = [
    "seed",
    "model_name",
    "pretrained",
    "attention_epoch",
    "kl_lambda",
    "kl_incr",
    "base_lr",
    "classifier_lr",
    "lr2_mult",
    "best_balanced_val_acc",
    "test_acc",
    "per_group",
    "worst_group",
    "checkpoint",
    "mask_root",
]
exists = os.path.exists(out_csv)
with open(out_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    if not exists:
        writer.writeheader()
    writer.writerow({
        "seed": seed,
        "model_name": "mobilenet_v2",
        "pretrained": 1,
        "attention_epoch": attn_epoch,
        "kl_lambda": kl_lambda,
        "kl_incr": kl_incr,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "lr2_mult": lr2_mult,
        "best_balanced_val_acc": best_val,
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "checkpoint": ckpt,
        "mask_root": gt_root,
    })
print(f"[SEED DONE] seed={seed} best_val={best_val:.4f} test_acc={test_acc:.2f}% per_group={per_group:.2f}% worst_group={worst_group:.2f}%")
PY
done

echo "[DONE] Waterbirds-95 guided MobileNetV2 sweep + seed reruns complete."
