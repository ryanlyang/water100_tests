#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/guided_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/guided_sweep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME=${ENV_NAME:-gals_a100}
BOOTSTRAP_ENV=${BOOTSTRAP_ENV:-0}
RECREATE_ENV=${RECREATE_ENV:-0}
REQ_FILE=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/requirements.txt

if [[ "$BOOTSTRAP_ENV" -eq 1 ]]; then
  if [[ "$RECREATE_ENV" -eq 1 ]]; then
    conda env remove -n "$ENV_NAME" -y || true
  fi
  if ! conda env list | grep -E "^${ENV_NAME}[[:space:]]" >/dev/null; then
    conda create -y -n "$ENV_NAME" python=3.8
    conda activate "$ENV_NAME"
    conda install -y pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -c nvidia -c conda-forge
    conda install -y -c conda-forge pycocotools
    REQ_TMP=/tmp/${ENV_NAME}_reqs_$$.txt
    grep -v -E '^(opencv-python|pycocotools|torch|torchvision|torchray)' "$REQ_FILE" > "$REQ_TMP"
    pip install -r "$REQ_TMP"
    rm -f "$REQ_TMP"
    pip install torchray==1.0.0.2 --no-deps
    pip install opencv-python==4.6.0.66 optuna
    conda deactivate
  fi
fi

conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2
SWEEP_GT_ROOT=/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
ALT1_GT_ROOT=/home/ryreu/guided_cnn/waterbirds/New_Teach/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
ALT2_GT_ROOT=/home/ryreu/guided_cnn/waterbirds/newCLIP/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
ALT3_GT_ROOT=/home/ryreu/guided_cnn/waterbirds/newCLIP/LearningToLook/code/WeCLIPPlus/results_siglip2/val/prediction_cmap

N_TRIALS=${N_TRIALS:-50}
SWEEP_OUT=${SWEEP_OUT:-$LOG_DIR/guided_waterbird_sweep_${SLURM_JOB_ID}.csv}
SEED_SWEEP_OUT=${SEED_SWEEP_OUT:-$LOG_DIR/guided_waterbird_sweep_best5_${SLURM_JOB_ID}.csv}
SEED_LIST=${SEED_LIST:-"0 1 2 3 4"}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing DATA_ROOT: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$SWEEP_GT_ROOT" ]]; then
  echo "Missing SWEEP_GT_ROOT: $SWEEP_GT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ALT1_GT_ROOT" ]]; then
  echo "Missing ALT1_GT_ROOT: $ALT1_GT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ALT2_GT_ROOT" ]]; then
  echo "Missing ALT2_GT_ROOT: $ALT2_GT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ALT3_GT_ROOT" ]]; then
  echo "Missing ALT3_GT_ROOT: $ALT3_GT_ROOT" >&2
  exit 1
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT"
echo "Sweep masks: $SWEEP_GT_ROOT"
echo "Alt1 masks: $ALT1_GT_ROOT"
echo "Alt2 masks: $ALT2_GT_ROOT"
echo "Alt3 masks: $ALT3_GT_ROOT"
echo "Trials: $N_TRIALS"
echo "Output CSV: $SWEEP_OUT"
echo "Seed rerun CSV: $SEED_SWEEP_OUT"
which python

srun --unbuffered python -u run_guided_waterbird_sweep.py \
  "$DATA_ROOT" \
  "$SWEEP_GT_ROOT" \
  --n-trials "$N_TRIALS" \
  --output-csv "$SWEEP_OUT"

if [[ ! -f "$SWEEP_OUT" ]]; then
  echo "Sweep CSV not found: $SWEEP_OUT" >&2
  exit 1
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

run_seed_reruns() {
  local phase_name="$1"
  local gt_root="$2"
  for seed in $SEED_LIST; do
    echo "[SEED-RERUN] phase=$phase_name seed=$seed masks=$gt_root"
    python - "$DATA_ROOT" "$gt_root" "$seed" "$phase_name" "$SEED_SWEEP_OUT" \
      "$BEST_ATTN" "$BEST_KL" "$BEST_KL_INCR" "$BEST_BASE_LR" "$BEST_CLS_LR" "$BEST_LR2_MULT" <<'PY'
import csv
import os
import sys
from types import SimpleNamespace

import run_guided_waterbird as rgw

data_root = sys.argv[1]
gt_root = sys.argv[2]
seed = int(sys.argv[3])
phase = sys.argv[4]
out_csv = sys.argv[5]
attn_epoch = int(float(sys.argv[6]))
kl_lambda = float(sys.argv[7])
kl_incr = float(sys.argv[8])
base_lr = float(sys.argv[9])
classifier_lr = float(sys.argv[10])
lr2_mult = float(sys.argv[11])

rgw.SEED = seed
rgw.base_lr = base_lr
rgw.classifier_lr = classifier_lr
rgw.lr2_mult = lr2_mult
args = SimpleNamespace(data_path=data_root, gt_path=gt_root)
best_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(args, attn_epoch, kl_lambda, kl_incr)

header = [
    "phase",
    "seed",
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
file_exists = os.path.exists(out_csv)
with open(out_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    if not file_exists:
        writer.writeheader()
    writer.writerow({
        "phase": phase,
        "seed": seed,
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
print(f"[SEED DONE] phase={phase} seed={seed} best_val={best_val:.4f} test_acc={test_acc:.2f}%")
PY
  done
}

run_seed_reruns "best5_ltl" "$SWEEP_GT_ROOT"
run_seed_reruns "best5_new_teach" "$ALT1_GT_ROOT"
run_seed_reruns "best5_newclip" "$ALT2_GT_ROOT"
run_seed_reruns "best5_siglip2" "$ALT3_GT_ROOT"

echo "[DONE] Sweep + seed reruns complete."
