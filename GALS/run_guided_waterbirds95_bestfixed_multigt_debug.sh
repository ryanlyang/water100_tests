#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/guided95_bestfixed_multigt_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/guided95_bestfixed_multigt_debug_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbirdSeeds}
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

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

GT_LTL=/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
GT_NEW_TEACH=/home/ryreu/guided_cnn/waterbirds/New_Teach/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
GT_NEWCLIP=/home/ryreu/guided_cnn/waterbirds/newCLIP/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
GT_SIGLIP2=/home/ryreu/guided_cnn/waterbirds/newCLIP/LearningToLook/code/WeCLIPPlus/results_siglip2/val/prediction_cmap

ATTENTION_EPOCH=109
KL_LAMBDA=295.3017825649997
KL_INCR=0
BASE_LR=4.81872261305513e-05
CLASSIFIER_LR=0.002932587987450572
LR2_MULT=0.40940723404327417

SEEDS=${SEEDS:-"0 1 2 3 4"}
SUMMARY_CSV=${SUMMARY_CSV:-$LOG_DIR/guided95_bestfixed_multigt_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] Missing DATA_ROOT: $DATA_ROOT" >&2
  exit 1
fi

PHASE_NAMES=("ltl" "new_teach" "newclip" "siglip2")
PHASE_PATHS=("$GT_LTL" "$GT_NEW_TEACH" "$GT_NEWCLIP" "$GT_SIGLIP2")

for i in "${!PHASE_NAMES[@]}"; do
  phase="${PHASE_NAMES[$i]}"
  root="${PHASE_PATHS[$i]}"
  if [[ ! -d "$root" ]]; then
    echo "[ERROR] Missing GT path for phase '$phase': $root" >&2
    exit 1
  fi
done

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT"
echo "Seeds: $SEEDS"
echo "Summary CSV: $SUMMARY_CSV"
echo "ATTENTION_EPOCH=$ATTENTION_EPOCH KL_LAMBDA=$KL_LAMBDA KL_INCR=$KL_INCR BASE_LR=$BASE_LR CLASSIFIER_LR=$CLASSIFIER_LR LR2_MULT=$LR2_MULT"
echo "GT_LTL=$GT_LTL"
echo "GT_NEW_TEACH=$GT_NEW_TEACH"
echo "GT_NEWCLIP=$GT_NEWCLIP"
echo "GT_SIGLIP2=$GT_SIGLIP2"
which python

echo "phase,seed,attention_epoch,kl_lambda,kl_incr,base_lr,classifier_lr,lr2_mult,best_balanced_val_acc,test_acc,per_group,worst_group,checkpoint,mask_root,log_path" > "$SUMMARY_CSV"

for i in "${!PHASE_NAMES[@]}"; do
  phase="${PHASE_NAMES[$i]}"
  gt_root="${PHASE_PATHS[$i]}"

  for seed in $SEEDS; do
    run_log="$LOG_DIR/guided95_bestfixed_${phase}_seed${seed}_${SLURM_JOB_ID}.log"
    echo "=== phase=$phase seed=$seed ==="

    python - "$DATA_ROOT" "$gt_root" "$seed" "$phase" "$SUMMARY_CSV" "$run_log" \
      "$ATTENTION_EPOCH" "$KL_LAMBDA" "$KL_INCR" "$BASE_LR" "$CLASSIFIER_LR" "$LR2_MULT" <<'PY' 2>&1 | tee "$run_log"
import csv
import os
import sys
from types import SimpleNamespace

import run_guided_waterbird as rgw

data_root = sys.argv[1]
gt_root = sys.argv[2]
seed = int(sys.argv[3])
phase = sys.argv[4]
summary_csv = sys.argv[5]
run_log = sys.argv[6]
attn_epoch = int(float(sys.argv[7]))
kl_lambda = float(sys.argv[8])
kl_incr = float(sys.argv[9])
base_lr = float(sys.argv[10])
classifier_lr = float(sys.argv[11])
lr2_mult = float(sys.argv[12])

rgw.SEED = seed
rgw.base_lr = base_lr
rgw.classifier_lr = classifier_lr
rgw.lr2_mult = lr2_mult

args = SimpleNamespace(data_path=data_root, gt_path=gt_root)
best_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(args, attn_epoch, kl_lambda, kl_incr)

row = {
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
    "log_path": run_log,
}

with open(summary_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "phase", "seed", "attention_epoch", "kl_lambda", "kl_incr",
        "base_lr", "classifier_lr", "lr2_mult",
        "best_balanced_val_acc", "test_acc", "per_group", "worst_group",
        "checkpoint", "mask_root", "log_path",
    ])
    writer.writerow(row)

print(
    f"[DONE] phase={phase} seed={seed} "
    f"best_val={best_val:.4f} test_acc={test_acc:.2f}% "
    f"per_group={per_group:.2f}% worst_group={worst_group:.2f}% "
    f"checkpoint={ckpt}"
)
PY
  done
done

echo "[DONE] All reruns complete."
echo "[DONE] Summary CSV: $SUMMARY_CSV"
