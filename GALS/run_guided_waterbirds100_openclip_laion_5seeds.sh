#!/bin/bash -l
# Guided Waterbirds100 fixed-hparam rerun (seeds 0-4) with OpenCLIP-LAION GT masks.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/guided100_openclip_laion_5seeds_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/guided100_openclip_laion_5seeds_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}
DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}
GT_PATH=${GT_PATH:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap}

ATTENTION_EPOCH=${ATTENTION_EPOCH:-73}
KL_LAMBDA=${KL_LAMBDA:-495.61}
KL_INCR=${KL_INCR:-0.0}
BASE_LR=${BASE_LR:-5.72e-5}
CLASSIFIER_LR=${CLASSIFIER_LR:-3.57e-3}
LR2_MULT=${LR2_MULT:-0.123}

SEEDS=${SEEDS:-"0 1 2 3 4"}
SUMMARY_CSV=${SUMMARY_CSV:-$LOG_DIR/guided100_openclip_laion_5seeds_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] Missing DATA_ROOT: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$GT_PATH" ]]; then
  echo "[ERROR] Missing GT_PATH: $GT_PATH" >&2
  exit 1
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT"
echo "GT_PATH: $GT_PATH"
echo "Seeds: $SEEDS"
echo "Summary CSV: $SUMMARY_CSV"
echo "ATTENTION_EPOCH=$ATTENTION_EPOCH KL_LAMBDA=$KL_LAMBDA KL_INCR=$KL_INCR BASE_LR=$BASE_LR CLASSIFIER_LR=$CLASSIFIER_LR LR2_MULT=$LR2_MULT"
which python

echo "seed,attention_epoch,kl_lambda,kl_incr,base_lr,classifier_lr,lr2_mult,best_balanced_val_acc,test_acc,per_group,worst_group,checkpoint,gt_path,log_path" > "$SUMMARY_CSV"

for seed in $SEEDS; do
  run_log="$LOG_DIR/guided100_openclip_laion_seed${seed}_${SLURM_JOB_ID}.log"
  echo "=== seed=$seed ==="

  python - "$DATA_ROOT" "$GT_PATH" "$seed" "$SUMMARY_CSV" "$run_log" \
    "$ATTENTION_EPOCH" "$KL_LAMBDA" "$KL_INCR" "$BASE_LR" "$CLASSIFIER_LR" "$LR2_MULT" <<'PY' 2>&1 | tee "$run_log"
import csv
import sys
from types import SimpleNamespace

import run_guided_waterbird as rgw

data_root = sys.argv[1]
gt_path = sys.argv[2]
seed = int(sys.argv[3])
summary_csv = sys.argv[4]
run_log = sys.argv[5]
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

args = SimpleNamespace(data_path=data_root, gt_path=gt_path)
best_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(args, attn_epoch, kl_lambda, kl_incr)

row = {
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
    "gt_path": gt_path,
    "log_path": run_log,
}

with open(summary_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)

print(
    f"[DONE] seed={seed} best_val={best_val:.4f} test_acc={test_acc:.2f}% "
    f"per_group={per_group:.2f}% worst_group={worst_group:.2f}% checkpoint={ckpt}"
)
PY
done

python - "$SUMMARY_CSV" <<'PY'
import csv
import math
import sys

csv_path = sys.argv[1]
rows = list(csv.DictReader(open(csv_path, "r", newline="")))
metrics = ["best_balanced_val_acc", "test_acc", "per_group", "worst_group"]

print("\n===== SUMMARY (mean +/- std over seeds) =====")
for m in metrics:
    vals = [float(r[m]) for r in rows]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(var)
    print(f"{m}: {mean:.4f} +/- {std:.4f}")
print("============================================")
PY

echo "[DONE] Guided WB100 rerun complete."
echo "[DONE] Summary CSV: $SUMMARY_CSV"
