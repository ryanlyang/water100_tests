#!/bin/bash -l
# Fixed five-seed rerun for the best guided RedMeat MobileNetV2 trial.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_mobilenetv2_fixed5_trial11_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/guided_redmeat_mobilenetv2_fixed5_trial11_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV_CANDIDATES=(
  "${SCRIPT_DIR}/common_env.sh"
  "${SBATCH_SUBMIT_DIR:-}/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/Food101/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
)
COMMON_ENV=""
for candidate in "${COMMON_ENV_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    COMMON_ENV="$candidate"
    break
  fi
done
if [[ -z "$COMMON_ENV" ]]; then
  echo "[ERROR] Could not locate common_env.sh" >&2
  exit 2
fi
source "$COMMON_ENV"

redmeat_set_defaults
redmeat_activate_env

REPO_ROOT="$PROJECT_ROOT"
GALS_REPO="$GALS_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"

PRIMARY_GT_ROOT=${PRIMARY_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_dinovit/val/prediction_cmap/}

SEED_LIST=${SEED_LIST:-"0 1 2 3 4"}
ATTENTION_EPOCH=${ATTENTION_EPOCH:-71}
KL_LAMBDA=${KL_LAMBDA:-43.53}
KL_INCR=${KL_INCR:-0.0}
BASE_LR=${BASE_LR:-0.00216}
CLASSIFIER_LR=${CLASSIFIER_LR:-0.00262}
LR2_MULT=${LR2_MULT:-0.849}
NUM_EPOCHS=${NUM_EPOCHS:-150}
GUIDED_MODEL_NAME=${GUIDED_MODEL_NAME:-mobilenet_v2}
GUIDED_TUNE_MODE=${GUIDED_TUNE_MODE:-full}
GUIDED_CLIP_MODEL=${GUIDED_CLIP_MODEL:-RN50}
GUIDED_PRETRAINED=${GUIDED_PRETRAINED:-1}

OUT_CSV=${OUT_CSV:-$LOG_DIR/guided_redmeat_mobilenetv2_fixed5_trial11_${SLURM_JOB_ID}.csv}
SUMMARY_CSV=${SUMMARY_CSV:-$LOG_DIR/guided_redmeat_mobilenetv2_fixed5_trial11_summary_${SLURM_JOB_ID}.csv}
CKPT_DIR=${CKPT_DIR:-$GALS_REPO/RedMeat_Guided_MobileNetV2_Checkpoints}

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}
export GUIDED_NUM_WORKERS=${GUIDED_NUM_WORKERS:-0}

cd "$GALS_REPO"
export PYTHONPATH="$REPO_ROOT:$GALS_REPO:${PYTHONPATH:-}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing DATASET_ROOT: $DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -d "$PRIMARY_GT_ROOT" ]]; then
  echo "[ERROR] Missing PRIMARY_GT_ROOT: $PRIMARY_GT_ROOT" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "Primary GT masks: $PRIMARY_GT_ROOT"
echo "Backbone: $GUIDED_MODEL_NAME tune_mode=$GUIDED_TUNE_MODE pretrained=$GUIDED_PRETRAINED"
echo "Fixed hyperparams from guided RedMeat MobileNetV2 trial 11:"
echo "  attention_epoch=$ATTENTION_EPOCH"
echo "  kl_lambda=$KL_LAMBDA kl_incr=$KL_INCR"
echo "  base_lr=$BASE_LR"
echo "  classifier_lr=$CLASSIFIER_LR"
echo "  lr2_mult=$LR2_MULT"
echo "Seeds: $SEED_LIST"
echo "Epochs: $NUM_EPOCHS workers=$GUIDED_NUM_WORKERS"
echo "Output CSV: $OUT_CSV"
echo "Summary CSV: $SUMMARY_CSV"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
which python

python - "$DATASET_ROOT" "$PRIMARY_GT_ROOT" "$OUT_CSV" "$SUMMARY_CSV" "$SEED_LIST" \
  "$ATTENTION_EPOCH" "$KL_LAMBDA" "$KL_INCR" "$BASE_LR" "$CLASSIFIER_LR" "$LR2_MULT" \
  "$NUM_EPOCHS" "$CKPT_DIR" "$GUIDED_MODEL_NAME" "$GUIDED_TUNE_MODE" "$GUIDED_CLIP_MODEL" "$GUIDED_PRETRAINED" <<'PY'
import argparse
import csv
import os
import sys
import time

import numpy as np

import RedMeat_Runs.run_guided_redmeat as rgm

(
    data_path,
    gt_path,
    out_csv,
    summary_csv,
    seed_list_s,
    attention_epoch_s,
    kl_lambda_s,
    kl_incr_s,
    base_lr_s,
    classifier_lr_s,
    lr2_mult_s,
    num_epochs_s,
    ckpt_dir,
    model_name,
    tune_mode,
    clip_model,
    pretrained_s,
) = sys.argv[1:18]

seeds = [int(s) for s in seed_list_s.split()]
attention_epoch = int(float(attention_epoch_s))
kl_lambda = float(kl_lambda_s)
kl_incr = float(kl_incr_s)
base_lr = float(base_lr_s)
classifier_lr = float(classifier_lr_s)
lr2_mult = float(lr2_mult_s)
num_epochs = int(num_epochs_s)
pretrained = str(pretrained_s).lower() in {"1", "true", "yes", "y", "on"}

classes = ["prime_rib", "pork_chop", "steak", "baby_back_ribs", "filet_mignon"]

header = [
    "seed",
    "gt_path",
    "model_name",
    "clip_model",
    "tune_mode",
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
    "seconds",
]

rows = []
for seed in seeds:
    rgm.SEED = seed
    rgm.base_lr = base_lr
    rgm.classifier_lr = classifier_lr
    rgm.lr2_mult = lr2_mult
    rgm.num_epochs = num_epochs
    rgm.checkpoint_dir = ckpt_dir

    run_args = argparse.Namespace(
        data_path=data_path,
        gt_path=gt_path,
        split_col="split",
        label_col="label",
        path_col="abs_file_path",
        classes=classes,
        model_name=model_name,
        clip_model=clip_model,
        tune_mode=tune_mode,
        pretrained=pretrained,
    )
    t0 = time.time()
    best_balanced_val, test_acc, per_group, worst_group, ckpt = rgm.run_single(
        run_args,
        attention_epoch,
        kl_lambda,
        kl_incr,
    )
    row = {
        "seed": seed,
        "gt_path": gt_path,
        "model_name": model_name,
        "clip_model": clip_model,
        "tune_mode": tune_mode,
        "pretrained": int(pretrained),
        "attention_epoch": attention_epoch,
        "kl_lambda": kl_lambda,
        "kl_incr": kl_incr,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "lr2_mult": lr2_mult,
        "best_balanced_val_acc": best_balanced_val,
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "checkpoint": ckpt,
        "seconds": int(time.time() - t0),
    }
    rows.append(row)
    exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(
        f"[SEED DONE] seed={seed} best_val={best_balanced_val:.4f} "
        f"test_acc={test_acc:.2f}% per_group={per_group:.2f}% worst_group={worst_group:.2f}%",
        flush=True,
    )

metrics = ["best_balanced_val_acc", "test_acc", "per_group", "worst_group"]
summary_header = ["metric", "mean", "std", "n"]
summary_rows = []
print("\n===== SUMMARY OVER SEEDS =====", flush=True)
for metric in metrics:
    vals = np.array([float(r[metric]) for r in rows], dtype=float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    summary_rows.append({"metric": metric, "mean": mean, "std": std, "n": len(vals)})
    print(f"{metric}: mean={mean:.4f} std={std:.4f} n={len(vals)}", flush=True)

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_header)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"[DONE] wrote {out_csv}", flush=True)
print(f"[DONE] wrote {summary_csv}", flush=True)
PY
