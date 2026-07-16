#!/bin/bash -l
# Guided Waterbirds mask-ablation runner (fixed subset corruption across full training).
#
# Usage examples:
#   sbatch run_guided_waterbirds_mask_ablation_5seeds.sh
#   sbatch --export=ALL,WB_PROFILE=100,ABLATION_MODE=blur,ABLATION_FRAC=0.15,BLUR_KSIZE=31,BLUR_SIGMA=8.0 run_guided_waterbirds_mask_ablation_5seeds.sh
#   sbatch --export=ALL,WB_PROFILE=95,ABLATION_MODE=invert,ABLATION_FRAC=0.30 run_guided_waterbirds_mask_ablation_5seeds.sh

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=13:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/guided_wb_mask_ablation_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/guided_wb_mask_ablation_%j.err
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

WB_PROFILE=${WB_PROFILE:-95}         # 95 or 100
ABLATION_MODE=${ABLATION_MODE:-invert} # invert | blur | none
ABLATION_FRAC=${ABLATION_FRAC:-0.15} # fraction of train masks to corrupt
ABLATION_SEED=${ABLATION_SEED:-1337} # fixed subset seed
BLUR_KSIZE=${BLUR_KSIZE:-31}         # odd integer
BLUR_SIGMA=${BLUR_SIGMA:-8.0}

if [[ "$WB_PROFILE" == "95" ]]; then
  DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}
  GT_PATH=${GT_PATH:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap}
  ATTENTION_EPOCH=${ATTENTION_EPOCH:-109}
  KL_LAMBDA=${KL_LAMBDA:-295.30}
  BASE_LR=${BASE_LR:-4.82e-5}
  CLASSIFIER_LR=${CLASSIFIER_LR:-2.93e-3}
  LR2_MULT=${LR2_MULT:-0.409}
elif [[ "$WB_PROFILE" == "100" ]]; then
  DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}
  GT_PATH=${GT_PATH:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap}
  ATTENTION_EPOCH=${ATTENTION_EPOCH:-73}
  KL_LAMBDA=${KL_LAMBDA:-495.61}
  BASE_LR=${BASE_LR:-5.72e-5}
  CLASSIFIER_LR=${CLASSIFIER_LR:-3.57e-3}
  LR2_MULT=${LR2_MULT:-0.123}
else
  echo "[ERROR] WB_PROFILE must be 95 or 100, got: $WB_PROFILE" >&2
  exit 2
fi

KL_INCR=${KL_INCR:-0.0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
SUMMARY_CSV=${SUMMARY_CSV:-$LOG_DIR/guided_wb${WB_PROFILE}_${ABLATION_MODE}_p${ABLATION_FRAC}_${SLURM_JOB_ID}.csv}

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
echo "WB_PROFILE=$WB_PROFILE data=$DATA_ROOT"
echo "GT_PATH=$GT_PATH"
echo "Ablation: mode=$ABLATION_MODE frac=$ABLATION_FRAC seed=$ABLATION_SEED blur_ksize=$BLUR_KSIZE blur_sigma=$BLUR_SIGMA"
echo "Seeds: $SEEDS"
echo "Summary CSV: $SUMMARY_CSV"
echo "ATTENTION_EPOCH=$ATTENTION_EPOCH KL_LAMBDA=$KL_LAMBDA KL_INCR=$KL_INCR BASE_LR=$BASE_LR CLASSIFIER_LR=$CLASSIFIER_LR LR2_MULT=$LR2_MULT"
which python

echo "profile,seed,attention_epoch,kl_lambda,kl_incr,base_lr,classifier_lr,lr2_mult,ablation_mode,ablation_frac,ablation_seed,blur_ksize,blur_sigma,best_balanced_val_acc,test_acc,per_group,worst_group,checkpoint,gt_path,log_path" > "$SUMMARY_CSV"

for seed in $SEEDS; do
  run_log="$LOG_DIR/guided_wb${WB_PROFILE}_${ABLATION_MODE}_seed${seed}_${SLURM_JOB_ID}.log"
  echo "=== profile=$WB_PROFILE seed=$seed ==="

  python - "$DATA_ROOT" "$GT_PATH" "$seed" "$SUMMARY_CSV" "$run_log" \
    "$ATTENTION_EPOCH" "$KL_LAMBDA" "$KL_INCR" "$BASE_LR" "$CLASSIFIER_LR" "$LR2_MULT" \
    "$WB_PROFILE" "$ABLATION_MODE" "$ABLATION_FRAC" "$ABLATION_SEED" "$BLUR_KSIZE" "$BLUR_SIGMA" <<'PY' 2>&1 | tee "$run_log"
import csv
import math
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

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
profile = sys.argv[12]
ablation_mode = str(sys.argv[13]).strip().lower()
ablation_frac = float(sys.argv[14])
ablation_seed = int(float(sys.argv[15]))
blur_ksize = int(float(sys.argv[16]))
blur_sigma = float(sys.argv[17])

if ablation_mode not in {"none", "invert", "blur"}:
    raise ValueError(f"ablation_mode must be one of none/invert/blur, got: {ablation_mode}")
ablation_frac = max(0.0, min(1.0, ablation_frac))
if blur_ksize % 2 == 0:
    blur_ksize += 1
if blur_ksize < 3:
    blur_ksize = 3


def _fixed_subset(n: int, frac: float, seed_value: int):
    k = int(round(frac * n))
    k = max(0, min(n, k))
    if k == 0:
        return set()
    rng = np.random.default_rng(seed_value)
    return set(rng.choice(n, size=k, replace=False).tolist())


def _renorm_distribution(mask_t: torch.Tensor) -> torch.Tensor:
    m = mask_t.float().clamp(min=0.0)
    s = float(m.sum().item())
    if s <= 1e-12:
        m.fill_(1.0 / float(m.numel()))
    else:
        m = m / s
    return m


def _ablate_mask(mask_t: torch.Tensor) -> torch.Tensor:
    m = mask_t.float()
    orig_shape = tuple(m.shape)

    if ablation_mode == "invert":
        m = 1.0 - m
        return _renorm_distribution(m)

    if ablation_mode == "blur":
        if m.dim() == 3 and m.shape[0] == 1:
            arr = m[0].cpu().numpy().astype(np.float32)
            arr = cv2.GaussianBlur(arr, (blur_ksize, blur_ksize), blur_sigma)
            out = torch.from_numpy(arr).unsqueeze(0).to(m.dtype)
        elif m.dim() == 2:
            arr = m.cpu().numpy().astype(np.float32)
            arr = cv2.GaussianBlur(arr, (blur_ksize, blur_ksize), blur_sigma)
            out = torch.from_numpy(arr).to(m.dtype)
        else:
            # Fallback for unexpected mask shapes.
            arr = m.squeeze().cpu().numpy().astype(np.float32)
            arr = cv2.GaussianBlur(arr, (blur_ksize, blur_ksize), blur_sigma)
            out = torch.from_numpy(arr).reshape(orig_shape).to(m.dtype)
        return _renorm_distribution(out)

    return m


def _patch_metadata_dataset():
    original_getitem = rgw.WaterbirdsMetadataDataset.__getitem__

    def wrapped_getitem(self, idx):
        out = original_getitem(self, idx)
        if not getattr(self, "return_mask", False):
            return out
        if ablation_mode == "none" or ablation_frac <= 0.0:
            return out
        if not hasattr(self, "_ablation_indices"):
            self._ablation_indices = _fixed_subset(len(self), ablation_frac, ablation_seed)
            print(
                f"[ABLATION] WaterbirdsMetadataDataset train subset: "
                f"{len(self._ablation_indices)}/{len(self)} "
                f"(mode={ablation_mode}, frac={ablation_frac}, seed={ablation_seed})",
                flush=True,
            )
        if idx not in self._ablation_indices:
            return out
        data = list(out)
        data[2] = _ablate_mask(data[2])
        return tuple(data)

    rgw.WaterbirdsMetadataDataset.__getitem__ = wrapped_getitem


def _patch_guided_imagefolder():
    original_getitem = rgw.GuidedImageFolder.__getitem__

    def wrapped_getitem(self, idx):
        out = original_getitem(self, idx)
        if ablation_mode == "none" or ablation_frac <= 0.0:
            return out
        if not hasattr(self, "_ablation_indices"):
            self._ablation_indices = _fixed_subset(len(self), ablation_frac, ablation_seed)
            print(
                f"[ABLATION] GuidedImageFolder train subset: "
                f"{len(self._ablation_indices)}/{len(self)} "
                f"(mode={ablation_mode}, frac={ablation_frac}, seed={ablation_seed})",
                flush=True,
            )
        if idx not in self._ablation_indices:
            return out
        img, label, mask, path = out
        return img, label, _ablate_mask(mask), path

    rgw.GuidedImageFolder.__getitem__ = wrapped_getitem


_patch_metadata_dataset()
_patch_guided_imagefolder()

rgw.SEED = seed
rgw.base_lr = base_lr
rgw.classifier_lr = classifier_lr
rgw.lr2_mult = lr2_mult

args = SimpleNamespace(data_path=data_root, gt_path=gt_path)
best_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(args, attn_epoch, kl_lambda, kl_incr)

row = {
    "profile": profile,
    "seed": seed,
    "attention_epoch": attn_epoch,
    "kl_lambda": kl_lambda,
    "kl_incr": kl_incr,
    "base_lr": base_lr,
    "classifier_lr": classifier_lr,
    "lr2_mult": lr2_mult,
    "ablation_mode": ablation_mode,
    "ablation_frac": ablation_frac,
    "ablation_seed": ablation_seed,
    "blur_ksize": blur_ksize,
    "blur_sigma": blur_sigma,
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
    f"[DONE] profile={profile} seed={seed} mode={ablation_mode} frac={ablation_frac:.3f} "
    f"best_val={best_val:.4f} test_acc={test_acc:.2f}% per_group={per_group:.2f}% "
    f"worst_group={worst_group:.2f}% checkpoint={ckpt}"
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

echo "[DONE] Guided Waterbirds mask-ablation run complete."
echo "[DONE] Summary CSV: $SUMMARY_CSV"
