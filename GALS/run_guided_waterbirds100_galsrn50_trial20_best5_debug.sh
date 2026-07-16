#!/bin/bash -l
# WB100 guided (RN50 attention maps) fixed Trial-20 hyperparams across 5 seeds.
# Reports per-seed metrics + aggregate mean/std for per_group and each test group.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/guided100_galsrn50_trial20_best5_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/guided100_galsrn50_trial20_best5_debug_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=0
export GUIDED_NUM_WORKERS="${GUIDED_NUM_WORKERS:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}
DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}
ATT_ROOT=${ATT_ROOT:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/clip_rn50_attention_gradcam}

# Fixed hyperparameters from user-provided best trial:
ATTENTION_EPOCH=${ATTENTION_EPOCH:-109}
KL_LAMBDA=${KL_LAMBDA:-288.89703257693924}
KL_INCR=${KL_INCR:-0.0}
BASE_LR=${BASE_LR:-3.982887304406873e-05}
CLASSIFIER_LR=${CLASSIFIER_LR:-0.0017759895768464104}
LR2_MULT=${LR2_MULT:-1.5564874514582572}

SEEDS=${SEEDS:-"0 1 2 3 4"}

# Attention-map loading knobs (match the sweep defaults).
ATT_KEY=${ATT_KEY:-unnormalized_attentions}
ATT_COMBINE=${ATT_COMBINE:-mean}
ATT_NORM01=${ATT_NORM01:-1}
ATT_BRIGHTEN=${ATT_BRIGHTEN:-1.0}

SUMMARY_CSV=${SUMMARY_CSV:-$LOG_DIR/guided100_galsrn50_trial20_best5_${SLURM_JOB_ID}.csv}
AGG_CSV=${AGG_CSV:-$LOG_DIR/guided100_galsrn50_trial20_best5_summary_${SLURM_JOB_ID}.csv}
SUMMARY_TXT=${SUMMARY_TXT:-$LOG_DIR/guided100_galsrn50_trial20_best5_summary_${SLURM_JOB_ID}.txt}

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[ERROR] Missing REPO_ROOT: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] Missing DATA_ROOT: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ATT_ROOT" ]]; then
  echo "[ERROR] Missing ATT_ROOT: $ATT_ROOT" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/run_guided_waterbird_gals_vitatt.py" ]]; then
  echo "[ERROR] Missing runner module: $REPO_ROOT/run_guided_waterbird_gals_vitatt.py" >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT"
echo "RN50 attention root: $ATT_ROOT"
echo "Seeds: $SEEDS"
echo "Summary CSV: $SUMMARY_CSV"
echo "Aggregate CSV: $AGG_CSV"
echo "Summary TXT: $SUMMARY_TXT"
echo "ATTENTION_EPOCH=$ATTENTION_EPOCH KL_LAMBDA=$KL_LAMBDA KL_INCR=$KL_INCR BASE_LR=$BASE_LR CLASSIFIER_LR=$CLASSIFIER_LR LR2_MULT=$LR2_MULT"
echo "ATT_KEY=$ATT_KEY ATT_COMBINE=$ATT_COMBINE ATT_NORM01=$ATT_NORM01 ATT_BRIGHTEN=$ATT_BRIGHTEN"
echo "GUIDED_NUM_WORKERS=$GUIDED_NUM_WORKERS"
which python

echo "seed,attention_epoch,kl_lambda,kl_incr,base_lr,classifier_lr,lr2_mult,test_acc,per_group,worst_group,Land_on_Land,Land_on_Water,Water_on_Land,Water_on_Water,checkpoint,run_log" > "$SUMMARY_CSV"

for seed in $SEEDS; do
  run_log="$LOG_DIR/guided100_galsrn50_trial20_seed${seed}_${SLURM_JOB_ID}.log"
  echo "=== seed=$seed ==="

  python - "$DATA_ROOT" "$ATT_ROOT" "$seed" "$SUMMARY_CSV" "$run_log" \
    "$ATTENTION_EPOCH" "$KL_LAMBDA" "$KL_INCR" "$BASE_LR" "$CLASSIFIER_LR" "$LR2_MULT" \
    "$ATT_KEY" "$ATT_COMBINE" "$ATT_NORM01" "$ATT_BRIGHTEN" <<'PY' 2>&1 | tee "$run_log"
import csv
import sys
from types import SimpleNamespace

import numpy as np

import run_guided_waterbird_gals_vitatt as rg

data_root = sys.argv[1]
att_root = sys.argv[2]
seed = int(sys.argv[3])
summary_csv = sys.argv[4]
run_log = sys.argv[5]
attn_epoch = int(float(sys.argv[6]))
kl_lambda = float(sys.argv[7])
kl_incr = float(sys.argv[8])
base_lr = float(sys.argv[9])
classifier_lr = float(sys.argv[10])
lr2_mult = float(sys.argv[11])
att_key = str(sys.argv[12])
att_combine = str(sys.argv[13])
att_norm01 = str(sys.argv[14]).strip() not in ("0", "false", "False", "no", "No", "")
att_brighten = float(sys.argv[15])

rg.SEED = seed
rg.base_lr = base_lr
rg.classifier_lr = classifier_lr
rg.lr2_mult = lr2_mult

captured = {}
orig_evaluate_test = rg.base.evaluate_test

def _wrapped_evaluate_test(model, loader):
    out = orig_evaluate_test(model, loader)
    captured["group_acc"] = out[2]
    return out

rg.base.evaluate_test = _wrapped_evaluate_test
try:
    run_args = SimpleNamespace(
        data_path=data_root,
        att_path=att_root,
        att_key=att_key,
        att_combine=att_combine,
        att_norm01=att_norm01,
        att_brighten=att_brighten,
    )
    best_balanced_val, test_acc, per_group, worst_group, ckpt = rg.run_single(
        run_args, attn_epoch, kl_lambda, kl_incr
    )
finally:
    rg.base.evaluate_test = orig_evaluate_test

group_acc = captured.get("group_acc")
if group_acc is None:
    raise RuntimeError("Failed to capture group_acc from evaluation.")
group_acc = np.asarray(group_acc, dtype=float).reshape(-1)
if group_acc.shape[0] != 4:
    raise RuntimeError(f"Expected 4 groups, got shape={group_acc.shape}.")

row = {
    "seed": seed,
    "attention_epoch": attn_epoch,
    "kl_lambda": kl_lambda,
    "kl_incr": kl_incr,
    "base_lr": base_lr,
    "classifier_lr": classifier_lr,
    "lr2_mult": lr2_mult,
    "test_acc": float(test_acc),
    "per_group": float(per_group),
    "worst_group": float(worst_group),
    "Land_on_Land": float(group_acc[0]),
    "Land_on_Water": float(group_acc[1]),
    "Water_on_Land": float(group_acc[2]),
    "Water_on_Water": float(group_acc[3]),
    "checkpoint": ckpt,
    "run_log": run_log,
}

with open(summary_csv, "a", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "seed",
            "attention_epoch",
            "kl_lambda",
            "kl_incr",
            "base_lr",
            "classifier_lr",
            "lr2_mult",
            "test_acc",
            "per_group",
            "worst_group",
            "Land_on_Land",
            "Land_on_Water",
            "Water_on_Land",
            "Water_on_Water",
            "checkpoint",
            "run_log",
        ],
    )
    writer.writerow(row)

print(
    "[DONE] seed={} test_acc={:.2f}% per_group={:.2f}% worst_group={:.2f}% groups=({:.2f}, {:.2f}, {:.2f}, {:.2f})".format(
        seed,
        row["test_acc"],
        row["per_group"],
        row["worst_group"],
        row["Land_on_Land"],
        row["Land_on_Water"],
        row["Water_on_Land"],
        row["Water_on_Water"],
    )
)
PY
done

python - "$SUMMARY_CSV" "$AGG_CSV" "$SUMMARY_TXT" <<'PY'
import csv
import sys

import numpy as np

summary_csv = sys.argv[1]
agg_csv = sys.argv[2]
summary_txt = sys.argv[3]

rows = []
with open(summary_csv, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if not rows:
    raise RuntimeError("No successful rows found in summary CSV; cannot aggregate.")

metrics = [
    "test_acc",
    "per_group",
    "worst_group",
    "Land_on_Land",
    "Land_on_Water",
    "Water_on_Land",
    "Water_on_Water",
]

def _mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std

agg_rows = []
for m in metrics:
    vals = [float(r[m]) for r in rows]
    mean, std = _mean_std(vals)
    agg_rows.append({"metric": m, "mean": mean, "std": std, "n": len(vals)})

with open(agg_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n"])
    writer.writeheader()
    writer.writerows(agg_rows)

with open(summary_txt, "w") as f:
    f.write("WB100 Guided + GALS-RN50 attention Trial-20 fixed hyperparams (5 seeds)\n")
    f.write(f"Per-seed CSV: {summary_csv}\n")
    f.write(f"Aggregate CSV: {agg_csv}\n")
    f.write("\n")
    for row in agg_rows:
        f.write(f"{row['metric']}: {row['mean']:.4f} +/- {row['std']:.4f} (n={row['n']})\n")

print("[SUMMARY] Aggregate mean/std across seeds:")
for row in agg_rows:
    print(f"  {row['metric']}: {row['mean']:.4f} +/- {row['std']:.4f} (n={row['n']})")
print(f"[DONE] Per-seed CSV: {summary_csv}")
print(f"[DONE] Aggregate CSV: {agg_csv}")
print(f"[DONE] Summary TXT: {summary_txt}")
PY

