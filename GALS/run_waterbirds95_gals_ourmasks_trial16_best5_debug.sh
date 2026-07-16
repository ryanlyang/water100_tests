#!/bin/bash -l
# Re-run Waterbirds95 GALS+ourmasks with fixed Trial-16 hyperparameters for 5 seeds.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_gals_ourmasks_trial16_best5_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_gals_ourmasks_trial16_best5_%j.err
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
    pip install opencv-python==4.6.0.66
    conda deactivate
  fi
fi

conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds
DATA_DIR=waterbird_complete95_forest2water2
MASK_DIR=${MASK_DIR:-/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
MASK_DIR="${MASK_DIR%\"}"
MASK_DIR="${MASK_DIR#\"}"
MASK_DIR="${MASK_DIR%\'}"
MASK_DIR="${MASK_DIR#\'}"

# Fixed Trial-16 hyperparameters (Waterbirds95 GALS+ourmasks sweep)
BASE_LR=${BASE_LR:-0.015122123432341057}
CLS_LR=${CLS_LR:-0.0005284097040082973}
GRAD_WEIGHT=${GRAD_WEIGHT:-1438.787318824726}
GRAD_CRITERION=${GRAD_CRITERION:-L2}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}

# Seeds to rerun (comma-separated list)
SEEDS_CSV=${SEEDS_CSV:-0,1,2,3,4}
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

if [[ ! -d "$MASK_DIR" ]]; then
  echo "[ERROR] MASK_DIR does not exist: $MASK_DIR" >&2
  exit 2
fi

python -c "import optuna" 2>/dev/null || {
  echo "[INFO] Installing optuna..."
  pip install -q optuna
}

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT/$DATA_DIR"
echo "MASK_DIR: $MASK_DIR"
echo "Seeds: $SEEDS_CSV"
echo "Fixed hparams: base_lr=$BASE_LR classifier_lr=$CLS_LR grad_weight=$GRAD_WEIGHT grad_criterion=$GRAD_CRITERION weight_decay=$WEIGHT_DECAY"
which python

OUT_CSV="$LOG_DIR/gals95_ourmasks_trial16_best5_${SLURM_JOB_ID}.csv"
TRIAL_LOGS="$LOG_DIR/gals95_ourmasks_trial16_best5_logs_${SLURM_JOB_ID}"
AGG_CSV="$LOG_DIR/gals95_ourmasks_trial16_best5_summary_${SLURM_JOB_ID}.csv"
SUMMARY_TXT="$LOG_DIR/gals95_ourmasks_trial16_best5_summary_${SLURM_JOB_ID}.txt"
mkdir -p "$TRIAL_LOGS"
rm -f "$OUT_CSV" "$AGG_CSV" "$SUMMARY_TXT"

for seed in "${SEEDS[@]}"; do
  seed="$(echo "$seed" | xargs)"
  if [[ -z "$seed" ]]; then
    continue
  fi
  RUN_PREFIX="gals95_ourmasks_trial16_seed${seed}_${SLURM_JOB_ID}"
  echo "[RUN] seed=$seed run_prefix=$RUN_PREFIX"
  srun --unbuffered python -u run_gals_sweep.py \
    --method gals \
    --config configs/waterbirds_95_gals_ourmasks.yaml \
    --data-root "$DATA_ROOT" \
    --waterbirds-dir "$DATA_DIR" \
    --n-trials 1 \
    --seed 0 \
    --train-seed "$seed" \
    --sampler random \
    --keep all \
    --output-csv "$OUT_CSV" \
    --logs-dir "$TRIAL_LOGS" \
    --run-name-prefix "$RUN_PREFIX" \
    --post-seeds 0 \
    --base-lr-min "$BASE_LR" \
    --base-lr-max "$BASE_LR" \
    --cls-lr-min "$CLS_LR" \
    --cls-lr-max "$CLS_LR" \
    --weight-min "$GRAD_WEIGHT" \
    --weight-max "$GRAD_WEIGHT" \
    --grad-criteria "$GRAD_CRITERION" \
    --weight-decay-min "$WEIGHT_DECAY" \
    --weight-decay-max "$WEIGHT_DECAY" \
    --tune-weight-decay \
    DATA.SEGMENTATION_DIR="$MASK_DIR"
done

python - "$OUT_CSV" "$AGG_CSV" "$SUMMARY_TXT" <<'PY'
import csv
import sys
import numpy as np

out_csv, agg_csv, summary_txt = sys.argv[1], sys.argv[2], sys.argv[3]

rows = []
with open(out_csv, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if not rows:
    raise RuntimeError("No successful rows found in output CSV; cannot aggregate.")

metrics = [
    "best_balanced_val_acc",
    "test_acc",
    "balanced_test_acc",
    "per_group",
    "worst_group",
]

agg_rows = []
for m in metrics:
    vals = []
    for r in rows:
        v = r.get(m)
        if v in (None, ""):
            continue
        vals.append(float(v))
    if not vals:
        continue
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    agg_rows.append({"metric": m, "mean": mean, "std": std, "n": int(arr.size)})

with open(agg_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n"])
    writer.writeheader()
    writer.writerows(agg_rows)

with open(summary_txt, "w") as f:
    f.write("WB95 GALS+ourmasks Trial-16 fixed hyperparams (5 seeds)\n")
    f.write(f"Per-seed CSV: {out_csv}\n")
    f.write(f"Aggregate CSV: {agg_csv}\n\n")
    for row in agg_rows:
        f.write(f"{row['metric']}: {row['mean']:.4f} +/- {row['std']:.4f} (n={row['n']})\n")

print("[SUMMARY] Aggregate mean/std across seeds:")
for row in agg_rows:
    print(f"  {row['metric']}: {row['mean']:.4f} +/- {row['std']:.4f} (n={row['n']})")
print(f"[DONE] Per-seed CSV: {out_csv}")
print(f"[DONE] Aggregate CSV: {agg_csv}")
print(f"[DONE] Summary TXT: {summary_txt}")
PY

echo "[DONE] Waterbirds95 GALS+ourmasks Trial-16 fixed reruns complete."
