#!/bin/bash -l
# One 50-trial Waterbirds-100 R4RR alignment-loss ablation.
# Submit through submit_waterbirds100_r4rr_alignment_sweeps.sh or set
# ALIGNMENT_LOSS to one of: reverse_kl, jensen_shannon, squared_l2, cosine.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=r4rr_align
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/%x_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

ALIGNMENT_LOSS=${ALIGNMENT_LOSS:?Set ALIGNMENT_LOSS before submission}
case "$ALIGNMENT_LOSS" in
  reverse_kl|jensen_shannon|squared_l2|cosine) ;;
  *) echo "[ERROR] Unsupported ALIGNMENT_LOSS=$ALIGNMENT_LOSS" >&2; exit 2 ;;
esac

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=0
export GUIDED_NUM_WORKERS=${GUIDED_NUM_WORKERS:-4}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs}
GALS_ROOT=${GALS_ROOT:-$REPO_ROOT/GALS}
DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}
TEACHER_MAP_ROOT=${TEACHER_MAP_ROOT:-/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
LOG_DIR=${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}
N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
RUN_ID=${RUN_ID:-${SLURM_JOB_ID:-manual}}
RESUME_CSV=${RESUME_CSV:-}
if [[ -n "$RESUME_CSV" && -z "${OUTPUT_CSV+x}" ]]; then
  OUTPUT_CSV="$RESUME_CSV"
else
  OUTPUT_CSV=${OUTPUT_CSV:-$LOG_DIR/wb100_r4rr_${ALIGNMENT_LOSS}_sweep_${RUN_ID}.csv}
fi

SWEEP_PY="$GALS_ROOT/RightForTheRightRegions/repro_runs/r4rr/sweeps/r4rr_waterbirds_sweep.py"
mkdir -p "$LOG_DIR"

for path in "$GALS_ROOT" "$DATA_ROOT" "$TEACHER_MAP_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$SWEEP_PY" ]]; then
  echo "[ERROR] Missing sweep runner: $SWEEP_PY" >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$GALS_ROOT:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
echo "Alignment loss: $ALIGNMENT_LOSS"
echo "Objective: best_balanced_val_acc"
echo "Trials: $N_TRIALS (TPE seed=$SWEEP_SEED, train seed=$SWEEP_SEED)"
echo "Ranges: attention_epoch=[0,199] alignment_weight=[1,500] base_lr=[1e-5,5e-2] classifier_lr=[1e-5,5e-2] lr2_mult=[0.1,3]"
echo "Data: $DATA_ROOT"
echo "Teacher maps: $TEACHER_MAP_ROOT"
echo "Output CSV: $OUTPUT_CSV"
[[ -n "$RESUME_CSV" ]] && echo "Resume CSV: $RESUME_CSV"
which python

cmd=(
  python -u "$SWEEP_PY"
  "$DATA_ROOT"
  "$TEACHER_MAP_ROOT"
  --alignment-loss "$ALIGNMENT_LOSS"
  --n-trials "$N_TRIALS"
  --seed "$SWEEP_SEED"
  --sampler tpe
  --attn-min 0 --attn-max 199
  --kl-min 1 --kl-max 500
  --base-lr-min 1e-5 --base-lr-max 5e-2
  --cls-lr-min 1e-5 --cls-lr-max 5e-2
  --lr2-mult-min 1e-1 --lr2-mult-max 3
  --output-csv "$OUTPUT_CSV"
)
if [[ -n "$RESUME_CSV" ]]; then
  cmd+=(--resume-csv "$RESUME_CSV")
fi

srun --unbuffered "${cmd[@]}"

echo "[DONE] alignment_loss=$ALIGNMENT_LOSS"
echo "[DONE] CSV: $OUTPUT_CSV"
