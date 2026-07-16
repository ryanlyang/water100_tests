#!/bin/bash -l
# Smoke test for all strategies submitted by submit_all_waterbirds_experiments.sh.
# Runs tiny, non-Optuna-heavy checks to validate training paths and dependencies.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_smoke_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_smoke_debug_%j.err

set -u -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

RUNS_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
GALS_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds

WB95_DIR=waterbird_complete95_forest2water2
WB100_DIR=waterbird_1.0_forest2water2

WB95_PATH="$DATA_ROOT/$WB95_DIR"
WB100_PATH="$DATA_ROOT/$WB100_DIR"

WB95_VIT_ATT="$WB95_PATH/clip_vit_attention"
WB100_VIT_ATT="$WB100_PATH/clip_vit_attention"
WB95_RN50_ATT="$WB95_PATH/clip_rn50_attention_gradcam"
WB100_RN50_ATT="$WB100_PATH/clip_rn50_attention_gradcam"

MASK95_DIR=/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
MASK100_DIR=/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
ABN_WEIGHT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/weights/resnet50_abn_imagenet.pth.tar

GUIDED_GALSVIT_SWEEP_PY="$RUNS_ROOT/run_guided_waterbird_gals_vitatt_sweep.py"
SKIP_RN50_PATH_SMOKE=${SKIP_RN50_PATH_SMOKE:-1}
SKIP_GUIDED_RN50_SMOKE=${SKIP_GUIDED_RN50_SMOKE:-0}

LOG_ROOT=/home/ryreu/guided_cnn/logsWaterbird
SMOKE_DIR="$LOG_ROOT/smoke_debug_${SLURM_JOB_ID}"
mkdir -p "$SMOKE_DIR"
SUMMARY_CSV="$SMOKE_DIR/summary.csv"
echo "name,status,seconds,log" > "$SUMMARY_CSV"

cd "$GALS_ROOT"
export PYTHONPATH="$RUNS_ROOT:$GALS_ROOT:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

for need_dir in \
  "$WB95_PATH" "$WB100_PATH" \
  "$WB95_VIT_ATT" "$WB100_VIT_ATT" \
  "$MASK95_DIR" "$MASK100_DIR"
do
  if [[ ! -d "$need_dir" ]]; then
    echo "[ERROR] Missing required directory: $need_dir" >&2
    exit 2
  fi
done

if [[ ! -f "$ABN_WEIGHT" ]]; then
  echo "[ERROR] Missing ABN pretrained weight: $ABN_WEIGHT" >&2
  exit 2
fi

PASS=0
FAIL=0

run_check() {
  local name="$1"
  shift
  local log="$SMOKE_DIR/${name}.log"
  local t0 t1 dt status
  t0="$(date +%s)"
  echo "===== [SMOKE] $name ====="
  if "$@" >"$log" 2>&1; then
    status="PASS"
    PASS=$((PASS + 1))
  else
    status="FAIL"
    FAIL=$((FAIL + 1))
  fi
  t1="$(date +%s)"
  dt=$((t1 - t0))
  echo "$name,$status,$dt,$log" >> "$SUMMARY_CSV"
  echo "[SMOKE] $name => $status (${dt}s)"
}

skip_check() {
  local name="$1"
  local reason="$2"
  local log="$SMOKE_DIR/${name}.log"
  printf "SKIPPED: %s\n" "$reason" > "$log"
  echo "$name,SKIP,0,$log" >> "$SUMMARY_CSV"
  echo "===== [SMOKE] $name ====="
  echo "[SMOKE] $name => SKIP ($reason)"
}

# Shared tiny sweep settings.
BASE_LR=1e-3
CLS_LR=1e-4
WD_MIN=1e-5
WD_MAX=1e-5
GRAD_W=10
CAM_W=1
ABN_W=1

# 1) GALS/ViT/RRR/GradCAM/ABN/UpWeight/OurMasks methods (both datasets) via run_gals_sweep.py.
if [[ "$SKIP_RN50_PATH_SMOKE" -eq 1 ]]; then
  skip_check gals95_rn50 "SKIP_RN50_PATH_SMOKE=1"
  skip_check gals100_rn50 "SKIP_RN50_PATH_SMOKE=1"
elif [[ ! -d "$WB95_RN50_ATT" || ! -d "$WB100_RN50_ATT" ]]; then
  skip_check gals95_rn50 "missing RN50 attention path(s)"
  skip_check gals100_rn50 "missing RN50 attention path(s)"
else
  run_check gals95_rn50 \
    python -u run_gals_sweep.py \
    --method gals --config configs/waterbirds_95_gals.yaml \
    --data-root "$DATA_ROOT" --waterbirds-dir "$WB95_DIR" \
    --sampler random --n-trials 1 --seed 0 --train-seed 0 \
    --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
    --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
    --weight-min "$GRAD_W" --weight-max "$GRAD_W" \
    --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
    --keep none --post-seeds 0 \
    --logs-dir "$SMOKE_DIR/gals95_rn50_logs" --output-csv "$SMOKE_DIR/gals95_rn50.csv" \
    EXP.NUM_EPOCHS=1

  run_check gals100_rn50 \
    python -u run_gals_sweep.py \
    --method gals --config configs/waterbirds_100_gals.yaml \
    --data-root "$DATA_ROOT" --waterbirds-dir "$WB100_DIR" \
    --sampler random --n-trials 1 --seed 0 --train-seed 0 \
    --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
    --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
    --weight-min "$GRAD_W" --weight-max "$GRAD_W" \
    --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
    --keep none --post-seeds 0 \
    --logs-dir "$SMOKE_DIR/gals100_rn50_logs" --output-csv "$SMOKE_DIR/gals100_rn50.csv" \
    EXP.NUM_EPOCHS=1
fi

run_check rrr95_vit \
  python -u run_gals_sweep.py \
  --method rrr --config configs/waterbirds_95_gals_vit.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB95_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --weight-min "$GRAD_W" --weight-max "$GRAD_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/rrr95_vit_logs" --output-csv "$SMOKE_DIR/rrr95_vit.csv" \
  EXP.NUM_EPOCHS=1

run_check rrr100_vit \
  python -u run_gals_sweep.py \
  --method rrr --config configs/waterbirds_100_gals_vit.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB100_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --weight-min "$GRAD_W" --weight-max "$GRAD_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/rrr100_vit_logs" --output-csv "$SMOKE_DIR/rrr100_vit.csv" \
  EXP.NUM_EPOCHS=1

run_check gradcam95_vit \
  python -u run_gals_sweep.py \
  --method gradcam --config configs/waterbirds_95_gradcam_vit.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB95_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --cam-weight-min "$CAM_W" --cam-weight-max "$CAM_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/gradcam95_vit_logs" --output-csv "$SMOKE_DIR/gradcam95_vit.csv" \
  EXP.NUM_EPOCHS=1

run_check gradcam100_vit \
  python -u run_gals_sweep.py \
  --method gradcam --config configs/waterbirds_100_gradcam_vit.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB100_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --cam-weight-min "$CAM_W" --cam-weight-max "$CAM_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/gradcam100_vit_logs" --output-csv "$SMOKE_DIR/gradcam100_vit.csv" \
  EXP.NUM_EPOCHS=1

run_check abn95_vit \
  python -u run_gals_sweep.py \
  --method abn_att --config configs/waterbirds_95_abn_vit.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB95_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --abn-att-weight-min "$ABN_W" --abn-att-weight-max "$ABN_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/abn95_vit_logs" --output-csv "$SMOKE_DIR/abn95_vit.csv" \
  EXP.NUM_EPOCHS=1

run_check abn100_vit \
  python -u run_gals_sweep.py \
  --method abn_att --config configs/waterbirds_100_abn_vit.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB100_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --abn-att-weight-min "$ABN_W" --abn-att-weight-max "$ABN_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/abn100_vit_logs" --output-csv "$SMOKE_DIR/abn100_vit.csv" \
  EXP.NUM_EPOCHS=1

run_check gals95_vit \
  python -u run_gals_sweep.py \
  --method gals --config configs/waterbirds_95_gals_vit.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB95_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --weight-min "$GRAD_W" --weight-max "$GRAD_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/gals95_vit_logs" --output-csv "$SMOKE_DIR/gals95_vit.csv" \
  EXP.NUM_EPOCHS=1

run_check gals100_vit \
  python -u run_gals_sweep.py \
  --method gals --config configs/waterbirds_100_gals_vit.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB100_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --weight-min "$GRAD_W" --weight-max "$GRAD_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/gals100_vit_logs" --output-csv "$SMOKE_DIR/gals100_vit.csv" \
  EXP.NUM_EPOCHS=1

run_check gals95_ourmasks \
  python -u run_gals_sweep.py \
  --method gals --config configs/waterbirds_95_gals_ourmasks.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB95_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --weight-min "$GRAD_W" --weight-max "$GRAD_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/gals95_ourmasks_logs" --output-csv "$SMOKE_DIR/gals95_ourmasks.csv" \
  DATA.SEGMENTATION_DIR="$MASK95_DIR" DATA.SEG_TRAIN_ONLY=True EXP.NUM_EPOCHS=1

run_check gals100_ourmasks \
  python -u run_gals_sweep.py \
  --method gals --config configs/waterbirds_100_gals_ourmasks.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB100_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --weight-min "$GRAD_W" --weight-max "$GRAD_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/gals100_ourmasks_logs" --output-csv "$SMOKE_DIR/gals100_ourmasks.csv" \
  DATA.SEGMENTATION_DIR="$MASK100_DIR" DATA.SEG_TRAIN_ONLY=True EXP.NUM_EPOCHS=1

run_check upweight95 \
  python -u run_gals_sweep.py \
  --method upweight --config configs/waterbirds_95_upweight.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB95_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/upweight95_logs" --output-csv "$SMOKE_DIR/upweight95.csv" \
  EXP.NUM_EPOCHS=1

run_check upweight100 \
  python -u run_gals_sweep.py \
  --method upweight --config configs/waterbirds_100_upweight.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB100_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/upweight100_logs" --output-csv "$SMOKE_DIR/upweight100.csv" \
  EXP.NUM_EPOCHS=1

run_check abn95_baseline \
  python -u run_gals_sweep.py \
  --method abn_cls --config configs/waterbirds_95_abn.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB95_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --abn-cls-weight-min "$ABN_W" --abn-cls-weight-max "$ABN_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/abn95_baseline_logs" --output-csv "$SMOKE_DIR/abn95_baseline.csv" \
  EXP.NUM_EPOCHS=1

run_check abn100_baseline \
  python -u run_gals_sweep.py \
  --method abn_cls --config configs/waterbirds_100_abn.yaml \
  --data-root "$DATA_ROOT" --waterbirds-dir "$WB100_DIR" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --base-lr-min "$BASE_LR" --base-lr-max "$BASE_LR" \
  --cls-lr-min "$CLS_LR" --cls-lr-max "$CLS_LR" \
  --abn-cls-weight-min "$ABN_W" --abn-cls-weight-max "$ABN_W" \
  --tune-weight-decay --weight-decay-min "$WD_MIN" --weight-decay-max "$WD_MAX" \
  --keep none --post-seeds 0 \
  --logs-dir "$SMOKE_DIR/abn100_baseline_logs" --output-csv "$SMOKE_DIR/abn100_baseline.csv" \
  EXP.NUM_EPOCHS=1

# 2) Guided strategy with GALS ViT attentions (both datasets), tiny non-Optuna mode.
run_check guided95_galsvit \
  python -u "$GUIDED_GALSVIT_SWEEP_PY" \
  "$WB95_PATH" "$WB95_VIT_ATT" \
  --sampler random --n-trials 1 --seed 0 \
  --num-epochs 1 \
  --attn-min 0 --attn-max 0 \
  --kl-min 1 --kl-max 1 \
  --base-lr-min 1e-4 --base-lr-max 1e-4 \
  --cls-lr-min 1e-3 --cls-lr-max 1e-3 \
  --lr2-mult-min 1.0 --lr2-mult-max 1.0 \
  --post-seeds 0 \
  --output-csv "$SMOKE_DIR/guided95_galsvit.csv"

run_check guided100_galsvit \
  python -u "$GUIDED_GALSVIT_SWEEP_PY" \
  "$WB100_PATH" "$WB100_VIT_ATT" \
  --sampler random --n-trials 1 --seed 0 \
  --num-epochs 1 \
  --attn-min 0 --attn-max 0 \
  --kl-min 1 --kl-max 1 \
  --base-lr-min 1e-4 --base-lr-max 1e-4 \
  --cls-lr-min 1e-3 --cls-lr-max 1e-3 \
  --lr2-mult-min 1.0 --lr2-mult-max 1.0 \
  --post-seeds 0 \
  --output-csv "$SMOKE_DIR/guided100_galsvit.csv"

if [[ "$SKIP_GUIDED_RN50_SMOKE" -eq 1 ]]; then
  skip_check guided95_galsrn50 "SKIP_GUIDED_RN50_SMOKE=1"
  skip_check guided100_galsrn50 "SKIP_GUIDED_RN50_SMOKE=1"
elif [[ ! -d "$WB95_RN50_ATT" || ! -d "$WB100_RN50_ATT" ]]; then
  skip_check guided95_galsrn50 "missing RN50 attention path(s)"
  skip_check guided100_galsrn50 "missing RN50 attention path(s)"
else
  run_check guided95_galsrn50 \
    python -u "$GUIDED_GALSVIT_SWEEP_PY" \
    "$WB95_PATH" "$WB95_RN50_ATT" \
    --sampler random --n-trials 1 --seed 0 \
    --num-epochs 1 \
    --attn-min 0 --attn-max 0 \
    --kl-min 1 --kl-max 1 \
    --base-lr-min 1e-4 --base-lr-max 1e-4 \
    --cls-lr-min 1e-3 --cls-lr-max 1e-3 \
    --lr2-mult-min 1.0 --lr2-mult-max 1.0 \
    --post-seeds 0 \
    --output-csv "$SMOKE_DIR/guided95_galsrn50.csv"

  run_check guided100_galsrn50 \
    python -u "$GUIDED_GALSVIT_SWEEP_PY" \
    "$WB100_PATH" "$WB100_RN50_ATT" \
    --sampler random --n-trials 1 --seed 0 \
    --num-epochs 1 \
    --attn-min 0 --attn-max 0 \
    --kl-min 1 --kl-max 1 \
    --base-lr-min 1e-4 --base-lr-max 1e-4 \
    --cls-lr-min 1e-3 --cls-lr-max 1e-3 \
    --lr2-mult-min 1.0 --lr2-mult-max 1.0 \
    --post-seeds 0 \
    --output-csv "$SMOKE_DIR/guided100_galsrn50.csv"
fi

# 3) CLIP+LR and Vanilla-CNN tiny smoke checks (both datasets).
run_check clip_lr95 \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -u run_clip_lr_sweep.py \
  "$WB95_PATH" \
  --clip-model RN50 --device cuda \
  --batch-size 512 --num-workers 0 \
  --sampler random --n-trials 1 --seed 0 \
  --penalty-solvers l2:liblinear \
  --C-min 1 --C-max 1 --max-iter 300 \
  --post-seeds 0 \
  --output-csv "$SMOKE_DIR/clip_lr95.csv"

run_check clip_lr100 \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -u run_clip_lr_sweep.py \
  "$WB100_PATH" \
  --clip-model RN50 --device cuda \
  --batch-size 512 --num-workers 0 \
  --sampler random --n-trials 1 --seed 0 \
  --penalty-solvers l2:liblinear \
  --C-min 1 --C-max 1 --max-iter 300 \
  --post-seeds 0 \
  --output-csv "$SMOKE_DIR/clip_lr100.csv"

run_check vanilla95 \
  python -u run_vanilla_waterbird_sweep.py \
  "$WB95_PATH" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --num-epochs 1 --batch-size 96 --num-workers 4 \
  --lr-min 1e-3 --lr-max 1e-3 \
  --wd-min 1e-5 --wd-max 1e-5 \
  --momentum-min 0.9 --momentum-max 0.9 \
  --post-seeds 0 \
  --output-csv "$SMOKE_DIR/vanilla95.csv"

run_check vanilla100 \
  python -u run_vanilla_waterbird_sweep.py \
  "$WB100_PATH" \
  --sampler random --n-trials 1 --seed 0 --train-seed 0 \
  --num-epochs 1 --batch-size 96 --num-workers 4 \
  --lr-min 1e-3 --lr-max 1e-3 \
  --wd-min 1e-5 --wd-max 1e-5 \
  --momentum-min 0.9 --momentum-max 0.9 \
  --post-seeds 0 \
  --output-csv "$SMOKE_DIR/vanilla100.csv"

echo
echo "===== [SMOKE SUMMARY] ====="
echo "PASS=$PASS FAIL=$FAIL"
echo "Summary CSV: $SUMMARY_CSV"
cat "$SUMMARY_CSV"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
