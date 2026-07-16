#!/bin/bash -l
# Smoke-test RedMeat strategies used by submit_all_redmeat_experiment.sh.
# Runs tiny checks in debug partition to validate end-to-end wiring.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_smoke_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_smoke_debug_%j.err
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
redmeat_prepare_food_layout "$DATA_ROOT" "$DATA_DIR"

RUNS_ROOT="$PROJECT_ROOT"
REPO_ROOT="$GALS_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"
META_CSV="$DATASET_ROOT/all_images.csv"
VIT_ATT_DIR="$DATASET_ROOT/clip_vit_attention"
RN50_ATT_DIR="$DATASET_ROOT/clip_rn50_attention_gradcam"
ABN_WEIGHT="$REPO_ROOT/weights/resnet50_abn_imagenet.pth.tar"
MASK_DIR=${MASK_DIR:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_dinovit/val/prediction_cmap/}
SIGLIP2_MASK_DIR=${SIGLIP2_MASK_DIR:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_siglip2_dinovit/val/prediction_cmap/}
SKIP_SIGLIP2_GUIDED_SMOKE=${SKIP_SIGLIP2_GUIDED_SMOKE:-1}
SKIP_CLIP_LR_SMOKE=${SKIP_CLIP_LR_SMOKE:-0}

LOG_ROOT="$LOG_DIR"
SMOKE_DIR="$LOG_ROOT/redmeat_smoke_debug_${SLURM_JOB_ID}"
mkdir -p "$SMOKE_DIR"
SUMMARY_CSV="$SMOKE_DIR/summary.csv"
echo "name,status,seconds,log" > "$SUMMARY_CSV"

cd "$REPO_ROOT"
export PYTHONPATH="$RUNS_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=0

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

for p in "$RUNS_ROOT" "$REPO_ROOT" "$DATASET_ROOT" "$VIT_ATT_DIR" "$RN50_ATT_DIR" "$MASK_DIR"; do
  if [[ ! -d "$p" ]]; then
    echo "[ERROR] Missing required directory: $p" >&2
    exit 2
  fi
done
if [[ ! -f "$META_CSV" ]]; then
  echo "[ERROR] Missing metadata file: $META_CSV" >&2
  exit 2
fi
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

# Shared tiny settings.
BASE_LR=1e-4
CLS_LR=1e-3
WD=1e-5
BATCH=16

COMMON_OVR=(
  "DATA.ROOT=$DATA_ROOT"
  "DATA.FOOD_SUBSET_DIR=$DATA_DIR"
  "DATA.SUBDIR=$DATA_DIR"
  "DATA.BATCH_SIZE=$BATCH"
  "EXP.NUM_EPOCHS=1"
  "EXP.AUX_LOSSES_ON_VAL=False"
  "LOGGING.SAVE_BEST=True"
  "LOGGING.SAVE_LAST=False"
)

run_check gals_vit \
  python -u main.py --config RedMeat_Runs/configs/redmeat_gals_vit.yaml --name smoke_gals_vit \
  "${COMMON_OVR[@]}" \
  "SEED=0" \
  "EXP.BASE.LR=$BASE_LR" \
  "EXP.CLASSIFIER.LR=$CLS_LR" \
  "EXP.WEIGHT_DECAY=$WD" \
  "EXP.LOSSES.GRADIENT_OUTSIDE.WEIGHT=10"

run_check rrr_vit \
  python -u main.py --config RedMeat_Runs/configs/redmeat_rrr_vit.yaml --name smoke_rrr_vit \
  "${COMMON_OVR[@]}" \
  "SEED=0" \
  "EXP.BASE.LR=$BASE_LR" \
  "EXP.CLASSIFIER.LR=$CLS_LR" \
  "EXP.WEIGHT_DECAY=$WD" \
  "EXP.LOSSES.GRADIENT_OUTSIDE.WEIGHT=10"

run_check gradcam_vit \
  python -u main.py --config RedMeat_Runs/configs/redmeat_gradcam_vit.yaml --name smoke_gradcam_vit \
  "${COMMON_OVR[@]}" \
  "SEED=0" \
  "EXP.BASE.LR=$BASE_LR" \
  "EXP.CLASSIFIER.LR=$CLS_LR" \
  "EXP.WEIGHT_DECAY=$WD" \
  "EXP.LOSSES.GRADCAM.WEIGHT=1"

run_check abn_vit \
  python -u main.py --config RedMeat_Runs/configs/redmeat_abn_vit.yaml --name smoke_abn_vit \
  "${COMMON_OVR[@]}" \
  "SEED=0" \
  "EXP.BASE.LR=$BASE_LR" \
  "EXP.CLASSIFIER.LR=$CLS_LR" \
  "EXP.WEIGHT_DECAY=$WD" \
  "EXP.LOSSES.ABN_SUPERVISION.WEIGHT=1"

run_check gals_ourmasks \
  python -u main.py --config RedMeat_Runs/configs/redmeat_gals_ourmasks.yaml --name smoke_gals_ourmasks \
  "${COMMON_OVR[@]}" \
  "SEED=0" \
  "DATA.SEGMENTATION_DIR=$MASK_DIR" \
  "DATA.SEG_TRAIN_ONLY=True" \
  "EXP.BASE.LR=$BASE_LR" \
  "EXP.CLASSIFIER.LR=$CLS_LR" \
  "EXP.WEIGHT_DECAY=$WD" \
  "EXP.LOSSES.GRADIENT_OUTSIDE.WEIGHT=10"

run_check upweight \
  python -u main.py --config RedMeat_Runs/configs/redmeat_upweight.yaml --name smoke_upweight \
  "${COMMON_OVR[@]}" \
  "SEED=0" \
  "EXP.BASE.LR=$BASE_LR" \
  "EXP.CLASSIFIER.LR=$CLS_LR" \
  "EXP.WEIGHT_DECAY=$WD"

run_check abn_baseline \
  python -u main.py --config RedMeat_Runs/configs/redmeat_abn.yaml --name smoke_abn_baseline \
  "${COMMON_OVR[@]}" \
  "SEED=0" \
  "EXP.BASE.LR=$BASE_LR" \
  "EXP.CLASSIFIER.LR=$CLS_LR" \
  "EXP.WEIGHT_DECAY=$WD" \
  "EXP.LOSSES.ABN_CLASSIFICATION.WEIGHT=1"

run_check vanilla_cnn \
  python -u RedMeat_Runs/run_vanilla_redmeat.py \
  "$DATASET_ROOT" \
  --model resnet50 \
  --pretrained \
  --batch-size "$BATCH" \
  --num-epochs 1 \
  --num-workers 4 \
  --lr "$BASE_LR" \
  --momentum 0.9 \
  --weight-decay "$WD" \
  --seed 0 \
  --checkpoint-dir "$SMOKE_DIR/vanilla_ckpts"

run_check guided \
  python -u RedMeat_Runs/run_guided_redmeat.py \
  "$DATASET_ROOT" \
  "$MASK_DIR" \
  --seed 0 \
  --num-epochs 1 \
  --attention-epoch 0 \
  --kl-lambda 10 \
  --kl-increment 1 \
  --base_lr "$BASE_LR" \
  --classifier_lr "$CLS_LR" \
  --lr2-mult 1.0 \
  --checkpoint-dir "$SMOKE_DIR/guided_ckpts"

run_check guided_galsvit \
  python -u RedMeat_Runs/run_guided_redmeat_gals_vitatt.py \
  "$DATASET_ROOT" \
  "$VIT_ATT_DIR" \
  --seed 0 \
  --num-epochs 1 \
  --attention-epoch 0 \
  --kl-lambda 10 \
  --kl-increment 1 \
  --base_lr "$BASE_LR" \
  --classifier_lr "$CLS_LR" \
  --lr2-mult 1.0 \
  --checkpoint-dir "$SMOKE_DIR/guided_galsvit_ckpts"

run_check guided_galsrn50 \
  python -u RedMeat_Runs/run_guided_redmeat_gals_vitatt.py \
  "$DATASET_ROOT" \
  "$RN50_ATT_DIR" \
  --seed 0 \
  --num-epochs 1 \
  --attention-epoch 0 \
  --kl-lambda 10 \
  --kl-increment 1 \
  --base_lr "$BASE_LR" \
  --classifier_lr "$CLS_LR" \
  --lr2-mult 1.0 \
  --checkpoint-dir "$SMOKE_DIR/guided_galsrn50_ckpts"

# Optional guided smoke against SigLIP2 WeCLIP masks.
# Default is skip since that mask path may not be available yet.
if [[ "$SKIP_SIGLIP2_GUIDED_SMOKE" -eq 1 ]]; then
  skip_check guided_siglip2 "SKIP_SIGLIP2_GUIDED_SMOKE=1"
elif [[ ! -d "$SIGLIP2_MASK_DIR" ]]; then
  skip_check guided_siglip2 "missing SIGLIP2 mask dir: $SIGLIP2_MASK_DIR"
else
  run_check guided_siglip2 \
    python -u RedMeat_Runs/run_guided_redmeat.py \
    "$DATASET_ROOT" \
    "$SIGLIP2_MASK_DIR" \
    --seed 0 \
    --num-epochs 1 \
    --attention-epoch 0 \
    --kl-lambda 10 \
    --kl-increment 1 \
    --base_lr "$BASE_LR" \
    --classifier_lr "$CLS_LR" \
    --lr2-mult 1.0 \
    --checkpoint-dir "$SMOKE_DIR/guided_siglip2_ckpts"
fi

# Single-trial CLIP+LR check (kept tiny; no post-seed reruns).
if [[ "$SKIP_CLIP_LR_SMOKE" -eq 1 ]]; then
  skip_check clip_lr "SKIP_CLIP_LR_SMOKE=1"
else
  run_check clip_lr \
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python -u RedMeat_Runs/run_clip_lr_sweep_redmeat.py \
    "$DATASET_ROOT" \
    --clip-model RN50 \
    --device cpu \
    --batch-size 128 \
    --num-workers 0 \
    --n-trials 1 \
    --sampler random \
    --seed 0 \
    --penalty-solvers l2:lbfgs \
    --C-min 1e-4 \
    --C-max 1e-4 \
    --max-iter 200 \
    --post-seeds 0 \
    --output-csv "$SMOKE_DIR/clip_lr_smoke.csv"
fi

echo
echo "===== [SMOKE] SUMMARY ====="
cat "$SUMMARY_CSV"
echo "PASS=$PASS FAIL=$FAIL"
echo "Logs: $SMOKE_DIR"

if (( FAIL > 0 )); then
  exit 1
fi
