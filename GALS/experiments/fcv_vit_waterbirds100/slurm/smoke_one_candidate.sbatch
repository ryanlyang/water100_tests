#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_smoke
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/smoke_one_candidate_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/smoke_one_candidate_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
CONFIG=$REPO/experiments/fcv_vit_waterbirds100/configs/waterbirds100_vit_s16_first_study.yaml
SMOKE_TAG=${SMOKE_TAG:-job_${SLURM_JOB_ID}}
SMOKE=$OUTPUT/smoke/$SMOKE_TAG
RESUMED=$SMOKE/candidate_models_resumed
REFERENCE=$SMOKE/candidate_models_uninterrupted
BANKS=$SMOKE/token_banks
FCV=$SMOKE/fcv_scores
CONTROLS=$SMOKE/control_scores
PREFLIGHT=$OUTPUT/preflight
MANIFEST=$OUTPUT/split_manifests/metadata_val.csv
MASKS=$OUTPUT/patch_masks/patch_masks_val.pt
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"
mkdir -p "$SMOKE" "$BANKS" "$FCV" "$CONTROLS" "$PREFLIGHT"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID} smoke=$SMOKE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Exercise the complete protocol under the production Python, Torch, timm,
# and aarch64 runtime. The synthetic downstream tests cover Oracle raw-result
# auditing, frozen Step-9 pool binding, final test evaluation, gap analysis,
# and rank analysis without inventing a noncanonical one-candidate pool.
srun python -m unittest discover \
  -s experiments/fcv_vit_waterbirds100/tests -v

srun python -u experiments/fcv_vit_waterbirds100/scripts/cache_pretrained_model.py

# Fail before GPU work if the public validation masks or current teacher-map
# bytes no longer match the accepted Step 3 summary.
srun python -u experiments/fcv_vit_waterbirds100/scripts/validate_patch_masks.py \
  --config "$CONFIG" --manifest "$MANIFEST" --patch-masks "$MASKS"

srun python -u experiments/fcv_vit_waterbirds100/scripts/verify_vit_patch_forward.py \
  --config "$CONFIG" --pretrained \
  --output-report "$PREFLIGHT/reconstruction_pretrained.json"

# Inject the exact first-epoch crash window identified in review: the resume
# state is committed, but metrics.csv has not yet been published.
if srun python -u experiments/fcv_vit_waterbirds100/scripts/train_candidate.py \
  --config "$CONFIG" --run-index 0 --candidate-root "$RESUMED" \
  --smoke-interrupt-after-resume-commit-epoch 1; then
  echo "Failure injection unexpectedly returned success." >&2
  exit 1
fi
test -f "$RESUMED/run_000_lr_1em05_wd_0p01_seed_0/resume_state.pt"
test ! -f "$RESUMED/run_000_lr_1em05_wd_0p01_seed_0/metrics.csv"

# Reinvocation must reconstruct metrics.csv from the committed epoch boundary,
# resume to epoch 20, and remain tensor-exact against uninterrupted training.
srun python -u experiments/fcv_vit_waterbirds100/scripts/train_candidate.py \
  --config "$CONFIG" --run-index 0 --candidate-root "$RESUMED"
# A third invocation must use the completed-run reuse gate.
srun python -u experiments/fcv_vit_waterbirds100/scripts/train_candidate.py \
  --config "$CONFIG" --run-index 0 --candidate-root "$RESUMED"

# Train the same locked candidate without pausing and compare every epoch's
# metrics and model tensors exactly on the GH200 runtime.
srun python -u experiments/fcv_vit_waterbirds100/scripts/train_candidate.py \
  --config "$CONFIG" --run-index 0 --candidate-root "$REFERENCE"
srun python -u experiments/fcv_vit_waterbirds100/scripts/verify_resume_equivalence.py \
  --config "$CONFIG" --run-index 0 \
  --resumed-root "$RESUMED" --reference-root "$REFERENCE" \
  --output-report "$SMOKE/resume_equivalence.json"

CHECKPOINT=$(python -c \
  'import pandas as pd,sys; f=pd.read_csv(sys.argv[1]).sort_values("epoch"); print(f.iloc[-1]["checkpoint_path"])' \
  "$RESUMED/run_000_lr_1em05_wd_0p01_seed_0/metrics.csv")

srun python -u experiments/fcv_vit_waterbirds100/scripts/verify_vit_patch_forward.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --output-report "$PREFLIGHT/reconstruction_candidate.json"

srun python -u experiments/fcv_vit_waterbirds100/scripts/build_background_token_banks.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" \
  --patch-masks "$MASKS" --output-dir "$BANKS" --batch-size 128 --workers 8
srun python -u experiments/fcv_vit_waterbirds100/scripts/prepare_opposite_donor_plan.py \
  --config "$CONFIG" --manifest "$MANIFEST" --patch-masks "$MASKS" \
  --token-bank-dir "$BANKS" --reference-run-index 0 --reference-epoch 20 \
  --output "$FCV/opposite_donor_plan.pt"
srun python -u experiments/fcv_vit_waterbirds100/scripts/score_fcv_candidates.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" \
  --patch-masks "$MASKS" --token-bank-dir "$BANKS" \
  --donor-plan "$FCV/opposite_donor_plan.pt" --output-dir "$FCV" \
  --batch-size 128 --workers 8
srun python -u experiments/fcv_vit_waterbirds100/scripts/prepare_control_plan.py \
  --config "$CONFIG" --manifest "$MANIFEST" --patch-masks "$MASKS" \
  --token-bank-dir "$BANKS" --reference-run-index 0 --reference-epoch 20 \
  --opposite-donor-plan "$FCV/opposite_donor_plan.pt" \
  --output "$CONTROLS/control_plan.pt"
srun python -u experiments/fcv_vit_waterbirds100/scripts/score_fcv_controls.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" \
  --patch-masks "$MASKS" --token-bank-dir "$BANKS" \
  --opposite-donor-plan "$FCV/opposite_donor_plan.pt" \
  --control-plan "$CONTROLS/control_plan.pt" --step7-score-dir "$FCV" \
  --output-dir "$CONTROLS" --batch-size 128 --workers 8

# Re-run every checkpoint-dependent stage without overwrite; all three must
# validate their complete summaries and return status=reused.
srun python -u experiments/fcv_vit_waterbirds100/scripts/build_background_token_banks.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" \
  --patch-masks "$MASKS" --output-dir "$BANKS" --batch-size 128 --workers 8
srun python -u experiments/fcv_vit_waterbirds100/scripts/score_fcv_candidates.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" \
  --patch-masks "$MASKS" --token-bank-dir "$BANKS" \
  --donor-plan "$FCV/opposite_donor_plan.pt" --output-dir "$FCV" \
  --batch-size 128 --workers 8
srun python -u experiments/fcv_vit_waterbirds100/scripts/score_fcv_controls.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" \
  --patch-masks "$MASKS" --token-bank-dir "$BANKS" \
  --opposite-donor-plan "$FCV/opposite_donor_plan.pt" \
  --control-plan "$CONTROLS/control_plan.pt" --step7-score-dir "$FCV" \
  --output-dir "$CONTROLS" --batch-size 128 --workers 8

echo "[DONE] GH200 full protocol tests, resume, reconstruction, Steps 6-8, diagnostics, and reuse passed."
echo "Artifacts: $SMOKE"
