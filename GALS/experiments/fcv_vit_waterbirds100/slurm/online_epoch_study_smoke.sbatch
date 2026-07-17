#!/usr/bin/env bash
#SBATCH --job-name=fcv_on_smoke
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/online_smoke_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/online_smoke_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
RUNNER=experiments/fcv_vit_waterbirds100/scripts/run_online_epoch_study.py
VALIDATOR=experiments/fcv_vit_waterbirds100/scripts/validate_online_smoke_prefix.py
STORAGE_CHECK=experiments/fcv_vit_waterbirds100/scripts/check_storage_budget.py
SMOKE_GATE=experiments/fcv_vit_waterbirds100/scripts/ensure_online_smoke_gate.py
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID} real online interruption/resume smoke"

# A completed gate is immutable for the current campaign. Reusing it makes the
# full launcher safe after run 0 has advanced or completed.
if python -u "$SMOKE_GATE" --run-index 0 --quiet-missing; then
  echo "[$(date)] current-campaign online smoke gate already passed"
  exit 0
fi

# Epoch 1 performs real training plus FCV, controls, Oracle validation, and
# analysis-only test inference, then exits through the production pause path.
# Its immutable receipt is independently reusable if a later smoke stage was
# interrupted, so the job never asks an epoch-2 resume to rewind to epoch 1.
if python -u "$SMOKE_GATE" --run-index 0 --receipt-epoch 1 --quiet-missing; then
  echo "[$(date)] current-campaign epoch-1 smoke receipt already passed"
else
  python -u "$STORAGE_CHECK" \
    --stage=online_smoke_epoch1_baseline \
    --output-json "$OUTPUT/preflight/online_smoke_epoch1_storage_baseline.json"
  srun python -u "$RUNNER" --run-index 0 --device cuda --stop-after-epoch 1
  python -u "$VALIDATOR" \
    --run-index 0 \
    --expected-epoch 1 \
    --storage-baseline-path "$OUTPUT/preflight/online_smoke_epoch1_storage_baseline.json"
fi

# A second process must consume the epoch-1 resume transaction and produce the
# exact next online candidate. The full array later continues run 0 at epoch 3.
if python -u "$SMOKE_GATE" --run-index 0 --receipt-epoch 2 --quiet-missing; then
  echo "[$(date)] current-campaign epoch-2 resume receipt already passed"
else
  python -u "$STORAGE_CHECK" \
    --stage=online_smoke_epoch2_baseline \
    --output-json "$OUTPUT/preflight/online_smoke_epoch2_storage_baseline.json"
  srun python -u "$RUNNER" --run-index 0 --device cuda --stop-after-epoch 2
  python -u "$VALIDATOR" \
    --run-index 0 \
    --expected-epoch 2 \
    --require-resumed-from-epoch 1 \
    --storage-baseline-path "$OUTPUT/preflight/online_smoke_epoch2_storage_baseline.json"
fi

python -u "$SMOKE_GATE" --run-index 0

echo "[$(date)] real online interruption/resume smoke passed"
