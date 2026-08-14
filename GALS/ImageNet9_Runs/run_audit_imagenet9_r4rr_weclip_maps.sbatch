#!/usr/bin/env bash
#SBATCH --job-name=audit_in9_r4rr_maps
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/audit_in9_r4rr_maps_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/audit_in9_r4rr_maps_%j.err

set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
TEACHER_ROOT="${TEACHER_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1}"
WORKSPACE_MANIFEST="${WORKSPACE_MANIFEST:-${TEACHER_ROOT}/voc_workspace/metadata/workspace_manifest.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TEACHER_ROOT}/inference/full/val}"
MAP_ROOT="${MAP_ROOT:-${OUTPUT_ROOT}/prediction_cmap}"
AUDIT_ROOT="${AUDIT_ROOT:-${TEACHER_ROOT}/inference/full/audit}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"

mkdir -p /home/ryreu/guided_cnn/logsImageNet9 "$AUDIT_ROOT"
cd "$REPO"

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gals_a100
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/ImageNet9_Runs:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[$(date)] Host: $(hostname)"
echo "Map root: $MAP_ROOT"
echo "Workspace manifest: $WORKSPACE_MANIFEST"
echo "Audit output: $AUDIT_ROOT"

srun --unbuffered "$PYTHON_BIN" -u ImageNet9_Runs/audit_imagenet9_r4rr_weclip_maps.py \
  --workspace-manifest "$WORKSPACE_MANIFEST" \
  --map-root "$MAP_ROOT" \
  --inference-contract "$OUTPUT_ROOT/inference_contract.json" \
  --output-dir "$AUDIT_ROOT"

echo "[$(date)] ImageNet-9 R4RR map audit complete."
