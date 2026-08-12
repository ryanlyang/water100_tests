#!/usr/bin/env bash
#SBATCH --job-name=prep_in9_r4rr_voc
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/ryreu/guided_cnn/logsImageNet9/prep_in9_r4rr_voc_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsImageNet9/prep_in9_r4rr_voc_%j.err

set -euo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL="${PROTOCOL:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL}/manifest.csv}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1/voc_workspace}"
PROMPT_CONFIG="${PROMPT_CONFIG:-${REPO}/RightForTheRightRegions/WeCLIPPlus/clip/clip_texts/clip_text_imagenet9.py}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"

mkdir -p /home/ryreu/guided_cnn/logsImageNet9
cd "$REPO"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO"
echo "Manifest: $MANIFEST"
echo "Workspace: $WORKSPACE_ROOT"
echo "Prompt config: $PROMPT_CONFIG"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -u ImageNet9_Runs/prepare_imagenet9_r4rr_voc_workspace.py \
  --manifest "$MANIFEST" \
  --workspace-root "$WORKSPACE_ROOT" \
  --prompt-config "$PROMPT_CONFIG"

echo "[$(date)] ImageNet-9 R4RR VOC workspace preparation complete."
