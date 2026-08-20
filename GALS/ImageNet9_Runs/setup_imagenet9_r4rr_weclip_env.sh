#!/usr/bin/env bash
# Create the isolated Python 3.8 environment used by ImageNet-9 WeCLIP+.

set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-r4rr-weclip}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
WECLIP_ROOT="${WECLIP_ROOT:-${REPO}/RightForTheRightRegions/WeCLIPPlus}"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "[ERROR] Conda initialization script not found under ${CONDA_ROOT}" >&2
  exit 2
fi

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
set -u

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[INFO] Reusing existing Conda environment: ${ENV_NAME}"
else
  conda create -y -n "$ENV_NAME" python=3.8 pip
fi

set +u
conda activate "$ENV_NAME"
set -u

python -m pip install --upgrade \
  'pip<25' \
  'setuptools<70' \
  wheel

# Install DenseCRF before the pip-pinned stack. Keep Python explicit because
# Conda may otherwise upgrade the environment to the newest Python build that
# satisfies pydensecrf, which is incompatible with the pinned PyTorch wheels.
conda install -y -c conda-forge python=3.8 pydensecrf

python - <<'PY'
import sys

assert sys.version_info[:2] == (3, 8), sys.version
print(f"[PYTHON CHECK] python={sys.version.split()[0]}")
PY

# DINOv2 uses torch.nn.functional.scaled_dot_product_attention, which requires
# PyTorch 2.x. CUDA 11.7 wheels run on the SPORC A100 nodes without changing
# the system CUDA installation.
python -m pip install --index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.0 \
  torchvision==0.15.1

# This training path uses MMCV's Python CNN/runner utilities, not mmcv.ops, so
# the pure-Python MMCV 1.x package avoids a fragile local CUDA extension build.
python -m pip install \
  numpy==1.23.5 \
  Pillow==9.5.0 \
  matplotlib==3.7.5 \
  tqdm==4.66.5 \
  omegaconf==2.1.2 \
  timm==0.9.16 \
  imageio==2.35.1 \
  mmcv==1.7.1 \
  scikit-learn==1.3.2 \
  tensorboard==2.13.0 \
  ftfy==6.2.3 \
  regex==2024.5.15 \
  ttach==0.0.3 \
  lxml==5.3.0 \
  colour==0.1.5 \
  open_clip_torch==2.20.0 \
  opencv-python-headless==4.8.1.78 \
  einops==0.7.0 \
  packaging==24.1

export PYTHONNOUSERSITE=1
export PYTHONPATH="${WECLIP_ROOT}:${REPO}:${PYTHONPATH:-}"
export XFORMERS_DISABLED=1

cd "$WECLIP_ROOT"
python - <<'PY'
import sys

import cv2
import imageio
import mmcv
import numpy
import omegaconf
import open_clip
import pydensecrf
import sklearn
import timm
import torch
import torchvision

from WeCLIP_Plus.model_attn_aff_voc import WeCLIP_Plus

assert sys.version_info[:2] == (3, 8), sys.version
assert torch.__version__.split("+")[0] == "2.0.0", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.15.1", torchvision.__version__

print("[ENV CHECK] ImageNet-9 WeCLIP imports: OK")
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available_on_this_host={torch.cuda.is_available()}")
print(f"mmcv={mmcv.__version__}")
print(f"open_clip={getattr(open_clip, '__version__', 'unknown')}")
PY

echo "[DONE] Environment ready: ${ENV_NAME}"
echo "[NOTE] cuda_available_on_this_host may be False on the submit node."
