#!/bin/bash
# Rebuild the legacy gals_a100 Conda environment on SPORC x86 nodes.

set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-gals_a100}"
RECREATE_ENV="${RECREATE_ENV:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${REQ_FILE:-${SCRIPT_DIR}/requirements_gals_a100_sporc.txt}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "[ERROR] This environment is for SPORC x86_64/A100 nodes, not $(uname -m)." >&2
  echo "Do not install it with the aarch64 Miniforge used on Tigris GH200 nodes." >&2
  exit 2
fi
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] Conda initialization script not found: $CONDA_SH" >&2
  exit 2
fi
if [[ ! -f "$REQ_FILE" ]]; then
  echo "[ERROR] Requirements file not found: $REQ_FILE" >&2
  exit 2
fi

# Conda activation hooks may inspect optional MKL variables before defining
# them. Keep nounset disabled only while Conda initializes or activates.
set +u
source "$CONDA_SH"
set -u

env_exists() {
  conda env list | awk -v name="$ENV_NAME" '$1 == name { found=1 } END { exit !found }'
}

echo "[INFO] host=$(hostname) architecture=$(uname -m)"
echo "[INFO] conda_base=$(conda info --base)"
echo "[INFO] environment=$ENV_NAME"
echo "[INFO] requirements=$REQ_FILE"
df -h "$HOME" | tail -n 1

if [[ "$RECREATE_ENV" == "1" ]] && env_exists; then
  echo "[INFO] Removing existing environment because RECREATE_ENV=1."
  conda env remove -y -n "$ENV_NAME"
fi

if ! env_exists; then
  echo "[STEP 1/5] Creating Python 3.8 environment."
  conda create -y -n "$ENV_NAME" python=3.8 pip
else
  echo "[STEP 1/5] Environment already exists; repairing it in place."
fi

set +u
conda activate "$ENV_NAME"
set -u
export PYTHONNOUSERSITE=1

echo "[STEP 2/5] Installing the established CUDA-enabled PyTorch stack."
conda install -y \
  pytorch==1.12.1 \
  torchvision==0.13.1 \
  cudatoolkit=11.3 \
  -c pytorch -c nvidia -c conda-forge

echo "[STEP 3/5] Installing compiled dataset utilities."
conda install -y -c conda-forge pycocotools

echo "[STEP 4/5] Installing GALS, R4RR, ElRep, sweep, and AFR dependencies."
python -m pip install --no-cache-dir -r "$REQ_FILE"
python -m pip install --no-cache-dir opencv-python==4.6.0.66
# TorchRay's dependency metadata targets an older torch release. Install its
# code without dependencies so it cannot replace the CUDA-enabled torch build.
python -m pip install --no-cache-dir torchray==1.0.0.2 --no-deps

echo "[STEP 5/5] Verifying imports and pinned core versions."
python - <<'PY'
import importlib
import sys

modules = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "PIL",
    "yaml",
    "cv2",
    "scipy",
    "sklearn",
    "skimage",
    "matplotlib",
    "omegaconf",
    "torchray",
    "pycocotools",
    "optuna",
    "wandb",
    "timm",
    "einops",
    "wilds",
    "transformers",
    "pytorch_transformers",
]
failed = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append((name, repr(exc)))

if failed:
    for name, error in failed:
        print(f"[IMPORT FAILED] {name}: {error}", file=sys.stderr)
    raise SystemExit(1)

import torch
import torchvision

if not torch.__version__.startswith("1.12.1"):
    raise SystemExit(f"Unexpected torch version: {torch.__version__}")
if not torchvision.__version__.startswith("0.13.1"):
    raise SystemExit(f"Unexpected torchvision version: {torchvision.__version__}")

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available_on_this_host={torch.cuda.is_available()}")
print("all required imports: OK")
PY

echo
echo "[DONE] Rebuilt Conda environment: $ENV_NAME"
echo "[NOTE] cuda_available_on_this_host may be False on the submit node."
echo "[NOTE] Optional cache cleanup after confirming jobs run: conda clean --all -y"
