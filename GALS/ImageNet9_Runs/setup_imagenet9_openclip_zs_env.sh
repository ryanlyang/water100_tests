#!/usr/bin/env bash
# Create the isolated OpenCLIP environment used by ImageNet-9 zero-shot jobs.

set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-openclip-zs}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "[ERROR] Conda initialization script not found under ${CONDA_ROOT}" >&2
  exit 2
fi

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
set -u

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[INFO] Reusing existing Conda environment: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python=3.10 pip
fi

set +u
conda activate "$ENV_NAME"
set -u

export PYTHONNOUSERSITE=1

python -m pip install --upgrade \
  'pip<26' \
  'setuptools<76' \
  wheel

# Keep this separate from r4rr-weclip. WeCLIP+ requires OpenCLIP 2.20.0,
# whereas the SigLIP2 model registry was added in OpenCLIP 2.31.0.
python -m pip install --index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.0 \
  torchvision==0.15.1

python -m pip install \
  open_clip_torch==2.31.0 \
  transformers==4.49.0 \
  tokenizers==0.21.0 \
  sentencepiece==0.2.0 \
  numpy==1.24.4 \
  Pillow==10.4.0

python - <<'PY'
import open_clip
import sys
import torch
import torchvision

required = {
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-16-SigLIP2-256", "webli"),
}
available = set(open_clip.list_pretrained())
missing = sorted(required - available)
if missing:
    raise RuntimeError(f"OpenCLIP registry is missing required model pairs: {missing}")

# This checkpoint uses a newer Gemma tokenizer JSON. Constructing and invoking
# it here catches an incompatible transformers/tokenizers stack before SBATCH.
siglip2_tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-256")
siglip2_tokens = siglip2_tokenizer(["a photo of a bird"])
if siglip2_tokens.shape[0] != 1:
    raise RuntimeError(f"Unexpected SigLIP2 token shape: {siglip2_tokens.shape}")

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split("+")[0] == "2.0.0", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.15.1", torchvision.__version__

print("[ENV CHECK] ImageNet-9 OpenCLIP zero-shot imports: OK")
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available_on_this_host={torch.cuda.is_available()}")
print(f"open_clip={getattr(open_clip, '__version__', 'unknown')}")
print("required model pairs: present")
print(f"SigLIP2 tokenizer: OK shape={tuple(siglip2_tokens.shape)}")
PY

echo "[DONE] Environment ready: $ENV_NAME"
echo "[NOTE] cuda_available_on_this_host may be False on the submit node."
