#!/usr/bin/env bash
#SBATCH --job-name=spuco_mask_audit
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/run_logs/audit_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/run_logs/audit_%j.err

set -Eeuo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
MASK_PICKLE=/home/ryreu/guided_cnn/data/spuco/spuco_animals_masks.pkl
CHECKSUM=/home/ryreu/guided_cnn/data/spuco/spuco_animals_masks.sha256
IMAGE_ROOT=/home/ryreu/guided_cnn/data/spuco/spuco_dogs
OUTPUT_ROOT=/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit
RUN_OUTPUT="${OUTPUT_ROOT}/audit_${SLURM_JOB_ID}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
echo "commit=$(git rev-parse HEAD 2>/dev/null || printf UNKNOWN)"
echo "mask_pickle=${MASK_PICKLE}"
echo "image_root=${IMAGE_ROOT}"
echo "run_output=${RUN_OUTPUT}"
python --version
python -c "import numpy, PIL; print('numpy', numpy.__version__, 'Pillow', PIL.__version__)"
df -h /home/ryreu

for required_file in "$MASK_PICKLE" "$CHECKSUM"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[ERROR] Missing required file: $required_file" >&2
    exit 2
  fi
done
if [[ ! -d "$IMAGE_ROOT" ]]; then
  echo "[ERROR] Missing SpuCoDogs image root: $IMAGE_ROOT" >&2
  exit 2
fi

srun --unbuffered python -u \
  experiments/spucodogs_mask_audit/audit_spucodogs_masks.py \
  --mask-pickle "$MASK_PICKLE" \
  --checksum-file "$CHECKSUM" \
  --compute-sha256 \
  --image-root "$IMAGE_ROOT" \
  --output-dir "$RUN_OUTPUT" \
  --preview-count 16

echo "[DONE] Audit output: $RUN_OUTPUT"
