#!/usr/bin/env bash
#SBATCH --job-name=spuco_deep_audit
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/run_logs/deep_audit_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/run_logs/deep_audit_%j.err

set -Eeuo pipefail

PROJECT_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
REPO="${PROJECT_ROOT}/GALS"
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
DATA_ROOT=/home/ryreu/guided_cnn/data/spuco
IMAGE_ROOT="${DATA_ROOT}/spuco_dogs"
MASK_PICKLE="${DATA_ROOT}/spuco_animals_masks.pkl"
MASK_CHECKSUM="${DATA_ROOT}/spuco_animals_masks.sha256"
ARCHIVE_CHECKSUM="${DATA_ROOT}/spuco_dogs_archive.sha256"
OUTPUT_ROOT=/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit
RUN_OUTPUT="${OUTPUT_ROOT}/deep_audit_${SLURM_JOB_ID}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
echo "commit=$(git rev-parse HEAD 2>/dev/null || printf UNKNOWN)"
echo "image_root=${IMAGE_ROOT}"
echo "mask_pickle=${MASK_PICKLE}"
echo "run_output=${RUN_OUTPUT}"
python --version
python -c "import numpy, PIL, scipy; print('numpy', numpy.__version__, 'Pillow', PIL.__version__, 'scipy', scipy.__version__)"
df -h /home/ryreu

for required_file in "$MASK_PICKLE" "$MASK_CHECKSUM"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[ERROR] Missing required file: $required_file" >&2
    exit 2
  fi
done
if [[ ! -d "$IMAGE_ROOT" ]]; then
  echo "[ERROR] Missing SpuCoDogs image root: $IMAGE_ROOT" >&2
  exit 2
fi

archive_args=()
if [[ -f "$ARCHIVE_CHECKSUM" ]]; then
  archive_args=(--archive-checksum "$ARCHIVE_CHECKSUM")
fi

srun --unbuffered python -u \
  experiments/spucodogs_mask_audit/deep_audit_spucodogs.py \
  --image-root "$IMAGE_ROOT" \
  --mask-pickle "$MASK_PICKLE" \
  --mask-checksum "$MASK_CHECKSUM" \
  "${archive_args[@]}" \
  --repo-root "$PROJECT_ROOT" \
  --output-dir "$RUN_OUTPUT" \
  --perceptual-hamming-threshold 4 \
  --quality-preview-count 40 \
  --storage-reserve-gib 10

echo "[DONE] Deep audit output: $RUN_OUTPUT"
