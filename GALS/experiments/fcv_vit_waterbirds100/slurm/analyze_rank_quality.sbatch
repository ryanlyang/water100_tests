#!/usr/bin/env bash
#SBATCH --job-name=fcv_vit_rank
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step12_rank_analysis_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/run_logs/step12_rank_analysis_%j.err

set -euo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
export MPLCONFIGDIR=/tmp/fcv_step12_matplotlib_${SLURM_JOB_ID}

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID}"
python --version
python -c "import pandas, scipy, matplotlib; print('pandas', pandas.__version__, 'scipy', scipy.__version__, 'matplotlib', matplotlib.__version__)"

srun python -u experiments/fcv_vit_waterbirds100/scripts/analyze_rank_quality.py
