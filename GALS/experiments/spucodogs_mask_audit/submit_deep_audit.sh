#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit
SBATCH=experiments/spucodogs_mask_audit/slurm/deep_audit_spucodogs.sbatch

mkdir -p "$OUTPUT/run_logs"
cd "$REPO"

for required in \
  /home/ryreu/guided_cnn/data/spuco/spuco_animals_masks.pkl \
  /home/ryreu/guided_cnn/data/spuco/spuco_animals_masks.sha256 \
  /home/ryreu/guided_cnn/data/spuco/spuco_dogs; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required input: $required" >&2
    exit 2
  fi
done

raw_job_id=$(sbatch --parsable "$SBATCH")
job_id=${raw_job_id%%;*}
echo "[DONE] Submitted SpuCoDogs deep audit job: $job_id"
echo "Monitor: squeue -j $job_id"
echo "Logs: $OUTPUT/run_logs/deep_audit_${job_id}.out"
echo "      $OUTPUT/run_logs/deep_audit_${job_id}.err"
echo "Output: $OUTPUT/deep_audit_${job_id}"
