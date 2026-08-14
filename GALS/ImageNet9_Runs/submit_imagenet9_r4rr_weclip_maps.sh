#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
cd "$REPO"

map_job_raw="$(sbatch --parsable ImageNet9_Runs/run_imagenet9_r4rr_weclip_maps.sbatch)"
map_job="${map_job_raw%%;*}"
audit_job="$(sbatch --parsable --dependency="afterok:${map_job}" \
  ImageNet9_Runs/run_audit_imagenet9_r4rr_weclip_maps.sbatch)"

echo "[SUBMITTED] map array: $map_job"
echo "[SUBMITTED] dependent audit: $audit_job"
echo "[INFO] Re-running this command is safe: valid maps are reused by sample ID."
