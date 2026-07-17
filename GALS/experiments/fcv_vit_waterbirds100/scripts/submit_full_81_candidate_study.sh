#!/usr/bin/env bash
set -euo pipefail

echo "This legacy 81-checkpoint launcher is disabled." >&2
echo "Use scripts/submit_full_online_540_study.sh for the all-epoch online protocol." >&2
exit 2

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
OUTPUT=/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
CONFIG=experiments/fcv_vit_waterbirds100/configs/waterbirds100_vit_s16_first_study.yaml
SLURM=experiments/fcv_vit_waterbirds100/slurm

mkdir -p \
  "$OUTPUT/run_logs" \
  "$OUTPUT/candidate_models" \
  "$OUTPUT/token_banks/cleanup_receipts" \
  "$OUTPUT/fcv_scores" \
  "$OUTPUT/control_scores" \
  "$OUTPUT/selection_results/oracle_scores" \
  "$OUTPUT/selection_results/final_test_scores" \
  "$OUTPUT/selection_results/candidate_pool_test_scores" \
  "$OUTPUT/selection_results/selector_scatter_plots" \
  "$OUTPUT/preflight"
cd "$REPO"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing FCV environment Python: $ENV/bin/python" >&2
  exit 1
fi
if ! "$ENV/bin/python" -c \
  "import torch, torchvision, timm, pandas, scipy, matplotlib"; then
  echo "The FCV environment is missing one or more full-study dependencies." >&2
  exit 1
fi
if [[ ! -f "$OUTPUT/split_manifests/metadata_train.csv" \
   || ! -f "$OUTPUT/split_manifests/metadata_val.csv" \
   || ! -f "$OUTPUT/split_manifests/metadata_oracle_val_analysis_only.csv" \
   || ! -f "$OUTPUT/split_manifests/metadata_test_analysis_only.csv" ]]; then
  echo "Missing one or more frozen Step 2 manifests under $OUTPUT/split_manifests." >&2
  exit 1
fi
if [[ ! -f "$OUTPUT/patch_masks/patch_masks_val.pt" ]]; then
  echo "Missing Step 3 patch masks: $OUTPUT/patch_masks/patch_masks_val.pt" >&2
  exit 1
fi

# Do not let an accidental second invocation create a duplicate campaign.
active_jobs=$(squeue --noheader --user "$USER" --format='%i %j %T' \
  | awk '$2 ~ /^fcv_vit_/ {print}')
if [[ -n "$active_jobs" && "${ALLOW_DUPLICATE_SUBMISSION:-0}" != "1" ]]; then
  echo "Existing FCV jobs are already active:" >&2
  echo "$active_jobs" >&2
  echo "Refusing a duplicate submission. Set ALLOW_DUPLICATE_SUBMISSION=1 only if intentional." >&2
  exit 1
fi

echo "Running the full-study preflight before creating the dependency chain..."
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/inspect_config.py \
  --config "$CONFIG" >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/validate_patch_masks.py \
  --config "$CONFIG" >/dev/null
"$ENV/bin/python" experiments/fcv_vit_waterbirds100/scripts/check_storage_budget.py \
  --config "$CONFIG" --stage submit_full_81_candidate_study >/dev/null
"$ENV/bin/python" -m unittest \
  experiments.fcv_vit_waterbirds100.tests.test_token_banks.TokenBanksTest.test_control_probability_statistics_use_serialized_draw_values \
  >/dev/null

submit_job() {
  local dependency=$1
  local script=$2
  local raw
  if [[ -n "$dependency" ]]; then
    raw=$(sbatch --parsable --dependency="$dependency" "$script")
  else
    raw=$(sbatch --parsable "$script")
  fi
  # Slurm may append a cluster name after a semicolon in federated setups.
  printf '%s\n' "${raw%%;*}"
}

echo "Submitting the complete 81-candidate FCV study..."

# Step 4: train 27 runs and retain epochs 5, 10, and 20.
cache_job=$(submit_job "" "$SLURM/cache_pretrained_model.sbatch")
candidate_job=$(submit_job "afterok:${cache_job}" "$SLURM/train_candidate_array.sbatch")
candidate_diag_job=$(submit_job "afterany:${candidate_job}" "$SLURM/aggregate_candidate_metrics.sbatch")
candidate_finalize_job=$(submit_job \
  "afterok:${candidate_job}:${candidate_diag_job}" \
  "$SLURM/finalize_candidate_pool.sbatch")

# Steps 6--8: reconstruction gate, shared plans, streamed FCV and controls.
reconstruction_job=$(submit_job \
  "afterok:${candidate_finalize_job}" \
  "$SLURM/verify_reconstruction_preflight.sbatch")
stream_plan_job=$(submit_job \
  "afterok:${reconstruction_job}" \
  "$SLURM/prepare_streaming_plans.sbatch")
stream_job=$(submit_job \
  "afterok:${stream_plan_job}" \
  "$SLURM/stream_score_run_array.sbatch")
fcv_diag_job=$(submit_job \
  "afterany:${stream_job}" \
  "$SLURM/aggregate_fcv_scores.sbatch")
control_diag_job=$(submit_job \
  "afterany:${stream_job}" \
  "$SLURM/aggregate_fcv_controls.sbatch")
stream_finalize_job=$(submit_job \
  "afterok:${stream_job}:${fcv_diag_job}:${control_diag_job}" \
  "$SLURM/aggregate_streaming_scores.sbatch")

# Step 9: analysis-only Oracle scoring and frozen selector construction.
oracle_job=$(submit_job \
  "afterok:${stream_finalize_job}" \
  "$SLURM/score_oracle_array.sbatch")
oracle_diag_job=$(submit_job \
  "afterany:${oracle_job}" \
  "$SLURM/aggregate_oracle_metrics.sbatch")
selection_job=$(submit_job \
  "afterok:${oracle_job}:${oracle_diag_job}" \
  "$SLURM/build_selection_table.sbatch")

# Step 10: evaluate only the frozen selected checkpoints on test.
final_test_job=$(submit_job \
  "afterok:${selection_job}" \
  "$SLURM/evaluate_selected_checkpoints.sbatch")

# Step 11: score the full pool post hoc and compute Oracle-gap closure.
pool_test_job=$(submit_job \
  "afterok:${final_test_job}" \
  "$SLURM/score_pool_test_array.sbatch")
pool_diag_job=$(submit_job \
  "afterany:${pool_test_job}" \
  "$SLURM/aggregate_pool_test_scores.sbatch")
gap_job=$(submit_job \
  "afterok:${pool_test_job}:${pool_diag_job}" \
  "$SLURM/compute_gap_closure.sbatch")

# Step 12: rank-quality analysis over the complete frozen pool.
rank_job=$(submit_job \
  "afterok:${gap_job}" \
  "$SLURM/analyze_rank_quality.sbatch")

timestamp=$(date +%Y%m%d_%H%M%S)
record="$OUTPUT/run_logs/full_study_submission_${timestamp}.txt"
commit=$(git rev-parse HEAD 2>/dev/null || printf 'UNKNOWN')
{
  printf 'submitted_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$commit"
  printf 'protocol=81 candidates (27 runs x epochs 5,10,20)\n'
  printf 'cache_job=%s\n' "$cache_job"
  printf 'candidate_job=%s\n' "$candidate_job"
  printf 'candidate_diag_job=%s\n' "$candidate_diag_job"
  printf 'candidate_finalize_job=%s\n' "$candidate_finalize_job"
  printf 'reconstruction_job=%s\n' "$reconstruction_job"
  printf 'stream_plan_job=%s\n' "$stream_plan_job"
  printf 'stream_job=%s\n' "$stream_job"
  printf 'fcv_diag_job=%s\n' "$fcv_diag_job"
  printf 'control_diag_job=%s\n' "$control_diag_job"
  printf 'stream_finalize_job=%s\n' "$stream_finalize_job"
  printf 'oracle_job=%s\n' "$oracle_job"
  printf 'oracle_diag_job=%s\n' "$oracle_diag_job"
  printf 'selection_job=%s\n' "$selection_job"
  printf 'final_test_job=%s\n' "$final_test_job"
  printf 'pool_test_job=%s\n' "$pool_test_job"
  printf 'pool_diag_job=%s\n' "$pool_diag_job"
  printf 'gap_job=%s\n' "$gap_job"
  printf 'rank_job=%s\n' "$rank_job"
} > "$record"

cat "$record"
echo
echo "[DONE] The complete study is queued as one fail-closed dependency chain."
echo "Final job: $rank_job"
echo "Submission record: $record"
echo "Monitor with: squeue --me -o '%.18i %.24j %.2t %.10M %R'"
echo "If a required stage fails, downstream afterok jobs will not run."
