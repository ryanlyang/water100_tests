#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUNNER="$REPO/ImageNet9_Runs/run_imagenet9_final_teacher_5seed.sbatch"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
VARIANTS="${VARIANTS:-gals gals_gradcam gals_abn gals_rn50 r4rr r4rr_reverse_kl r4rr_jensen_shannon r4rr_squared_l2 r4rr_cosine}"

mkdir -p "$LOG_ROOT"
cd "$REPO"
read -r -a variant_array <<< "$VARIANTS"

record="$LOG_ROOT/submitted_imagenet9_final_teacher_5seed_$(date +%Y%m%d_%H%M%S).csv"
echo "variant,job_name,job_id,status" > "$record"

for variant in "${variant_array[@]}"; do
  sweep_dir="$variant"
  [[ "$variant" == "r4rr" ]] && sweep_dir="r4rr"
  summary="$LOG_ROOT/sweeps/$sweep_dir/main/summary.json"
  final_summary="$LOG_ROOT/final/$variant/main/summary.json"

  if [[ -f "$final_summary" ]] && "$PYTHON_BIN" - "$final_summary" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
raise SystemExit(0 if int(s.get("n", 0)) == 5 else 1)
PY
  then
    echo "$variant,NONE,NONE,FINAL_COMPLETE" >> "$record"
    echo "[SKIP] final result already has five seeds: $variant"
    continue
  fi

  if [[ ! -f "$summary" ]] || ! "$PYTHON_BIN" - "$summary" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
complete = int(s.get("complete_trials", 0))
target = int(s.get("target_complete_trials", 50))
raise SystemExit(0 if complete >= target else 1)
PY
  then
    echo "$variant,NONE,NONE,SWEEP_INCOMPLETE" >> "$record"
    echo "[SKIP] sweep has not reached its target: $variant"
    continue
  fi

  job_name="in9ft_${variant}"
  if [[ "${FORCE_SUBMIT:-0}" != "1" ]] && \
     squeue -h -u "$USER" -n "$job_name" | grep -q .; then
    echo "$variant,$job_name,ALREADY_QUEUED,ALREADY_QUEUED" >> "$record"
    echo "[SKIP] already queued: $job_name"
    continue
  fi
  output="$(sbatch --parsable \
    --job-name="$job_name" \
    --export="ALL,VARIANT=${variant}" \
    "$RUNNER")"
  job_id="${output%%;*}"
  echo "$variant,$job_name,$job_id,SUBMITTED" >> "$record"
  echo "[SUBMITTED] variant=$variant job=$job_id"
done

echo "[DONE] submission record: $record"
echo "[INFO] Stable outputs: $LOG_ROOT/final/<variant>/main"
echo "[INFO] Re-running this submitter skips completed finals and incomplete sweeps."
