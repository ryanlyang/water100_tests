#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUNNER="$REPO/ImageNet9_Runs/run_imagenet9_final_teacher_5seed.sbatch"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
VARIANTS=(
  r4rr
  r4rr_reverse_kl
  r4rr_jensen_shannon
  r4rr_squared_l2
  r4rr_cosine
  r4rr_trial13
)

mkdir -p "$LOG_ROOT"
cd "$REPO"
record="$LOG_ROOT/submitted_imagenet9_final_r4rr_klincr0_$(date +%Y%m%d_%H%M%S).csv"
echo "variant,output_variant,job_name,job_id,status" > "$record"

for variant in "${VARIANTS[@]}"; do
  sweep_dir="$variant"
  case "$variant" in
    r4rr|r4rr_trial13) sweep_dir="r4rr" ;;
  esac
  summary="$LOG_ROOT/sweeps/$sweep_dir/main/summary.json"
  output_variant="${variant}_klincr0"
  final_summary="$LOG_ROOT/final/$output_variant/main/summary.json"

  if [[ ! -f "$summary" ]] || ! "$PYTHON_BIN" - "$summary" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
complete = int(s.get("complete_trials", 0))
target = int(s.get("target_complete_trials", 50))
raise SystemExit(0 if complete >= target else 1)
PY
  then
    echo "$variant,$output_variant,NONE,NONE,SWEEP_INCOMPLETE" >> "$record"
    echo "[SKIP] sweep has not reached its target: $variant"
    continue
  fi

  if [[ -f "$final_summary" ]] && "$PYTHON_BIN" - "$final_summary" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
raise SystemExit(0 if int(s.get("n", 0)) == 5 else 1)
PY
  then
    echo "$variant,$output_variant,NONE,NONE,FINAL_COMPLETE" >> "$record"
    echo "[SKIP] zero-increment final already has five seeds: $variant"
    continue
  fi

  job_name="in9f0_${variant}"
  if [[ "${FORCE_SUBMIT:-0}" != "1" ]] && \
     squeue -h -u "$USER" -n "$job_name" | grep -q .; then
    echo "$variant,$output_variant,$job_name,ALREADY_QUEUED,ALREADY_QUEUED" >> "$record"
    echo "[SKIP] already queued: $job_name"
    continue
  fi
  output="$(sbatch --parsable \
    --job-name="$job_name" \
    --export="ALL,VARIANT=${variant},KL_INCREMENT=0" \
    "$RUNNER")"
  job_id="${output%%;*}"
  echo "$variant,$output_variant,$job_name,$job_id,SUBMITTED" >> "$record"
  echo "[SUBMITTED] variant=$variant output=$output_variant job=$job_id"
done

echo "[DONE] submission record: $record"
echo "[INFO] Stable outputs: $LOG_ROOT/final/<variant>_klincr0/main"
echo "[INFO] These finals transfer ramp-selected hyperparameters to kl_increment=0."
