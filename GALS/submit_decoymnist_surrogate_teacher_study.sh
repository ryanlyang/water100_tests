#!/bin/bash
# Submit surrogate-data preparation, clean R4RR, and all corruption conditions.

set -Eeuo pipefail

# Avoid Slurm's unreliable submit-node environment reconstruction. Each worker
# activates the project environment itself.
unset SBATCH_GET_USER_ENV SBATCH_EXPORT SBATCH_EXPORT_FILE
export SLURM_EXPORT_ENV=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREP_WORKER="${PREP_WORKER:-$SCRIPT_DIR/run_prepare_decoymnist_surrogate_teacher_data.sh}"
TRAIN_WORKER="${TRAIN_WORKER:-$SCRIPT_DIR/run_decoymnist_r4rr_systematic_corruption_condition.sh}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
SURROGATE_ROOT="${SURROGATE_ROOT:-/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyMNIST_surrogate_oracle_v1}"
PNG_ROOT="${PNG_ROOT:-$SURROGATE_ROOT/DecoyMNIST_png}"
TEACHER_MAP_PATH="${TEACHER_MAP_PATH:-$SURROGATE_ROOT/teacher_maps/prediction_cmap}"
RUN_ROOT="${RUN_ROOT:-$LOG_DIR/r4rr_surrogate_teacher_study/decoymnist}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$RUN_ROOT/corruption_manifests}"
CORRUPTION_SEED="${CORRUPTION_SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"

CONDITIONS=(
  clean
  random_10pct
  digit_0
  digit_1
  digit_2
  digit_3
  digit_4
  digit_5
  digit_6
  digit_7
  digit_8
  digit_9
)

[[ -f "$PREP_WORKER" ]] || { echo "[ERROR] Missing preparation worker: $PREP_WORKER" >&2; exit 2; }
[[ -f "$TRAIN_WORKER" ]] || { echo "[ERROR] Missing training worker: $TRAIN_WORKER" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT" "$MANIFEST_ROOT"

submission_file="$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
printf 'stage,condition,status,job_id,dependency\n' > "$submission_file"

preparation_is_complete() {
  python3 - "$SURROGATE_ROOT/metadata/completion_manifest.json" \
    "$PNG_ROOT" "$TEACHER_MAP_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, png_root_raw, teacher_root_raw = sys.argv[1:]
png_root = Path(png_root_raw)
teacher_root = Path(teacher_root_raw)
try:
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    contract = manifest["contract"]
    counts = manifest["counts"]
    valid = (
        manifest.get("status") == "complete"
        and int(manifest.get("protocol_version", -1)) == 1
        and contract.get("dataset_variant") == "CDEP_5x5_label_code_reversed_test"
        and int(contract.get("dataset_seed", -1)) == 0
        and contract.get("teacher_source") == "oracle_clean_torchvision_mnist_foreground"
        and counts == {"train_images": 60000, "test_images": 10000, "teacher_maps": 60000}
        and (png_root / "train").is_dir()
        and (png_root / "test").is_dir()
        and teacher_root.is_dir()
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

condition_is_complete() {
  local condition="$1"
  python3 - "$RUN_ROOT/$condition/summary.json" \
    "$RUN_ROOT/$condition/run_contract.json" "$condition" "$CORRUPTION_SEED" \
    "$PNG_ROOT" "$TEACHER_MAP_PATH" <<'PY'
import json
import sys
from pathlib import Path

summary_path, contract_path, condition, corruption_seed, png_root, teacher_root = sys.argv[1:]
try:
    summary = json.load(open(summary_path, "r", encoding="utf-8"))
    contract = json.load(open(contract_path, "r", encoding="utf-8"))
    valid = (
        summary.get("protocol_version") == 1
        and summary.get("dataset") == "decoymnist"
        and summary.get("condition") == condition
        and int(summary.get("corruption_seed", -1)) == int(corruption_seed)
        and summary.get("completed_seeds") == [0, 1, 2, 3, 4]
        and int(summary.get("n_completed", -1)) == 5
        and Path(contract.get("png_root", "")).resolve() == Path(png_root).resolve()
        and Path(contract.get("teacher_map_path", "")).resolve() == Path(teacher_root).resolve()
        and int(contract.get("epochs", -1)) == 19
        and float(contract.get("learning_rate", -1)) == 1e-3
        and int(contract.get("attention_epoch", -1)) == 7
        and float(contract.get("kl_lambda", -1)) == 495.61
        and float(contract.get("kl_increment", -1)) == 0.0
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

job_name_for_condition() {
  local condition="$1"
  case "$condition" in
    clean) printf 'r4s_dec_clean' ;;
    random_10pct) printf 'r4s_dec_rand' ;;
    digit_[0-9]) printf 'r4s_dec_d%s' "${condition#digit_}" ;;
    *) return 2 ;;
  esac
}

prep_job_id=""
dependency_args=()
if preparation_is_complete; then
  echo "[SKIP-COMPLETE] surrogate dataset and maps are already prepared"
  printf 'prepare,,COMPLETE,,\n' >> "$submission_file"
elif [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY RUN] sbatch --export=ALL,OUTPUT_ROOT=$SURROGATE_ROOT $PREP_WORKER"
  prep_job_id="PREP_JOB_ID"
  dependency_args=(--dependency="afterok:$prep_job_id")
  printf 'prepare,,DRY_RUN,%s,\n' "$prep_job_id" >> "$submission_file"
else
  queued_prep="$(squeue -h -u "$USER" -n prep_decoy_surrogate -o '%A' | head -n 1)"
  if [[ -n "$queued_prep" ]]; then
    prep_job_id="$queued_prep"
    echo "[REUSE-QUEUED] preparation job=$prep_job_id"
    prep_status="QUEUED"
  else
    prep_job_id="$(sbatch --parsable \
      --export="ALL,OUTPUT_ROOT=$SURROGATE_ROOT" \
      "$PREP_WORKER")"
    echo "[SUBMITTED] preparation job=$prep_job_id"
    prep_status="SUBMITTED"
  fi
  dependency_args=(--dependency="afterok:$prep_job_id")
  printf 'prepare,,%s,%s,\n' "$prep_status" "$prep_job_id" >> "$submission_file"
fi

for condition in "${CONDITIONS[@]}"; do
  job_name="$(job_name_for_condition "$condition")"
  dependency_text="${prep_job_id:+afterok:$prep_job_id}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] sbatch ${dependency_args[*]} --job-name=$job_name condition=$condition"
    status="DRY_RUN"
    job_id="DRY_RUN"
  elif condition_is_complete "$condition"; then
    echo "[SKIP-COMPLETE] condition=$condition"
    status="COMPLETE"
    job_id=""
  else
    queued_job_ids="$(squeue -h -u "$USER" -n "$job_name" -o '%A' | paste -sd ';' -)"
    if [[ -n "$queued_job_ids" ]]; then
      echo "[SKIP-QUEUED] condition=$condition jobs=$queued_job_ids"
      status="QUEUED"
      job_id="$queued_job_ids"
    else
      job_id="$(sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="$job_name" \
        --export="ALL,CONDITION=$condition,PNG_ROOT=$PNG_ROOT,TEACHER_MAP_PATH=$TEACHER_MAP_PATH,RUN_ROOT=$RUN_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT,CORRUPTION_SEED=$CORRUPTION_SEED" \
        "$TRAIN_WORKER")"
      echo "[SUBMITTED] condition=$condition job=$job_id dependency=${dependency_text:-none}"
      status="SUBMITTED"
    fi
  fi
  printf 'train,%s,%s,%s,%s\n' \
    "$condition" "$status" "$job_id" "$dependency_text" >> "$submission_file"
done

echo
echo "[DONE] submission record: $submission_file"
echo "[DONE] surrogate root: $SURROGATE_ROOT"
echo "[DONE] result root: $RUN_ROOT"
echo "[INFO] clean and every corruption condition run optimized R4RR over seeds 0-4."
echo "[INFO] Aggregate completed results with:"
echo "  cd $SCRIPT_DIR"
echo "  python summarize_decoymnist_r4rr_systematic_corruption.py --include-clean --run-root $RUN_ROOT"
