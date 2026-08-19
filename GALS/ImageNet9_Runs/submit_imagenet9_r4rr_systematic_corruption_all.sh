#!/usr/bin/env bash
# Submit nine class conditions and one exactly count-matched random control.
set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
STUDY_ROOT="${STUDY_ROOT:-${LOG_ROOT}/r4rr_systematic_teacher_corruption/imagenet9}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL="${PROTOCOL:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL}/manifest.csv}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${STUDY_ROOT}/corruption_manifests}"
TEACHER_MAP_ROOT="${TEACHER_MAP_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1/inference/full/val/prediction_cmap}"
SWEEP_SUMMARY="${SWEEP_SUMMARY:-${LOG_ROOT}/sweeps/r4rr/main/summary.json}"
RUNNER="${RUNNER:-${REPO}/ImageNet9_Runs/run_imagenet9_r4rr_systematic_corruption_condition.sbatch}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
CORRUPTION_SEED="${CORRUPTION_SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONDITIONS=(
  class_dog class_bird class_vehicle class_reptile class_carnivore
  class_insect class_instrument class_primate class_fish random_matched
)

mkdir -p "$LOG_ROOT" "$STUDY_ROOT" "$MANIFEST_ROOT"
cd "$REPO"
[[ -f "$RUNNER" ]] || { echo "[ERROR] Missing worker: $RUNNER" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "[ERROR] Missing ImageNet-9 manifest: $MANIFEST" >&2; exit 2; }
[[ -f "$SWEEP_SUMMARY" ]] || { echo "[ERROR] Missing R4RR sweep summary: $SWEEP_SUMMARY" >&2; exit 2; }
[[ -d "$TEACHER_MAP_ROOT" ]] || { echo "[ERROR] Missing teacher maps: $TEACHER_MAP_ROOT" >&2; exit 2; }

"$PYTHON_BIN" ImageNet9_Runs/imagenet9_systematic_corruption.py \
  --manifest "$MANIFEST" \
  --output-root "$MANIFEST_ROOT" \
  --conditions "${CONDITIONS[@]}" \
  --corruption-seed "$CORRUPTION_SEED"

record="$STUDY_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
echo "condition,status,job_name,job_id" > "$record"

condition_complete() {
  local condition="$1" path="$STUDY_ROOT/$condition/corruption_summary.json"
  "$PYTHON_BIN" - "$path" "$condition" "$CORRUPTION_SEED" <<'PY'
import json, sys
path, condition, corruption_seed = sys.argv[1:]
try:
    row = json.load(open(path, "r", encoding="utf-8"))
    valid = (
        row.get("protocol_version") == 1
        and row.get("dataset") == "imagenet9"
        and row.get("condition") == condition
        and int(row.get("corruption_seed", -1)) == int(corruption_seed)
        and int(row.get("corrupted_example_count", -1)) == 5045
        and row.get("completed_seeds") == [0, 1, 2, 3, 4]
        and int(row.get("n_completed", -1)) == 5
        and float(row.get("kl_increment", -1)) == 0.0
        and row.get("official_variants_used_for_selection") is False
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

job_name() {
  case "$1" in
    class_dog) echo in9c_dog ;;
    class_bird) echo in9c_bird ;;
    class_vehicle) echo in9c_vehicle ;;
    class_reptile) echo in9c_reptile ;;
    class_carnivore) echo in9c_carn ;;
    class_insect) echo in9c_insect ;;
    class_instrument) echo in9c_instr ;;
    class_primate) echo in9c_primate ;;
    class_fish) echo in9c_fish ;;
    random_matched) echo in9c_random ;;
  esac
}

for condition in "${CONDITIONS[@]}"; do
  name="$(job_name "$condition")"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] condition=$condition job=$name"
    echo "$condition,DRY_RUN,$name,DRY_RUN" >> "$record"
  elif condition_complete "$condition"; then
    echo "[SKIP] complete condition=$condition"
    echo "$condition,COMPLETE,$name,SKIPPED" >> "$record"
  elif [[ "${FORCE_SUBMIT:-0}" != "1" ]] && squeue -h -u "$USER" -n "$name" | grep -q .; then
    ids="$(squeue -h -u "$USER" -n "$name" -o '%A' | paste -sd ';' -)"
    echo "[SKIP] queued condition=$condition jobs=$ids"
    echo "$condition,QUEUED,$name,$ids" >> "$record"
  else
    output="$(sbatch --parsable \
      --job-name="$name" \
      --export="ALL,CONDITION=${condition},STUDY_ROOT=${STUDY_ROOT},DATA_ROOT=${DATA_ROOT},MANIFEST=${MANIFEST},MANIFEST_ROOT=${MANIFEST_ROOT},TEACHER_MAP_ROOT=${TEACHER_MAP_ROOT},SWEEP_SUMMARY=${SWEEP_SUMMARY},CORRUPTION_SEED=${CORRUPTION_SEED}" \
      "$RUNNER")"
    id="${output%%;*}"
    echo "[SUBMITTED] condition=$condition job=$id"
    echo "$condition,SUBMITTED,$name,$id" >> "$record"
  fi
done

echo "[DONE] submission record: $record"
echo "[INFO] Stable output root: $STUDY_ROOT"
echo "[INFO] Each job resumes from completed seed artifacts."
echo "[INFO] Aggregate after completion with:"
echo "  $PYTHON_BIN ImageNet9_Runs/summarize_imagenet9_r4rr_systematic_corruption.py --run-root $STUDY_ROOT"
