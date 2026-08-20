#!/usr/bin/env bash
# Submit nine class corruptions and one matched random control with WB95 params.
set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
PROTOCOL="${PROTOCOL:-reconstructed_original_bbox1_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL}/manifest.csv}"
TEACHER_MAP_ROOT="${TEACHER_MAP_ROOT:-${DATA_ROOT}/r4rr_teacher/weclipplus_clip_dino_v1/inference/full/val/prediction_cmap}"
TRANSFER_CONFIG="${TRANSFER_CONFIG:-${REPO}/ImageNet9_Runs/configs/waterbirds95_hparam_transfer.yaml}"
BASE_STUDY_ROOT="${BASE_STUDY_ROOT:-${LOG_ROOT}/r4rr_systematic_teacher_corruption}"
STUDY_ROOT="${STUDY_ROOT:-${BASE_STUDY_ROOT}/imagenet9_wb95_transfer}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${BASE_STUDY_ROOT}/imagenet9/corruption_manifests}"
RUNNER="${RUNNER:-${REPO}/ImageNet9_Runs/run_imagenet9_r4rr_wb95_systematic_corruption_condition.sbatch}"
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
[[ -f "$TRANSFER_CONFIG" ]] || { echo "[ERROR] Missing transfer config: $TRANSFER_CONFIG" >&2; exit 2; }
[[ -d "$TEACHER_MAP_ROOT" ]] || { echo "[ERROR] Missing teacher maps: $TEACHER_MAP_ROOT" >&2; exit 2; }

# This is idempotent and verifies that the transfer study uses the same
# corruption selections and checksums as the completed ImageNet-tuned study.
"$PYTHON_BIN" ImageNet9_Runs/imagenet9_systematic_corruption.py \
  --manifest "$MANIFEST" \
  --output-root "$MANIFEST_ROOT" \
  --conditions "${CONDITIONS[@]}" \
  --corruption-seed "$CORRUPTION_SEED"

record="$STUDY_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
echo "condition,status,job_name,job_id" > "$record"

condition_complete() {
  local condition="$1" path="$STUDY_ROOT/$condition/corruption_summary.json"
  "$PYTHON_BIN" - "$path" "$condition" "$CORRUPTION_SEED" "$TRANSFER_CONFIG" <<'PY'
import hashlib, json, sys
path, condition, corruption_seed, config = sys.argv[1:]
digest = hashlib.sha256(open(config, "rb").read()).hexdigest()
try:
    row = json.load(open(path, "r", encoding="utf-8"))
    valid = (
        row.get("protocol_version") == 1
        and row.get("dataset") == "imagenet9"
        and row.get("condition") == condition
        and row.get("hyperparameter_protocol") == "waterbirds95_transfer"
        and row.get("hyperparameter_selection") == "waterbirds95_validation_transfer"
        and row.get("source_transfer_config_sha256") == digest
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
    class_dog) echo in9c95_dog ;;
    class_bird) echo in9c95_bird ;;
    class_vehicle) echo in9c95_vehicle ;;
    class_reptile) echo in9c95_reptile ;;
    class_carnivore) echo in9c95_carn ;;
    class_insect) echo in9c95_insect ;;
    class_instrument) echo in9c95_instr ;;
    class_primate) echo in9c95_primate ;;
    class_fish) echo in9c95_fish ;;
    random_matched) echo in9c95_random ;;
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
      --export="ALL,CONDITION=${condition},STUDY_ROOT=${STUDY_ROOT},DATA_ROOT=${DATA_ROOT},MANIFEST=${MANIFEST},MANIFEST_ROOT=${MANIFEST_ROOT},TEACHER_MAP_ROOT=${TEACHER_MAP_ROOT},TRANSFER_CONFIG=${TRANSFER_CONFIG},CORRUPTION_SEED=${CORRUPTION_SEED}" \
      "$RUNNER")"
    id="${output%%;*}"
    echo "[SUBMITTED] condition=$condition job=$id"
    echo "$condition,SUBMITTED,$name,$id" >> "$record"
  fi
done

echo "[DONE] submission record: $record"
echo "[INFO] Stable output root: $STUDY_ROOT"
echo "[INFO] Shared corruption manifests: $MANIFEST_ROOT"
echo "[INFO] Each job resumes from completed seed artifacts."
echo "[INFO] Aggregate after completion with:"
echo "  $PYTHON_BIN ImageNet9_Runs/summarize_imagenet9_r4rr_systematic_corruption.py --run-root $STUDY_ROOT"
