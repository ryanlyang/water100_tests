#!/usr/bin/env bash
# Submit one resumable job per WB95-transfer method, seed, and requested variant.
set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUN_ROOT="${RUN_ROOT:-${LOG_ROOT}/pointing_game_rise_wb95_transfer}"
SOURCE_ROOT="${SOURCE_ROOT:-${LOG_ROOT}/transfer/waterbirds95}"
PROTOCOL="${PROTOCOL:-reconstructed_original_bbox1_v1}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-${DATA_ROOT}/official_test/bg_challenge}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL}/official_test_manifest.csv}"
MASK_ROOT="${MASK_ROOT:-${OFFICIAL_ROOT}/fg_mask/val}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/gals_a100/bin/python}"
RUNNER="$REPO/ImageNet9_Runs/run_imagenet9_wb95_transfer_rise.sbatch"
METHODS="${METHODS:-erm upweight abn elrep gals afr clip_lr r4rr}"
SEEDS="${SEEDS:-0 1 2 3 4}"
# Primary protocol by default. Set VARIANTS to add shifted-background results.
VARIANTS="${VARIANTS:-original}"
RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"

mkdir -p "$LOG_ROOT" "$RUN_ROOT"
cd "$REPO"
read -r -a method_array <<< "$METHODS"
read -r -a seed_array <<< "$SEEDS"
read -r -a variant_array <<< "$VARIANTS"

variant_token="$(IFS=-; echo "${variant_array[*]}")"
audit="$RUN_ROOT/foreground_mask_audit_${variant_token}.json"
"$PYTHON_BIN" ImageNet9_Runs/audit_imagenet9_foreground_masks.py \
  --official-manifest "$OFFICIAL_MANIFEST" \
  --official-test-root "$OFFICIAL_ROOT" \
  --mask-root "$MASK_ROOT" \
  --variants "${variant_array[@]}" \
  --output-json "$audit"

record="$LOG_ROOT/submitted_imagenet9_wb95_transfer_rise_$(date +%Y%m%d_%H%M%S).csv"
echo "method,seed,variant,job_name,job_id" > "$record"

summary_is_complete() {
  local path="$1" method="$2" seed="$3" variant="$4" num_masks="$5" grid_size="$6" p1="$7" rise_seed="$8"
  "$PYTHON_BIN" - "$path" "$method" "$seed" "$variant" "$num_masks" "$grid_size" "$p1" "$rise_seed" <<'PY'
import json, sys
path, method, seed, variant, num_masks, grid_size, p1, rise_seed = sys.argv[1:]
try:
    row = json.load(open(path, "r", encoding="utf-8"))
    valid = (
        row.get("dataset") == "imagenet9"
        and row.get("transfer_source") == "waterbirds95"
        and row.get("method") == method
        and int(row.get("seed", -1)) == int(seed)
        and row.get("variant") == variant
        and row.get("target_mode") == "label"
        and row.get("explainer") == "rise"
        and row.get("mask_source") == "backgrounds_challenge_fg_mask"
        and int(row.get("rise_num_masks", -1)) == int(num_masks)
        and int(row.get("rise_grid_size", -1)) == int(grid_size)
        and abs(float(row.get("rise_p1", -1)) - float(p1)) < 1e-12
        and int(row.get("rise_seed", -1)) == int(rise_seed)
        and int(row.get("pg_total", 0)) == 4050
        and int(row.get("errors", 1)) == 0
        and int(row.get("missing_images", 1)) == 0
        and int(row.get("missing_masks", 1)) == 0
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

for variant in "${variant_array[@]}"; do
  case "$variant" in
    original) variant_tag="orig" ;;
    mixed_same) variant_tag="ms" ;;
    mixed_rand) variant_tag="mr" ;;
    mixed_next) variant_tag="mn" ;;
    *) echo "[ERROR] Unsupported variant: $variant" >&2; exit 2 ;;
  esac
  for method in "${method_array[@]}"; do
    case "$method" in
      erm|upweight|abn|elrep|gals|afr|clip_lr|r4rr) ;;
      *) echo "[ERROR] Unsupported method: $method" >&2; exit 2 ;;
    esac
    for seed in "${seed_array[@]}"; do
      source_json="$SOURCE_ROOT/$method/main/seed_${seed}/official_evaluation.json"
      [[ -f "$source_json" ]] || { echo "[ERROR] Missing source result: $source_json" >&2; exit 2; }
      result_json="$RUN_ROOT/$method/$variant/seed_${seed}/pointing_game_summary.json"
      if [[ "${FORCE_SUBMIT:-0}" != "1" ]] && \
         summary_is_complete "$result_json" "$method" "$seed" "$variant" \
           "$RISE_NUM_MASKS" "$RISE_GRID_SIZE" "$RISE_P1" "$RISE_SEED"; then
        echo "$method,$seed,$variant,COMPLETE,SKIPPED" >> "$record"
        echo "[SKIP] complete method=$method seed=$seed variant=$variant"
        continue
      fi
      job_name="in9pg_${method}_${variant_tag}_s${seed}"
      if [[ "${FORCE_SUBMIT:-0}" != "1" ]] && squeue -h -u "$USER" -n "$job_name" | grep -q .; then
        echo "$method,$seed,$variant,$job_name,ALREADY_QUEUED" >> "$record"
        echo "[SKIP] already queued: $job_name"
        continue
      fi
      output="$(sbatch --parsable \
        --job-name="$job_name" \
        --export="ALL,METHOD=${method},SEED=${seed},VARIANT=${variant},RUN_ROOT=${RUN_ROOT},SOURCE_ROOT=${SOURCE_ROOT},OFFICIAL_ROOT=${OFFICIAL_ROOT},OFFICIAL_MANIFEST=${OFFICIAL_MANIFEST},MASK_ROOT=${MASK_ROOT},PYTHON_BIN=${PYTHON_BIN},RISE_NUM_MASKS=${RISE_NUM_MASKS},RISE_GRID_SIZE=${RISE_GRID_SIZE},RISE_P1=${RISE_P1},RISE_SEED=${RISE_SEED}" \
        "$RUNNER")"
      job_id="${output%%;*}"
      echo "$method,$seed,$variant,$job_name,$job_id" >> "$record"
      echo "[SUBMITTED] method=$method seed=$seed variant=$variant job=$job_id"
    done
  done
done

echo "[DONE] submission record: $record"
echo "[INFO] Stable output root: $RUN_ROOT"
echo "[INFO] Re-running this command resumes partial jobs and skips completed jobs."
echo "[INFO] Summarize after completion with:"
echo "  $PYTHON_BIN ImageNet9_Runs/summarize_imagenet9_wb95_transfer_rise.py --run-root $RUN_ROOT --variants $(IFS=,; echo "${variant_array[*]}")"
