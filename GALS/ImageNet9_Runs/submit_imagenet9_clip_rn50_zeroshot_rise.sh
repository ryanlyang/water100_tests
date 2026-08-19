#!/usr/bin/env bash
# Submit one deterministic CLIP RN50 zero-shot RISE job per foreground variant.
set -Eeuo pipefail

REPO="${REPO:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/data/imagenet9}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsImageNet9}"
RUN_ROOT="${RUN_ROOT:-${LOG_ROOT}/pointing_game_rise_clip_rn50_zeroshot}"
PROTOCOL="${PROTOCOL:-reconstructed_original_bbox1_v1}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-${DATA_ROOT}/official_test/bg_challenge}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:-${DATA_ROOT}/metadata/${PROTOCOL}/official_test_manifest.csv}"
MASK_ROOT="${MASK_ROOT:-${OFFICIAL_ROOT}/fg_mask/val}"
SOURCE_CONTRACT="${SOURCE_CONTRACT:-${LOG_ROOT}/clip_rn50_zeroshot_openai/evaluation_contract.json}"
CONDA_ENV="${CONDA_ENV:-r4rr-weclip}"
PYTHON_BIN="${PYTHON_BIN:-/home/ryreu/miniconda3/envs/${CONDA_ENV}/bin/python}"
RUNNER="${RUNNER:-${REPO}/ImageNet9_Runs/run_imagenet9_clip_rn50_zeroshot_rise.sbatch}"
VARIANTS="${VARIANTS:-original mixed_same mixed_rand mixed_next}"
RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_SEED="${RISE_SEED:-0}"

mkdir -p "$LOG_ROOT" "$RUN_ROOT"
cd "$REPO"
[[ -x "$PYTHON_BIN" ]] || { echo "[ERROR] Missing Python environment: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$SOURCE_CONTRACT" ]] || { echo "[ERROR] Missing CLIP-ZS contract: $SOURCE_CONTRACT" >&2; exit 2; }
[[ -f "$RUNNER" ]] || { echo "[ERROR] Missing worker: $RUNNER" >&2; exit 2; }

read -r -a variant_array <<< "$VARIANTS"
"$PYTHON_BIN" ImageNet9_Runs/audit_imagenet9_foreground_masks.py \
  --official-manifest "$OFFICIAL_MANIFEST" \
  --official-test-root "$OFFICIAL_ROOT" \
  --mask-root "$MASK_ROOT" \
  --variants "${variant_array[@]}" \
  --output-json "$RUN_ROOT/foreground_mask_audit.json"

record="$RUN_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).csv"
echo "variant,status,job_name,job_id" > "$record"

summary_is_complete() {
  local path="$1" variant="$2"
  "$PYTHON_BIN" - "$path" "$variant" "$RISE_NUM_MASKS" "$RISE_GRID_SIZE" "$RISE_P1" "$RISE_SEED" <<'PY'
import json, sys
path, variant, num_masks, grid_size, p1, rise_seed = sys.argv[1:]
try:
    row = json.load(open(path, "r", encoding="utf-8"))
    valid = (
        row.get("method") == "clip_zs_rn50"
        and int(row.get("seed", -1)) == 0
        and row.get("variant") == variant
        and row.get("transfer_source") == "none_frozen_zero_shot"
        and row.get("target_mode") == "label"
        and row.get("explainer") == "rise"
        and int(row.get("rise_num_masks", -1)) == int(num_masks)
        and int(row.get("rise_grid_size", -1)) == int(grid_size)
        and abs(float(row.get("rise_p1", -1)) - float(p1)) < 1e-12
        and int(row.get("rise_seed", -1)) == int(rise_seed)
        and int(row.get("pg_total", 0)) == 4050
        and int(row.get("errors", 1)) == 0
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

for variant in "${variant_array[@]}"; do
  case "$variant" in
    original) tag=orig ;;
    mixed_same) tag=ms ;;
    mixed_rand) tag=mr ;;
    mixed_next) tag=mn ;;
    *) echo "[ERROR] Unsupported variant: $variant" >&2; exit 2 ;;
  esac
  result="$RUN_ROOT/clip_zs_rn50/$variant/seed_0/pointing_game_summary.json"
  name="in9pg_zsr50_${tag}"
  if summary_is_complete "$result" "$variant"; then
    echo "[SKIP] complete variant=$variant"
    echo "$variant,COMPLETE,$name,SKIPPED" >> "$record"
  elif squeue -h -u "$USER" -n "$name" | grep -q .; then
    ids="$(squeue -h -u "$USER" -n "$name" -o '%A' | paste -sd ';' -)"
    echo "[SKIP] queued variant=$variant jobs=$ids"
    echo "$variant,QUEUED,$name,$ids" >> "$record"
  else
    output="$(sbatch --parsable \
      --job-name="$name" \
      --export="ALL,VARIANT=${variant},RUN_ROOT=${RUN_ROOT},SOURCE_CONTRACT=${SOURCE_CONTRACT},CONDA_ENV=${CONDA_ENV},PYTHON_BIN=${PYTHON_BIN},RISE_NUM_MASKS=${RISE_NUM_MASKS},RISE_GRID_SIZE=${RISE_GRID_SIZE},RISE_P1=${RISE_P1},RISE_SEED=${RISE_SEED}" \
      "$RUNNER")"
    id="${output%%;*}"
    echo "[SUBMITTED] variant=$variant job=$id"
    echo "$variant,SUBMITTED,$name,$id" >> "$record"
  fi
done

echo "[DONE] submission record: $record"
echo "[INFO] Deterministic outputs: $RUN_ROOT/clip_zs_rn50/<variant>/seed_0"
echo "[INFO] Re-running this command resumes partial evaluations and skips completed variants."
echo "[INFO] Summarize with:"
echo "  $PYTHON_BIN ImageNet9_Runs/summarize_imagenet9_clip_rn50_zeroshot_rise.py --run-root $RUN_ROOT"
