#!/usr/bin/env bash
set -Eeuo pipefail

# Submit the full MobileNetV2 experiment set:
#   guided KL / R4RR MobileNetV2: Waterbirds-95, Waterbirds-100, RedMeat
#   vanilla MobileNetV2:          Waterbirds-95, Waterbirds-100, RedMeat
#
# Run from the GALS/ directory on RC:
#   bash submit_mobilenetv2_r4rr_sweeps.sh
#
# Useful overrides:
#   N_TRIALS=50
#   SEED_LIST="0 1 2 3 4"
#   SBATCH_TIME=8-00:00:00
#   DRY_RUN=1
#
# Skips:
#   SKIP_GUIDED=1
#   SKIP_VANILLA=1
#   SKIP_WATERBIRDS=1
#   SKIP_REDMEAT=1

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] Missing command: $1" >&2
    exit 2
  }
}

if [[ "${DRY_RUN:-0}" -ne 1 ]]; then
  need_cmd sbatch
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N_TRIALS=${N_TRIALS:-50}
SEED_LIST=${SEED_LIST:-"0 1 2 3 4"}
SBATCH_TIME=${SBATCH_TIME:-8-00:00:00}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}

export N_TRIALS
export SEED_LIST
export_vars="ALL"

submit_job() {
  local label="$1"
  local script="$2"
  shift 2

  if [[ ! -f "$script" ]]; then
    echo "[ERROR] Missing runner for $label: $script" >&2
    exit 2
  fi

  local args=(--parsable --time="$SBATCH_TIME" --export="$export_vars")
  if [[ -n "$SBATCH_PARTITION" ]]; then
    args+=(--partition="$SBATCH_PARTITION")
  fi
  if [[ -n "$SBATCH_ACCOUNT" ]]; then
    args+=(--account="$SBATCH_ACCOUNT")
  fi
  args+=("$@" "$script")

  if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
    printf '[DRY-RUN] %-28s sbatch' "$label"
    printf ' %q' "${args[@]}"
    printf '\n'
    return
  fi

  local jid
  jid="$(sbatch "${args[@]}")"
  printf '[SUBMIT] %-28s %s\n' "$label" "$jid"
}

echo "[INFO] repo=$SCRIPT_DIR"
echo "[INFO] N_TRIALS=$N_TRIALS"
echo "[INFO] SEED_LIST=$SEED_LIST"
echo "[INFO] SBATCH_TIME=$SBATCH_TIME"
if [[ -n "$SBATCH_PARTITION" ]]; then echo "[INFO] SBATCH_PARTITION=$SBATCH_PARTITION"; fi
if [[ -n "$SBATCH_ACCOUNT" ]]; then echo "[INFO] SBATCH_ACCOUNT=$SBATCH_ACCOUNT"; fi

echo "=============================="
echo "[SUBMIT] MobileNetV2 sweeps"
echo "=============================="

if [[ "${SKIP_GUIDED:-0}" -ne 1 ]]; then
  if [[ "${SKIP_WATERBIRDS:-0}" -ne 1 ]]; then
    submit_job "guided_wb95_mobilenetv2" "run_guided_waterbirds95_mobilenetv2_sweep.sh"
    submit_job "guided_wb100_mobilenetv2" "run_guided_waterbirds100_mobilenetv2_sweep.sh"
  fi
  if [[ "${SKIP_REDMEAT:-0}" -ne 1 ]]; then
    submit_job "guided_redmeat_mobilenetv2" "RedMeat_Runs/run_guided_redmeat_mobilenetv2_sweep.sh"
  fi
else
  echo "[SKIP] guided MobileNetV2 sweeps"
fi

if [[ "${SKIP_VANILLA:-0}" -ne 1 ]]; then
  if [[ "${SKIP_WATERBIRDS:-0}" -ne 1 ]]; then
    submit_job "vanilla_wb95_mobilenetv2" "run_waterbirds95_vanilla_mobilenetv2_sweep.sh"
    submit_job "vanilla_wb100_mobilenetv2" "run_waterbirds100_vanilla_mobilenetv2_sweep.sh"
  fi
  if [[ "${SKIP_REDMEAT:-0}" -ne 1 ]]; then
    submit_job "vanilla_redmeat_mobilenetv2" "RedMeat_Runs/run_redmeat_vanilla_mobilenetv2_sweep_optuna.sh"
  fi
else
  echo "[SKIP] vanilla MobileNetV2 sweeps"
fi

echo "[DONE] Submission commands issued."
