#!/usr/bin/env bash
set -Eeuo pipefail

# Submits (optionally) ViT attention generation, then queues all GALS ViT sweeps:
#   - RRR (gradient_outside) for WB95 + WB100
#   - GradCAM for WB95 + WB100
#   - ABN (attention supervision) for WB95 + WB100
#
# By default, it will submit the generator job *only if* the attention folders
# are missing. You can force generation with FORCE_GEN=1 or skip with SKIP_GEN=1.
#
# Example:
#   bash submit_vit_attention_then_all_gals_vit_method_sweeps.sh
#   GEN_JOB_ID=21038942 bash submit_vit_attention_then_all_gals_vit_method_sweeps.sh
#   FORCE_GEN=1 bash submit_vit_attention_then_all_gals_vit_method_sweeps.sh
#   SKIP_GEN=1  bash submit_vit_attention_then_all_gals_vit_method_sweeps.sh

DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds}
WB95_DIR=${WB95_DIR:-waterbird_complete95_forest2water2}
WB100_DIR=${WB100_DIR:-waterbird_1.0_forest2water2}
ATT_DIR=${ATT_DIR:-clip_vit_attention}

need_gen=0
if [[ -n "${GEN_JOB_ID:-}" ]]; then
  need_gen=0
elif [[ "${SKIP_GEN:-0}" -eq 1 ]]; then
  need_gen=0
elif [[ "${FORCE_GEN:-0}" -eq 1 ]]; then
  need_gen=1
else
  [[ -d "${DATA_ROOT}/${WB95_DIR}/${ATT_DIR}" ]] || need_gen=1
  [[ -d "${DATA_ROOT}/${WB100_DIR}/${ATT_DIR}" ]] || need_gen=1
fi

dep=()
if [[ -n "${GEN_JOB_ID:-}" ]]; then
  echo "[SUBMIT] Using existing generator job id: ${GEN_JOB_ID}"
  dep=(--dependency="afterok:${GEN_JOB_ID}")
elif [[ "$need_gen" -eq 1 ]]; then
  echo "[SUBMIT] ViT attention maps missing (or forced). Submitting generator..."
  gen_jid="$(sbatch --parsable run_generate_waterbirds_vit_attentions_95_100.sh)"
  echo "[SUBMIT] Generator job id: ${gen_jid}"
  dep=(--dependency="afterok:${gen_jid}")
else
  echo "[SUBMIT] Found existing ViT attention dirs:"
  echo "  - ${DATA_ROOT}/${WB95_DIR}/${ATT_DIR}"
  echo "  - ${DATA_ROOT}/${WB100_DIR}/${ATT_DIR}"
  echo "[SUBMIT] Skipping generator."
fi

echo "[SUBMIT] Queuing sweeps (all depend on generator if submitted)..."

j_rrr95="$(sbatch --parsable "${dep[@]}" run_waterbirds95_rrr_sweep_vit.sh)"
j_rrr100="$(sbatch --parsable "${dep[@]}" run_waterbirds100_rrr_sweep_vit.sh)"
j_gc95="$(sbatch --parsable "${dep[@]}" run_waterbirds95_gradcam_sweep_vit.sh)"
j_gc100="$(sbatch --parsable "${dep[@]}" run_waterbirds100_gradcam_sweep_vit.sh)"
j_abn95="$(sbatch --parsable "${dep[@]}" run_waterbirds95_abn_sweep_vit.sh)"
j_abn100="$(sbatch --parsable "${dep[@]}" run_waterbirds100_abn_sweep_vit.sh)"

echo "[SUBMIT] Jobs:"
echo "  RRR WB95:    ${j_rrr95}"
echo "  RRR WB100:   ${j_rrr100}"
echo "  GradCAM WB95:${j_gc95}"
echo "  GradCAM WB100:${j_gc100}"
echo "  ABN WB95:    ${j_abn95}"
echo "  ABN WB100:   ${j_abn100}"
