# ViT-FCV Waterbirds100 First Study

This directory contains the isolated implementation of the first
Feature-Counterfactual Validation (FCV) study. It tests whether FCV can select
a more group-robust vanilla ViT checkpoint when natural validation examples
have the same complete shortcut correlation as training.

## Locked online protocol

- Train 27 ImageNet-pretrained ViT-S/16 models: 3 learning rates x 3 weight
  decays x 3 seeds.
- Train each model for 20 epochs. Every live epoch state is a candidate, giving
  540 candidates.
- Use a single class-stratified 80/20 split of Waterbirds100 training data.
  Vanilla and FCV see only this train-derived holdout; they never use the
  original mixed-group validation split.
- Score biased validation, FCV, all four controls, and the privileged Oracle
  validation criterion online at every epoch.
- Evaluate test at every epoch for post-hoc analysis, but write it to a
  separate analysis-only artifact stream. The retention decision is made and
  staged before test inference, and test values are never returned to training
  or retention code.
- Retain at most three unique checkpoints per training run: current winners
  for biased validation accuracy, the primary equal-weight FCV score, and
  Oracle balanced-group validation accuracy.
- After global validation selections are frozen and hashed, delete every
  non-global local winner and keep at most the three unique globally selected
  primary checkpoints. All 540 test records remain available independently.
- Build each model-specific token bank under node-local scratch, score FCV and
  controls, and immediately delete the temporary checkpoint and token banks.
- Freeze and hash every validation selector before the post-hoc analyzer is
  allowed to open test artifacts.
- Guard the external output tree at 35 GiB against a 40 GiB study budget.

The complete locked settings are in
`configs/waterbirds100_vit_s16_first_study.yaml`.

## Prepare inputs

The online campaign reuses the frozen manifests and patch masks produced by
the audited preflight/smoke pipeline. The following must already exist under
the configured output root:

```text
split_manifests/metadata_train.csv
split_manifests/metadata_val.csv
split_manifests/metadata_oracle_val_analysis_only.csv
split_manifests/metadata_test_analysis_only.csv
patch_masks/patch_masks_val.pt
```

`metadata_train.csv` and `metadata_val.csv` intentionally omit context and
group labels. The Oracle and test manifests are explicitly analysis-only.

## Submit the complete study on Tigris

From the GALS repository root:

```bash
bash experiments/fcv_vit_waterbirds100/scripts/submit_full_online_540_study.sh
```

The launcher uses account `reu-aisocial`, partition `tigris`, one GH200 per
training task, and the `fcv_gh200` Miniforge environment. It submits this
dependency chain:

```text
cache pretrained model
  -> real run-0 epoch-1 interruption / epoch-2 resume smoke
  -> 27-task online training/scoring array (maximum 8 concurrent)
  -> freeze validation selections (no test access)
  -> post-hoc test/gap/rank analysis
```

The old `submit_full_81_candidate_study.sh` entry point is deliberately
disabled so an outdated persistent-checkpoint campaign cannot be launched by
mistake.

The cache job also writes one compact campaign-provenance receipt. It binds
the seeded initial model states, pretrained backbone, all manifests and image
inventories, teacher maps, patch masks, preprocessing, locked configuration,
software, and source tree. The real online smoke recomputes FCV, controls,
Oracle, and analysis-only test evidence before the full array is released;
its second process must resume exactly from the first epoch's atomic commit.
For both smoke epochs it also records complete online-epoch wall time and a
storage breakdown that separates FCV/control/Oracle/test evidence from model
state. The gate rejects the launch unless (i) observed growth plus one
transient checkpoint fits the 5 GiB eight-writer concurrency reserve, (ii) a
2x extrapolation of all durable evidence, compact indexes, bounded retained
checkpoints, resumes, and plans for all 540 candidates remains below the 35
GiB launch guard, and (iii) the slowest measured epoch projects to finish 20
epochs within the seven-day array limit with a 1.5x safety factor.

The two smoke receipts are combined into a campaign-bound
`online_smoke_gate_receipt.json`. Re-running the launcher reuses this immutable
gate, so the smoke does not attempt to rewind run 0 after the array has already
advanced or completed. If the smoke process is restarted exactly at a
committed prefix, the online producer returns that prefix unchanged instead of
training an extra epoch.

## Online artifacts

Each `online_runs/<run_id>/` contains compact validation/test indexes, one
atomic resume state while the run is active, deterministic donor/control
plans, and three steady-state retained checkpoints at most. A crash-safe
winner replacement can briefly stage one fourth checkpoint before the atomic
resume commit; it is pruned immediately afterward. Detailed FCV, control,
Oracle, and analysis-only test records live in separate trees and are bound by
SHA-256 in the compact indexes.

The selection-freeze job writes a checkpoint cleanup plan before deleting any
local winner, then writes a completion receipt. This makes cleanup restartable
and leaves only the unique global winners for the three primary selectors
(one to three model files total). The post-hoc analyzer validates the receipt
and surviving model hashes before reading test artifacts.

FCV and control probability-draw lists are the canonical raw records. Their
saved mean and standard deviation are derived from those exact serialized
values and re-derived during validation, avoiding accelerator-specific
float32 reduction differences without relaxing artifact checks.
Likewise, biased validation loss is selected from the canonical per-example
FCV record; the ordinary batch-reduced loss is retained only as a diagnostic
and must agree within the locked tolerance.

After all 27 runs finish, `selection_results/` contains the frozen
validation-only candidate matrix and selector table. Only after those files
are written and hashed does the post-hoc phase recompute test metrics from all
per-image records, report selected-checkpoint performance and gap closure,
compute rank correlations/top-k overlap with run-cluster bootstrap intervals,
and generate selector scatter plots.

## Restart behavior

The full launcher and array are safe to resubmit with the same output root.
The launcher reuses only a smoke gate whose campaign/source/config hashes and
two underlying receipts still validate. A completed run is strictly validated
and reused. An interrupted run resumes from its atomic
model/optimizer/scheduler/RNG state and reconstructs its compact CSV indexes.
Temporary node-local candidate checkpoints and token banks are never treated
as resumable state.
