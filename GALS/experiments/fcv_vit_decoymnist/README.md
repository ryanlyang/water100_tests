# DecoyMNIST ViT shortcut-susceptibility pilot

This pilot answers one question before FCV is transferred to DecoyMNIST:
does the same ImageNet-pretrained ViT-S/16 used by the Waterbirds study learn
the **unmodified** DecoyMNIST corner shortcut?

The source benchmark is not recolored, enlarged, or regenerated. Its original
28x28 grayscale PNGs are converted to RGB and resized directly to 224x224 for
ViT. No crop or flip is used because this diagnostic must not intermittently
remove the corner shortcut.

## Protocol

- Model: `vit_small_patch16_224.augreg_in21k_ft_in1k`
- Optimizer: AdamW
- Learning rates: `1e-5`, `3e-5`, `1e-4`
- Weight decay: `0.05`
- Seeds: `0`, `1`, `2`
- Epochs: 10, evaluated online at every epoch
- Candidate training data: 90% of the original biased training split
- Biased validation: one fixed, class-stratified 10% training holdout
- Test: the official reversed-decoy test split
- Checkpoints saved: none

For biased validation and reversed test, the pilot measures:

1. **Original:** unchanged image.
2. **Digit-only:** the detected 5x5 class-coded corner patch is erased.
3. **Patch-only:** every pixel except that corner patch is erased.

The detector validates the published encoding (`255-25*y` in train and
`25*y` in test) against source PNG pixels. It does not infer a more convenient
or strengthened shortcut.

An epoch passes the preregistered susceptibility diagnostic when:

- biased validation accuracy is at least 95%;
- biased-validation minus reversed-test accuracy is at least 10 points; and
- either patch erasure drops biased-validation accuracy by at least 10 points
  or patch-only biased-validation accuracy is at least 80%.

The aggregate recommendation proceeds only if every seed has at least one
passing epoch. All underlying metrics remain available regardless of this
summary gate.

## Tigris launch

From the GALS repository root on Tigris:

```bash
bash experiments/fcv_vit_decoymnist/scripts/submit_susceptibility_pilot.sh
```

The launcher submits a pretrained/data preflight, a nine-task GH200 array, and
an aggregation job with `afterok` dependencies. Outputs are written under:

```text
/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility
```

The concise final result is `pilot_summary.json`; the complete 90-candidate
matrix is `all_epoch_metrics.csv`.

## Full online FCV campaign

The susceptibility result motivates the larger campaign specified in
[`FCV_DECOYMNIST_FULL_CAMPAIGN_IMPLEMENTATION_PLAN.md`](FCV_DECOYMNIST_FULL_CAMPAIGN_IMPLEMENTATION_PLAN.md).
The production protocol uses 108 training runs and treats every one of ten
epochs as an online candidate. It does not retain model checkpoints, optimizer
states, resume states, token banks, embeddings, or per-image predictions.

Steps 1 and 2 are implemented by the frozen configuration and immutable split
manifests. Inspect the configuration with:

```bash
python experiments/fcv_vit_decoymnist/scripts/inspect_full_campaign_config.py
```

On Tigris, construct and authenticate the 48k/6k/6k train-derived partition
and untouched 10k official-test manifest with:

```bash
python experiments/fcv_vit_decoymnist/scripts/prepare_full_campaign_manifests.py
```

Manifest preparation audits the published corner encoding and computes a
SHA-256 for every source PNG. It writes no transformed images. The trust root
is `split_manifests/manifest_bundle.json`; it binds the frozen YAML, split
assignment, source inventory, all four visibility-separated manifests, and
their image hashes.

Steps 3 and 4 add the privileged Oracle view and primary teacher-map preflight.
The Oracle dataset rewrites the class-coded corner patch to the official test
encoding only after reading an authenticated Oracle-source PNG into memory.
It never saves reversed images.

After creating the split manifests, project and audit the primary
OpenCLIP+DINO maps with:

```bash
python experiments/fcv_vit_decoymnist/scripts/prepare_full_campaign_teacher_masks.py
```

This requires complete coverage of the exact 6,000 biased-validation IDs,
decodes foreground class 1 from the VOC colormap, applies direct 224x224
geometry, and average-pools each map into the 14x14 ViT grid. The compact
`projected_teacher_masks.npz` artifact contains only patch occupancies,
categories, labels, sample IDs, and eligibility flags. It is loaded with
`allow_pickle=False`; no model state, token features, or transformed dataset
is stored.

Targets whose teacher map does not classify every decoy-overlapping ViT cell
as safe background are explicitly audited and excluded from both FCV target
and donor pools. Preflight still fails unless the retained pool satisfies the
locked overall and per-class eligibility minima.

If coverage is incomplete, preprocessing fails and writes
`missing_teacher_maps.csv` plus an exact regeneration request. Missing maps
must be produced by the frozen OpenCLIP+DINO pipeline; the code will not
silently substitute another teacher. Fixed qualitative overlays and
exact-digit-mask quality statistics are audit-only.

Steps 5 and 6 implement the augmentation-diverse candidate grid and the shared
multiclass donor plan. The grid can be inspected without loading PyTorch with:

```bash
python experiments/fcv_vit_decoymnist/scripts/inspect_full_campaign_candidate_grid.py
```

Candidate training uses the four locked square RandomResizedCrop regimes, but
all validation and test views use the same direct bicubic resize. The module
contains no checkpoint-writing path: each epoch state is intended to be scored
online by the later orchestration step.

After the accepted projected teacher masks exist, create the donor plan with:

```bash
python experiments/fcv_vit_decoymnist/scripts/prepare_full_campaign_donor_plan.py
```

The plan is deterministic and shared by all 1,080 online candidates. Every
eligible target receives five same-corner donors from five distinct non-target
classes, exclusively from the teacher-audited eligible portion of biased
validation. This guarantees that mutually safe replacement includes the
class-coded decoy cells for both target and donor. The persisted JSON contains IDs,
labels, corners, and cryptographic bindings only—never images, token features,
logits, model weights, or optimizer state.

Steps 7 and 8 implement the model-facing FCV path. Raw ViT patch embeddings are
intervened on before position embeddings and transformer blocks. A donor token
is copied only at the identical spatial index where both target and donor maps
mark safe background; target evidence and ambiguous patches remain unchanged.
The production path verifies native-versus-reconstructed forward equivalence,
token integrity, and inclusion of the known decoy cells.

Each candidate epoch is scored in memory using donor-expanded accuracy over
five donors per eligible target. The primary selector is the locked,
parameter-free harmonic mean of original and counterfactual validation
accuracy. The same-context, matched-size random-mask, shuffled-teacher-mask,
evidence-swap, and exact-synthetic-mask controls are also computed online.
Controls are warning-only and cannot affect selection. Returned results contain
aggregate accuracies, confidence statistics, eligibility, and replacement
counts only; no per-image or per-donor records are persisted.

Steps 9 and 10 add visibility-separated Oracle/test evaluation and the complete
ten-epoch online loop. Run one campaign member with:

```bash
python experiments/fcv_vit_decoymnist/scripts/run_full_campaign_online.py \
  --run-index 0
```

Each epoch trains once, builds a memory-only validation token bank, evaluates
biased validation, harmonic FCV, the five controls, privileged Oracle
validation, and finally the official test split. Evaluation RNG state is
isolated from subsequent training. The token bank is deleted immediately after
the epoch and the model is discarded after epoch 10.

Aggregate rows are stored in five physically separate namespaces under
`online_metrics/`: `biased_validation`, `fcv`, `controls`,
`oracle_analysis_only`, and `test_analysis_only`. The selector-facing loader
enforces that Vanilla sees only biased validation, FCV sees only biased
validation plus FCV, Oracle sees only its privileged validation view, and test
is available only to post-hoc analysis. No checkpoint, optimizer state, resume
state, token bank, embedding cache, or per-image prediction is written. If a
task is interrupted, rerun it from epoch 1 with `--restart-partial`; completed
runs are authenticated and reused without retraining.

Steps 11 and 12 add leakage-separated selection/reporting and the complete
Tigris launch chain. Submit the campaign from the GALS repository root with:

```bash
bash experiments/fcv_vit_decoymnist/scripts/submit_full_campaign.sh
```

The launcher runs a provenance/data/model preflight, then one real GH200 epoch
through the complete online path. The smoke finalizer checks runtime and disk
projections, verifies that no checkpoint-like artifact survives, and writes a
launch gate. Only a successful gate releases the 108-task production array;
selection freezing and post-hoc analysis likewise use `afterok` dependencies.
All jobs use account `reu-aisocial`, partition `tigris`, and GH200 GPUs.

The selector freeze authenticates all 108 run receipts and all 1,080 candidate
rows before selecting Vanilla, harmonic FCV, and privileged Oracle winners.
It never parses test or control values. Post-hoc analysis subsequently joins
the frozen selections to official test results and reports gap closure,
headroom, rank correlations, top-k retrieval, crop-regime behavior, seed-level
results, and warning-only control diagnostics. The principal outputs are:

```text
selection/frozen_selector_matrix.csv
selection/frozen_selections.csv
selection/freeze_summary.json
reports/selector_outcomes.csv
reports/gap_closure.csv
reports/rank_report.csv
reports/crop_regime_report.csv
reports/seed_selections.csv
reports/final_summary.json
```

Useful monitoring commands after submission are:

```bash
squeue --me -o "%.18i %.20j %.2t %.10M %R"
sacct -X --starttime today \
  --format=JobID,JobName%22,State,ExitCode,Elapsed
```

The submission receipt under `submissions/` records every Slurm job ID and its
dependency. Completed production tasks are safely reusable. Partial tasks must
be explicitly restarted because the no-checkpoint protocol deliberately does
not retain resumable model or optimizer state.
