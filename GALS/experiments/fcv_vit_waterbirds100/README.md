# ViT-FCV Waterbirds100 First Study

This directory contains the isolated implementation of the first
Feature-Counterfactual Validation (FCV) study. The study asks whether a
feature-counterfactual selector can choose a more robust vanilla ViT checkpoint
than ordinary validation accuracy when validation contains no natural
shortcut-breaking examples.

## Locked first-study protocol

- Candidate models are vanilla, fully fine-tuned, ImageNet-pretrained ViT-S/16
  classifiers.
- The Waterbirds100 training split is divided once using a class-stratified
  80/20 split with split seed 0.
- Both the ordinary (Vanilla) selector and FCV use only the held-out 20% of the
  fully biased training split.
- FCV background banks are built only from that same held-out split.
- The original mixed-group validation split is unavailable to Vanilla and FCV.
  It is used only by the explicitly privileged Oracle selector for analysis.
- The test split is never used for selection. It is used only to report the
  robust performance of selected checkpoints and post-hoc rank correlations.
- FCV intervenes on raw ViT patch embeddings before positional embeddings.
- The primary selector is fixed to equal weights on original held-out accuracy
  and opposite-context counterfactual accuracy.
- Candidate training is exactly locked to batch size 128, two warm-up epochs,
  ImageNet normalization, random-resized-crop scale `[0.8, 1.0]`, horizontal
  flip probability 0.5, and `Resize(256)+CenterCrop(224)` evaluation geometry.
- Token extraction, FCV/control forward chunking, Oracle inference, and final
  test inference use locked batch/worker settings. Those settings are stored
  in summaries and checked by every reuse and aggregation gate.
- Production artifacts bind the exact Torch/Torchvision/timm runtime, the
  executable source-tree hash, the persisted pretrained-backbone hash, and
  SHA-256 hashes of every source image recorded in the split manifests.

The complete locked settings live in
`configs/waterbirds100_vit_s16_first_study.yaml`.

## Candidate pool

The first pool contains 27 training runs:

```text
3 learning rates x 3 weight decays x 3 seeds = 27 runs
```

Every epoch is a candidate, yielding:

```text
27 runs x 20 epochs = 540 candidate checkpoints
```

Step 4 retains all 540 float32 epoch checkpoints until FCV and Oracle selector
analysis is complete. This requires roughly 45--50 GB for ViT-S/16 weights,
plus transient resume states. Non-selected candidates may be pruned only after
Step 12 rank/scatter analysis completes successfully; Steps 11--12 require all
540 original checkpoint paths and hashes.

## Inspect the configuration

From the GALS repository root:

```bash
python experiments/fcv_vit_waterbirds100/scripts/inspect_config.py
```

On Tigris, add `--check-paths` to verify the dataset, metadata, teacher-map,
environment, and output-root parents:

```bash
python experiments/fcv_vit_waterbirds100/scripts/inspect_config.py --check-paths
```

## Prepare the locked metadata split

On Tigris:

```bash
python experiments/fcv_vit_waterbirds100/scripts/prepare_metadata.py
```

The command writes the following beneath the configured output root:

```text
split_manifests/metadata_train.csv
split_manifests/metadata_val.csv
split_manifests/metadata_oracle_val_analysis_only.csv
split_manifests/metadata_test_analysis_only.csv
split_manifests/split_indices.json
split_manifests/split_summary.json
split_manifests/manifest_bundle.json
```

`metadata_train.csv` and `metadata_val.csv` intentionally omit context and
group labels. The latter two CSV files are analysis-only and must never be
loaded by Vanilla or FCV selection code. `manifest_bundle.json` binds all four
manifest hashes to the original metadata hash, deterministic split indices,
split summary, and locked holdout configuration. Every downstream stage
validates both this bundle and the role-specific `source_split` before use.

## Preprocess teacher maps into ViT patches

After preparing the metadata split, run Step 3 on Tigris:

```bash
python experiments/fcv_vit_waterbirds100/scripts/prepare_patch_masks.py
```

The command loads only the train-derived `metadata_val.csv` maps, decodes the
VOC RGB colormap categorically (class ID 1 is bird foreground), applies the
same aspect-preserving `Resize(256)` and `CenterCrop(224)` geometry used by
validation images, converts each map to the locked 14x14 ViT patch grid, and
writes:

```text
patch_masks/patch_masks_val.pt
patch_masks/patch_masks_val_audit.csv
patch_masks/patch_masks_val_summary.json
patch_masks/preflight_overlays/*.png
```

Patch scores are foreground fractions pooled from the categorical mask.
Evidence, background, and ambiguous indices use the locked thresholds 0.60
and 0.10. Samples with fewer than 20 safe background patches are retained for
audit but marked `fcv_eligible=false`. The preflight fails before training if
overall or per-class eligibility is too low, but still preserves the complete
audit, summary, and 20 image/mask/token-grid overlays for diagnosis. If an
individual decode/dimension/overlay operation fails earlier, it instead
preserves the completed diagnostic prefix, failing sample ID, label,
image/map paths, processing stage, error, and all prior overlays.

An overwrite attempt invalidates the old `.pt` before decoding begins. Any
decode, native-dimension, overlay, or acceptance failure leaves no usable mask
artifact and persists a failed summary plus diagnostics. Step 4 and Step 6
also re-hash every current teacher map and require `status=complete` in the
Step 3 summary before proceeding.

## Train the vanilla ViT candidate pool

Step 4 requires `timm` in the Tigris environment:

```bash
source /home/ryreu/miniforge3-aarch64/bin/activate
conda activate fcv_gh200
python -m pip install 'timm==1.0.28'
```

Submit the 27-run resumable Slurm array from the GALS repository root:

```bash
bash experiments/fcv_vit_waterbirds100/scripts/submit_step4_candidate_pool.sh
```

The array ordering is fixed as learning rate, then weight decay, then seed.
Each task trains for 20 epochs and writes one candidate checkpoint and one
biased-validation metric row per epoch. Re-submitting an interrupted array task
automatically resumes from its last committed epoch. The epoch commit order is
checkpoint, resume state, then public metrics CSV; if interruption occurs after
the resume commit but before CSV publication, the next invocation validates
the checkpoint bytes and reconstructs the missing CSV prefix exactly.

Before training, one cache job persists and hashes the exact pretrained ViT
backbone. Every epoch checkpoint and run summary binds that artifact, the
source tree, public manifests (which themselves contain image-byte hashes),
and the full seed-specific initial state. Aggregation reproduces every mutable
`metrics.csv` value from the metric row embedded in its checkpoint before any
selector may consume it. The executable-source hash includes Python entry
points, shell submission wrappers, and Slurm job files.

The dependent aggregation job writes a diagnostic table even when an array
task fails. Once all tasks are complete, enforce the full 540-candidate check:

```bash
python experiments/fcv_vit_waterbirds100/scripts/aggregate_candidate_metrics.py
```

Step 4 never loads the Oracle validation or test manifests.

## Verify raw-patch forwarding

Step 5 provides a narrow adapter for the locked timm ViT. It extracts the
`[B,196,384]` patch embeddings immediately after `patch_embed`, before CLS and
positional embeddings. The resumed route then calls the model's own
`_pos_embed`, patch-dropout, pre-normalization, transformer blocks, final
normalization, and classifier head. This ensures that counterfactual donor
content receives the target patch position in later steps.

Verify the architecture with cached pretrained weights on Tigris:

```bash
source /home/ryreu/miniforge3-aarch64/bin/activate
conda activate fcv_gh200
python experiments/fcv_vit_waterbirds100/scripts/verify_vit_patch_forward.py \
  --pretrained \
  --output-report /home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/preflight/reconstruction_pretrained.json \
  --device cuda
```

Verify any saved Step 4 candidate by supplying its checkpoint:

```bash
python experiments/fcv_vit_waterbirds100/scripts/verify_vit_patch_forward.py \
  --checkpoint /path/to/candidate_models/run_.../checkpoints/epoch_001.pt \
  --output-report /home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/preflight/reconstruction_candidate.json \
  --device cuda
```

Candidate loading is strict: artifact type, schema, training fingerprint,
model metadata, and every state-dict key must match the active configuration.
FCV verification requires `model.eval()` and passes only when normal and
reconstructed logits differ by less than `1e-5` in maximum absolute value.
The Step 6 submission wrapper schedules both reconstruction checks inside a
GH200 preflight job. The token-bank array has an `afterok` dependency on that
job, and Steps 6--8 refuse to run if either report, its configuration
fingerprint, or the candidate bytes have changed.

## Build model-specific background-token banks

Step 6 creates two raw-patch banks for every candidate checkpoint:

```text
token_banks/<candidate_id>_land_context.pt
token_banks/<candidate_id>_water_context.pt
token_banks/<candidate_id>_summary.json
```

Only the train-derived `metadata_val.csv` public manifest is loaded. Class label
acts as the context proxy because the Waterbirds100 source training split is
fully correlated: label 0 supplies land-context background tokens and label 1
supplies water-context tokens. No group or context column is accepted.

Donor sources include every public-validation image with at least one safe
background patch; target eligibility remains the stricter 20-background plus
evidence criterion. Each bank stores float32 raw patch tokens before positional
embeddings, plus compact
provenance arrays for source image, class, patch index, patch row/column, and
teacher-map patch score. A source-image table maps those integer indices back
to sample IDs, allowing Step 7 to exclude self-donors without duplicating a
string for every token.

After all Step 4 candidates are complete, submit Step 6 from the repository
root:

```bash
bash experiments/fcv_vit_waterbirds100/scripts/submit_step6_token_banks.sh
```

The submission script first enforces the complete 540-candidate Step 4 pool,
then submits a GH200 reconstruction-preflight job for the locked pretrained
model and a real Step 4 checkpoint. Only after that job passes does Slurm
release the 27-task GH200 token-bank array, one task per
LR/weight-decay/seed run; each task processes that run's 20 epoch checkpoints.
Their hashes are required and recorded by Steps 6--8. Re-submission reuses
completed banks whose checkpoint/config/manifest/mask/map/software provenance
still matches. Partial writes are rebuilt atomically, while a stale completed
summary requires `--overwrite`.
Because the protocol retains float32 raw tokens for every candidate, the full
bank pool can require substantial storage; check the output filesystem quota
before submitting all 27 tasks.

The dependent aggregation job intentionally permits incomplete output for
diagnostics. After the array succeeds, enforce all 540 candidates and 1,080
context banks with:

```bash
python experiments/fcv_vit_waterbirds100/scripts/aggregate_token_banks.py
```

To build or inspect one candidate directly:

```bash
python experiments/fcv_vit_waterbirds100/scripts/build_background_token_banks.py \
  --checkpoint /path/to/candidate_models/run_.../checkpoints/epoch_001.pt \
  --device cuda
```

Before launching the full 540-candidate campaign, run the executable GH200
end-to-end smoke workflow:

```bash
bash experiments/fcv_vit_waterbirds100/scripts/submit_smoke_one_candidate.sh
```

For one locked hyperparameter candidate, this deliberately injects an abrupt
first-epoch failure after the resume-state commit but before `metrics.csv`,
recovers and resumes through epoch 20, compares every metric and model tensor
with an uninterrupted GH200 reference, verifies pretrained and real-candidate ViT
reconstruction, runs Steps 6--8, and invokes those stages again to exercise
their provenance-aware reuse gates. It first runs the full protocol unit suite
in the production aarch64/Torch/timm environment; synthetic downstream tests
cover Oracle raw-artifact validation, frozen Step-9 pool binding, final-test
evaluation, gap analysis, and rank analysis. Outputs are isolated under
`<output_root>/smoke/<timestamp>/`; only the two reconstruction reports are
written to the shared preflight directory used by the production gates.

## Score opposite-context feature counterfactuals

Step 7 first creates one shared, deterministic donor-index plan. The plan uses
seed 0 and records five donor-token indices per safe target-background patch,
sampling globally with replacement from the opposite class/context bank. The
same integer draws are reused for all 540 checkpoints. Every candidate bank
must have exactly the same source-image and source-patch provenance layout as
the reference bank or scoring stops rather than silently changing donors.

For each eligible validation image, Step 7 extracts the target's raw patch
embeddings and changes only the Step 3 `background_idx` positions. Landbird
targets receive water-context donor content and waterbird targets receive
land-context donor content. Ambiguous and evidence tokens remain untouched.
The existing Step 5 resumed forward applies target positional embeddings after
the replacement and computes all five counterfactual predictions.

After the complete Step 6 pool is available, submit the resumable pipeline:

```bash
bash experiments/fcv_vit_waterbirds100/scripts/submit_step7_fcv_scoring.sh
```

The dependency chain is:

```text
shared donor-plan job -> 27-task GH200 scoring array -> diagnostic aggregation
```

Candidate outputs are:

```text
fcv_scores/opposite_donor_plan.pt
fcv_scores/<candidate_id>.csv
fcv_scores/<candidate_id>_summary.json
fcv_scores/candidate_fcv_scores.csv
fcv_scores/candidate_fcv_scores_summary.json
```

Each per-image CSV contains original true-class probability/correctness, mean
and standard deviation over the five counterfactual probabilities, draw-level
and majority-vote accuracy, confidence drop, swapped-patch count, mask
coverage, and compact JSON arrays of the five probabilities and predictions.
The candidate summary reports the primary FCV accuracy (mean correctness over
all eligible image/draw pairs), majority-vote accuracy, confidence metrics,
conditional FCV accuracy among originally correct images, and the locked
equal-weight original/FCV selector score. It also records per-class FCV values,
target/donor token norms and cosine similarities (global quantiles and
per-class breakdowns), donor-source diversity, the
normal-versus-resumed error, and an explicit own-token identity-swap error.
For the real opposite-context swaps it additionally asserts and records exact
foreground preservation, exact reconstruction of the selected donor tokens,
the changed-token fraction, and mean/max replacement magnitude. A foreground
or donor mismatch—or a completely no-op intervention cohort—is fatal.
Step 7 does not load Oracle or test
metadata and does not implement the Step 8 control interventions.

The dependent aggregation job allows incomplete output so it remains useful
when an array task fails. After all scoring tasks finish, enforce all 540
candidates explicitly:

```bash
python experiments/fcv_vit_waterbirds100/scripts/aggregate_fcv_scores.py
```

Aggregation verifies each per-image CSV hash and then recomputes draw
accuracies, probabilities, confidence drops, conditional/per-class metrics,
and the primary selector score from its JSON draw arrays. Summary values that
do not reproduce are rejected. The pool summary also reports cohort-level
identity-swap failures and token-diagnostic coverage across all candidates.

To inspect one candidate directly:

```bash
python experiments/fcv_vit_waterbirds100/scripts/score_fcv_candidates.py \
  --checkpoint /path/to/candidate_models/run_.../checkpoints/epoch_001.pt \
  --device cuda
```

## Run the essential FCV controls

Step 8 implements four controls over the same public validation examples and
five-draw protocol:

```text
same_context   target teacher-background positions + same-context background tokens
random_mask    matched-count random positions + the exact Step 7 opposite donors
shuffled_mask  another image's teacher-background positions + opposite donors
evidence_swap  target teacher-evidence positions + opposite-class evidence tokens
```

All control randomness is cached once in `control_scores/control_plan.pt`.
Same-context donors exclude the target image. Random masks contain unique,
uniform positions and match each target's background count. Shuffled masks use
a deterministic no-fixed-point permutation of eligible masks. Evidence-bank
layouts come entirely from Step 3 records; each candidate reconstructs its own
evidence token values in that exact source-image/patch order. Evidence-swap
targets are restricted to the main FCV-eligible cohort so sensitivity gaps are
not confounded by evaluating different image sets.

Submit Step 8 after the complete Step 7 pool exists:

```bash
bash experiments/fcv_vit_waterbirds100/scripts/submit_step8_fcv_controls.sh
```

The dependency chain is:

```text
shared control-plan job -> 27-task GH200 control array -> diagnostic aggregation
```

Each candidate writes:

```text
control_scores/<candidate_id>_same_context.csv
control_scores/<candidate_id>_random_mask.csv
control_scores/<candidate_id>_shuffled_mask.csv
control_scores/<candidate_id>_evidence_swap.csv
control_scores/<candidate_id>_controls_summary.json
```

The summary reports same-minus-opposite accuracy, random/shuffled-mask gaps,
and the evidence-versus-background confidence-sensitivity gap. A fixed,
warning-only policy flags implausible semantic ordering without altering any
selector. These remain diagnostics in Step 8; this stage defines no selectors
and loads no Oracle or test metadata. Reuse and aggregation re-hash all four
control CSVs plus the corresponding Step 7 CSV, reproduce every draw-level and
aggregate value, and regenerate the warning status from the fixed policy;
mutable summary fields are never treated as authoritative. Every real control
swap also verifies that all non-replaced tokens are bit-identical, every
replaced token exactly equals its planned donor, and each control cohort is not
a complete no-op. Replacement counts and norm deltas are retained in the
hashed per-image CSVs. After the array completes, enforce all
540 candidates and
2,160 control CSVs with:

```bash
python experiments/fcv_vit_waterbirds100/scripts/aggregate_fcv_controls.py
```

To inspect one candidate directly:

```bash
python experiments/fcv_vit_waterbirds100/scripts/score_fcv_controls.py \
  --checkpoint /path/to/candidate_models/run_.../checkpoints/epoch_001.pt \
  --device cuda
```

## Build the validation selector table

Step 9 evaluates every saved epoch on the explicitly privileged original
mixed validation split, then compares ordinary, FCV, control-normalized, and
Oracle selectors. Oracle scoring is isolated in `fcv.selectors`: its dataset
requires the analysis-only manifest marker, all four group labels, and
consistent label/context/group identities. It cannot accept the test manifest.
Oracle inference is locked to float32.

Submit the resumable Tigris pipeline after Steps 4, 7, and 8 are complete:

```bash
bash experiments/fcv_vit_waterbirds100/scripts/submit_step9_selectors.sh
```

The submit script first strictly validates all 540 unprivileged candidate,
FCV, and control rows. It then launches:

```text
27-task GH200 Oracle array
  -> partial diagnostic aggregate after any outcome
  -> strict selector-table job only after all Oracle tasks succeed
```

Re-submission reuses Oracle summaries only when checkpoint, configuration,
training fingerprint, Oracle-manifest provenance, execution settings, and a
hashed per-example result file all match. Oracle loss, accuracy, group
counts/correct counts, per-group accuracy, balanced-group accuracy, and
worst-group accuracy are recomputed from the raw per-example records before
reuse or aggregation. After a successful run, Step 9 writes:

```text
selection_results/oracle_scores/<candidate_id>_oracle_summary.json
selection_results/oracle_scores/<candidate_id>_oracle_per_image.csv
selection_results/candidate_oracle_scores.csv
selection_results/candidate_oracle_scores_summary.json
selection_results/candidate_selector_scores.csv
selection_results/selection_table.csv
selection_results/selection_table_summary.json
```

The selector comparison includes ordinary biased-validation accuracy/loss,
opposite-context accuracy, mean counterfactual true-class probability,
probability retention, the locked equal-weight primary FCV score, fixed
accuracy-weight ablations with lambda in `{0.25, 0.5, 1.0}`, a same-context
control-normalized score, and Oracle worst/balanced-group accuracy. The
control-normalized score is

```text
original accuracy
  - [(opposite-context confidence drop) - (same-context confidence drop)].
```

Exact score ties are resolved only by ascending candidate ID. This fixed rule
does not introduce another performance metric. `selection_table.csv` contains
the selected checkpoint and all validation diagnostics, but deliberately has
no test columns. Step 10 is the first stage allowed to evaluate these selected
checkpoints on test data.

To strictly rebuild the table after Oracle scoring:

```bash
python experiments/fcv_vit_waterbirds100/scripts/aggregate_oracle_metrics.py
python experiments/fcv_vit_waterbirds100/scripts/build_selection_table.py
```

## Evaluate frozen selections on test data

Step 10 is the first stage allowed to open
`split_manifests/metadata_test_analysis_only.csv`. Before doing so, it verifies
the Step 9 selection-table hash, selector-to-candidate mapping, candidate
selector-matrix hash, selector-analysis fingerprint, and the path and SHA-256
of every selected checkpoint. A selection table containing any `test_*`
column is rejected.

Submit the final selected-checkpoint evaluation on Tigris with:

```bash
bash experiments/fcv_vit_waterbirds100/scripts/submit_step10_final_test.sh
```

The submission script validates and freezes Step 9 without opening the test
manifest, then submits one GH200 job. If multiple selectors chose the same
candidate, that checkpoint is evaluated only once. Its metrics are copied back
to each corresponding selector row in the original Step 9 order. Re-submission
reuses per-candidate test summaries only when checkpoint, configuration,
manifest-bundle, test-manifest, and hashed per-example prediction records still
match. All aggregate metrics are recomputed from those records.

Step 10 writes:

```text
selection_results/final_test_scores/<candidate_id>_test_summary.json
selection_results/final_test_scores/<candidate_id>_test_per_image.csv
selection_results/final_test_results.csv
selection_results/final_test_results_summary.json
```

`final_test_results.csv` reports average accuracy, balanced-group accuracy,
worst-group accuracy, and Landbird-on-Land, Landbird-on-Water,
Waterbird-on-Land, and Waterbird-on-Water accuracy for every selector. The
primary reported metric is test worst-group accuracy. Step 10 does not sort,
select, or filter candidates using these values. Oracle-gap closure and
post-hoc pool analysis remain later steps.

To validate Step 9 manually without test access:

```bash
python experiments/fcv_vit_waterbirds100/scripts/evaluate_selected_checkpoints.py \
  --validate-selection-only
```

## Compute Oracle selection gap closure

Step 11 measures how much of the robust-test gap from ordinary biased
validation selection to the privileged Oracle selector is recovered by the
locked primary FCV selector. It uses exactly these Step 9 rows:

```text
biased: biased_validation_accuracy
FCV:    equal_weight_original_and_opposite_fcv_accuracy
Oracle: oracle_validation_balanced_group_accuracy
```

With robust performance defined as test worst-group accuracy, the reported raw
fraction is:

```text
(R_fcv - R_biased) / (R_oracle - R_biased)
```

The fraction is not clipped. A zero denominator is reported as undefined, and
an Oracle value below the biased selector is retained with an explicit
negative-gap status rather than hidden.

Step 11 also evaluates all 540 candidates on test to report the candidate-pool
upper bound. This maximum is explicitly post-hoc and unfair: it diagnoses
whether a strong checkpoint exists in the pool but is never allowed to alter
the frozen selector choices. The scorer itself—not only its submission
wrapper—must load a completed Step 9 selection before opening the test
manifest. Every per-candidate and aggregate pool-test artifact stores the
selection table, selection summary, and selector-matrix paths and hashes, so a
later selection change invalidates the post-hoc pool.
Before evaluating or reusing a candidate, Step 11 also requires its current
checkpoint path and bytes to match the candidate-by-candidate mapping frozen
in the Step-9 selector matrix. This includes candidates selected by no method.
Each pool-test summary is also bound to a hashed per-example logits/predictions
CSV, and the strict aggregate recomputes every reported metric from those raw
records.

Submit the resumable Tigris pipeline after Step 10 completes:

```bash
bash experiments/fcv_vit_waterbirds100/scripts/submit_step11_gap_closure.sh
```

The dependency chain is:

```text
27-task GH200 pool-test array
  -> partial diagnostic aggregate after any outcome
  -> strict full-pool aggregate and gap report only after all tasks succeed
```

The final artifacts are:

```text
selection_results/candidate_pool_test_scores/<candidate_id>_pool_test_summary.json
selection_results/candidate_pool_test_scores/<candidate_id>_pool_test_per_image.csv
selection_results/candidate_pool_test_scores.csv
selection_results/candidate_pool_test_scores_summary.json
selection_results/gap_closure_summary.csv
selection_results/gap_closure_summary.json
```

## Analyze selector correlation and rank quality

Step 12 asks whether each validation-only criterion is predictive across the
entire candidate pool, rather than judging it from one selected checkpoint.
It joins the immutable Step 9 selector matrix to Step 11's complete post-hoc
test table by candidate ID and verifies every run, epoch, seed, hyperparameter,
checkpoint path, and checkpoint SHA-256 before computing statistics.

The six locked criteria are biased validation accuracy/loss, the equal-weight
FCV main score, opposite-context counterfactual accuracy, FCV stability, and
Oracle balanced-group validation accuracy. FCV stability is the existing mean
counterfactual true-class probability. Validation loss is multiplied by `-1`
for correlation and ranking so that higher oriented scores always mean a
better selector.

For each criterion, Step 12 reports Spearman correlation, Kendall tau-b,
selected test worst-group accuracy, regret to the post-hoc pool maximum, and
top-k overlap recall for `k={1,5,10,25}`. Selector and robust-test top-k sets
both use descending score with candidate ID ascending as the fixed tie break:

```text
top-k recall = |selector top-k intersect robust-test top-k| / k
```

Submit the CPU-only Tigris analysis after Step 11 completes:

```bash
bash experiments/fcv_vit_waterbirds100/scripts/submit_step12_rank_analysis.sh
```

The outputs are:

```text
selection_results/rank_correlation_results.csv
selection_results/candidate_rank_analysis.csv
selection_results/rank_correlation_results_summary.json
selection_results/selector_scatter_plots/*_scatter.png
selection_results/selector_scatter_plots/selector_rank_scatter_grid.png
```

Step 12 is strictly post-hoc. It recomputes every frozen selector choice and
fails if any result differs, but never updates the selection table.
