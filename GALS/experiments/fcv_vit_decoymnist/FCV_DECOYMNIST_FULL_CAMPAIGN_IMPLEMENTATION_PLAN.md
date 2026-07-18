# ViT-FCV on DecoyMNIST: Full Online Campaign Implementation Plan

## Implementation progress

Steps 1--12 are implemented. The campaign now has the frozen configuration,
authenticated manifests, in-memory Oracle transformation, projected teacher
masks, augmentation-diverse candidates, shared donor plans, spatial token
intervention, harmonic FCV and controls, visibility-separated Oracle/test
evaluation, a ten-epoch no-checkpoint online runner, leakage-separated selector
freezing and post-hoc reports, and the provenance-gated Tigris smoke/full
launch chain.

Pre-result feasibility amendment (2026-07-18): the first real Tigris
preflight found that 65 of 6,000 OpenCLIP+DINO maps overlap the synthetic
decoy region. Such samples are now explicitly marked FCV-ineligible and
excluded from both target and donor pools. Their IDs, counts, classes, and
reasons remain in the teacher audit. The campaign still fails if the retained
pool violates the precommitted overall or per-class eligibility minima. This
does not alter teacher maps, thresholds, donor rules, candidate models, or any
selection metric, and was made before training or test-result inspection.

## Status and purpose

This document locks the implementation protocol for the second
Feature-Counterfactual Validation (FCV) study. The study transfers the
Waterbirds-100 ViT-FCV model-selection framework to **unmodified
DecoyMNIST**.

The preceding susceptibility pilot established that an ImageNet-pretrained
ViT-S/16 learns the standard DecoyMNIST shortcut extremely strongly:

- all 90 pilot candidate epochs passed the shortcut-susceptibility gate;
- biased-validation accuracy approached 100%;
- reversed-decoy test accuracy was approximately 19% at the end of training;
- erasing the corner patch reduced biased-validation accuracy by roughly 78
  percentage points.

The open question is therefore not whether ViT can learn the shortcut. It is:

> Can FCV select a ViT training state and hyperparameter configuration with
> better reversed-decoy test accuracy than ordinary biased-validation
> selection, without using Oracle-validation or test information?

The harmonic FCV aggregation rule is locked before this campaign. It must not
be changed after inspecting DecoyMNIST Oracle or test results.

---

# 1. Non-negotiable protocol constraints

The implementation must satisfy all of the following:

1. Use the original DecoyMNIST PNG data without recoloring, enlarging,
   strengthening, or regenerating its shortcut.
2. Use the same pretrained ViT-S/16 family as the Waterbirds-100 FCV study.
3. Give Vanilla, FCV, and Oracle selectors the exact same candidate pool.
4. Give Vanilla and FCV access only to the biased training holdout.
5. Give Oracle access only to a separate, distribution-matched privileged
   validation split.
6. Never use the official test split to train a model, construct a donor bank,
   choose a selector, tune an aggregation rule, or alter the candidate grid.
7. Compute all candidate metrics online while each model state is in memory.
8. Do **not** save per-epoch model checkpoints or retained selector winners.
9. Do **not** save optimizer states or resume checkpoints. Individual jobs are
   short enough that a failed array task can be rerun from the beginning.
10. Store only compact aggregate CSV/JSON results, manifests, provenance, and
    logs. Do not persist token banks, per-image embeddings, logits, or
    per-donor predictions.
11. Keep any temporary token banks on node-local scratch and delete them after
    the corresponding epoch is scored.
12. Treat test metrics as analysis-only columns that cannot be read by any
    selector implementation.

These constraints make the campaign both scientifically auditable and small
on disk.

---

# 2. Dataset and frozen paths

## 2.1 Tigris paths

Repository:

```text
/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
```

DecoyMNIST PNG root:

```text
/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png
```

Primary OpenCLIP+DINO teacher-map root:

```text
/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist_openclip/val/prediction_cmap
```

Campaign output root:

```text
/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_full_campaign
```

Cluster configuration:

```text
account: reu-aisocial
partition: tigris
GPU: NVIDIA GH200
environment: /home/ryreu/miniforge3-aarch64/envs/fcv_gh200
python: /home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python
```

## 2.2 Original benchmark construction

The source images remain 28x28 grayscale PNGs. The published generator places
a 5x5 block in one randomly selected corner.

Training patch intensity:

\[
I_{\mathrm{train}}(y)=255-25y.
\]

Official test patch intensity:

\[
I_{\mathrm{test}}(y)=25y.
\]

Every source PNG used by the campaign must pass an encoding audit against
these definitions. Test-class zero has a zero-valued patch indistinguishable
from black background; deterministic first-corner tie handling is acceptable
because all tied views are pixel-identical.

## 2.3 Preprocessing

For ViT, convert grayscale to three identical RGB channels and resize directly
from 28x28 to 224x224 with bicubic interpolation. Apply ImageNet normalization.

Evaluation preprocessing is deterministic:

```text
grayscale -> RGB -> Resize((224,224), bicubic) -> tensor -> ImageNet normalize
```

Do not use horizontal flips. Mirrored digits are not label-preserving.

---

# 3. Frozen data split and information boundaries

Create one deterministic, class-stratified partition of the original 60,000
biased training examples using split seed 0 and largest-remainder
apportionment:

| Study split | Count | Visibility |
|---|---:|---|
| Candidate training | 48,000 | Training only |
| Biased validation | 6,000 | Vanilla and FCV |
| Oracle validation source | 6,000 | Oracle analysis only |

The official 10,000-image reversed-decoy test split remains intact.

## 3.1 Candidate training

Candidate models train only on the 48,000 biased examples. No Oracle or test
image may enter a training loader.

## 3.2 Biased validation

The 6,000 biased-validation images retain their original training patch
encoding. They are the only images used by:

- ordinary biased-validation selection;
- FCV target examples;
- FCV donor examples;
- FCV controls;
- teacher-map projection and eligibility checks.

## 3.3 Oracle validation

The 6,000 Oracle-source images are disjoint from candidate training and biased
validation. At Oracle-evaluation time only, replace their 5x5 training patch
with the official reversed-test encoding `25*y`, in memory. Do not write
modified Oracle PNGs to disk.

This produces an independent, distribution-matched privileged validation set
without using any official test image.

## 3.4 Official test

The complete 10,000-image official reversed-decoy test set is evaluated online
at every epoch for analysis only. Test values must be written to a separate
analysis namespace and must never enter selector code.

## 3.5 Frozen manifests

Persist compact manifests containing:

```text
sample_id
relative image path
label
source split
study split
image SHA-256
```

Do not include Oracle or test membership in the public manifests consumed by
Vanilla or FCV. Bind all manifests into one provenance receipt and require
every array task to verify it before training.

---

# 4. Model and online candidate pool

## 4.1 Model

Use:

```text
timm model: vit_small_patch16_224.augreg_in21k_ft_in1k
pretrained: true
num_classes: 10
image_size: 224
patch_size: 16
patch_grid: 14 x 14
classification head: CLS token
fine-tune mode: full
```

Cache and hash the pretrained backbone once. Every run must verify that its
pretrained backbone matches the frozen cache provenance before training.

## 4.2 Optimizer and schedule

```text
optimizer: AdamW
precision: bfloat16 AMP
epochs: 10
scheduler: linear warmup for 1 epoch, then cosine decay to zero
batch size: 256, subject only to a smoke-tested memory reduction
label smoothing: 0
mixup: 0
cutmix: 0
```

## 4.3 Candidate grid

The susceptibility pilot showed that learning rate and weight decay alone
produce almost no robustness diversity. Add one standard, non-targeted
augmentation parameter: RandomResizedCrop minimum scale.

```text
learning_rate in {1e-5, 3e-5, 1e-4}
weight_decay in {0.01, 0.05, 0.10}
crop_scale_min in {1.0, 0.8, 0.6, 0.4}
seed in {0, 1, 2}
```

Training transform for crop scale `s`:

```text
RandomResizedCrop(
    size=224,
    scale=(s, 1.0),
    ratio=(1.0, 1.0),
    interpolation=bicubic,
)
```

`crop_scale_min=1.0` is the direct-resize shortcut-heavy condition.
Progressively smaller values are ordinary crop augmentation that may omit a
corner while usually retaining the centered digit. This creates candidate
diversity without explicitly detecting or deleting the known shortcut during
training.

Total training runs:

\[
3\ \mathrm{LRs}\times3\ \mathrm{WDs}\times4\ \mathrm{crop\ regimes}
\times3\ \mathrm{seeds}=108.
\]

Every epoch is an online candidate:

\[
108\ \mathrm{runs}\times10\ \mathrm{epochs}=1080\ \mathrm{candidates}.
\]

Use a stable candidate ID containing run index, LR, weight decay, crop scale,
seed, and epoch.

## 4.4 No-checkpoint online evaluation

For each training run:

```text
for epoch in 1..10:
    train one epoch
    score biased validation in memory
    construct ephemeral FCV donor features in memory/node-local scratch
    score FCV and controls in memory
    score Oracle validation in memory
    score official test in analysis-only mode
    append one aggregate metric row
    delete donor features
continue training
discard model after epoch 10
```

Persist no model state at any point. If a run fails, rerun that full array task.
The susceptibility pilot indicates that a complete ten-epoch run is short
enough for this policy.

---

# 5. Teacher maps and ViT patch masks

## 5.1 Primary maps

Use the existing OpenCLIP+DINO maps as the primary evidence maps. Before
training, audit exact sample-ID coverage for all 6,000 biased-validation
examples.

If coverage is incomplete, generate only the missing maps with the same frozen
OpenCLIP+DINO pipeline. Do not silently substitute maps from another teacher.

## 5.2 Projection to patch space

Apply the same deterministic 224x224 evaluation transform to the map and image.
Average-pool the map over each 16x16 ViT patch.

Lock the Waterbirds thresholds unless a pre-result geometric audit proves they
are invalid:

```text
evidence patch: map occupancy >= 0.60
background patch: map occupancy <= 0.10
ambiguous patch: otherwise
```

Preserve evidence and ambiguous target patches. Only safe-background patches
may be replaced.

## 5.3 Exact synthetic masks as analysis only

The known synthetic digit mask may be used for:

- an Oracle-mask control;
- teacher-map quality measurement;
- qualitative overlays;
- diagnosing a failed primary teacher intervention.

It must not replace the primary OpenCLIP+DINO maps after results are observed.

## 5.4 Mask preflight requirements

Before launching the array, require:

- 100% teacher-map coverage;
- valid 14x14 projected masks;
- at least one safe evidence patch per eligible target;
- enough safe background patches to include the decoy region;
- explicit exclusion and auditing of targets whose decoy cells are not safe
  background;
- no image-transform or sample-ID mismatch;
- saved overlays for a fixed audit subset.

---

# 6. Multiclass donor construction

Waterbirds has one binary opposite context. DecoyMNIST has nine possible
class-conflicting contexts. Lock the following donor protocol.

## 6.1 Context proxy

In the completely confounded biased validation set, the target class determines
the patch intensity. Therefore, class label is the shortcut-context proxy, as
class label was the water/land context proxy in Waterbirds-100.

This uses target labels but no group labels.

## 6.2 Corner matching

Detect each example's source 5x5 patch corner from the frozen benchmark
encoding. For a target with label `y` and corner `c`, donors must:

```text
have label y' != y
have the same corner c
be a different sample from the target
belong to biased validation only
have an accepted FCV teacher mask
```

Same-corner matching ensures that a spatially aligned donor actually provides
a conflicting class-coded patch instead of a black token from an unrelated
position.

## 6.3 Fixed donor count

Use five donors per target. Deterministically sample five distinct donor labels
from the nine labels other than `y`, then one donor image per selected label
with the same corner. Use donor-plan seed 0.

Persist only donor sample IDs in one compact plan. Do not persist donor
embeddings.

## 6.4 Mutually safe spatial replacement

For each target-donor pair, consider patch position `p` replaceable only when:

```text
target patch p is safe background
donor patch p is safe background
```

Replace donor content at the same spatial patch position, then apply the target
position embedding. Leave target evidence, target ambiguous, and donor-evidence
conflict positions unchanged.

The exact decoy patch-grid cells must be represented among the replaceable
positions. If a teacher map marks any decoy cell as evidence or ambiguous, the
target is explicitly excluded from both target and donor pools and recorded in
the audit. The preflight fails if these exclusions leave fewer than the locked
overall or per-class minimums; it never silently performs FCV without replacing
the shortcut region.

---

# 7. FCV forward pass and metrics

Intervene on raw patch embeddings before position embeddings and transformer
blocks:

```text
target image
 -> target patch embeddings
 -> replace mutually safe background positions with donor content
 -> add target positional embeddings
 -> prepend target CLS token
 -> transformer blocks
 -> classifier
```

For each candidate and target, evaluate five counterfactual donor draws.

Define counterfactual accuracy as donor-expanded accuracy:

\[
A_{\mathrm{cf}}=
\frac{1}{5N}\sum_{i=1}^{N}\sum_{k=1}^{5}
\mathbf{1}\!\left[\hat y_{ik}^{\mathrm{cf}}=y_i\right].
\]

Also record compact aggregate forms of:

- majority-vote counterfactual accuracy;
- mean true-class counterfactual probability;
- original-to-counterfactual confidence drop;
- eligible target fraction;
- mean number of replaced patches.

Do not save per-target or per-donor rows in the production campaign.

---

# 8. Locked selectors

## 8.1 Vanilla selector

\[
S_{\mathrm{Vanilla}}=A_{\mathrm{orig}},
\]

where `A_orig` is ordinary biased-validation accuracy.

## 8.2 Primary FCV selector

Use the harmonic mean locked after the Waterbirds development study:

\[
S_{\mathrm{FCV}}=
\frac{2A_{\mathrm{orig}}A_{\mathrm{cf}}}
{A_{\mathrm{orig}}+A_{\mathrm{cf}}+10^{-12}}.
\]

This selector is symmetric, parameter-free, and penalizes candidates that are
excellent on one validation view but poor on the other.

## 8.3 Oracle selector

\[
S_{\mathrm{Oracle}}=A_{\mathrm{oracle\ val}},
\]

using accuracy on the independent reversed-decoy Oracle validation split.

## 8.4 Post-hoc ceiling

The analysis-only ceiling selects maximum official test accuracy over all 1080
candidates. It is never presented as a deployable selector.

## 8.5 Tie breaking

For exact score ties, select ascending candidate ID. Do not use validation loss,
Oracle metrics, or test metrics to break an accuracy tie unless that secondary
rule was explicitly defined as a separate reported selector.

---

# 9. Counterfactual controls

Compute controls online using the same candidate state and target set.

## 9.1 Same-context donor control

Use donors with the same label and same corner. A semantic opposite-context
intervention should be more challenging than this control.

## 9.2 Random-mask control

Replace a matched number of eligible patch positions sampled independently of
the teacher map.

## 9.3 Shuffled-teacher-mask control

Assign another validation image's teacher mask while preserving the target
image and matched mask-area distribution.

## 9.4 Evidence-swap control

Swap safe evidence positions rather than safe background positions. This is a
diagnostic expected to damage digit classification and must not become a
selector.

## 9.5 Exact-mask analysis control

Repeat the primary counterfactual using the known synthetic digit mask. This
measures whether primary teacher quality limits FCV, but it remains explicitly
Oracle-mask analysis.

Control diagnostics are warning-only after the campaign starts. They cannot
abort or alter candidate selection after test results exist.

---

# 10. Online Oracle and test evaluation

At every epoch, evaluate the in-memory model on:

1. biased validation;
2. primary FCV counterfactuals;
3. FCV controls;
4. privileged Oracle validation;
5. official reversed-decoy test.

Store one aggregate row per candidate in each visibility-separated namespace.
Selector code must load only its authorized namespace:

```text
Vanilla -> biased validation aggregates
FCV -> biased validation + FCV aggregates
Oracle -> Oracle validation aggregates
Post-hoc analysis -> test aggregates
```

The official test evaluation reports:

- overall accuracy, primary;
- balanced-class accuracy, secondary;
- worst-class accuracy, secondary;
- ten per-class accuracies, analysis only.

Online test computation does not authorize test-driven stopping, grid changes,
or selector changes.

---

# 11. Final analyses

## 11.1 Selector outcomes

Report official test accuracy for the candidates selected by:

- biased-validation accuracy;
- harmonic FCV;
- privileged Oracle validation;
- post-hoc test ceiling.

## 11.2 Accuracy gap closure

Use official test accuracy:

\[
\mathrm{GapClosed}=
\frac{A_{\mathrm{test}}^{\mathrm{FCV}}-
      A_{\mathrm{test}}^{\mathrm{Vanilla}}}
     {A_{\mathrm{test}}^{\mathrm{Oracle}}-
      A_{\mathrm{test}}^{\mathrm{Vanilla}}}.
\]

Report the unclipped value. If the denominator is zero or negative, report the
three component accuracies and mark gap closure undefined rather than forcing
a percentage.

## 11.3 Candidate-pool headroom

Always report:

- test-accuracy range across all candidates;
- Oracle-selected test accuracy;
- post-hoc maximum test accuracy;
- performance by crop regime;
- whether the pool contains meaningfully robust candidates.

This separates selector failure from candidate-pool failure.

## 11.4 Rank analysis

Compute Spearman and Kendall association between each selector score and
official test accuracy. Report top-k recall for `k in {1, 5, 10, 25}`.

Rank analysis is post-hoc and must not change the primary selector.

## 11.5 Seed-stratified analysis

In addition to global selection, report each selector's best candidate within
each training seed and summarize selected test accuracy across seeds. This is
secondary because deployment selection operates over the full candidate pool.

---

# 12. Output and storage contract

Allowed persistent artifacts:

```text
split_manifests/*.csv
split_manifests/*.json
teacher_mask_audit/*.json
teacher_mask_audit/overlays/*.png
donor_plans/*.json (IDs and provenance only)
online_metrics/biased_validation/*.csv
online_metrics/fcv/*.csv
online_metrics/controls/*.csv
online_metrics/oracle_analysis_only/*.csv
online_metrics/test_analysis_only/*.csv
selection_results/*.csv
selection_results/*.json
plots/*.pdf or *.png
run_logs/*.out
run_logs/*.err
```

Forbidden persistent artifacts:

```text
candidate model checkpoints
retained selector-winner checkpoints
optimizer/resume states
per-epoch state dictionaries
token banks
patch embeddings
per-image logits
per-donor predictions
duplicate transformed datasets
```

Node-local token banks must be deleted in a `finally` block after each epoch.
The full campaign should remain far below 1 GiB excluding the already-existing
teacher maps and shared pretrained-model cache.

---

# 13. Tigris execution graph

Recommended dependency chain:

```text
preflight/cache
    -> one-run, one-epoch end-to-end smoke
        -> 108-task full online array (maximum 8 concurrent)
            -> completeness/provenance freeze
                -> selector and post-hoc report generation
```

Recommended full-array request:

```text
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --array=0-107%8
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
```

The one-day limit is deliberately generous. Based on the susceptibility pilot,
each full ten-epoch run should be short, although FCV and control scoring add
additional forward passes.

---

# 14. Required preflight and smoke gates

The full array must not launch unless all gates pass.

## 14.1 Data gate

- exactly 60,000 source training PNGs;
- exactly 10,000 official test PNGs;
- exact 48,000/6,000/6,000 disjoint partition;
- class-stratified counts within one example of proportional allocation;
- frozen image hashes;
- valid original patch encoding.

## 14.2 Teacher gate

- 100% biased-validation coverage;
- valid sample-ID resolution;
- valid 14x14 patch masks;
- eligible digit and background regions;
- decoy patch included in safe context for the required fraction;
- fixed qualitative overlays generated.

## 14.3 Donor gate

- five donors per eligible target;
- all donor labels differ from target label;
- all donors share target corner;
- no target self-donation;
- donor IDs limited to biased validation;
- deterministic plan regeneration produces the same hash.

## 14.4 Model gate

- exact timm version and model name;
- pretrained backbone hash matches cache;
- output shape `[B,10]`;
- 14x14 raw patch grid;
- reconstructed unmodified forward matches native forward numerically.

## 14.5 One-epoch production-path smoke

Run one representative candidate through:

- one real training epoch;
- biased validation;
- FCV donor construction and counterfactual forward;
- every control;
- Oracle validation;
- official test analysis;
- aggregate row serialization;
- token-bank cleanup;
- verification that no checkpoint or optimizer file exists.

The smoke must also project runtime and output size for all 108 runs. A failed
smoke blocks the full array through `afterok` dependencies.

---

# 15. Implementation steps

Implementation status (2026-07-18): Steps 1--12 are complete and covered by
focused unit, integration, provenance, launch-chain, and storage tests.

## Step 1 — Add and validate the full-campaign configuration

Create one frozen YAML configuration containing all paths, split rules, model
settings, candidate values, donor rules, selector definitions, controls,
storage constraints, and Tigris resources. Validate types, allowed values, and
the expected 108-run/1080-candidate cardinality.

## Step 2 — Build immutable data manifests

Audit the unmodified benchmark, create the exact 48k/6k/6k split, preserve the
official 10k test split, hash all source images, and write visibility-separated
manifests with one provenance bundle.

## Step 3 — Implement Oracle-view construction

Add an in-memory transform that locates the source training patch and replaces
it with `25*y`. Unit-test all labels, corners, and non-mutation of source PNGs.

## Step 4 — Audit and project teacher maps

Resolve maps for the exact biased-validation sample IDs, regenerate missing
maps if needed, project maps to 14x14 patch masks, validate eligibility, and
write fixed audit overlays.

## Step 5 — Implement the augmentation-diverse candidate grid

Extend candidate training with the four locked crop-scale regimes while
preserving the fixed ViT, optimizer, scheduler, precision, and deterministic
seed behavior. Keep evaluation preprocessing identical across regimes.

## Step 6 — Implement multiclass same-corner donor plans

Build deterministic five-donor plans using five distinct non-target labels and
same-corner matching. Add strict provenance and leakage checks.

## Step 7 — Implement spatially aligned FCV feature intervention

At raw patch embeddings, replace only mutually safe target/donor background
positions, preserve target evidence and ambiguous positions, apply target
position embeddings, and verify equivalence when no replacement occurs.

## Step 8 — Implement harmonic FCV and controls

Compute donor-expanded counterfactual accuracy, harmonic FCV, confidence
statistics, and all locked controls online. Store only aggregate metrics.

## Step 9 — Implement online Oracle and test analysis

Score the privileged Oracle view and official test while the candidate model is
in memory. Enforce namespace separation so selectors cannot load test values.

## Step 10 — Implement no-checkpoint orchestration

Create the ten-epoch in-memory training/evaluation loop. Persist one compact
row per candidate, delete ephemeral token banks every epoch, discard the model
after each run, and verify that no checkpoint-like files are produced.

## Step 11 — Implement selection, gap closure, and rank reports

Freeze the completed 1080-row matrix, select Vanilla/FCV/Oracle winners with
locked tie breaking, attach analysis-only test metrics afterward, and generate
gap-closure, headroom, crop-regime, rank, and seed-stratified reports.

## Step 12 — Add tests, Tigris smoke, and full launcher

Add unit and integration coverage for split exactness, patch reversal, donor
validity, forward reconstruction, harmonic scoring, visibility boundaries,
storage cleanup, and cardinality. Add the `afterok` Tigris dependency chain and
a concise launch/monitoring README.

---

# 16. Success criteria and interpretation

## Technical success

- all 108 runs and 1080 online candidates complete;
- one split, teacher, donor, model, and source provenance contract is used;
- no checkpoint, optimizer state, token bank, or embedding cache remains;
- selector outputs are reproducible from aggregate validation metrics;
- test metrics cannot affect selection.

## Scientific success

The primary confirmatory result is positive if harmonic FCV selects a candidate
with higher official reversed-test accuracy than ordinary biased-validation
selection.

Stronger supporting outcomes are:

- FCV approaches the privileged Oracle selector;
- FCV closes a positive and meaningful fraction of the Oracle accuracy gap;
- FCV ranks candidates better than biased-validation accuracy;
- semantic teacher masks outperform random and shuffled controls;
- results are directionally consistent across training seeds.

## Honest failure interpretations

- If Oracle and post-hoc ceilings are low, the candidate pool lacks headroom.
- If Oracle succeeds but FCV does not, the selector/intervention is the likely
  bottleneck.
- If exact masks succeed but teacher masks fail, teacher quality is the likely
  bottleneck.
- If random or shuffled masks match semantic masks, FCV may be measuring generic
  feature corruption rather than semantic shortcut sensitivity.
- If stronger crop regimes dominate every selector, report that interaction
  rather than hiding it.

No outcome authorizes changing the harmonic rule or candidate grid after test
results are inspected. Any follow-up protocol must be labeled as a separate
development study.
