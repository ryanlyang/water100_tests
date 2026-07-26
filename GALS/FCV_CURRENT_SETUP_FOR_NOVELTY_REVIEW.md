# Feature Counterfactual Validation (FCV)

## Current Setup, Implementation, Results, and Novelty-Review Brief

This document describes the current Feature Counterfactual Validation (FCV)
setup in enough detail for an outside researcher or coding agent to evaluate:

1. what the method actually does;
2. which parts are implementation choices rather than conceptual claims;
3. how the completed Waterbirds-100 and DecoyMNIST studies were run;
4. what the current evidence supports;
5. where the method may overlap with prior work; and
6. which novelty claims would and would not be defensible.

The intended use of this document is a technical and literature-review handoff.
It is deliberately candid about the current method's mathematical relationship
to patchwise pixel replacement.

---

## 1. Research question

FCV addresses model selection under spurious correlation.

Suppose the training data and the available validation data contain the same
shortcut correlation. For example, in Waterbirds-100:

```text
waterbird -> water background
landbird  -> land background
```

A classifier that recognizes birds and a classifier that recognizes
backgrounds can both achieve excellent validation accuracy. Ordinary biased
validation therefore cannot reliably determine which model will generalize
when the shortcut changes.

The ideal validation question is:

> If the target object's evidence is preserved but its shortcut context is
> changed to support a conflicting class, does the model's prediction still
> follow the target object?

FCV creates this missing validation condition inside a Vision Transformer's
patch representation. It then uses performance on both the original biased
validation examples and the feature counterfactuals to select a model.

FCV is currently a **model-selection method**, not a training objective. It
does not update the candidate model using counterfactual examples.

---

## 2. High-level method

For every model-epoch candidate, FCV performs the following operations:

1. Evaluate the candidate on an unchanged biased-validation holdout.
2. Use a frozen teacher map to identify target-object and safe-background
   patches in each validation image.
3. Run target and donor images through the candidate model's ViT patch
   projection.
4. Preserve target object tokens and uncertain boundary tokens.
5. Replace only safe target-background tokens with safe tokens from a
   deliberately conflicting donor context.
6. Add the target positions and run the modified sequence through the
   remaining transformer and classifier.
7. Measure whether predictions continue to follow the original target labels.
8. Select the candidate maximizing the harmonic mean of original validation
   accuracy and counterfactual accuracy.

The canonical selector is:

\[
S_{\mathrm{FCV}}=
\frac{2A_{\mathrm{orig}}A_{\mathrm{cf}}}
{A_{\mathrm{orig}}+A_{\mathrm{cf}}+10^{-12}}.
\]

The harmonic mean is symmetric and has no fitted mixing weight. It penalizes a
candidate that performs well on only one of the two validation views.

---

## 3. Candidate training and visibility boundaries

All selectors receive exactly the same candidate pool. Differences in final
performance therefore come from model selection rather than different model
training.

The data visible to each component are separated as follows:

| Component | Candidate train | Biased validation | Privileged oracle validation | Test |
|---|---:|---:|---:|---:|
| Candidate training | Yes | No | No | No |
| Vanilla selector | No | Yes | No | No |
| FCV selector | No | Yes | No | No |
| Oracle selector | No | No | Yes | No |
| Post-hoc analysis | No | No | Analysis only | Analysis only |

The official test metrics are not permitted to influence donor construction,
mask thresholds, candidate training, selector scores, ties, or candidate
selection. In the completed online campaigns, test metrics were written as
analysis-only columns and attached to frozen selections for final reporting.

The principal selectors are:

- **Vanilla:** maximize biased-validation accuracy.
- **FCV:** maximize the harmonic mean of biased-validation accuracy and FCV
  counterfactual accuracy.
- **Oracle:** maximize accuracy on a privileged counter-biased validation set.
- **Post-hoc ceiling:** maximize test accuracy over the complete candidate
  pool. This is not a usable selector; it measures available headroom.

---

## 4. What "token space" means in the current implementation

The current model is an ImageNet-pretrained ViT-S/16. A 224 x 224 RGB image is
divided into a 14 x 14 grid of non-overlapping 16 x 16 patches:

```text
224 x 224 x 3 image
    -> 196 patches
    -> each patch contains 16 x 16 x 3 = 768 pixel values
```

For patch `i`, let `x_i` be its flattened 768-dimensional pixel vector. The
ViT patch projection is a learned linear map:

\[
z_i = E(x_i) = W x_i + b,
\]

where the ViT-S/16 patch token `z_i` has 384 dimensions. In practice, this
operation is implemented as a convolution whose kernel and stride both equal
16, but it is equivalent to independently projecting every non-overlapping
patch.

The resulting tensor has shape:

```text
196 patches x 384 feature dimensions
```

FCV intervenes at this point: after patch projection but before target
positional embeddings, the CLS token, and transformer attention blocks are
applied.

These are **local, uncontextualized patch tokens**. Calling them
"contextualized tokens" would be inaccurate. Tokens become contextualized
only after transformer attention mixes information across image locations.

The candidate's patch projection is used for both target and donor images at
the same training epoch. Donor vectors therefore have the expected dimension,
scale, and candidate-specific representation and originate from real image
patches.

---

## 5. Teacher maps and patch eligibility

Each biased-validation image has a frozen spatial evidence map generated by an
external teacher pipeline. The teacher map is intended to localize the target
object rather than the shortcut context.

The image-space map is average-pooled onto the ViT's 14 x 14 patch grid. Two
thresholds divide patches into:

- **Evidence patches:** confidently overlap the target object.
- **Safe-background patches:** confidently outside the target object.
- **Ambiguous patches:** fall between the evidence and background thresholds.

FCV preserves evidence and ambiguous target tokens. Only safe-background
tokens are replaceable.

This conservative policy is necessary for causal interpretability of the
intervention. If an object patch were removed and the prediction failed, the
failure could not be attributed specifically to changed context.

The teacher is frozen. Candidate models cannot modify the teacher maps, and
test data are not used to construct or audit them.

### DecoyMNIST eligibility amendment

The DecoyMNIST preflight found 65 of 6,000 teacher maps whose evidence regions
overlapped the synthetic decoy cells. Those samples were explicitly marked
FCV-ineligible and excluded from both target and donor pools. The exclusion was
implemented before candidate training or test-result inspection. Coverage,
class counts, sample IDs, and exclusion reasons remain in the audit artifacts.

The method does not silently replace the wrong region when the teacher map
cannot distinguish object evidence from the shortcut.

---

## 6. Donor selection: the main intervention is not arbitrary mixing

The primary FCV intervention deliberately selects a context that conflicts
with the target's shortcut. It does not sample indiscriminately from every
validation image.

Donor identities are sampled reproducibly within a valid conflicting-context
pool. Thus, the random element chooses *which* eligible opposing donor is used;
it does not decide whether the donor context opposes the target.

### 6.1 Waterbirds-100 donor rule

Waterbirds-100 is completely confounded:

```text
waterbird label -> water context
landbird label  -> land context
```

Class label can therefore serve as a shortcut-context proxy without explicit
group annotations.

- A waterbird target receives safe-background tokens pooled from landbird
  validation images, which provide land context.
- A landbird target receives safe-background tokens pooled from waterbird
  validation images, which provide water context.

Conceptually:

\[
\text{waterbird object tokens}+\text{land-context tokens},
\]

or:

\[
\text{landbird object tokens}+\text{water-context tokens}.
\]

Background banks are assembled from safe donor patches across many images. A
single donor image is not required to provide a complete object-free scene.
The donor plan is fixed and reused across candidates so selector comparisons
do not receive different random counterfactual draws.

### 6.2 DecoyMNIST donor rule

In the biased DecoyMNIST data, a 5 x 5 corner patch encodes the digit class.
For a target with class `y` and corner `c`, every main-method donor must:

```text
have label y' != y
have the same corner c
be a different image from the target
come only from biased validation
pass the teacher-mask eligibility audit
```

Five distinct conflicting donor labels are deterministically selected for
each eligible target. Mutually safe replacement is performed only where the
same patch location is safe background for both target and donor.

Same-corner matching ensures that FCV replaces the actual class-coded decoy
location rather than inserting a black or unrelated patch from another corner.

Conceptually:

\[
\text{visual evidence for digit 3}
+
\text{corner-patch feature associated with digit 7}.
\]

The correct counterfactual label remains the target digit, 3.

---

## 7. Counterfactual forward pass

Let `Z_i` be the target image's uncontextualized patch-token sequence. Let
`B_i` denote its safe-background positions. FCV creates a copy:

\[
Z_i^{\mathrm{cf}} \leftarrow Z_i.
\]

At replaceable positions, it inserts donor content:

\[
Z_i^{\mathrm{cf}}[B_i] \leftarrow D_i[B_i].
\]

Evidence and ambiguous target positions are unchanged. The model then:

1. adds the target positional embeddings;
2. prepends the target CLS token;
3. applies the normal transformer blocks and classification head; and
4. evaluates the output against the target's original class label.

Using target positional embeddings means donor visual content is interpreted
at the target location rather than carrying the donor image's original
position encoding.

DecoyMNIST evaluates five donor-expanded counterfactuals per eligible target:

\[
A_{\mathrm{cf}}=
\frac{1}{5N}\sum_{i=1}^{N}\sum_{k=1}^{5}
\mathbf{1}[f(Z_{i,k}^{\mathrm{cf}})=y_i].
\]

The production reports also retain aggregate diagnostics such as majority-vote
counterfactual accuracy, true-class probability, confidence drop, eligible
target fraction, and replacement coverage. They do not retain per-target
embeddings or per-donor predictions.

---

## 8. Why use a feature-space implementation?

The current implementation was motivated by comparison with **full image-level
background compositing**, not by a claim that linear patch projection makes
patch swapping mathematically more powerful.

Whole-image counterfactual generation introduces several confounds:

- Removing the donor object leaves a hole that requires inpainting, blur,
  masking, texture synthesis, or a generative model.
- Pasting an object can create seams, scale mismatches, lighting differences,
  perspective errors, and color inconsistencies.
- A prediction change may be caused by editing artifacts rather than the
  intended context change.
- One donor must often provide an entire plausible object-free scene.
- A separate generator introduces additional learned assumptions and failure
  modes.

The FCV token-bank implementation instead:

- preserves the target object representation exactly at the intervention
  boundary;
- uses representations of real donor patches rather than zeros or synthetic
  noise;
- pools safe context patches across many donor images;
- avoids donor-object removal and full-scene reconstruction;
- creates counterfactuals online without storing synthetic images; and
- directly measures sensitivity to changed context at a defined model
  interface.

These are operational and experimental-control advantages over full-scene
image synthesis.

---

## 9. Critical mathematical equivalence to patchwise pixel replacement

The current intervention occurs immediately after a non-overlapping linear
patch projection. Therefore, exact aligned replacement of complete pixel
patches is mathematically equivalent to replacing their projected tokens.

Let target patch `x_i` be replaced by donor patch `d_i` in pixel space. The
token received by the transformer is:

\[
E(d_i)=Wd_i+b.
\]

If FCV first computes the donor token and then swaps it, the result is also:

\[
z_i^{\mathrm{cf}}=E(d_i)=Wd_i+b.
\]

After adding the same target positional embedding `p_i`, both procedures pass:

\[
E(d_i)+p_i
\]

to the transformer. Under the current standard patch projection, the later
contextualized transformer states and logits should therefore be the same,
apart from numerical or preprocessing differences.

This means the present method should **not** claim that pre-attention token
swapping is inherently more expressive or more natural than exact, aligned
16 x 16 pixel-patch swapping.

The defensible distinction is:

1. FCV is much cleaner than full-object cut-and-paste or generated-background
   counterfactuals.
2. The current token implementation is a convenient and auditable realization
   of patch-level context intervention.
3. The main conceptual contribution under review is counterfactual
   model selection under biased validation, not the proposition that a linear
   patch projection creates a fundamentally new intervention.

A direct exact pixel-patch swap comparison should be added as an implementation
equivalence control. A full-scene compositing or inpainting comparison would
address the separate artifact-avoidance motivation.

---

## 10. Controls in the implemented studies

The main selector uses opposing-context donor tokens and teacher-defined
background positions. The implementation also computes controls intended to
test whether the signal is specific to that design:

- **Same-context donors:** replace background using context that agrees with
  the target shortcut.
- **Random masks:** alter randomly selected spatial locations.
- **Shuffled masks:** break the correspondence between target image and teacher
  mask.
- **Evidence swapping:** replace target-evidence rather than background tokens.

These controls are diagnostic and are not used to select the candidate. Their
purpose is to distinguish meaningful shortcut intervention from generic
sensitivity to feature perturbation.

---

## 11. Completed Waterbirds-100 study

### 11.1 Candidate pool

The reduced Waterbirds-100 campaign used:

- Model: ImageNet-pretrained ViT-S/16.
- Learning rates: `1e-5`, `3e-5`, `1e-4`.
- Weight decays: `0.01`, `0.05`, `0.1`.
- Seeds: `0`, `1`, `2`.
- Training epochs: 20.
- Training runs: `3 x 3 x 3 = 27`.
- Model-epoch candidates: `27 x 20 = 540`.

The original Waterbirds-100 training split was deterministically partitioned
into candidate training and biased validation. The official validation split
was privileged oracle analysis only, and the official test split was analysis
only.

### 11.2 Selection results

Using the current harmonic FCV definition for candidate analysis:

| Selector | Selected test accuracy |
|---|---:|
| Biased-validation accuracy | 69.93% |
| Harmonic FCV | 78.82% |
| Oracle-validation accuracy | 77.11% |
| Post-hoc pool maximum | 78.82% |

The oracle validation set is a finite proxy for test performance rather than a
guaranteed test upper bound. Therefore, an FCV-selected candidate can exceed
the candidate chosen by oracle validation while remaining bounded by the
post-hoc pool maximum.

Across all 540 candidates, Spearman rank correlation with test accuracy was:

| Signal | Spearman rho |
|---|---:|
| Biased-validation accuracy | 0.080 |
| Harmonic FCV score | 0.824 |
| Oracle-validation accuracy | 0.931 |

The harmonic formulation was developed during the Waterbirds study and locked
before the confirmatory DecoyMNIST campaign. Waterbirds should therefore be
described as development evidence for the current selector, not as a
preregistered confirmation of that exact aggregation rule.

---

## 12. Completed DecoyMNIST confirmatory study

### 12.1 Candidate pool

The DecoyMNIST campaign used:

- Model: the same ImageNet-pretrained ViT-S/16 family.
- Candidate training images: 48,000 biased examples.
- Biased-validation images: 6,000 biased examples.
- Oracle-validation source images: 6,000 disjoint examples transformed in
  memory to the reversed-decoy distribution.
- Official reversed-decoy test images: 10,000, analysis only.
- Learning rates: three values matching the Waterbirds scale.
- Weight decays: `0.01`, `0.05`, `0.1`.
- Crop minimum scales: `1.0`, `0.8`, `0.6`, `0.4`.
- Seeds: `0`, `1`, `2`.
- Training epochs: 10.
- Training runs: `3 x 3 x 4 x 3 = 108`.
- Model-epoch candidates: `108 x 10 = 1,080`.

All candidate metrics were computed online. Per-epoch model checkpoints,
optimizer states, token banks, and retained selector winners were not saved.
Temporary donor features were deleted after each epoch.

### 12.2 Selection results

| Selector | Selected test accuracy |
|---|---:|
| Biased-validation accuracy | 18.75% |
| Harmonic FCV | 71.67% |
| Oracle-validation accuracy | 74.13% |
| Post-hoc pool maximum | 74.13% |

FCV closed:

\[
\frac{71.67-18.75}{74.13-18.75}=95.56\%
\]

of the available vanilla-to-oracle accuracy gap.

Across all 1,080 candidates:

| Signal | Spearman rho with test accuracy | Kendall tau-b with test accuracy |
|---|---:|---:|
| Biased-validation accuracy | -0.328 | -0.252 |
| Harmonic FCV score | 0.904 | 0.735 |
| Oracle-validation accuracy | 0.993 | 0.967 |

FCV recovered 6 of the true top-10 candidates in its own top 10 and 16 of the
true top-25 candidates in its top 25. Test metrics were attached only after
selector freezing.

### 12.3 Crop-regime headroom

| Minimum crop scale | Mean test accuracy | Maximum test accuracy |
|---|---:|---:|
| 1.0 | 19.05% | 26.79% |
| 0.8 | 20.80% | 41.03% |
| 0.6 | 41.71% | 70.16% |
| 0.4 | 50.73% | 74.13% |

The augmentation sweep created a candidate pool with substantial variation in
shortcut susceptibility. FCV's role was to identify a strong candidate from
that pool without reading oracle or test performance.

---

## 13. What the completed evidence currently supports

The current experiments support the following statements:

1. Biased validation can rank shortcut-dependent candidates incorrectly.
2. FCV provides candidate-ordering information strongly associated with
   reversed-context test accuracy on two datasets.
3. FCV selected substantially stronger candidates than biased-validation
   accuracy from the same candidate pools.
4. On confirmatory DecoyMNIST, FCV recovered most of the performance available
   to a privileged oracle selector.
5. The framework can be executed online without retaining a large collection
   of model checkpoints or feature banks.

The current evidence does not by itself establish:

1. that token swapping is superior to exact aligned pixel-patch swapping;
2. that FCV works when class label is not a reliable context proxy;
3. that FCV is robust to poor or systematically biased teacher maps;
4. that hybrid token sequences are fully in-distribution internal states;
5. that the method transfers beyond ERM-trained ViTs; or
6. that FCV improves the trained representation rather than only selecting a
   better existing candidate.

---

## 14. Important assumptions and limitations

### 14.1 Complete confounding makes donor construction unusually simple

In Waterbirds-100 and DecoyMNIST, class label determines the shortcut context.
This allows FCV to construct conflicting donor pools using class labels without
group annotations.

In datasets where each class appears across multiple contexts, opposite class
is not necessarily opposite context. A broader group-agnostic method will need
an inferred context representation, such as:

- VLM-based context prompts;
- clustering of safe-background features;
- an unsupervised context classifier; or
- a separately declared oracle-context analysis.

Using true background or shortcut labels for the main selector would weaken
the group-agnostic claim unless those labels are part of the stated problem
setting.

### 14.2 Teacher-map dependence

FCV assumes that the teacher can conservatively separate object evidence from
safe context. Teacher errors can cause object deletion, failure to alter the
shortcut, or selection bias toward models aligned with the same teacher.

Future evaluation should vary teacher backbones and include independent,
human, or oracle masks for analysis.

### 14.3 Hybrid token distributions

Every donor token comes from a real patch through the same candidate patch
projection, but the assembled sequence may still be an unusual combination of
features. Same-context and mask controls partly diagnose generic perturbation
sensitivity, but they do not prove that every counterfactual state lies on the
model's natural feature manifold.

### 14.4 FCV needs candidate diversity

FCV cannot select a robust candidate if training never produces one. The large
gain on DecoyMNIST depended on augmentation creating candidates with a wide
range of reversed-decoy performance.

---

## 15. Planned extensions

### 15.1 UrbanCars

UrbanCars contains two controlled shortcuts: background and a co-occurring
object. It is intended to expose a Whac-A-Mole failure mode in which mitigating
one shortcut increases reliance on another.

An FCV study should define and compare:

- background-only intervention;
- co-occurring-object-only intervention;
- joint intervention of both shortcut regions; and
- corresponding same-context and random controls.

The main unresolved design question is how to infer conflicting context pools
without using privileged shortcut labels in the primary selector.

### 15.2 SpuCoDogs

SpuCoDogs is the dog-only subset of SpuCoAnimals, curated from ImageNet to
contain naturally occurring class-context correlations. It tests FCV under
less synthetic object boundaries and less perfectly controlled contexts.

This dataset will require a defensible group-agnostic context-inference rule
rather than relying automatically on opposite-class donors.

### 15.3 Training-family transfer

The proposed core comparison is to apply Vanilla, FCV, and Oracle selectors
independently within candidate pools trained using:

- ERM;
- Right for the Right Reasons (RRR) explanation regularization; and
- Elastic Representation (ElRep) representation regularization.

AFR is a lower-cost optional fourth family. GroupDRO can serve as an explicitly
group-labeled oracle training reference but should not be included in the
group-agnostic claim.

For RRR, the study should address possible circularity if training and FCV use
the same teacher maps. At least one independent teacher or alternative mask
source should be evaluated.

---

## 16. Potentially novel components to investigate

The novelty review should separate the following possible contributions.

### 16.1 Counterfactual validation as model selection

The strongest candidate contribution may be using teacher-localized,
context-conflicting feature interventions specifically as a **validation and
checkpoint-selection signal** when the available validation distribution is
itself spuriously correlated.

This should be compared against prior work in:

- counterfactual validation;
- group-free robust model selection;
- environment inference;
- feature intervention diagnostics;
- causal representation validation;
- explanation-guided validation; and
- validation under distribution shift.

### 16.2 ViT patch-token context intervention

The implementation uses candidate-specific real donor patch embeddings,
teacher-constrained replacement, fixed conflicting donors, and target
positional embeddings. Similar mechanisms may already appear in TokenMix,
TransMix, CutMix-like training, token-labeling, representation mixup, causal
intervention, or ViT interpretability papers.

The review must distinguish methods that use token replacement for training
augmentation from methods that use it solely for model selection.

### 16.3 Harmonic original/counterfactual selector

The harmonic selector is simple and likely not independently novel. Its value
is that it is symmetric, parameter-free, and precommitted in the confirmatory
study. Any novelty claim should concern the complete validation framework, not
the harmonic mean alone.

### 16.4 Online leakage-separated campaign design

The implementation evaluates every epoch in memory, freezes selectors before
attaching test metrics, uses shared candidates and donor plans, and avoids
retaining checkpoints. This is strong experimental methodology, but it is
probably not a standalone algorithmic novelty.

### 16.5 Genuinely contextualized feature interventions

Replacing tokens after one or more attention blocks would no longer be
equivalent to pixel-patch replacement. It could become a distinct
feature-space intervention, but later tokens are spatially mixed and may no
longer cleanly separate object from context. A layer sweep would need to study
that tradeoff rather than assume deeper is better.

---

## 17. Questions for the outside novelty reviewer

Please perform a deep literature and implementation review addressing these
questions:

1. Has prior work used feature counterfactuals specifically to select models or
   checkpoints under biased validation data?
2. Has prior work used object-preserving, opposing-context ViT token swaps as a
   validation criterion rather than a training augmentation?
3. Which prior methods are closest conceptually: CutMix/TokenMix variants,
   representation mixup, causal feature intervention, explanation-guided
   validation, environment inference, or group-free model selection?
4. Does any prior work combine ordinary validation and counterfactual feature
   accuracy for selection using a harmonic or worst-view objective?
5. Is the current pre-attention intervention too mathematically close to
   patchwise pixel replacement to support a feature-space novelty claim?
6. Would an after-block-`k` contextualized-token intervention be meaningfully
   more novel, and what validity risks would it introduce?
7. What is the strongest accurate novelty statement supported by the current
   method and evidence?
8. What additional baseline would reviewers expect before accepting the model
   selection claim?
9. Which controls are still missing, especially exact pixel-patch replacement,
   full image compositing, random donors, independent masks, and inferred
   contexts?
10. Does class-as-context-proxy in the current datasets materially limit the
    group-agnostic claim?
11. Are the Waterbirds development and DecoyMNIST confirmatory results reported
    with appropriate separation and statistical caution?
12. What papers must be cited to avoid overstating novelty?

The reviewer should prioritize finding conceptual or experimental overlap over
suggesting superficial terminology changes.

---

## 18. Code and design-document locations

Current repository root:

```text
/home/ryan/ComputerScience/LearnToLook/SwitchVLM/GALS
```

Waterbirds-100 implementation:

```text
experiments/fcv_vit_waterbirds100/
```

DecoyMNIST implementation:

```text
experiments/fcv_vit_decoymnist/
```

Important Waterbirds modules include:

```text
experiments/fcv_vit_waterbirds100/src/fcv/token_banks.py
experiments/fcv_vit_waterbirds100/src/fcv/fcv_scoring.py
experiments/fcv_vit_waterbirds100/src/fcv/vit_counterfactual_forward.py
experiments/fcv_vit_waterbirds100/src/fcv/online_study.py
experiments/fcv_vit_waterbirds100/configs/waterbirds100_vit_s16_first_study.yaml
```

Important DecoyMNIST modules include:

```text
experiments/fcv_vit_decoymnist/src/decoy_donor_plans.py
experiments/fcv_vit_decoymnist/src/decoy_fcv_scoring.py
experiments/fcv_vit_decoymnist/src/decoy_online_study.py
experiments/fcv_vit_decoymnist/src/decoy_selection_analysis.py
experiments/fcv_vit_decoymnist/configs/decoymnist_vit_s16_fcv_full_online.yaml
```

Design documents:

```text
/home/ryan/ComputerScience/LearnToLook/SwitchVLM/feature_counterfactual_validation_blueprint.md
/home/ryan/ComputerScience/LearnToLook/SwitchVLM/vit_feature_counterfactual_validation_implementation.md
/home/ryan/ComputerScience/LearnToLook/SwitchVLM/vit_fcv_first_study_implementation_plan.md
experiments/fcv_vit_decoymnist/FCV_DECOYMNIST_FULL_CAMPAIGN_IMPLEMENTATION_PLAN.md
```

Downloaded completed-result artifacts currently used for the numerical summary
are under:

```text
/home/ryan/ComputerScience/LearnToLook/SwitchVLM/download_logs/logsWaterbird/fcv_vit_waterbirds100_first_study/
/home/ryan/ComputerScience/LearnToLook/SwitchVLM/download_logs/logsMNIST/fcv_vit_decoymnist_full_campaign/
```

---

## 19. Concise current claim

A conservative summary of the current contribution is:

> Feature Counterfactual Validation is a group-label-free model-selection
> framework for settings in which ordinary validation preserves the training
> shortcut. FCV uses frozen evidence maps to preserve target-object ViT patch
> representations while replacing safe contextual representations with
> conflicting donor context. It selects candidates that perform well on both
> original and counterfactual validation views. In Waterbirds-100 development
> experiments and a locked DecoyMNIST confirmation, FCV ranked candidate test
> performance substantially better than ordinary biased validation and selected
> candidates close to privileged oracle selection.

Whether this complete framework is novel, and which prior work most closely
anticipates it, is the subject of the requested external review.
