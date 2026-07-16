# Feature-Counterfactual Validation for Spatial Shortcut Robustness

## Working title

**Feature-Counterfactual Validation for Spatial Shortcut Robustness**

Alternative titles:

- **Validation Without Counterexamples: Feature-Counterfactual Model Selection for Spatial Bias**
- **When Validation Is Biased: Feature-Counterfactual Model Selection under Spatial Shortcut Correlation**
- **Do Predictions Follow Evidence or Background? Feature-Counterfactual Validation for Robust Recognition**
- **Spatial Teacher Maps Enable Counterfactual Model Selection under Complete Shortcut Bias**

---

## One-sentence idea

When validation data is itself spuriously correlated and may contain no natural counterexamples, ordinary validation accuracy cannot distinguish evidence-based models from shortcut-based models. We propose **Feature-Counterfactual Validation (FCV)**: use spatial teacher maps to split a model's internal spatial representations into evidence and background components, construct missing counterexamples by swapping background activations in feature space, and select models whose predictions follow evidence rather than background.

---

## The core research niche

Many spurious-correlation and domain-generalization papers focus on training methods. But the practical question is often:

> Given a pool of trained models, checkpoints, seeds, losses, and hyperparameters, how do we choose the one that will generalize when the validation set is also shortcut-contaminated?

This is especially hard in a **no-counterexample validation** setting:

- Every waterbird validation image appears on a water background.
- Every landbird validation image appears on a land background.
- Ordinary validation accuracy rewards both:
  - a true bird-shape model, and
  - a water/land-background shortcut model.
- A selector using only original image-label pairs may have no way to distinguish them.

The proposed paper attacks that model-selection gap directly.

---

## Why this is not just another training method

The main object is not a new classifier architecture.

The main object is a **model-selection criterion**:

> Which trained model should we select when validation accuracy is contaminated by the same spatial shortcut as training?

The proposed answer:

> Select the model whose predictions are stable when background features are counterfactually replaced, and whose predictions follow object/evidence features rather than background/context features.

A training regularizer can be included, but the paper should be **validation-first**.

---

## Why this is not just saliency-mask alignment

A nearby but weaker idea would be:

> Generate a saliency map for the candidate model and compare it to a segmentation mask.

That is useful, but it overlaps heavily with XAI localization literature. It asks:

> Does the explanation map overlap the object mask?

FCV asks a different behavioral question:

> If we intervene on the model's background representations, does the prediction change? Does the prediction follow the foreground evidence or the swapped background?

This is closer to a causal intervention than a post-hoc explanation overlap score.

Saliency alignment is about *where the model appears to look*. FCV is about *whether changing background activations changes the model's decision*.

---

## Why this is not pixel-space background swapping

Pixel-space background swapping has a major problem: to create a waterbird-on-land counterfactual, we need a clean land background. But in a landbird image, the landbird may occupy most of the image. Removing it creates holes, and filling those holes requires inpainting or generation. The validation metric may then measure sensitivity to artifacts rather than background reliance.

FCV avoids this by swapping **background activations**, not pixels.

Instead of needing a full clean donor background image, FCV builds a bank of background tokens/features from all non-evidence locations across many validation images. If a donor image only has a few reliable background tokens, those tokens can still be used. There is no image-level hole filling.

---

## High-level method

For each candidate model and each validation image:

1. Run the original RGB image through the model normally up to a chosen layer.
2. Use a spatial teacher map to identify evidence locations and background locations.
3. Keep the target image's evidence activations fixed.
4. Replace the target image's background activations with background activations sampled from another context.
5. Continue the forward pass from the patched layer.
6. Measure whether the model's prediction follows the evidence label or the swapped background context.
7. Use this feature-counterfactual behavior as a model-selection signal.

For Waterbirds100-style validation:

- Target: waterbird evidence + water background.
- Counterfactual: waterbird evidence + land-background features.
- Desired behavior: still predict waterbird.

For the reverse:

- Target: landbird evidence + land background.
- Counterfactual: landbird evidence + water-background features.
- Desired behavior: still predict landbird.

A shortcut model may flip or lose confidence when the background features are swapped.

---

## The best paper version

The strongest version is:

> **Feature-Counterfactual Validation is a validation-first framework for no-counterexample spatial shortcut settings. It is teacher-map agnostic, architecture-aware, and optionally strengthened by counterfactual-compatible training.**

The paper should have three layers:

### 1. Main contribution: FCV as a model-selection criterion

Use feature-space background swaps to rank/select candidate models when validation data is fully or heavily shortcut-correlated.

### 2. Teacher-map study

Evaluate FCV with:

- oracle/human/dataset-provided masks,
- VLM-generated masks,
- noisy masks,
- wrong-class masks,
- random same-area masks.

This makes the method scientifically grounded and avoids overcommitting to a particular VLM.

### 3. Optional training extension

Train models with a feature-counterfactual consistency regularizer so that internal counterfactuals become less out-of-distribution. This is not the main claim, but it addresses the concern that feature-swapped activations may be unnatural.

---

## Problem formulation

We have a dataset of images and labels:

```text
D_train = {(x_i, y_i)}
D_val   = {(x_i, y_i)}
D_test  = {(x_i, y_i, g_i)} optional group labels for evaluation only
```

There is an unobserved or unavailable spurious attribute `s`, often background/context:

```text
s = water / land
s = indoor / outdoor
s = snow / grass / forest
s = camera-trap location
```

In ordinary validation:

```text
P_val(s | y) is highly correlated
```

In the hard case:

```text
P_val(s = water | y = waterbird) = 1
P_val(s = land  | y = landbird)  = 1
```

No validation example naturally breaks the shortcut.

A candidate model pool:

```text
M = {f_1, f_2, ..., f_K}
```

is produced by different methods, seeds, checkpoints, hyperparameters, and architectures.

Goal:

> Select the model with best true robust performance, such as worst-group accuracy or OOD test accuracy, without using validation group labels.

---

## Why ordinary validation can fail

If validation is completely spuriously correlated, two models can have identical validation predictions:

- `f_evidence`: classifies by object/bird evidence.
- `f_shortcut`: classifies by water/land background.

On the biased validation set, both are correct.

But on a shortcut-breaking test set:

- `f_evidence` generalizes.
- `f_shortcut` fails.

Therefore, any selector that only observes original validation images and labels may be unable to distinguish the two.

This motivates an external intervention signal.

FCV supplies that signal by constructing feature-space counterexamples that the original validation set lacks.

---

## Spatial teacher maps

FCV assumes access to spatial teacher maps:

```text
M_i ∈ [0,1]^{H×W}
```

where `M_i` indicates likely class evidence / foreground / object-relevant pixels.

Important framing:

> The method is **not** a VLM method at its core. It is a spatial-teacher-map method.

Teacher maps can come from:

1. **Human/dataset-provided masks**
   - upper-bound / oracle analysis.
   - Example: Waterbirds has segmentation-style masks in many setups.

2. **VLM or open-vocabulary segmentation**
   - practical/scalable version.
   - Examples: WeCLIP+, Grounded-SAM, CLIP-based segmenters, SAM with class prompts, etc.

3. **Class-agnostic objectness maps**
   - SAM or object proposal methods.

4. **Noisy maps**
   - generated by perturbing oracle/VLM masks.

5. **Negative-control maps**
   - random same-area masks,
   - shuffled masks,
   - wrong-class masks,
   - background masks.

The paper should repeatedly emphasize:

> FCV is only as good as the spatial teacher signal, so we explicitly measure teacher-map quality and robustness.

---

## Spatial teacher maps: notation

For image `x_i`, label `y_i`, and teacher mask `M_i`:

- Evidence region: locations where `M_i = 1` or high.
- Background region: locations where `M_i = 0` or low.

At model layer `ℓ`, the model produces a spatial representation:

### CNN-like model

```text
h_i^ℓ ∈ R^{H_ℓ × W_ℓ × C}
```

### ViT-like model

```text
h_i^ℓ ∈ R^{N_ℓ × C}
```

where `N_ℓ` is the number of patch tokens.

The mask is resized or pooled to the representation grid:

```text
m_i^ℓ ∈ [0,1]^{H_ℓ × W_ℓ}
```

or

```text
m_i^ℓ ∈ [0,1]^{N_ℓ}
```

---

## Feature-counterfactual construction

Given a target image `i` and donor image `j`, construct a patched representation:

```text
h_cf = m_i * h_i + (1 - m_i) * B_j
```

where:

- `h_i` is the target representation,
- `m_i` preserves target evidence locations,
- `B_j` supplies donor background activations.

The forward pass then continues from layer `ℓ`:

```text
p_cf = f_{>ℓ}(h_cf)
```

Desired behavior:

```text
argmax p_cf = y_i
```

and ideally:

```text
p_cf(y_i) remains high.
```

---

## Background banks

Instead of using one donor image to provide a complete background, build background banks.

For each validation image, collect background activations:

```text
B_i = {h_i[k] : m_i[k] < τ_bg}
```

Group background banks by inferred or known context.

In Waterbirds100-style validation, because labels and backgrounds are perfectly correlated, labels can approximate context for bank construction:

```text
Bank_water = background tokens from waterbird validation images
Bank_land  = background tokens from landbird validation images
```

For a waterbird target:

```text
sample replacement background tokens from Bank_land
```

For a landbird target:

```text
sample replacement background tokens from Bank_water
```

This creates the missing counterexamples:

```text
waterbird evidence + land-background features
landbird evidence + water-background features
```

For multi-class or nonbinary datasets, background banks can be constructed using:

- metadata if available,
- VLM-inferred context labels,
- clustering of background embeddings,
- class-conditional opposite-background sampling,
- nearest-neighbor background retrieval with context dissimilarity.

---

## Donor sampling strategies

### Simple sampling

Randomly sample donor background activations from the opposite context bank.

Pros:

- easy,
- fast,
- simple baseline.

Cons:

- may create unnatural spatial arrangements.

### Position-aware sampling

For each target background position, sample donor background tokens from similar spatial positions.

Example:

```text
top-row target background token → top-row donor background token
bottom-row target background token → bottom-row donor background token
```

Pros:

- preserves coarse layout,
- reduces feature distribution shock.

### Texture/statistic-aware sampling

Match donor background tokens by low-level or mid-level similarity while ensuring context differs.

Pros:

- smoother internal counterfactuals,
- better control.

Cons:

- more complex,
- risk of preserving too much of the original background type.

### Multi-donor sampling

Fill target background positions using background tokens from multiple donor images.

Pros:

- avoids relying on one donor,
- handles large donor objects.

Cons:

- more internal distribution shift.

### Best initial implementation

Start with:

1. position-aware sampling,
2. multiple random donor samples per target,
3. aggregation over 5–10 counterfactual draws per image.

---

## FCV scoring

A good selector should prefer models that:

1. maintain accuracy on original validation images,
2. remain correct when background features are swapped,
3. do not collapse under opposite-background swaps more than under control swaps,
4. show higher sensitivity to evidence swaps than background swaps.

### Basic FCV score

For model `f`:

```text
FCV(f) = Acc_original(f) + λ * Acc_bg_swap(f)
```

where `Acc_bg_swap` is accuracy on feature-counterfactual validation examples.

This is simple but may be too crude.

### Better: confidence-stability score

For each validation image:

```text
S_i = p_f(y_i | original) - p_f(y_i | background-swapped)
```

A robust model should have small drop under background swap.

Define:

```text
Stability_bg(f) = - mean_i S_i
```

Higher is better.

### Better: differential control-normalized score

Feature swaps may create unnatural hidden states. So compare opposite-background swaps to control swaps.

For each image:

```text
drop_opposite = p_y(original) - p_y(opposite_background_swap)
drop_same     = p_y(original) - p_y(same_background_swap)
```

Then:

```text
ShortcutSensitivity = drop_opposite - drop_same
```

A model overly sensitive to opposite shortcut backgrounds will have high `ShortcutSensitivity`.

Then select with:

```text
FCV(f) = Acc_original(f) - λ * ShortcutSensitivity(f)
```

### Evidence-vs-background sensitivity gap

Also test evidence swaps:

- Background swap should not change prediction much.
- Evidence swap/removal should change prediction.

Define:

```text
drop_bg = p_y(original) - p_y(background_swapped)
drop_ev = p_y(original) - p_y(evidence_swapped_or_removed)
```

Then:

```text
EvidenceRelianceGap = drop_ev - drop_bg
```

A robust model should have high gap.

Final score:

```text
FCV(f) =
  Acc_original(f)
  + α * Acc_bg_swap(f)
  + β * EvidenceRelianceGap(f)
  - γ * ControlNormalizedShortcutSensitivity(f)
```

In practice, the first paper should keep the main score simpler and use other terms as ablations.

Recommended primary score:

```text
FCV(f) = Acc_original(f) + λ * Acc_bg_counterfactual(f)
```

Recommended mature score:

```text
FCV(f) =
  Acc_original(f)
  + λ1 * Acc_opposite_bg_swap(f)
  - λ2 * [Drop_opposite_bg(f) - Drop_same_bg(f)]
```

---

## Model-selection protocol

Given candidate pool `M = {f_1, ..., f_K}`:

1. Compute ordinary biased validation accuracy for each model.
2. Compute FCV score for each model.
3. Select:

```text
f_FCV = argmax_f FCV(f)
```

4. Evaluate selected model on real held-out robust test data:
   - worst-group accuracy,
   - OOD accuracy,
   - counterexample accuracy,
   - test group-balanced accuracy.

Compare against:

```text
f_val = argmax_f Acc_val(f)
f_oracle = argmax_f WorstGroupVal(f)  # if group labels available for analysis only
```

Primary evaluation metric:

```text
Selection regret = RobustTest(f_oracle_pool_best) - RobustTest(f_selected)
```

or:

```text
Oracle-selection gap = RobustTest(f_oracle_group_val) - RobustTest(f_selected)
```

---

## Why selection regret should be central

Final model accuracy alone can be misleading because a strong training method may dominate.

The paper is about model selection. Therefore, report:

1. **Selected model robust test accuracy**
2. **Selection regret**
3. **Rank correlation** between validation criterion and robust test performance
4. **Top-k hit rate**
5. **Stability across candidate pools and seeds**

Examples:

```text
Selector                  Selected WGA    Regret vs best    Spearman ρ
Biased val accuracy       62.1             24.5              0.12
Val loss                  60.4             26.2              0.08
Saliency-mask alignment   70.5             16.1              0.29
EVaLS-style selector      78.3              8.3              0.45
FCV                       84.9              1.7              0.71
Oracle group validation   86.0              0.6              0.81
Best in pool              86.6              0.0              —
```

This is the kind of table that would make the paper compelling.

---

## Architecture-specific implementation

### CNNs / ConvNeXt-like models

Use spatial feature maps from intermediate or late blocks.

Candidate layers:

- ResNet:
  - layer2,
  - layer3,
  - layer4.
- ConvNeXt:
  - stage2,
  - stage3,
  - stage4.

Procedure:

1. Register forward hook at layer `ℓ`.
2. Run image to get activation `h_i^ℓ`.
3. Resize mask `M_i` to activation grid.
4. Construct `h_cf^ℓ`.
5. Continue model forward from layer `ℓ`.

Implementation detail:

- For ResNet, easiest is to split model into:
  - stem + layers up to `ℓ`,
  - layers after `ℓ` + global average pooling + classifier.
- For ConvNeXt, similarly split at stage boundaries.

Recommended initial layer:

- ResNet layer3 or layer4.
- Layer3 may preserve spatial specificity better.
- Layer4 may be more semantic but more globally mixed.

Run a layer sweep.

### ViTs

Use patch tokens at intermediate transformer blocks.

Candidate layers:

- 25%, 50%, 75% depth.
- Example for ViT-B/16:
  - block 4,
  - block 8,
  - block 11.

Procedure:

1. Patchify original image normally.
2. Run through transformer blocks up to layer `ℓ`.
3. Use teacher mask pooled to patch grid.
4. Replace background patch tokens with donor background patch tokens.
5. Continue transformer blocks after `ℓ`.
6. Read classifier output.

Important caveat:

- The CLS token may already contain global information.
- Options:
  - keep target CLS token unchanged,
  - patch background tokens only,
  - optionally reset CLS or use patch-pooling heads in controlled experiments.

Best initial ViT strategy:

- Keep CLS token from target image.
- Patch only non-evidence patch tokens.
- Evaluate across layers.
- Include CLS-control ablation.

Potential issue:

- If the CLS token already encoded the background shortcut before patching, swapping later tokens may not affect prediction. This is why early/mid layers matter.

### Hybrid / general API

Define a model wrapper:

```python
class CounterfactualWrapper:
    def forward_to_layer(self, x, layer_id): ...
    def forward_from_layer(self, h, layer_id): ...
    def resize_mask_to_layer(self, mask, layer_id): ...
```

This wrapper enables FCV across architectures.

---

## Training extension: counterfactual-compatible training

The core paper should be validation-first, but include an optional training regularizer.

Motivation:

Feature counterfactuals may create hidden states the model was not trained to process. Training with similar interventions can make FCV more reliable and improve robustness.

### Training objective

For training sample `(x_i, y_i)`:

1. Compute feature representation `h_i`.
2. Construct counterfactual `h_cf` by replacing background activations with donor background activations.
3. Continue forward pass.
4. Apply consistency/classification loss:

```text
L_cf = CE(f_cf(x_i), y_i)
```

or:

```text
L_cons = KL(p_original || p_counterfactual)
```

Also optionally:

```text
L_bg = entropy / low-confidence loss when evidence is replaced
```

Final training loss:

```text
L = L_CE(original)
  + λ_cf L_CE(background_feature_swap)
  + λ_cons KL(original || bg_swap)
  + λ_ev L_evidence_sensitivity
```

### Important separation from validation

To avoid circularity:

- Train with one donor sampling strategy; validate with another.
- Train using one set of teacher masks/prompts; validate using held-out prompts or perturbed masks.
- Train using one layer; validate across multiple layers.
- Train with background swaps; validate with both background swaps and evidence swaps.
- Final performance is always measured on real natural OOD/worst-group test data.

### How to present the training extension

Do **not** make the training method the main identity of the paper.

Present it as:

> FCV can be applied post-hoc to ordinary models. However, feature-counterfactual-compatible training improves both robustness and the reliability of FCV selection. We include it as an optional extension and as a stronger practical recipe.

---

## Candidate model pool

The paper should evaluate FCV as a selector over a large candidate pool.

Include models from:

### Standard

- ERM
- Early stopping variations
- Different seeds
- Different learning rates
- Different weight decay
- Different data augmentations

### Spurious-correlation methods

- GroupDRO if group labels are available as oracle training baseline
- JTT-style methods
- AFR-style methods
- MaskTune-like methods
- GALS-style spatial supervision
- R4RR-style attention/region alignment
- Background augmentation methods where feasible

### Architectures

- ResNet-50
- ResNet-18 / ResNet-34 for speed
- ConvNeXt-Tiny
- ViT-B/16
- DeiT / DINOv2-style backbones if feasible

### Checkpoints

Include multiple checkpoints per training run.

This is important because model selection often fails across checkpoints, not just methods.

---

## Datasets

### 1. Waterbirds / Waterbirds100-style

Primary benchmark.

Why:

- canonical spatial shortcut benchmark,
- bird class correlated with land/water background,
- no-counterexample variants are easy to construct,
- segmentation masks are available or can be created,
- worst-group accuracy is well established.

Experiments:

- Standard Waterbirds.
- Waterbirds100:
  - training fully correlated,
  - validation fully correlated,
  - test contains counterexamples.
- Partial-correlation sweeps:
  - 100%, 95%, 90%, 80% correlation in validation.

### 2. MetaShift cat/dog

Why:

- natural context shift,
- cat/dog with indoor/outdoor/contextual subpopulations,
- aligns with the motivating example.

Experiments:

- cat indoor / dog outdoor shortcut,
- no-counterexample validation,
- OOD test with reversed or mixed contexts.

### 3. NICO / NICO++

Why:

- object recognition under context bias,
- many object-context combinations,
- useful for multi-class setting.

Experiments:

- context-held-out validation,
- context-biased validation,
- multi-background banks.

### 4. ImageNet-9 / Backgrounds Challenge

Why:

- directly probes foreground/background reliance,
- accepted benchmark for background bias.

Experiments:

- use foreground/background masks where available,
- create feature counterfactuals across background categories,
- compare selected model performance on mixed-random/background challenge variants.

### 5. CelebA

Optional / caution.

Why:

- widely used spurious-correlation benchmark,
- but less spatially clean than object-background settings.

Potential issue:

- spurious attributes like gender/hair color are not always cleanly separable into foreground/background regions.
- FCV is primarily for **spatial shortcut** bias, so CelebA may be a secondary experiment only.

### 6. iWildCam-WILDS

Optional realism benchmark.

Why:

- camera-trap background/camera context likely matters.
- real-world distribution shift.

Potential issue:

- many classes,
- small/occluded animals,
- masks may be noisy.

Good for demonstrating scalability if early experiments work.

---

## Baselines for model selection

### Basic selectors

- Biased validation accuracy
- Biased validation loss
- Training loss
- Last checkpoint
- Average confidence
- Calibration metrics

### Oracle selectors

- Oracle group-balanced validation
- Oracle worst-group validation
- Oracle OOD validation if available

These are not realistic but provide upper bounds.

### Group-inference / environment selectors

- EVaLS-style environment validation
- Loss-based high/low groups
- Clustering-based pseudo-groups
- Background embedding clusters

### Spatial/XAI selectors

- Grad-CAM overlap with teacher mask
- Integrated gradients overlap
- Relevance Rank Accuracy / similar localization metrics
- Dual-polarity or foreground/background attribution precision
- GALS/R4RR validation losses if available

### Perturbation selectors

- Pixel-space background blur stability
- Evidence-only accuracy
- Background-only confidence
- Inpainting/background swap validation

These are important to show FCV is better than simpler intervention validation.

### FCV variants

- FCV with oracle masks
- FCV with VLM masks
- FCV with noisy masks
- FCV with random masks
- FCV with same-background control
- FCV with opposite-background swaps
- FCV with feature-token banks
- FCV with training extension

---

## Ablations

### 1. Mask source

Compare:

- oracle masks,
- VLM masks,
- SAM/objectness masks,
- weak localization maps,
- random masks,
- wrong-class masks,
- shuffled masks.

Expected result:

```text
oracle > VLM > noisy > random/wrong
```

### 2. Layer choice

For CNNs:

- layer2,
- layer3,
- layer4.

For ViTs:

- early,
- middle,
- late blocks.

Expected:

- early layers may be too low-level,
- late layers may be too mixed,
- middle layers may be best.

### 3. Donor sampling

Compare:

- random opposite-context tokens,
- position-aware tokens,
- nearest-neighbor matched tokens,
- multi-donor tokens,
- same-context tokens,
- random tokens.

### 4. Counterfactual type

Compare:

- swap background only,
- swap evidence only,
- swap both,
- remove background,
- remove evidence,
- same-background control.

### 5. Number of counterfactual draws

Evaluate stability with:

```text
1, 3, 5, 10, 20 draws per validation image
```

### 6. Candidate pool size

Does FCV remain effective when selecting among:

```text
10, 50, 100, 500 candidate models?
```

### 7. Validation correlation severity

Waterbirds-style validation correlation:

```text
100%, 95%, 90%, 80%, 70%
```

FCV should be most valuable at 100–95%, where ordinary validation is most misleading.

### 8. Training extension

Compare:

- no counterfactual-compatible training,
- feature-counterfactual training,
- image-space augmentation training,
- attention alignment training,
- combined training.

Key question:

> Does FCV work post-hoc, and does it work better when models are trained to be counterfactual-compatible?

### 9. Artifact / unnatural-state controls

Controls:

- same-background feature swaps,
- random feature swaps,
- noise feature swaps,
- wrong-mask swaps,
- patching at irrelevant regions,
- patching with tokens from same image.

The FCV signal should be strongest for semantic opposite-background swaps.

---

## Expected failure modes

### 1. Feature swaps produce unnatural hidden states

Risk:

- model performance drops because activations are out of distribution, not because of background reliance.

Mitigations:

- same-background controls,
- random-token controls,
- position-aware sampling,
- layer sweeps,
- counterfactual-compatible training,
- use differential rather than absolute score.

### 2. Spatial teacher maps are poor

Risk:

- evidence/background split is wrong.

Mitigations:

- oracle mask upper bound,
- prompt ensembles,
- mask stability scoring,
- noisy mask ablations,
- downweight low-confidence masks,
- VLM vs human mask comparison.

### 3. Background features already mixed into evidence tokens

Risk:

- by the chosen layer, foreground/evidence tokens already encode background context.

Mitigations:

- intervene at earlier/mid layers,
- compare multiple layers,
- patch attention blocks before global mixing,
- analyze token mixing.

### 4. CLS token leakage in ViTs

Risk:

- CLS token already encodes background before patching.

Mitigations:

- test early/mid layers,
- optionally patch/reset CLS,
- use patch-pooling ViT variants,
- report CLS-specific ablations.

### 5. Multi-class background banks are hard

Risk:

- in nonbinary datasets, "opposite background" is less obvious.

Mitigations:

- use background clustering,
- use context labels if available for analysis,
- define donor backgrounds by low class-conditional similarity,
- use nearest context mismatch.

### 6. Method appears too architecture-specific

Risk:

- FCV implementation differs for CNNs and ViTs.

Mitigations:

- present general abstraction: spatial representation + teacher mask + background feature bank + counterfactual continuation.
- provide concrete adapters for common architectures.

### 7. It is not fully universal

Risk:

- not every model exposes easy spatial features.

Mitigations:

- state scope clearly:
  - FCV targets models with spatial feature maps or patch tokens.
  - Most modern vision classifiers satisfy this.

---

## Why FCV could be top-tier-worthy if results are strong

The case for top-tier publishability:

### 1. It addresses a real unsolved gap

The field has many robust training methods, but model selection under biased validation remains weak. This is especially important when validation contains no shortcut-breaking examples.

### 2. It targets the hard regime

Not merely "validation is imbalanced," but:

> validation lacks natural counterexamples.

That is a sharper and more difficult problem.

### 3. It avoids pixel artifacts

Unlike blacking out, blurring, or inpainting backgrounds, FCV creates counterfactuals in feature space using real activations from real images.

### 4. It is not saliency overlap

It measures prediction behavior under internal interventions, not just attribution-map localization.

### 5. It is model-selection-first

The contribution is not another training recipe. It is a criterion for choosing robust models from a pool.

### 6. It is teacher-map agnostic

The method can use oracle masks, VLM masks, human masks, or objectness maps.

### 7. It can be tested rigorously

The core evaluation is clean:

> Does FCV reduce selection regret relative to biased validation accuracy and approach oracle group validation?

That is an objective and compelling test.

---

## Possible reviewer objections and responses

### Objection: "Activation patching already exists."

Response:

Activation patching has been used mainly for attribution or interpretability. FCV uses mask-guided evidence/background activation recombination for **model selection under spatial shortcut bias**, especially when validation lacks natural counterexamples.

### Objection: "Background swapping already exists."

Response:

Pixel-space background swapping requires clean background synthesis and can introduce image artifacts. FCV swaps background activations, not pixels, and uses the resulting behavior as a validation selector rather than training augmentation.

### Objection: "This creates unnatural hidden states."

Response:

We agree this is a risk. FCV uses differential controls: same-background swaps, random-token swaps, wrong-mask swaps, and layer sweeps. We also introduce optional counterfactual-compatible training to reduce hidden-state shift. The final validation of the metric is whether it predicts real natural OOD/worst-group test performance.

### Objection: "Masks may be wrong."

Response:

FCV is teacher-map agnostic. We evaluate oracle, VLM, noisy, and random maps, quantify degradation as map quality worsens, and report mask stability/coverage. The method does not require perfect masks; it requires maps that are better aligned with class evidence than chance.

### Objection: "This only works for spatial shortcuts."

Response:

Correct. FCV is specifically designed for **spatial shortcut robustness**, where evidence and background/context can be spatially separated. This scope is narrower than all spurious correlations, but important and common in vision.

### Objection: "Why not just train with background augmentation?"

Response:

Training augmentation and model selection solve different problems. FCV can select among many training methods, including background augmentation methods. It is useful precisely when the validation set cannot reveal which model actually learned robust evidence.

---

## Recommended implementation plan

### Phase 0: sanity prototype

Dataset:

- Waterbirds100-style split.

Models:

- ResNet-50 ERM with multiple seeds/checkpoints/hyperparameters.

Masks:

- oracle/human masks if available.
- VLM masks later.

Implement:

- feature extraction at ResNet layer3/layer4,
- background banks,
- opposite-background feature swaps,
- FCV score,
- selection regret.

Goal:

> Does FCV select better ERM checkpoints than biased validation accuracy?

### Phase 1: candidate pool expansion

Add:

- ERM variants,
- AFR/JTT-style variants,
- GALS/R4RR-style variants if available,
- MaskTune-like variants.

Goal:

> Does FCV select robust methods from a heterogeneous candidate pool?

### Phase 2: mask source study

Compare:

- oracle masks,
- VLM masks,
- noisy masks,
- random masks.

Goal:

> How much spatial teacher quality is needed?

### Phase 3: architecture study

Add:

- ResNet,
- ConvNeXt,
- ViT.

Goal:

> Does the principle transfer across architecture families?

### Phase 4: datasets

Add:

- MetaShift,
- NICO/NICO++,
- ImageNet-9.

Goal:

> Does FCV generalize beyond Waterbirds?

### Phase 5: optional training extension

Train with feature-counterfactual consistency.

Goal:

> Does counterfactual-compatible training reduce hidden-state-shift concerns and improve both robustness and FCV reliability?

---

## Pseudocode: FCV scoring

```python
def compute_fcv_score(
    model,
    val_loader,
    masks,
    layer_id,
    bank_builder,
    donor_sampler,
    lambda_cf=1.0,
    n_draws=5,
):
    model.eval()

    # Step 1: collect activations and resized masks
    records = []
    for x, y, idx in val_loader:
        h = model.forward_to_layer(x, layer_id)
        m = resize_mask_to_activation_grid(masks[idx], h)
        records.append({
            "x": x,
            "y": y,
            "idx": idx,
            "h": h.detach(),
            "mask": m.detach(),
        })

    # Step 2: build background banks
    banks = bank_builder(records)

    original_correct = []
    cf_correct = []
    cf_conf_drops = []
    same_conf_drops = []

    for rec in records:
        y = rec["y"]
        h_target = rec["h"]
        m = rec["mask"]

        # Original prediction from stored layer
        logits_orig = model.forward_from_layer(h_target, layer_id)
        p_orig = softmax(logits_orig)
        original_correct.append(argmax(p_orig) == y)

        for _ in range(n_draws):
            # Opposite-context background
            bg_opposite = donor_sampler.sample_opposite_background(
                target_record=rec,
                banks=banks,
                shape=h_target.shape,
            )

            h_cf = m * h_target + (1 - m) * bg_opposite
            logits_cf = model.forward_from_layer(h_cf, layer_id)
            p_cf = softmax(logits_cf)

            cf_correct.append(argmax(p_cf) == y)
            cf_conf_drops.append(p_orig[y] - p_cf[y])

            # Same-context control
            bg_same = donor_sampler.sample_same_background(
                target_record=rec,
                banks=banks,
                shape=h_target.shape,
            )

            h_same = m * h_target + (1 - m) * bg_same
            logits_same = model.forward_from_layer(h_same, layer_id)
            p_same = softmax(logits_same)

            same_conf_drops.append(p_orig[y] - p_same[y])

    acc_orig = mean(original_correct)
    acc_cf = mean(cf_correct)
    drop_opposite = mean(cf_conf_drops)
    drop_same = mean(same_conf_drops)

    # Control-normalized shortcut sensitivity
    shortcut_sensitivity = drop_opposite - drop_same

    fcv_score = acc_orig + lambda_cf * acc_cf - shortcut_sensitivity

    return {
        "fcv_score": fcv_score,
        "acc_orig": acc_orig,
        "acc_cf": acc_cf,
        "drop_opposite": drop_opposite,
        "drop_same": drop_same,
        "shortcut_sensitivity": shortcut_sensitivity,
    }
```

---

## Pseudocode: counterfactual-compatible training

```python
def train_step_counterfactual_compatible(
    model,
    batch,
    masks,
    layer_id,
    background_bank,
    optimizer,
    lambda_cf=1.0,
    lambda_kl=0.5,
):
    x, y, idx = batch

    # Original forward
    h = model.forward_to_layer(x, layer_id)
    logits_orig = model.forward_from_layer(h, layer_id)
    loss_orig = cross_entropy(logits_orig, y)

    # Resize masks
    m = resize_mask_to_activation_grid(masks[idx], h)

    # Sample donor background features
    bg = background_bank.sample_opposite_or_random(y, shape=h.shape)

    # Feature counterfactual
    h_cf = m * h + (1 - m) * bg

    # Continue forward
    logits_cf = model.forward_from_layer(h_cf, layer_id)

    # Counterfactual consistency/classification loss
    loss_cf = cross_entropy(logits_cf, y)
    loss_kl = kl_divergence(
        softmax(logits_orig.detach()),
        softmax(logits_cf)
    )

    loss = loss_orig + lambda_cf * loss_cf + lambda_kl * loss_kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

---

## Figures the paper should include

### Figure 1: Problem setup

Show:

- biased validation set:
  - waterbirds on water,
  - landbirds on land.
- two models:
  - evidence model,
  - background shortcut model.
- both get high biased validation accuracy.
- only evidence model works on OOD test.

### Figure 2: FCV concept

Show:

- target image,
- teacher mask,
- target feature map,
- donor background token bank,
- patched representation:
  - target evidence features + donor background features.
- prediction should follow evidence.

### Figure 3: Pixel swap vs feature swap

Show why pixel-space donor backgrounds create holes and artifacts, while feature-space background banks avoid needing full clean backgrounds.

### Figure 4: Selection regret plot

Compare selectors:

- biased val,
- val loss,
- saliency alignment,
- EVaLS-style,
- FCV,
- oracle group val.

### Figure 5: Layer sweep

FCV effectiveness across layers.

### Figure 6: Teacher-map quality curve

Oracle → VLM → noisy → random.

### Figure 7: Control-normalized effects

Opposite-background swaps should correlate with robust failure more than same-background or random swaps.

---

## Main tables

### Table 1: Model selection on Waterbirds100

Rows:

- selectors.

Columns:

- selected average test accuracy,
- selected worst-group accuracy,
- selection regret,
- Spearman rank correlation,
- top-k hit rate.

### Table 2: Cross-method candidate pool

Rows:

- candidate training methods.

Columns:

- best-in-pool robust performance,
- selected by biased val,
- selected by FCV,
- selected by oracle.

### Table 3: Dataset generalization

Rows:

- Waterbirds100,
- MetaShift,
- NICO,
- ImageNet-9.

Columns:

- biased val selector WGA,
- FCV selector WGA,
- oracle selector WGA,
- regret reduction.

### Table 4: Mask source ablation

Rows:

- oracle,
- VLM,
- noisy,
- random,
- wrong-class.

Columns:

- selection regret,
- correlation with robust test accuracy.

### Table 5: Architecture ablation

Rows:

- ResNet,
- ConvNeXt,
- ViT.

Columns:

- best FCV layer,
- regret reduction,
- robustness gain.

---

## What the abstract could say

> Model selection under spurious correlations is often treated as an afterthought: even methods that avoid group labels during training may rely on validation data that contains group labels or natural counterexamples to the shortcut. We study a harder setting, no-counterexample validation, where the validation set is itself fully shortcut-correlated and ordinary validation accuracy cannot distinguish evidence-based models from shortcut-based models. We propose Feature-Counterfactual Validation (FCV), a model-selection criterion for spatial shortcut robustness. Given spatial teacher maps, FCV partitions a candidate model's internal representation into evidence and background components, constructs missing counterexamples by replacing background activations with background activations from other validation images, and selects models whose predictions follow evidence rather than context. FCV is teacher-map agnostic and can use human masks, dataset masks, or VLM-generated maps. Across spatially biased recognition benchmarks, FCV reduces selection regret relative to biased validation accuracy and approaches oracle group-balanced validation, with strong controls showing that semantic feature counterfactuals outperform random or same-context swaps. Our results suggest that robust model selection can be improved not by collecting group labels, but by constructing counterfactual validation signals inside the model's own spatial representation.

---

## Suggested contribution bullets

1. We formalize **no-counterexample model selection** for spatial shortcut robustness, where validation data is itself fully shortcut-correlated.

2. We propose **Feature-Counterfactual Validation**, which constructs missing counterexamples in representation space by recombining evidence and background activations using spatial teacher maps.

3. We show that FCV reduces model-selection regret relative to biased validation accuracy, validation loss, and spatial attribution selectors, approaching oracle group-balanced validation across multiple benchmarks.

4. We study teacher-map dependence using oracle, human/dataset, VLM-generated, noisy, and random maps.

5. We introduce an optional counterfactual-compatible training extension that improves robustness and reduces concerns about activation-distribution shift.

---

## Positioning against nearby work

### DomainBed / domain-generalization model selection

DomainBed emphasizes that model selection is a central, under-specified part of domain generalization. FCV is narrower and more targeted: spatial shortcut robustness under biased/no-counterexample validation.

### EVaLS and group-label-free validation

EVaLS targets group-label-free validation/model selection through inferred environments and loss-based sampling. FCV targets spatial shortcut bias through internal feature counterfactuals, especially when natural validation counterexamples are missing.

### GALS / language-guided attention supervision

GALS uses language grounding to supervise model attention during training. FCV uses spatial teacher maps to create feature-space counterfactual validation examples for model selection. The teacher maps can come from VLMs, but the focus is validation rather than attention supervision.

### MaskTune

MaskTune masks discovered features to force exploration during training. FCV swaps background activations to test model reliance for selection. MaskTune is a training method; FCV is a selector.

### Pixel-space background swapping / augmentation

Background-swapping methods synthesize new images for training by recombining foregrounds and backgrounds. FCV avoids pixel-space hole filling and uses representation-space background recombination for model selection.

### Activation patching / causal attribution

Activation patching directly intervenes on internal activations to estimate causal patch contributions. FCV adapts the intervention idea to spatial shortcut robustness and model selection: we patch background activations to construct validation counterexamples and select models whose predictions follow evidence.

### XAI localization metrics

XAI localization metrics compare explanations to segmentation masks. FCV is not a saliency overlap metric; it measures behavioral response to feature-level counterfactuals.

---

## References to anchor the related work

These are starting references to cite and compare against:

1. **DomainBed / In Search of Lost Domain Generalization**  
   Gulrajani and Lopez-Paz, 2020/2021.  
   https://arxiv.org/abs/2007.01434

2. **EVaLS: Trained Models Tell Us How to Make Them Robust to Spurious Correlation without Group Annotation**  
   Ghaznavi et al., 2024.  
   https://arxiv.org/abs/2410.05345

3. **On Guiding Visual Attention with Language Specification (GALS)**  
   Petryk et al., CVPR 2022.  
   https://arxiv.org/abs/2202.08926

4. **MaskTune: Mitigating Spurious Correlations by Forcing to Explore**  
   Taghanaki et al., 2022.  
   https://arxiv.org/abs/2210.00055

5. **Causal Attribution via Activation Patching**  
   Izadi et al., 2026.  
   https://arxiv.org/abs/2603.13652

6. **Automated Background Swapping for Robustness against Spurious Backgrounds**  
   Roder and Schweighofer, 2026.  
   https://arxiv.org/abs/2606.32018

7. **MixStyle: Domain Generalization via Randomized Feature Statistics Mixing**  
   Zhou et al., 2021.  
   https://arxiv.org/abs/2107.02053

8. **ASPIRE: Language-Guided Data Augmentation for Improving Robustness Against Spurious Correlations**  
   Ghosh et al., 2023.  
   https://arxiv.org/abs/2308.10103

9. **Group Robust Classification Without Any Group Information**  
   https://arxiv.org/abs/2310.18555

10. **Causal Component Analysis / Decompose-and-Compose style counterfactual debiasing**  
   https://arxiv.org/abs/2402.18919

---

## Minimal viable experiment

If only one experiment can be run first, run this:

### Dataset

Waterbirds100-style validation.

### Candidate models

Train 50–100 ResNet-50 ERM candidates:

- different seeds,
- learning rates,
- weight decay,
- epochs/checkpoints,
- augmentations.

### Masks

Use oracle bird masks first.

### FCV

- patch at ResNet layer3 and layer4,
- background banks:
  - water background tokens from waterbird val images,
  - land background tokens from landbird val images,
- sample 5 opposite-background counterfactuals per image,
- compute FCV score.

### Compare selectors

- biased validation accuracy,
- validation loss,
- saliency-mask alignment,
- FCV,
- oracle group validation.

### Test

Evaluate selected models on real Waterbirds test with counterexamples.

### Decision criterion

If FCV substantially reduces selection regret relative to biased validation, the idea is alive.

If oracle masks work but VLM masks fail, mask quality is the bottleneck.

If oracle masks fail, the feature-counterfactual idea is likely not strong enough.

---

## The most important early diagnostic plot

Plot each candidate model as a point:

```text
x-axis: biased validation accuracy
y-axis: robust/worst-group test accuracy
color: FCV score
```

If biased validation accuracy is flat/uninformative but FCV color tracks robust test accuracy, that is the visual proof of concept.

Also plot:

```text
x-axis: FCV score
y-axis: robust/worst-group test accuracy
```

You want strong monotonicity.

---

## Final recommendation

The best version of this project is:

> **A validation-first paper on no-counterexample spatial shortcut model selection, using feature-space counterfactuals created from spatial teacher maps.**

Do not lead with VLMs. Do not lead with background swapping. Do not lead with a new training loss.

Lead with the problem:

> Validation can be spuriously correlated too, and sometimes contains no counterexamples.

Then the method:

> Build the missing counterexamples inside the model's representation by swapping background activations while preserving evidence activations.

Then the evidence:

> FCV selects models closer to oracle robust selection than ordinary validation accuracy.

Then the extension:

> Counterfactual-compatible training improves reliability but is not required for the central validation story.

If the empirical results are strong across datasets, architectures, and mask sources, this is plausibly a serious CVPR/ECCV-level contribution.
