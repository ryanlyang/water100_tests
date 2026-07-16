# ViT Feature-Counterfactual Validation: First Study Implementation Plan

## Working title

**Feature-Counterfactual Validation for Model Selection under Complete Spatial Shortcut Bias**

## Goal of the first study

The first study should answer one narrow question:

> In a Waterbirds100-style setting where the validation set is itself spuriously correlated, can a ViT feature-counterfactual validation score select better models than ordinary biased validation accuracy?

The goal is **not** yet to prove the full paper. The goal is to determine whether the idea has a strong enough signal to justify building the full benchmark.

The first study should use:

- **Dataset:** Waterbirds100 / fully spuriously correlated Waterbirds validation setting.
- **Model family:** vanilla pretrained ViT fine-tuning setups.
- **Teacher maps:** the spatial teacher maps already used for R4RR.
- **Main method:** feature-space background counterfactuals at the ViT patch-token level.
- **Primary outcome:** how much of the gap between biased validation selection and oracle validation selection is closed by Feature-Counterfactual Validation.

---

# 1. Core idea in one paragraph

Standard validation accuracy fails when validation data has the same spatial shortcut as training. In Waterbirds100, if all waterbirds appear on water backgrounds and all landbirds appear on land backgrounds, validation accuracy cannot tell whether a model learned bird evidence or background context. Feature-Counterfactual Validation creates missing shortcut-breaking examples **inside the ViT representation**: keep the target image's evidence/object patch tokens, replace its background patch tokens with background patch tokens from the opposite shortcut context, and test whether the model's prediction follows the evidence rather than the swapped background. A good selector should choose models whose predictions remain stable when background tokens are counterfactually swapped.

---

# 2. Study hypotheses

## Main hypothesis

Feature-Counterfactual Validation will select ViT checkpoints/hyperparameters with better robust test performance than ordinary biased validation accuracy.

## More specific hypotheses

1. **Biased validation accuracy will often select shortcut-heavy models.**  
   These models will perform well on the spuriously correlated validation set but poorly on rare or counter-spurious test groups.

2. **Feature-counterfactual stability will correlate better with worst-group test accuracy.**  
   Models that remain correct after opposite-background feature swaps should be less dependent on background context.

3. **FCV will close a nontrivial fraction of the oracle selection gap.**  
   The oracle selector uses group-balanced or worst-group validation labels. FCV should recover part of that advantage without using group labels.

4. **Semantic teacher masks should outperform random or shuffled masks.**  
   If random masks work just as well as teacher maps, then FCV is probably measuring generic token corruption rather than evidence/background reliance.

---

# 3. Important definitions

## Candidate model

A candidate model is a trained ViT checkpoint. The candidate pool should include different:

- learning rates,
- weight decays,
- augmentation settings,
- random seeds,
- training durations/checkpoints,
- possibly linear-probe versus full fine-tune settings.

The first study can use only vanilla ViT fine-tuning. Later studies can add R4RR, GALS, AFR, MaskTune-style methods, and others.

## Teacher map

A teacher map is a binary or soft spatial map indicating the likely class-relevant evidence region. For this first study, use the maps already used in the R4RR experiments.

For Waterbirds, the teacher map should roughly identify the bird region.

## Evidence patch

A ViT patch whose area overlaps strongly with the teacher map.

Example criterion:

```text
patch_evidence_score >= 0.60
```

## Background patch

A ViT patch whose area has little or no overlap with the teacher map.

Example criterion:

```text
patch_evidence_score <= 0.10
```

## Ambiguous patch

A patch near the object boundary or with mixed foreground/background content.

Example criterion:

```text
0.10 < patch_evidence_score < 0.60
```

Ambiguous patches should be left unchanged in the first implementation.

## Opposite-background token bank

A collection of background patch embeddings from validation images with the opposite shortcut context.

For Waterbirds100:

- Waterbird images are assumed to provide mostly water-background tokens.
- Landbird images are assumed to provide mostly land-background tokens.

So:

```text
waterbird target → sample donor background tokens from landbird validation images
landbird target → sample donor background tokens from waterbird validation images
```

This uses class labels, but not group labels. In the fully confounded Waterbirds100 validation split, class label is a proxy for shortcut background context.

---

# 4. Recommended first-study architecture

Use a pretrained ViT with patch size 16.

Recommended starting options:

```text
ViT-S/16 or ViT-B/16
Image size: 224 × 224
Patch grid: 14 × 14 = 196 patch tokens
Classification: binary waterbird vs landbird
```

For ease of implementation, use a `timm` ViT model if the existing codebase supports it.

The first counterfactual intervention should occur at the **raw patch embedding stage**, before transformer blocks.

Why this is best for the first prototype:

- patch-token to image-region mapping is clean;
- no self-attention mixing has occurred yet;
- donor tokens and target tokens have the same embedding dimension;
- positional embeddings can be applied after replacement so donor content occupies the target spatial position;
- the intervention is easier to reason about than late-layer activation patching.

Later, sweep intervention depth:

```text
patch embedding
block 2
block 4
block 6
block 8
```

But do **patch embedding first**.

---

# 5. Exact ViT intervention design

## Normal ViT forward pass

A simplified ViT forward pass is:

```python
patch_tokens = patch_embed(image)        # [B, N, D]
patch_tokens = patch_tokens + pos_embed  # position added
cls_token = cls + cls_pos
x = concat(cls_token, patch_tokens)
x = transformer_blocks(x)
logits = head(x)
```

## FCV counterfactual forward pass

For a target image `x_i`:

1. Compute target raw patch embeddings:

```python
target_tokens = patch_embed(x_i)  # [N, D]
```

2. Identify target background patch positions using the teacher map.

3. Sample donor background patch embeddings from the opposite-background token bank:

```python
donor_bg_tokens = sample(bank_opposite, k=num_target_background_patches)
```

4. Replace only the target's safe background patch tokens:

```python
cf_tokens = target_tokens.clone()
cf_tokens[background_positions] = donor_bg_tokens
```

5. Add the **target positional embeddings**, not the donor positional embeddings:

```python
cf_tokens = cf_tokens + target_pos_embed
```

6. Run the rest of the ViT normally:

```python
logits_cf = transformer_blocks_and_head(cf_tokens)
```

The prediction should remain the target label if the model relies on object evidence.

---

# 6. Why positional embeddings must come from the target

The donor background token provides **content**, not location.

If a donor background token came from the lower-left patch of a donor image but is inserted into the upper-right background position of the target image, it should receive the target position embedding.

The counterfactual should mean:

```text
target evidence + donor background content arranged in target layout
```

not:

```text
target evidence + donor background content with donor spatial identity attached
```

So the clean implementation is:

```text
replace raw patch embeddings first
then add target positional embeddings
```

---

# 7. Step-by-step implementation plan

## Step 1 — Create the experiment branch and config structure

Create a dedicated experiment branch/folder so the first study is isolated.

Suggested folder structure:

```text
experiments/fcv_vit_waterbirds100/
    configs/
    scripts/
    src/
        fcv/
            masks.py
            token_banks.py
            vit_counterfactual_forward.py
            selectors.py
            metrics.py
    outputs/
        candidate_models/
        token_banks/
        fcv_scores/
        selection_results/
```

Create a single YAML config format for the full study:

```yaml
dataset: waterbirds100
image_size: 224
patch_size: 16
model: vit_s_16
pretrained: imagenet
teacher_maps: /path/to/r4rr_teacher_maps
candidate_pool: vanilla_vit_sweep
intervention_layer: patch_embed
background_patch_threshold: 0.10
evidence_patch_threshold: 0.60
ambiguous_policy: keep_target
num_donor_samples: 5
selector_lambda: 1.0
```

Output of this step:

```text
A reproducible config-driven experiment skeleton.
```

---

## Step 2 — Prepare Waterbirds100 splits and metadata

Load the Waterbirds100 training, validation, and test splits.

For each image, store:

```text
image_path
class_label: waterbird or landbird
split: train / val / test
optional group label: waterbird-water, waterbird-land, landbird-water, landbird-land
teacher_map_path
```

For this first study:

- training may be spuriously correlated depending on the Waterbirds100 setup;
- validation should be fully or near-fully spuriously correlated;
- test should include counter-spurious groups so robust performance can be measured.

Important:

The FCV selector should **not** use validation group labels. Group labels are only used for oracle selector comparison and final test analysis.

Output of this step:

```text
metadata_train.csv
metadata_val.csv
metadata_test.csv
```

---

## Step 3 — Load and preprocess R4RR teacher maps

Use the existing R4RR teacher maps as the spatial teacher maps.

For each image:

1. Load the teacher map.
2. Resize it to the ViT input size, e.g. `224 × 224`.
3. Normalize to `[0, 1]` if soft.
4. Convert it to patch-level scores.

Patch-level score:

```python
patch_score[p] = mean(mask[pixels inside patch p])
```

For a `224 × 224` image and patch size `16`, this gives:

```text
14 × 14 patch scores
```

Define patch categories:

```python
evidence_patches = patch_score >= evidence_threshold
background_patches = patch_score <= background_threshold
ambiguous_patches = otherwise
```

Recommended first thresholds:

```text
evidence_threshold = 0.60
background_threshold = 0.10
```

Be conservative at first. It is better to use fewer clean background patches than many contaminated boundary patches.

Output of this step:

```text
patch_masks_val.pt
patch_masks_test.pt  # optional for analysis only
```

Each item should contain:

```python
{
    "image_id": str,
    "patch_scores": Tensor[196],
    "evidence_idx": LongTensor,
    "background_idx": LongTensor,
    "ambiguous_idx": LongTensor,
    "coverage": {
        "evidence_frac": float,
        "background_frac": float,
        "ambiguous_frac": float,
    }
}
```

---

## Step 4 — Train a vanilla ViT candidate pool

Train a pool of vanilla ViT models on the Waterbirds100 training set.

Start simple. Use pretrained ViT and fine-tune it.

Recommended initial sweep:

```text
model: ViT-S/16 or ViT-B/16
pretraining: ImageNet / DeiT / timm pretrained
optimizer: AdamW
learning rates: [1e-5, 3e-5, 1e-4]
weight decay: [0.01, 0.05, 0.1]
seeds: [0, 1, 2]
epochs: 20–50
checkpoint frequency: every epoch
augmentations:
    basic resize/crop/flip
    optionally weak vs moderate augmentation
```

This gives enough candidate diversity without becoming too expensive.

Example candidate count:

```text
3 learning rates × 3 weight decays × 3 seeds × 20 checkpoints = 540 candidate checkpoints
```

That is probably enough to test model selection.

For each checkpoint, record:

```text
training loss
training accuracy
biased validation loss
biased validation accuracy
optional group validation metrics if available, hidden from FCV selector
final test metrics, computed after selection analysis
```

Output of this step:

```text
candidate_checkpoints/
candidate_metrics_biased_val.csv
```

---

## Step 5 — Implement custom ViT patch-token extraction

Modify or wrap the ViT forward pass so you can access raw patch embeddings before positional embeddings.

Required functions:

```python
def extract_raw_patch_tokens(model, images):
    """
    Returns raw patch embeddings before positional embeddings.
    Shape: [B, N, D]
    """
```

```python
def forward_from_patch_tokens(model, raw_patch_tokens):
    """
    Takes raw patch tokens, adds the model's positional embeddings and CLS token,
    then runs the transformer blocks and classifier head.
    Returns logits.
    """
```

Sanity check:

```python
normal_logits = model(images)
raw_tokens = extract_raw_patch_tokens(model, images)
reconstructed_logits = forward_from_patch_tokens(model, raw_tokens)
```

The outputs should match up to tiny numerical differences.

Acceptance criterion:

```text
max_abs(normal_logits - reconstructed_logits) < 1e-5
```

or close enough depending on dropout/eval mode.

Important:

Set the model to eval mode during FCV scoring:

```python
model.eval()
```

Output of this step:

```text
A verified ViT forward wrapper that can reproduce normal model logits from raw patch tokens.
```

---

## Step 6 — Build model-specific background token banks

For each candidate checkpoint, build background token banks from the validation set.

This has to be done **per model**, because each model has its own patch embedding weights.

For each validation image:

1. Run `extract_raw_patch_tokens(model, image)`.
2. Use the patch-level teacher mask to find safe background patch positions.
3. Add those raw patch tokens to a class/context-specific bank.

For Waterbirds100, use two banks:

```text
water_context_bank: safe background tokens from waterbird-labeled validation images
land_context_bank: safe background tokens from landbird-labeled validation images
```

Because the validation set is fully spuriously correlated, class label acts as the shortcut context label.

Important:

This is not using group labels. It uses known class labels and the known experimental fact that the validation split is fully shortcut-correlated.

For each token, optionally store metadata:

```python
{
    "token": Tensor[D],
    "source_image_id": str,
    "source_class": int,
    "source_patch_idx": int,
    "source_patch_row": int,
    "source_patch_col": int,
    "patch_score": float,
}
```

Recommended first implementation:

- sample donor tokens with replacement;
- avoid using tokens from the same image as the target;
- optionally match spatial row/column loosely later.

Output of this step:

```text
token_banks/{checkpoint_id}_water_context.pt
token_banks/{checkpoint_id}_land_context.pt
```

---

## Step 7 — Implement feature-counterfactual validation forward pass

For each target validation image:

1. Extract target raw patch tokens.
2. Identify target safe background positions.
3. Determine opposite-context donor bank.
4. Sample one donor token per target background position.
5. Replace target background tokens.
6. Run the model forward from the modified patch tokens.
7. Record the counterfactual prediction.

For Waterbirds100:

```python
if target_label == WATERBIRD:
    donor_bank = land_context_bank
elif target_label == LANDBIRD:
    donor_bank = water_context_bank
```

Pseudo-code:

```python
def make_counterfactual_tokens(target_tokens, target_background_idx, donor_bank):
    cf_tokens = target_tokens.clone()
    sampled = donor_bank.sample(k=len(target_background_idx), replace=True)
    cf_tokens[target_background_idx] = sampled
    return cf_tokens
```

For stochastic donor sampling, use multiple samples per image:

```text
num_donor_samples = 5 initially
```

Then average logits or average scores across samples.

Recommended outputs per image/model:

```text
p_y_original
p_y_counterfactual_mean
correct_original
correct_counterfactual_majority
counterfactual_confidence_drop
num_background_patches_swapped
coverage
```

Output of this step:

```text
fcv_raw_scores/{checkpoint_id}.csv
```

---

## Step 8 — Add essential controls

Controls are necessary to prove FCV is not merely measuring sensitivity to arbitrary token replacement.

Implement at least these controls in the first study:

## Control A: same-context background swap

Replace background tokens with donor background tokens from the same shortcut context.

```text
waterbird target → water-context donor tokens
landbird target → land-context donor tokens
```

This tests whether the model is sensitive to token replacement itself.

A good FCV signal should show:

```text
opposite-context swaps reveal more shortcut dependence than same-context swaps
```

## Control B: random patch mask swap

Instead of using teacher-defined background positions, replace random patch positions with matched count.

This controls for generic patch replacement.

## Control C: shuffled teacher masks

Use teacher masks from other images.

This controls for mask area and shape distribution.

## Control D: evidence-token swap

Replace evidence tokens instead of background tokens.

A model should be much more sensitive to evidence-token swaps than background-token swaps.

This is not necessarily part of the selection metric at first, but it is a sanity check.

Output of this step:

```text
fcv_control_scores/{checkpoint_id}_same_context.csv
fcv_control_scores/{checkpoint_id}_random_mask.csv
fcv_control_scores/{checkpoint_id}_shuffled_mask.csv
fcv_control_scores/{checkpoint_id}_evidence_swap.csv
```

---

## Step 9 — Define the model-selection scores

Compute several selectors.

## Selector 1: biased validation accuracy

This is the standard baseline.

```text
select checkpoint with highest ordinary validation accuracy
```

## Selector 2: biased validation loss

```text
select checkpoint with lowest ordinary validation loss
```

## Selector 3: FCV counterfactual accuracy

```text
select checkpoint with highest accuracy after opposite-background feature swaps
```

This is the simplest FCV selector.

## Selector 4: FCV stability score

For each checkpoint:

```text
stability = mean[p_y(counterfactual)]
```

or:

```text
stability = mean[p_y(counterfactual) / p_y(original)]
```

This measures confidence retention under opposite-background swaps.

## Selector 5: control-normalized FCV score

This is likely the best first serious selector.

Example:

```text
FCV_score = original_val_accuracy
            + λ * opposite_context_counterfactual_accuracy
            - β * random_mask_counterfactual_accuracy_bonus
```

A cleaner version:

```text
FCV_gap = same_context_counterfactual_accuracy
          - opposite_context_counterfactual_drop
```

But for the first study, keep it simple and evaluate multiple variants.

Recommended initial selector:

```text
FCV_main = original_val_accuracy + λ * opposite_context_counterfactual_accuracy
```

with:

```text
λ ∈ {0.25, 0.5, 1.0}
```

To avoid tuning λ unfairly, report all three and maybe use a fixed default λ = 1.0.

## Selector 6: oracle group-balanced validation

This is the target upper-bound selection baseline.

Examples:

```text
select checkpoint with highest worst-group validation accuracy
select checkpoint with highest balanced-group validation accuracy
```

This selector uses group labels and is not available to FCV.

Output of this step:

```text
selection_table.csv
```

Columns:

```text
selector_name
selected_checkpoint_id
selected_hparams
biased_val_acc
fcv_score
oracle_group_val_score
test_avg_acc
test_worst_group_acc
```

---

## Step 10 — Evaluate selected checkpoints on real test data

After selectors choose checkpoints, evaluate each selected checkpoint on the real Waterbirds test set.

Report:

```text
test average accuracy
test worst-group accuracy
test waterbird-water accuracy
test waterbird-land accuracy
test landbird-water accuracy
test landbird-land accuracy
```

The main comparison is:

```text
biased validation selector vs FCV selector vs oracle validation selector
```

Primary robust metric:

```text
worst-group test accuracy
```

Secondary metric:

```text
balanced test accuracy across groups
```

Output of this step:

```text
final_test_results.csv
```

---

## Step 11 — Compute oracle selection gap closure

Define:

```text
R_biased = robust test performance of model selected by biased validation accuracy
R_fcv = robust test performance of model selected by FCV
R_oracle = robust test performance of model selected by oracle group-balanced validation
```

Then compute:

```text
gap_closed = (R_fcv - R_biased) / (R_oracle - R_biased)
```

Report as a percentage:

```text
gap_closed_percent = 100 * gap_closed
```

Example interpretation:

```text
Biased val selection: 62% worst-group test acc
FCV selection: 74% worst-group test acc
Oracle group-val selection: 82% worst-group test acc

Gap closed = (74 - 62) / (82 - 62) = 60%
```

Also report the oracle-pool upper bound:

```text
best possible test robust accuracy among all candidate checkpoints
```

This is not a fair selector, but it tells us whether the candidate pool contains good models at all.

Output of this step:

```text
gap_closure_summary.csv
```

---

## Step 12 — Analyze correlation and rank quality

Beyond selecting one checkpoint, evaluate whether FCV is genuinely predictive of robust performance.

For every candidate checkpoint, compute:

```text
biased_val_accuracy
biased_val_loss
FCV_main_score
FCV_counterfactual_accuracy
FCV_stability
oracle_group_val_score
test_worst_group_accuracy
```

Then measure:

```text
Spearman correlation between selector score and test worst-group accuracy
Kendall tau between selector score and test worst-group accuracy
Top-k recall: whether the selector's top-k checkpoints contain high robust-test checkpoints
```

This matters because a single selected checkpoint may be noisy. Rank correlation shows whether the metric has a real signal.

Expected successful pattern:

```text
FCV correlation with worst-group test acc > biased validation accuracy correlation
```

Output of this step:

```text
rank_correlation_results.csv
selector_scatter_plots/
```

---

# 8. Minimal viable first experiment

If time is limited, do this smallest possible version first:

1. Train 30–50 ViT candidate checkpoints from a small hyperparameter sweep.
2. Use R4RR teacher maps to identify bird/background patches.
3. Intervene at patch embedding only.
4. Build water-context and land-context token banks per checkpoint.
5. Compute opposite-background FCV accuracy for each checkpoint.
6. Select by:
   - biased validation accuracy,
   - FCV counterfactual accuracy,
   - oracle group validation.
7. Evaluate selected models on Waterbirds test worst-group accuracy.
8. Compute gap closure.

This is enough to know whether the idea is alive.

---

# 9. What would count as a promising result?

The method is promising if we see something like:

```text
Selector                         Worst-group test acc
-----------------------------------------------------
Biased validation accuracy        55–65
FCV selector                      70–80
Oracle group validation           80–85
Best-in-pool upper bound          85+
```

Or, in gap terms:

```text
FCV closes at least 40–60% of the oracle selection gap.
```

Even if absolute numbers differ, the key result is:

```text
FCV selector consistently beats biased validation selector.
```

A very strong result would be:

```text
FCV approaches oracle group-validation selection without using group labels.
```

---

# 10. What would count as a failure?

The idea is probably weak if:

1. FCV score has no correlation with worst-group test accuracy.
2. Random-mask controls perform as well as teacher maps.
3. Same-context swaps hurt as much as opposite-context swaps.
4. FCV selects models with worse robust test performance than biased validation.
5. The candidate pool does not contain any models with good robust performance.
6. The teacher maps are so sparse/noisy that too few background patches are available.

Important distinction:

If FCV fails because the candidate pool contains no robust models, that is not necessarily a failure of the selector. We need to inspect the oracle-pool upper bound.

---

# 11. Recommended implementation details

## Batch handling

Counterfactual scoring can be expensive. Start simple:

- precompute token banks per checkpoint;
- score validation images in batches;
- sample donor tokens on CPU or GPU depending on speed;
- cache donor token indices for reproducibility.

## Multiple donor samples

Use multiple donor samples per target image to reduce noise.

Recommended first setting:

```text
num_donor_samples = 5
```

Later:

```text
num_donor_samples ∈ {1, 3, 5, 10}
```

## Patch coverage filtering

Skip images with too few safe background patches.

Example:

```text
require at least 20 safe background patches out of 196
```

Also skip images with too few safe evidence patches for evidence-swap diagnostics.

## Ambiguous patches

Leave ambiguous patches unchanged in the first implementation.

Do not swap boundary patches at first. Boundary patches are likely to contain both bird and background information.

## Donor image exclusion

Avoid sampling donor tokens from the same image as the target.

For strictness, also avoid sampling from images in the same mini-batch if that simplifies reproducibility.

## Spatial matching

First version:

```text
sample any donor background token from opposite-context bank
```

Second version:

```text
sample donor tokens from similar patch rows/columns
```

Spatial matching might reduce unnatural hidden states. But do not add it until the simple version is working.

---

# 12. First-study report outline

The first internal report should include:

## Table 1: Candidate pool summary

```text
number of checkpoints
hyperparameter grid
range of biased validation accuracy
range of test worst-group accuracy
oracle-pool upper bound
```

## Table 2: Selector comparison

```text
selector
selected checkpoint
biased val acc
test avg acc
test worst-group acc
gap closed
```

## Table 3: FCV variants

```text
FCV counterfactual accuracy
FCV confidence retention
FCV original + counterfactual score
FCV control-normalized score
```

## Table 4: Controls

```text
semantic opposite-context swap
same-context swap
random-mask swap
shuffled-mask swap
evidence-token swap
```

## Figure 1: Method diagram

Show:

```text
target image → patch tokens → preserve evidence tokens → swap background tokens → prediction should stay target label
```

## Figure 2: Selector scatter plot

Plot:

```text
x-axis: selector score
y-axis: test worst-group accuracy
```

Compare biased validation accuracy vs FCV.

## Figure 3: Example counterfactual token masks

Show image, teacher map, evidence patches, background patches, ambiguous patches.

No need to visualize feature-swapped images because they are not pixel-space images.

---

# 13. Key risks and mitigations

## Risk 1: Feature swaps create unnatural hidden states

Mitigation:

- start at raw patch embedding where tokens are still local;
- compare same-context and opposite-context swaps;
- use random-mask controls;
- use multiple donor samples;
- optionally add spatially matched donor sampling.

## Risk 2: The method just measures token replacement robustness

Mitigation:

- semantic teacher masks must outperform random masks;
- opposite-context swaps should be more diagnostic than same-context swaps;
- evidence swaps should hurt more than background swaps for robust models.

## Risk 3: Teacher maps leak or encode bias

Mitigation:

- use existing R4RR maps known to be good;
- later compare oracle/human maps, VLM maps, noisy maps, and random maps;
- do not claim teacher maps are perfect.

## Risk 4: Candidate pool has no robust model

Mitigation:

- compute best-in-pool robust test performance;
- ensure hyperparameter sweep is wide enough;
- include checkpoints across training epochs;
- possibly include simple augmentation variants later.

## Risk 5: Patch embedding intervention is too early

Mitigation:

- after first proof of concept, sweep transformer blocks 2/4/6;
- compare which layer's FCV score best correlates with robust test performance.

---

# 14. Exact deliverables for the first study

The first study is complete when we have:

1. A trained vanilla ViT candidate pool.
2. Patch-level teacher masks for validation images.
3. Per-checkpoint background token banks.
4. FCV scores for every checkpoint.
5. Control scores for every checkpoint.
6. Selector comparison table.
7. Oracle gap-closure calculation.
8. Rank-correlation analysis.
9. A short internal report with plots.
10. A decision: continue, revise, or abandon.

---

# 15. Decision criteria after the first study

## Continue aggressively if:

- FCV clearly beats biased validation selection;
- FCV closes a meaningful fraction of the oracle selection gap;
- semantic masks beat random/shuffled masks;
- FCV score correlates with worst-group test accuracy;
- same-context controls show the signal is not just token corruption.

## Revise if:

- FCV is noisy but has some correlation;
- patch embedding swaps are too unstable;
- later-layer swaps look more promising;
- confidence-retention scores work better than counterfactual accuracy.

## Abandon or pivot if:

- FCV has no robust-test correlation;
- random masks work just as well;
- same-context swaps are equally disruptive;
- oracle masks do not help;
- the method only works by exploiting artifacts.

---

# 16. My recommended first run

Run this exact first experiment:

```text
Dataset: Waterbirds100
Model: pretrained ViT-S/16
Candidate pool:
    learning_rate ∈ {1e-5, 3e-5, 1e-4}
    weight_decay ∈ {0.01, 0.05, 0.1}
    seed ∈ {0, 1, 2}
    epochs = 20
    save every epoch
Teacher maps: existing R4RR maps
Intervention layer: raw patch embedding
Background threshold: 0.10
Evidence threshold: 0.60
Ambiguous patches: keep target
Donor samples: 5
Selectors:
    biased val acc
    biased val loss
    FCV counterfactual acc
    FCV original + counterfactual score
    oracle group val
Test metric:
    worst-group accuracy
Main result:
    oracle gap closure
```

This is the simplest serious test of the idea.

---

# 17. Final summary

The first study should stay focused. Do not try to prove every version of the paper yet.

The core question is:

> Can feature-space background counterfactuals reveal which vanilla ViT checkpoints are less background-dependent when the validation set itself contains no shortcut-breaking examples?

If the answer is yes, then the project has a strong foundation. The next phase can add more datasets, model families, teacher-map sources, intervention layers, and optional counterfactual-compatible training.

