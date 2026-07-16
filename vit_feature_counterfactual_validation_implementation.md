# ViT Feature-Counterfactual Validation: Implementation Blueprint

## Working title

**Feature-Counterfactual Validation for Spatial Shortcut Robustness in Vision Transformers**

Short name: **ViT-FCV**.

This document describes the ViT-first version of the idea: use spatial teacher maps to create feature-space counterfactuals inside a Vision Transformer, then use those counterfactuals as a model-selection signal when validation data is spatially spuriously correlated.

The goal is not to create a new image. The goal is to ask:

> If we preserve the target image's object/evidence tokens but replace its background tokens with background tokens from another context, does the model's prediction still follow the object evidence?

For Waterbirds100-style validation, this becomes:

> If a waterbird's bird tokens are combined with land-background tokens, does the model still predict waterbird?

If yes, the model is more likely to rely on evidence. If no, the model may be background-dependent.

---

# 1. Core problem setting

We care about validation/model selection under spatial shortcut bias.

In ordinary Waterbirds, a model can be selected using a validation set that contains group diversity or group labels. But in a harder setting like **Waterbirds100**, the validation data may be completely spuriously correlated:

```text
waterbird validation images -> water backgrounds
landbird validation images  -> land backgrounds
```

Then ordinary validation accuracy cannot distinguish these two models:

```text
Model A: learned bird evidence.
Model B: learned water/land background shortcut.
```

Both can score highly on biased validation.

The proposed solution is to create **missing counterexamples in feature space**, not pixel space.

Instead of synthesizing:

```text
waterbird pixels pasted into land image
```

we synthesize:

```text
waterbird evidence tokens + land-background tokens
```

inside the ViT representation.

This avoids black masks, blur artifacts, inpainting holes, and pasted-image seams.

---

# 2. Why ViT is the simplest first implementation

A ViT naturally represents an image as a sequence of patch tokens.

For a standard `224 x 224` image with patch size `16`, the image becomes:

```text
14 x 14 = 196 patch tokens
```

Each patch token corresponds to a known spatial patch of the image. That makes the bookkeeping much cleaner than in CNNs, where feature-map cells have overlapping receptive fields.

For the first implementation, use a ViT variant where we can access:

```text
patch_embed
pos_embed
cls_token, if used
transformer blocks
norm
classifier head
```

Good first choices:

```text
vit_small_patch16_224
vit_base_patch16_224
deit_small_patch16_224
```

Prefer a model where patch tokens are accessible and the forward pass can be restarted from patch embeddings.

---

# 3. Main version to implement first

The cleanest first version is:

> **Patch-embedding-level background token swapping.**

This means we intervene **after patch embedding but before transformer blocks**.

Pipeline:

```text
image -> patch embedding -> patch tokens -> swap background tokens -> add positional embeddings -> transformer blocks -> classifier
```

Why start here?

1. Patch tokens are still local.
2. CLS/global attention has not mixed evidence and background yet.
3. Donor tokens are real patch embeddings from real validation images.
4. The spatial teacher mask maps directly onto the patch grid.
5. We avoid late-layer ambiguity where a “background token” may already contain object information.

Later, we can run layer sweeps:

```text
intervene after patch_embed
intervene after block 2
intervene after block 4
intervene after block 6
```

But the first MVP should use patch-embedding-level swaps.

---

# 4. Data assumptions

Each validation sample should provide:

```python
sample = {
    "image": Tensor[3, H, W],
    "label": int,
    "mask": Tensor[1, H, W],  # spatial teacher map for class evidence
    "group": optional,        # only for oracle analysis, not for FCV selection
}
```

The mask is a spatial teacher map. It can come from:

1. oracle/human/dataset segmentation;
2. VLM/open-vocabulary segmentation;
3. SAM + class prompt filtering;
4. noisy maps for robustness analysis;
5. random maps for controls.

The main method should be **teacher-map agnostic**.

For Waterbirds, the mask should mark the bird, not the water/land background.

---

# 5. Projecting image masks to patch masks

For a ViT with patch size `P = 16`, convert the image-space mask into a patch-level mask.

If the image is `224 x 224`, the patch grid is `14 x 14`.

Let:

```text
M_img: 1 x 224 x 224
M_patch: 14 x 14
```

Compute `M_patch` using average pooling:

```python
M_patch = avg_pool2d(M_img.float(), kernel_size=P, stride=P)
```

Each patch receives a value in `[0, 1]` representing the fraction of that patch covered by the evidence mask.

Then define:

```python
evidence_patch   = M_patch >= tau_evidence
background_patch = M_patch <= tau_background
ambiguous_patch  = otherwise
```

Recommended first thresholds:

```python
tau_evidence = 0.60 or 0.70
tau_background = 0.05 or 0.10
```

For high precision, use stricter thresholds:

```python
tau_evidence = 0.80
tau_background = 0.05
```

Important: ambiguous patches should be left unchanged. Do **not** swap boundary patches in the first version.

This creates three patch sets:

```text
E_i = safe evidence patches for image i
B_i = safe background patches for image i
A_i = ambiguous patches for image i
```

For a target counterfactual, preserve evidence and ambiguous patches from the target image. Replace only safe background patches.

---

# 6. What gets swapped?

For target image `i`, we compute target patch embeddings:

```text
Z_i = patch_embed(x_i)  # shape: [N, D], where N = 196
```

For donor images, we compute donor patch embeddings:

```text
Z_j = patch_embed(x_j)
```

We build a bank of background patch embeddings:

```text
background_bank[c] = all safe-background patch embeddings from images associated with context c
```

For Waterbirds100, the shortcut is perfectly correlated with the class:

```text
waterbird images -> water backgrounds
landbird images  -> land backgrounds
```

So the simplest first implementation can use:

```text
water-background bank = background tokens from waterbird validation images
land-background bank  = background tokens from landbird validation images
```

Then:

```text
waterbird target -> replace its water-background tokens with land-background tokens
landbird target  -> replace its land-background tokens with water-background tokens
```

The label should follow the target evidence, not the donor background.

---

# 7. Why this avoids the donor-hole problem

Pixel-space background swapping has a huge issue:

```text
If the donor landbird is large, removing it leaves a giant hole.
```

Then we need inpainting, cloning, blur, or some other artifact-prone repair.

ViT-FCV avoids this completely.

We never remove the donor object from the donor image. We only collect donor patch embeddings from donor locations that the teacher mask says are safe background.

If a donor image has only a few safe background patches, that is fine. We take those few patches and add them to the background bank. If an image has no safe background patches, skip it.

The background counterfactual for a target image is built from a bank pooled across many images:

```text
land-background token bank = safe background tokens from many land-background images
water-background token bank = safe background tokens from many water-background images
```

So no single donor image has to provide a complete full-image background.

---

# 8. Basic counterfactual construction

For a target image `i` with label `y_i`, define:

```text
Z_i: target patch embeddings, shape [N, D]
E_i: evidence patch indices
B_i: background patch indices
A_i: ambiguous patch indices
```

Choose a donor background bank that breaks the shortcut.

For Waterbirds100:

```python
if y_i == "waterbird":
    donor_bank = land_background_bank
else:
    donor_bank = water_background_bank
```

Sample `len(B_i)` donor background embeddings:

```text
D_sampled: [len(B_i), D]
```

Construct the counterfactual patch embeddings:

```python
Z_cf = Z_i.clone()
Z_cf[B_i] = D_sampled
```

Keep:

```text
evidence patches from target
ambiguous patches from target
background patches from donor bank
```

Then run the rest of the ViT:

```text
Z_cf -> add target positional embeddings -> transformer blocks -> classifier head
```

The predicted label should remain `y_i` if the model follows evidence.

---

# 9. Position handling

For the first implementation, store donor tokens **before positional embeddings**.

This is important.

Patch embeddings before position encodings represent local visual content without being tied to a specific absolute position. After replacement, add the **target** positional embeddings.

That means a donor land-background patch sampled from some image becomes background content placed at the target background location.

Forward pass:

```python
patch_tokens = model.patch_embed(images)  # no pos yet
patch_tokens_cf = swap_background_tokens(patch_tokens, masks, donor_bank)
patch_tokens_cf = patch_tokens_cf + target_positional_embeddings
logits = forward_blocks_and_head(patch_tokens_cf)
```

Later variants can test position-matched sampling:

```text
replace top-left target background token with donor background tokens from top-left-ish positions
replace lower-row target background token with donor background tokens from lower-row-ish positions
```

But the first version should be global sampling before position embeddings.

---

# 10. Forward pass design

A practical wrapper should expose two functions:

```python
patch_tokens = model_to_patch_tokens(images)
logits = model_from_patch_tokens(patch_tokens)
```

For a timm-like ViT, pseudocode:

```python
def model_to_patch_tokens(model, images):
    # Output shape: [B, N, D]
    return model.patch_embed(images)


def model_from_patch_tokens(model, patch_tokens):
    B, N, D = patch_tokens.shape

    if model_has_cls_token(model):
        cls = model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patch_tokens], dim=1)
        x = x + model.pos_embed[:, :N + 1, :]
    else:
        x = patch_tokens + model.pos_embed[:, :N, :]

    x = model.pos_drop(x)

    for block in model.blocks:
        x = block(x)

    x = model.norm(x)

    if model_uses_cls(model):
        feat = x[:, 0]
    else:
        feat = x.mean(dim=1)

    logits = model.head(feat)
    return logits
```

Different ViT implementations vary, so the wrapper should be implemented for the chosen model family first, then generalized later.

---

# 11. First model choice

Use a simple, accessible ViT implementation.

Recommended first pass:

```text
timm vit_small_patch16_224
or
timm deit_small_patch16_224
```

Reasons:

1. patch size is clear;
2. patch grid is manageable;
3. model is small enough for many candidate checkpoints;
4. patch embeddings and transformer blocks are easy to access;
5. pretrained weights are available.

Try both:

```text
CLS-token head
mean-pooled patch-token head
```

For the cleanest interpretability, mean pooling may be easier. But CLS-token ViTs are common, so the method should eventually support them.

---

# 12. Candidate model pool

The paper is about model selection, so we need a candidate pool.

Each candidate is a trained model checkpoint.

Sources of candidates:

```text
ERM checkpoints at different epochs
ERM with different seeds
ERM with different learning rates
ERM with different weight decay
ERM with different augmentations
GALS/R4RR-style models if available
AFR/JTT-style variants if available
MaskTune/foreground-masking variants if available
FCV-training extension variants if implemented
```

For the first experiment, a smaller pool is fine:

```text
20-50 ViT checkpoints trained on Waterbirds100-style data
```

For a serious paper, aim for:

```text
100-500 candidate checkpoints per dataset
```

The evaluation question is:

> Which validation selector chooses the candidate with the best real robust test performance?

---

# 13. Validation selectors to compare

FCV should be compared against simpler selectors.

Minimum selectors:

```text
1. Standard biased validation accuracy
2. Standard biased validation loss
3. Last checkpoint
4. FCV counterfactual accuracy
5. FCV counterfactual confidence score
6. Oracle group-balanced validation accuracy, if available
7. Oracle worst-group validation accuracy, if available
```

Stronger baselines:

```text
8. Saliency-mask alignment selector
9. Grad-CAM/attention overlap selector
10. Group-discovery/inferred-group selector
11. EVaLS-style selector, if feasible
12. Random mask feature-swap selector
```

The main result should be selection regret.

```text
selection regret = best candidate robust test score - selected candidate robust test score
```

Lower is better.

---

# 14. Main FCV score

For each candidate model `f`, validation image `x_i`, label `y_i`, and `K` counterfactual background swaps:

```text
p_i_orig = softmax(f(x_i))[y_i]
p_i_cf,k = softmax(f_counterfactual(x_i, bg_swap_k))[y_i]
```

The simplest score:

```text
FCV_confidence = mean_i,k p_i_cf,k
```

Counterfactual accuracy:

```text
FCV_accuracy = mean_i,k 1[argmax f_counterfactual(x_i, bg_swap_k) == y_i]
```

A robust candidate should keep predicting the target label after background-token replacement.

But avoid selecting a model that is bad on original validation. Combine with original validation accuracy:

```text
SelectorScore = Acc_original + lambda * FCV_accuracy
```

or:

```text
SelectorScore = Acc_original + lambda * mean_i,k log p_i_cf,k
```

Recommended first version:

```text
SelectorScore = 0.5 * Acc_original + 0.5 * FCV_accuracy
```

Then tune `lambda` only on development datasets, not on the target benchmark.

---

# 15. Conditional FCV score

A useful diagnostic is to score only images that the model originally gets right:

```text
FCV_given_correct = mean_{i,k: pred_orig_i == y_i} 1[pred_cf_i,k == y_i]
```

This asks:

> When the model is right on the biased original image, does it remain right when the background tokens are counterfactually changed?

This isolates background reliance among successful predictions.

But do not use this alone as the final selector because a bad model with few correct original samples could get a misleading score. Pair it with original accuracy.

---

# 16. Evidence-swap diagnostic

To verify that predictions actually depend on object evidence, also run the opposite intervention:

```text
keep target background tokens
replace target evidence tokens with evidence tokens from another class
```

For a good model:

```text
background swap -> prediction should stay
foreground/evidence swap -> prediction should change or confidence should drop
```

Define:

```text
EvidenceSensitivity = p_y(original) - p_y(evidence_swapped)
BackgroundSensitivity = p_y(original) - p_y(background_swapped)
SensitivityGap = EvidenceSensitivity - BackgroundSensitivity
```

A robust evidence-based model should have:

```text
SensitivityGap > 0
```

However, evidence swapping is more delicate than background swapping because evidence tokens carry class-defining information and can create unnatural mixtures. Use this first as a diagnostic/ablation, not necessarily the main selector.

---

# 17. Control swaps

Controls are essential. They prove FCV is not merely measuring sensitivity to arbitrary token corruption.

Implement these controls:

## 17.1 Same-background swap

Replace target background tokens with background tokens from the same shortcut context.

For Waterbirds100:

```text
waterbird target -> water-background donor tokens
landbird target  -> land-background donor tokens
```

The model should remain stable here. If same-background swaps are as damaging as opposite-background swaps, then the method may just be measuring token replacement artifacts.

## 17.2 Opposite-background swap

The main FCV intervention.

```text
waterbird target -> land-background donor tokens
landbird target  -> water-background donor tokens
```

This should expose shortcut reliance.

## 17.3 Random-token swap

Replace target background tokens with random patch tokens from anywhere, matched in count.

This tests whether semantic donor choice matters.

## 17.4 Random-mask swap

Use random patch positions with the same count as the background mask, rather than teacher-map background positions.

This tests whether teacher maps are doing real work.

## 17.5 Wrong-mask swap

Use an intentionally wrong or shuffled mask from another image.

This tests whether mask semantics matter.

## 17.6 No-swap original

Ordinary validation.

The expected ordering for a shortcut-dependent model:

```text
original accuracy high
same-background swap stable
opposite-background swap unstable
foreground/evidence swap very unstable
```

For an evidence-reliant model:

```text
original accuracy high
same-background swap stable
opposite-background swap mostly stable
foreground/evidence swap unstable
```

---

# 18. Donor bank construction

For each candidate model and each intervention layer, build a background token bank.

For patch-embedding-level FCV:

```python
for images, labels, masks in val_loader:
    patch_tokens = model.patch_embed(images)  # [B, N, D]
    patch_masks = masks_to_patch_masks(masks) # [B, N]

    for b in range(B):
        bg_indices = patch_masks[b] <= tau_background
        label = labels[b]
        bank[label].append(patch_tokens[b, bg_indices])
```

For Waterbirds100:

```text
bank[waterbird] contains mostly water-background patch embeddings
bank[landbird] contains mostly land-background patch embeddings
```

If group labels are available for analysis, build cleaner oracle banks:

```text
bank[water_background]
bank[land_background]
```

But the practical no-group version can use class-correlated banks in the fully biased setting.

---

# 19. Donor sampling strategies

Start with simple global sampling.

## 19.1 Global sampling

For each target background patch position, sample a random donor token from the donor bank.

Pros:

```text
simple
fast
works even if donor images have few background patches
```

Cons:

```text
may put sky-like tokens in lower image regions or grass-like tokens in upper regions
```

## 19.2 Position-matched sampling

Store row/column metadata for each donor token.

Sample donor tokens from similar spatial positions:

```text
target row r, col c -> donor tokens from rows near r and columns near c
```

This preserves rough layout.

Recommended second version.

## 19.3 Feature-statistic matching

Sample donor tokens whose low-level feature statistics are compatible with the target background position.

This is more complex and should not be part of the MVP.

---

# 20. Counterfactual batch generation

For efficiency, generate multiple counterfactuals per image.

Let:

```text
K = number of donor samples per target image
```

Recommended first value:

```text
K = 4
```

For stronger validation:

```text
K = 8 or 16
```

Each target image produces:

```text
original logits
K same-background counterfactual logits
K opposite-background counterfactual logits
optional K random-token counterfactual logits
optional K evidence-swap counterfactual logits
```

Use vectorization by repeating target patch tokens `K` times and sampling donor tokens in batch.

---

# 21. Exact MVP algorithm

## Inputs

```text
candidate models F = {f_1, ..., f_M}
biased validation set V = {(x_i, y_i, M_i)}
teacher masks M_i
patch size P
number of swaps K
```

## For each candidate model f

1. Compute original validation accuracy.
2. Build patch-token background banks from validation data.
3. For each validation image:
   - compute patch tokens;
   - compute safe evidence/background/ambiguous patch sets;
   - sample opposite-background donor tokens;
   - replace safe background tokens;
   - forward through ViT blocks and head;
   - record counterfactual prediction and true-class probability.
4. Compute FCV score.
5. Select the model with the highest FCV selector score.

## Output

```text
selected checkpoint / hyperparameter / method
```

Then evaluate the selected model on real robust test data:

```text
worst-group accuracy
OOD accuracy
minority-group accuracy
selection regret
```

---

# 22. Pseudocode: mask to patch labels

```python
import torch
import torch.nn.functional as F


def masks_to_patch_scores(masks, patch_size):
    """
    masks: [B, 1, H, W], values in [0, 1]
    returns: [B, N], where N = (H/P) * (W/P)
    """
    pooled = F.avg_pool2d(masks.float(), kernel_size=patch_size, stride=patch_size)
    return pooled.flatten(1)


def patch_partitions(patch_scores, tau_evidence=0.7, tau_background=0.1):
    """
    patch_scores: [B, N]
    returns boolean masks: evidence, background, ambiguous
    """
    evidence = patch_scores >= tau_evidence
    background = patch_scores <= tau_background
    ambiguous = ~(evidence | background)
    return evidence, background, ambiguous
```

---

# 23. Pseudocode: build background bank

```python
@torch.no_grad()
def build_background_bank(model, val_loader, patch_size, tau_background=0.1, device="cuda"):
    """
    Builds class-indexed background token banks for patch-embedding-level FCV.
    For Waterbirds100, class-indexed banks act as shortcut-background banks.
    """
    model.eval()
    bank = {}

    for images, labels, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        patch_tokens = model.patch_embed(images)  # [B, N, D], before pos embed
        patch_scores = masks_to_patch_scores(masks, patch_size)  # [B, N]
        _, background, _ = patch_partitions(
            patch_scores,
            tau_evidence=0.7,
            tau_background=tau_background,
        )

        B = images.size(0)
        for b in range(B):
            y = int(labels[b].item())
            bg_tokens = patch_tokens[b, background[b]].detach().cpu()
            if bg_tokens.numel() == 0:
                continue
            if y not in bank:
                bank[y] = []
            bank[y].append(bg_tokens)

    for y in bank:
        bank[y] = torch.cat(bank[y], dim=0)  # [num_bg_tokens, D]

    return bank
```

---

# 24. Pseudocode: sample donor background tokens

```python
def sample_bank_tokens(bank_tokens, num_tokens, device):
    """
    bank_tokens: [B_bank, D]
    returns: [num_tokens, D]
    """
    n = bank_tokens.size(0)
    idx = torch.randint(low=0, high=n, size=(num_tokens,))
    return bank_tokens[idx].to(device)
```

For position-matched sampling, store `(row, col)` with each bank token and sample from nearby rows/columns.

---

# 25. Pseudocode: construct counterfactual patch tokens

```python
def make_bg_swap_counterfactuals(
    patch_tokens,
    labels,
    background_mask,
    bank,
    opposite_label_fn,
    K=4,
):
    """
    patch_tokens: [B, N, D], before pos embeddings
    labels: [B]
    background_mask: [B, N], bool
    bank: dict[label -> Tensor[num_tokens, D]]
    opposite_label_fn: maps target label to donor-bank label

    returns: [B*K, N, D]
    """
    device = patch_tokens.device
    B, N, D = patch_tokens.shape
    cf_tokens = []

    for b in range(B):
        y = int(labels[b].item())
        donor_key = opposite_label_fn(y)
        donor_bank = bank[donor_key]
        bg_idx = background_mask[b].nonzero(as_tuple=False).flatten()
        num_bg = len(bg_idx)

        for _ in range(K):
            z = patch_tokens[b].clone()
            if num_bg > 0:
                sampled = sample_bank_tokens(donor_bank, num_bg, device=device)
                z[bg_idx] = sampled
            cf_tokens.append(z)

    return torch.stack(cf_tokens, dim=0)
```

For Waterbirds binary labels:

```python
def opposite_label_fn(y):
    return 1 - y
```

This assumes label `0` and `1` correspond to the two shortcut-correlated backgrounds.

---

# 26. Pseudocode: forward from patch tokens

```python
def forward_from_patch_tokens(model, patch_tokens):
    """
    patch_tokens: [B, N, D], before pos embeddings
    returns logits: [B, num_classes]
    This pseudocode assumes a timm-like ViT with cls token.
    """
    B, N, D = patch_tokens.shape

    if hasattr(model, "cls_token") and model.cls_token is not None:
        cls = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, patch_tokens), dim=1)
        x = x + model.pos_embed[:, :N + 1, :]
    else:
        x = patch_tokens + model.pos_embed[:, :N, :]

    x = model.pos_drop(x)

    for block in model.blocks:
        x = block(x)

    x = model.norm(x)

    # CLS-token model
    if hasattr(model, "cls_token") and model.cls_token is not None:
        feat = x[:, 0]
    else:
        feat = x.mean(dim=1)

    logits = model.head(feat)
    return logits
```

Need to adapt this to the exact ViT implementation. Some timm models use `forward_head`, `fc_norm`, or global pooling differently.

---

# 27. Pseudocode: compute FCV score

```python
@torch.no_grad()
def compute_fcv_score(
    model,
    val_loader,
    bank,
    patch_size,
    opposite_label_fn,
    K=4,
    tau_evidence=0.7,
    tau_background=0.1,
    lambda_fcv=0.5,
    device="cuda",
):
    model.eval()

    total = 0
    correct_orig = 0
    correct_cf = 0
    cf_total = 0
    true_prob_cf_sum = 0.0

    for images, labels, masks in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        patch_tokens = model.patch_embed(images)
        logits_orig = forward_from_patch_tokens(model, patch_tokens)
        pred_orig = logits_orig.argmax(dim=1)
        correct_orig += (pred_orig == labels).sum().item()
        total += labels.numel()

        patch_scores = masks_to_patch_scores(masks, patch_size)
        _, background, _ = patch_partitions(
            patch_scores,
            tau_evidence=tau_evidence,
            tau_background=tau_background,
        )

        cf_tokens = make_bg_swap_counterfactuals(
            patch_tokens=patch_tokens,
            labels=labels,
            background_mask=background,
            bank=bank,
            opposite_label_fn=opposite_label_fn,
            K=K,
        )

        labels_cf = labels.repeat_interleave(K)
        logits_cf = forward_from_patch_tokens(model, cf_tokens)
        probs_cf = logits_cf.softmax(dim=1)
        pred_cf = logits_cf.argmax(dim=1)

        correct_cf += (pred_cf == labels_cf).sum().item()
        cf_total += labels_cf.numel()
        true_prob_cf_sum += probs_cf[torch.arange(labels_cf.numel(), device=device), labels_cf].sum().item()

    acc_orig = correct_orig / max(total, 1)
    acc_cf = correct_cf / max(cf_total, 1)
    mean_true_prob_cf = true_prob_cf_sum / max(cf_total, 1)

    selector_score = (1.0 - lambda_fcv) * acc_orig + lambda_fcv * acc_cf

    return {
        "acc_orig": acc_orig,
        "acc_cf": acc_cf,
        "mean_true_prob_cf": mean_true_prob_cf,
        "selector_score": selector_score,
    }
```

---

# 28. How to validate whether FCV actually works

The method is not judged by counterfactual validation accuracy itself.

It is judged by whether FCV selects models with better **real robust test performance**.

For each candidate selector:

```text
selector chooses one candidate model
chosen model is evaluated on real test data
report worst-group/OOD accuracy
```

Key metric:

```text
selection regret = robust_test_score(best candidate) - robust_test_score(selected candidate)
```

If FCV has low selection regret compared to biased validation accuracy, it is doing something useful.

---

# 29. First experiment: Waterbirds100 MVP

## Goal

Show that FCV selects better models than ordinary biased validation when validation has no natural counterexamples.

## Dataset setup

Use Waterbirds-style data.

Construct biased validation:

```text
waterbird validation -> water background only
landbird validation  -> land background only
```

Keep real test groups for evaluation:

```text
waterbird on water
waterbird on land
landbird on land
landbird on water
```

## Masks

Start with oracle bird masks if available.

Then repeat with VLM-generated bird masks.

## Candidate pool

Train 20-50 ViT candidates:

```text
different seeds
different epochs
different learning rates
different weight decay
different augmentation strengths
possibly different debiasing losses
```

## Selectors

Compare:

```text
biased val accuracy
biased val loss
FCV opposite-background counterfactual accuracy
FCV combined score
oracle group-balanced validation
oracle worst-group validation
random-mask FCV control
```

## Expected result

Biased validation may select high-average, shortcut-heavy models.

FCV should select models with better minority/worst-group test accuracy.

---

# 30. Important ablations

## 30.1 Mask source

```text
oracle masks
VLM masks
noisy oracle masks
random same-area masks
wrong-class masks
shuffled masks
```

Expected ordering:

```text
oracle > VLM > noisy > random/shuffled
```

## 30.2 Intervention layer

```text
patch embedding
block 2
block 4
block 6
late block
```

Expected:

```text
patch embedding or early blocks are cleanest
middle blocks may be strongest
late blocks may be too mixed
```

## 30.3 Donor sampling

```text
global sampling
position-matched sampling
same-background sampling
opposite-background sampling
random-token sampling
```

## 30.4 Number of swaps K

```text
K = 1, 2, 4, 8, 16
```

Check stability and compute cost.

## 30.5 Patch thresholds

```text
tau_background = 0.01, 0.05, 0.10, 0.20
tau_evidence   = 0.50, 0.70, 0.90
```

Using stricter thresholds should improve semantic purity but reduce coverage.

## 30.6 Original accuracy anchoring

Compare:

```text
FCV only
original val accuracy only
original val accuracy + FCV
original val loss + FCV
```

---

# 31. Training extension

The paper should be validation-first, but an optional training extension may be valuable.

Name:

```text
Feature-Counterfactual Consistency Training
```

During training, perform stochastic background-token swaps and require the label to remain the target label.

Training objective:

```text
L = CE(original, y)
  + lambda_bg * CE(background_swapped_features, y)
  + lambda_cons * KL(pred_original || pred_background_swapped)
```

Optional evidence-swap loss:

```text
remove/swap evidence tokens -> true-class confidence should drop
```

But keep this optional because evidence swaps may create unstable hidden states.

## Why include training?

1. It can improve robustness.
2. It reduces the “unnatural hidden state” concern because models learn to handle feature counterfactuals.
3. It creates an intervention-ready candidate family.
4. It can be combined with ERM, GALS, R4RR, AFR/JTT-style methods.

## How to avoid circularity

Do not train and validate with exactly the same intervention distribution.

Possible separation:

```text
train with global donor sampling
validate with position-matched donor sampling

train with patch-embedding swaps
validate with early-block swaps

train with VLM masks
validate with oracle masks where available

train with prompt set A
validate with prompt set B
```

Final evaluation must always be real natural OOD/worst-group test data.

---

# 32. How to argue fairness

The fairness argument for ViT-FCV:

1. **No corrupted images.** Candidate models are not fed blacked-out, blurred, or inpainted images.
2. **Real features only.** Donor background tokens come from real validation images passed through the same model.
3. **Same protocol for all candidates.** Every model uses the same teacher masks, patch thresholds, donor-bank rules, and selector formula.
4. **Control-normalized.** Same-background, random-token, and random-mask controls are included.
5. **Judged by real test data.** FCV is only useful if it predicts real worst-group/OOD test performance.
6. **Teacher-map agnostic.** The framework can use oracle, human, VLM, or noisy maps; the mask source is explicitly studied.

The method does not claim the feature counterfactual is a perfect natural sample. It claims the counterfactual is a useful diagnostic intervention for model selection.

---

# 33. Main risks and mitigations

## Risk 1: Feature counterfactuals create unnatural hidden states

Mitigations:

```text
compare to same-background swaps
compare to random-token swaps
use early-layer swaps
optionally train with feature-counterfactual consistency
use selection regret on real test data as final proof
```

## Risk 2: Donor background tokens leak object information

Mitigations:

```text
strict background threshold
mask erosion
skip ambiguous patches
oracle-mask ablation
VLM-mask quality analysis
wrong-mask controls
```

## Risk 3: VLM masks are poor

Mitigations:

```text
teacher-map agnostic framing
oracle masks for upper bound
VLM masks for practical setting
noisy-mask sensitivity study
prompt ensembles
mask stability filtering
```

## Risk 4: Patch-level masks are too coarse

Mitigations:

```text
use ViT patch size 16 first
test patch size 14 or 8 if available
strict thresholds
leave boundary patches unchanged
```

## Risk 5: FCV rewards robustness to feature swaps rather than real OOD robustness

Mitigations:

```text
main metric is real test selection regret
include many candidate model types
include multiple datasets
include negative controls
```

---

# 34. Suggested code organization

```text
fcv/
  data/
    waterbirds.py
    masks.py
  models/
    vit_wrapper.py
  fcv/
    patch_masks.py
    banks.py
    swaps.py
    scoring.py
    selectors.py
  train/
    train_vit.py
    train_vit_fcv_consistency.py
  eval/
    build_candidate_pool.py
    compute_fcv_scores.py
    select_models.py
    evaluate_selected.py
  analysis/
    selection_regret.py
    mask_quality.py
    layer_sweep.py
    controls.py
```

---

# 35. Minimal viable experiment plan

## Phase 1: Does the intervention execute?

Use one trained ViT on Waterbirds.

Run:

```text
original validation
same-background swap
opposite-background swap
random-token swap
```

Check whether logits change in sensible ways.

## Phase 2: Does FCV distinguish models?

Train several ERM ViTs with different seeds/epochs.

Compute:

```text
biased val accuracy
FCV score
real worst-group test accuracy
```

Check rank correlation.

## Phase 3: Does FCV select better?

Build a candidate pool.

Select by:

```text
biased validation accuracy
FCV score
oracle group validation
```

Report test worst-group accuracy and selection regret.

## Phase 4: Mask source study

Repeat with:

```text
oracle masks
VLM masks
random masks
noisy masks
```

## Phase 5: Training extension

Train ViT with feature-counterfactual consistency.

Ask:

```text
Does FCV-trained ViT get better robust test performance?
Does FCV validation select better checkpoints among FCV-trained candidates?
```

---

# 36. What success should look like

A strong result would show:

```text
biased validation accuracy selects a model with high average validation accuracy but poor worst-group test accuracy

FCV selects a model with similar validation accuracy but much better worst-group/OOD test accuracy

oracle group validation remains best or comparable

FCV closes much of the gap between biased validation and oracle validation
```

A clean table:

```text
Selector                    Selected worst-group test acc    Selection regret
Biased val accuracy          62.1                             18.4
Biased val loss              64.5                             16.0
Random-mask FCV              65.0                             15.5
Saliency overlap             68.2                             12.3
FCV with VLM masks           76.8                              3.7
FCV with oracle masks         79.1                              1.4
Oracle group validation       80.5                              0.0
```

Numbers above are illustrative only.

---

# 37. Best paper framing

Do not frame the paper as:

```text
We swap ViT features.
```

Frame it as:

```text
We study model selection when validation contains no natural counterexamples to a spatial shortcut. We propose Feature-Counterfactual Validation, which constructs missing counterexamples inside candidate models' spatial representations by recombining evidence and background patch tokens. Models are selected based on whether predictions follow evidence rather than background.
```

The key sentence:

> FCV creates shortcut-breaking validation interventions without editing pixels, requiring group labels, or relying on saliency-map overlap.

---

# 38. What to implement first, exactly

The first implementation should be narrow and clean:

```text
Architecture: timm ViT-S/16 or DeiT-S/16
Dataset: Waterbirds100-style validation
Masks: oracle bird masks first
Intervention layer: patch embedding before positional embeddings
Donor banks: class-indexed background token banks
Swap: opposite-label background token replacement
K: 4 swaps per image
Selector: 0.5 * original val accuracy + 0.5 * opposite-background counterfactual accuracy
Controls: same-background swap, random-token swap, random-mask swap
Evaluation: selection regret on real Waterbirds group test set
```

Once this works, expand to:

```text
VLM masks
more candidate models
layer sweeps
MetaShift/NICO/ImageNet-9
training extension
CNN version
```

---

# 39. Final concise version

ViT-FCV is best understood as a validation-time causal stress test in patch-token space:

```text
Keep object/evidence tokens fixed.
Swap background tokens with tokens from the opposite shortcut context.
Ask whether the prediction follows the object or the background.
Use that behavior to select models when normal validation accuracy is shortcut-contaminated.
```

This is the simplest, cleanest, and most defensible first implementation of the broader Feature-Counterfactual Validation idea.
