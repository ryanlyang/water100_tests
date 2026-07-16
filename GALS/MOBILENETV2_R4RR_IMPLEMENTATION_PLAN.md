# MobileNetV2 R4RR Implementation Plan

## Goal

Add MobileNetV2 as the lightweight CNN student for the guided KL / R4RR experiments, with the implementation designed specifically to preserve the CAM assumptions used by R4RR.

The target experiment set is:

- Guided KL / R4RR MobileNetV2 for Waterbirds-95, Waterbirds-100, and RedMeat.
- Vanilla MobileNetV2 for Waterbirds-95, Waterbirds-100, and RedMeat.
- 50-trial Optuna sweeps for each dataset/method pair.
- Best-hyperparameter reruns over seeds 0-4.

DecoyMNIST should remain LeNet-based for the main paper comparisons. A MobileNetV2 DecoyMNIST stress test can be added later, but it should be separate because it changes the input regime from small synthetic digits to 224x224 ImageNet-style images.

## Why MobileNetV2 Instead Of MobileNetV3

R4RR uses CAM-style guidance. In the current guided code, the attention map is computed as:

```text
CAM_y(h, w) = classifier_weight[y] dot feature_map(:, h, w)
```

This is most faithful when the student has the classic CAM-compatible structure:

```text
spatial features -> global average pooling -> linear classifier
```

ResNet-50 has this structure. MobileNetV2 also has this structure up to its optional dropout before the classifier. MobileNetV3-Large is less clean because torchvision MobileNetV3 uses an MLP-style classifier head:

```text
features -> avgpool -> linear -> hardswish -> dropout -> linear
```

That extra nonlinear classifier head means the final classifier weights do not directly correspond to raw convolutional feature channels. We patched around that for MobileNetV3 by applying the classifier prefix spatially, but that is inherently less direct. MobileNetV2 gives a cleaner story: it is a standard lightweight CNN, and its representation-to-logit path can be made exactly CAM-compatible.

## Core Design Choice

Use MobileNetV2 with a CAM-exact head:

```text
MobileNetV2.features -> adaptive average pool -> flatten -> Linear(num_classes)
```

Do not keep active classifier dropout in the guided MobileNetV2 wrapper. The reason is not to improve performance arbitrarily; it is to keep logits and CAMs mathematically aligned. If dropout is active during training, logits are computed from dropped pooled features, while CAMs are computed from the undropped spatial feature tensor. That creates another mismatch in the guidance loss.

For consistency, vanilla MobileNetV2 should use the same classifier replacement:

```python
model.classifier = nn.Linear(1280, num_classes)
```

or an equivalent wrapper whose forward pass is exactly:

```python
features = model.features(images)
pooled = adaptive_avg_pool2d(features, 1).flatten(1)
logits = classifier(pooled)
```

This makes the guided-vs-vanilla comparison fair: both use the same MobileNetV2 student, and only the R4RR guidance differs.

## Fixed Experimental Choices

Keep these aligned with the current ResNet-50 and MobileNetV3 runs:

- Input size: `224x224`.
- Normalization: ImageNet mean/std.
- Initialization: ImageNet-pretrained MobileNetV2 weights when available.
- Batch size: `96`.
- Optimizer: SGD.
- Weight decay: `1e-5`.
- Guided momentum: fixed `0.9`, matching the current guided runner.
- Vanilla momentum: swept if matching existing vanilla sweeps, otherwise fixed only if we deliberately choose a stricter controlled setup.
- Waterbirds epochs: `200`.
- RedMeat epochs: `150`.
- Final reporting seeds: `0,1,2,3,4`.
- Checkpoint saving: disabled by default during sweeps through `SAVE_CHECKPOINTS=0`, same as recent runners.

## Validation Objectives

Use the same model-selection protocol as the existing sweeps:

- Waterbirds-95: maximize validation balanced accuracy over classes during training, then report test average group and worst group.
- Waterbirds-100: same as Waterbirds-95.
- RedMeat: maximize validation balanced class accuracy, reported through the same `best_balanced_val_acc` convention used by the RedMeat runners.

Do not tune on test metrics.

## Implementation Plan

### Step 1: Add MobileNetV2 CAM Backbone

Edit:

- `models/cam_backbones.py`

Add:

- `_mobilenet_v2(pretrained: bool)`
- `MobileNetV2CAM`
- `mobilenet_v2` support in `make_cam_backbone`

Implementation details:

1. Build torchvision MobileNetV2 with compatibility across APIs:

   ```python
   if hasattr(models, "MobileNet_V2_Weights"):
       weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
       base = models.mobilenet_v2(weights=weights)
   else:
       base = models.mobilenet_v2(pretrained=pretrained)
   ```

2. Validate that `base.features` exists.

3. Replace the classifier with a single linear layer:

   ```python
   in_features = base.classifier[-1].in_features
   self.classifier = nn.Linear(in_features, num_classes)
   ```

4. Keep `base.features` and `base.avgpool` behavior, but avoid `base.classifier` if it contains dropout.

5. Forward should return `(logits, feature_maps)`:

   ```python
   feature_maps = self.base.features(images)
   pooled = nn.functional.adaptive_avg_pool2d(feature_maps, 1).flatten(1)
   logits = self.classifier(pooled)
   return logits, feature_maps
   ```

6. Expose `.classifier` as the final linear classifier, so the existing CAM code can continue using:

   ```python
   weights = model.classifier.weight[labels]
   cams = torch.einsum("bc,bchw->bhw", weights, feats)
   ```

Expected shape for 224x224 inputs:

```text
feature_maps:      B x 1280 x 7 x 7
classifier.weight: C x 1280
```

This should be tested explicitly.

### Step 2: Wire MobileNetV2 Into Guided Waterbirds

Edit:

- parent-level `../run_guided_waterbird.py`
- parent-level `../run_guided_waterbird_sweep.py`

Changes:

1. Add `mobilenet_v2` to the valid `model_name` choices.
2. In `make_cam_model`, route `mobilenet_v2` to `GALS.models.cam_backbones.MobileNetV2CAM`.
3. Keep the rest of the training loop unchanged.

The current guided training loop is already backbone-agnostic enough once the model returns `(outputs, feats)` and exposes `.classifier.weight`.

Do not change the KL loss yet. The first MobileNetV2 implementation should test the same R4RR objective.

### Step 3: Wire MobileNetV2 Into Guided RedMeat

Edit:

- `RedMeat_Runs/run_guided_redmeat.py`
- `RedMeat_Runs/run_guided_redmeat_sweep.py`

Changes:

1. Add `mobilenet_v2` to `--model-name` choices.
2. In `make_redmeat_cam_model`, route `mobilenet_v2` through `base.make_cam_model`.
3. Ensure `tune_mode=full` remains the default for the MobileNetV2 guided sweeps.
4. Keep RedMeat data/mask handling unchanged.

The RedMeat guided runner delegates most R4RR logic to the shared Waterbirds guided code, so this should be a small change once the shared model factory supports MobileNetV2.

### Step 4: Wire MobileNetV2 Into Vanilla Runners

Edit:

- `run_vanilla_waterbird_clip.py`
- `run_vanilla_waterbird_clip_sweep.py`
- `RedMeat_Runs/run_vanilla_redmeat.py`
- `RedMeat_Runs/run_vanilla_redmeat_sweep.py`

Add a MobileNetV2 constructor that mirrors the guided head:

```python
class MobileNetV2LinearHead(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        ...
    def forward(self, images):
        feats = self.features(images)
        pooled = adaptive_avg_pool2d(feats, 1).flatten(1)
        return self.classifier(pooled)
```

or directly modify torchvision MobileNetV2 so the active classifier is only a linear layer.

The key is that vanilla and guided use the same effective student architecture.

### Step 5: Add Guided MobileNetV2 Sweep Scripts

Create new scripts rather than modifying the existing MobileNetV3 scripts:

- `run_guided_waterbirds95_mobilenetv2_sweep.sh`
- `run_guided_waterbirds100_mobilenetv2_sweep.sh`
- `RedMeat_Runs/run_guided_redmeat_mobilenetv2_sweep.sh`

Shared guided sweep space:

| Hyperparameter | Range | Sampling |
|---|---:|---|
| `attention_epoch` | `0` to `num_epochs - 1` | integer |
| `kl_lambda` | `[1, 500]` | log |
| `base_lr` | `[1e-5, 5e-2]` | log |
| `classifier_lr` | `[1e-5, 5e-2]` | log |
| `lr2_mult` | `[1e-1, 3.0]` | log |

Fixed:

- `kl_incr=0.0`
- `model_name=mobilenet_v2`
- `pretrained=True`
- `tune_mode=full`

Each script should:

1. Run 50 Optuna trials.
2. Write a sweep CSV.
3. Pick best by validation objective.
4. Rerun best hyperparameters over seeds 0-4.
5. Write a best5 CSV.

### Step 6: Add Vanilla MobileNetV2 Sweep Scripts

Create:

- `run_waterbirds95_vanilla_mobilenetv2_sweep.sh`
- `run_waterbirds100_vanilla_mobilenetv2_sweep.sh`
- `RedMeat_Runs/run_redmeat_vanilla_mobilenetv2_sweep_optuna.sh`

Shared vanilla sweep space:

| Hyperparameter | Range | Sampling |
|---|---:|---|
| `base_lr` | `[1e-5, 5e-2]` | log |
| `classifier_lr` | `[1e-5, 5e-2]` | log |
| `momentum` | `[0.85, 0.95]` | uniform |

Fixed:

- `model=mobilenet_v2`
- `pretrained=True`
- `tune_mode=full`
- `weight_decay=1e-5`
- `nesterov=False`

Each script should run 50 Optuna trials and then seeds 0-4, matching the MobileNetV3 vanilla scripts.

### Step 7: Add Minimal Smoke Tests

Before submitting full sweeps, run local or short GPU checks:

1. Instantiate MobileNetV2CAM with `pretrained=False`.
2. Forward a dummy batch:

   ```text
   logits: B x num_classes
   feats:  B x 1280 x 7 x 7
   classifier.weight: num_classes x 1280
   ```

3. Compute one CAM:

   ```python
   cams = torch.einsum("bc,bchw->bhw", weight[labels], feats)
   ```

4. Verify CAM is finite and has nonzero spatial variance for at least some samples.
5. Run a one-epoch guided command with `SAVE_CHECKPOINTS=0` and `GUIDED_NUM_WORKERS=0`.

Suggested one-epoch guided smoke run:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate gals_a100

SAVE_CHECKPOINTS=0 GUIDED_NUM_WORKERS=0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python -u run_guided_waterbird.py \
  /home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2 \
  /home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap \
  --model-name mobilenet_v2 \
  --attention_epoch 1 \
  --kl_lambda 10 \
  --kl_increment 0 \
  --base_lr 1e-4 \
  --classifier_lr 1e-3 \
  --lr2_mult 1.0 \
  --seed 0
```

If the full runner does not expose a quick epoch override, add one before running the smoke test.

### Step 8: Add CAM Diagnostics

Because MobileNet runs have already underperformed with V3, add lightweight diagnostics before launching large sweeps.

Log once per epoch or every few epochs:

- CAM min/mean/max.
- CAM spatial standard deviation before min-max normalization.
- fraction of samples with near-flat CAMs, e.g. `max - min < 1e-6`.
- attention loss mean.

This can be optional behind an environment variable:

```bash
GUIDED_CAM_DIAGNOSTICS=1
```

Do not make diagnostics part of the scientific method; use them as a safety check for broken CAM behavior.

### Step 9: Submit Full Sweeps

Expected six main scripts:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS

sbatch run_guided_waterbirds95_mobilenetv2_sweep.sh
sbatch run_guided_waterbirds100_mobilenetv2_sweep.sh
sbatch RedMeat_Runs/run_guided_redmeat_mobilenetv2_sweep.sh

sbatch run_waterbirds95_vanilla_mobilenetv2_sweep.sh
sbatch run_waterbirds100_vanilla_mobilenetv2_sweep.sh
sbatch RedMeat_Runs/run_redmeat_vanilla_mobilenetv2_sweep_optuna.sh
```

This is the same six-run structure as the current MobileNetV3 setup.

## Potential Failure Modes To Watch

### Flat CAMs

If many CAMs are flat, the KL term will not give useful spatial guidance. This is the first thing to inspect if guided MobileNetV2 underperforms.

### Guidance Too Strong For A Smaller Model

MobileNetV2 has much less capacity than ResNet-50. Even if CAMs are mechanically correct, the same `kl_lambda` range may over-constrain it. If first sweeps look unstable, a faithful follow-up is to keep the method unchanged but narrow/lower the KL range, for example:

```text
kl_lambda in [0.1, 200]
```

This should be treated as a second diagnostic sweep, not the first protocol.

### Attention Starts Too Early

If early MobileNetV2 representations are weaker than ResNet-50, early guidance may harm learning. The existing sweep over `attention_epoch` should catch this, but if many trials collapse, consider restricting:

```text
attention_epoch >= 10 or 20
```

Again, do this only after the first fair sweep if needed.

### Dropout Mismatch

Avoid active dropout in the guided MobileNetV2 head. If dropout remains in the classifier path, logits and CAMs are no longer perfectly aligned during training.

### Hidden Path Mismatch

Waterbirds guided scripts currently run from the parent `Waterbird_Runs` root, while many helper files live under `GALS/`. Keep `PYTHONPATH` and imports consistent. The existing V3 setup already depends on importing `GALS.models.cam_backbones` from the parent runner, so V2 should follow the same pattern.

## Acceptance Criteria

The implementation is ready for full sweeps when:

1. `python -m py_compile` passes for all modified Python files.
2. `bash -n` passes for all six new sbatch scripts.
3. Dummy MobileNetV2CAM forward pass confirms:

   ```text
   feats.shape[1] == classifier.weight.shape[1]
   ```

4. One short guided smoke run completes at least one epoch.
5. The smoke log prints finite loss, finite attention loss, and non-flat CAM diagnostics if diagnostics are enabled.

## Paper Framing

A concise explanation for the paper or appendix:

MobileNetV2 is used as the lightweight CNN student because its architecture preserves the same CAM-compatible structure as ResNet-50: a spatial convolutional representation followed by global average pooling and a linear classifier. This lets R4RR use the same classifier-weighted CAM construction without changing the explanation mechanism. MobileNetV3, although newer, includes a nonlinear classifier head that weakens this direct CAM interpretation, so MobileNetV2 is the cleaner controlled test of whether R4RR transfers from a large CNN to a compact mobile CNN.

