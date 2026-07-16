# MobileNetV3 R4RR Implementation Plan

## Goal

Add MobileNetV3-Large as an alternate student backbone for the image-scale R4RR/LearnToLook experiments while keeping the experimental protocol as close as possible to the current ResNet-50 setup.

The first implementation target is:

- Guided KL / R4RR MobileNetV3-Large for Waterbirds-95, Waterbirds-100, and RedMeat.
- Vanilla MobileNetV3-Large for Waterbirds-95, Waterbirds-100, and RedMeat.
- 50-trial Optuna sweeps for each dataset/method pair.
- Best-config reruns over seeds 0-4, matching the existing reporting pattern.

DecoyMNIST should remain a standalone LeNet experiment for the main comparison. A MobileNet DecoyMNIST variant can be added later as a separate stress test, but it should not be mixed into the main architecture-swap story because it changes both the model family and the input regime.

## Rationale

Waterbirds and RedMeat already use ImageNet-scale RGB student models, so replacing ResNet-50 with MobileNetV3-Large is a clean architecture swap. DecoyMNIST is different: the existing setup is intentionally LeNet-style on small synthetic digits. Resizing DecoyMNIST to 224x224 RGB and using ImageNet-pretrained MobileNet would test a different representation regime rather than simply testing R4RR under another CNN backbone.

## Fixed Choices

- Backbone: `torchvision.models.mobilenet_v3_large`.
- Initialization: ImageNet pretrained weights when available.
- Input preprocessing: keep the current ImageNet-style preprocessing used by ResNet-50 runs.
- Waterbirds epochs: 200.
- RedMeat epochs: 150.
- Optimizer family: keep the same optimizer behavior used by the existing guided/vanilla runners.
- Validation objective: match existing runs.
  - Waterbirds: validation balanced group accuracy.
  - RedMeat: validation balanced class accuracy, kept under the existing `balanced_group` field name for consistency.
- Seeds for final reruns: 0, 1, 2, 3, 4.

## Key Architecture Work

### 1. Add A Backbone-Agnostic CAM Wrapper

Create a small wrapper that exposes the common interface needed by guided KL:

```python
logits = model(images)
feature_maps = model.feature_maps
classifier_weights = model.classifier_weight()
base_parameters = model.base_parameters()
classifier_parameters = model.classifier_parameters()
```

For ResNet-50, this maps to:

- final feature maps: `layer4`
- classifier: `fc`

For MobileNetV3-Large, this maps to:

- final feature maps: `features[-1]`
- classifier: final linear layer in `classifier`

This avoids scattering architecture-specific checks through the guided loss code.

### 2. Implement `MobileNetV3CAM`

The MobileNet wrapper should:

1. Build `torchvision.models.mobilenet_v3_large(pretrained=True)` or the equivalent weights API for the installed torchvision version.
2. Replace the final classifier layer with `nn.Linear(in_features, num_classes)`.
3. Run `features(images)` to get spatial features.
4. Pool features with adaptive average pooling.
5. Run the classifier head.
6. Store the final spatial tensor as `self.feature_maps`.
7. Expose classifier weights for CAM construction.

Expected final feature map size for 224x224 inputs should be small spatially, likely 7x7, which is compatible with the existing low-resolution CAM alignment pattern.

### 3. Make Guided KL Use The Wrapper Interface

The current guided runners assume ResNet-style attributes such as `layer4` and `fc`. Replace those assumptions with helper functions:

- `get_cam_features(model)`
- `get_classifier_weight(model)`
- `iter_base_params(model)`
- `iter_classifier_params(model)`

The guided KL code should not need to know whether the student is ResNet-50 or MobileNetV3-Large.

### 4. Add `model_name="mobilenet_v3_large"`

Extend the relevant model factories:

- Waterbirds guided model factory.
- RedMeat guided model factory.
- Waterbirds vanilla model factory.
- RedMeat vanilla model factory.

The new `model_name` option should support at least:

- `resnet50`
- `mobilenet_v3_large`

Any existing `clip_rn50` support should remain untouched.

### 5. Replace `layer4_head` With A General Fine-Tuning Mode

For ResNet-50, `layer4_head` means train only the last ResNet block and the classifier.

MobileNet does not have `layer4`, so use a more general mode:

- `full`: train all parameters.
- `classifier_only`: train only the final classifier.
- `last_blocks_head`: train the final MobileNet feature blocks plus classifier.

If preserving CLI compatibility matters, map old `layer4_head` to the architecture-specific equivalent:

- ResNet-50: `layer4 + fc`.
- MobileNetV3-Large: last N blocks of `features` plus classifier.

For the first fair R4RR MobileNet runs, use `full` unless there is a strong reason to mirror a partial fine-tuning setup.

## Guided KL MobileNet Sweeps

### Shared Sweep Space

Use the same R4RR-specific search space as the current guided KL sweeps:

| Hyperparameter | Range / Values | Sampling |
|---|---:|---|
| `attention_epoch` | `0` to `num_epochs - 1` | integer |
| `kl_lambda` | `[1, 500]` | log |
| `base_lr` | `[1e-5, 5e-2]` | log |
| `classifier_lr` | `[1e-5, 5e-2]` | log |
| `lr2_mult` | `[1e-1, 3.0]` | log |

Keep `kl_increment=0.0` unless deliberately testing a different schedule.

### Waterbirds-95 Guided MobileNet

Create:

- `run_guided_waterbirds95_mobilenet_sweep.sh`

Behavior:

- Dataset: `waterbird_complete95_forest2water2`.
- Teacher maps: same Waterbirds-95 R4RR teacher-map root used by the corresponding ResNet-50 guided run.
- Trials: 50.
- Epochs: 200.
- Objective: validation balanced group accuracy.
- Post-rerun best config over seeds 0-4.
- Output CSVs to `logsWaterbird`.

### Waterbirds-100 Guided MobileNet

Create:

- `run_guided_waterbirds100_mobilenet_sweep.sh`

Behavior:

- Dataset: `waterbird_1.0_forest2water2`.
- Teacher maps: same Waterbirds-100 R4RR teacher-map root used by the corresponding ResNet-50 guided run.
- Trials: 50.
- Epochs: 200.
- Objective: validation balanced group accuracy.
- Post-rerun best config over seeds 0-4.
- Output CSVs to `logsWaterbird`.

### RedMeat Guided MobileNet

Create:

- `RedMeat_Runs/run_guided_redmeat_mobilenet_sweep.sh`

Behavior:

- Dataset: `food-101-redmeat`.
- Teacher maps: same RedMeat R4RR teacher-map root used by the corresponding ResNet-50 guided run.
- Trials: 50.
- Epochs: 150.
- Objective: validation balanced class accuracy using the existing `best_balanced_val_acc` / `balanced_group` reporting convention.
- Post-rerun best config over seeds 0-4.
- Output CSVs to `logsRedMeat`.

## Vanilla MobileNet Sweeps

Vanilla means ordinary supervised MobileNetV3-Large training with no teacher-map KL loss.

### Shared Vanilla Sweep Space

Mirror the current vanilla CNN sweeps:

| Hyperparameter | Range / Values | Sampling |
|---|---:|---|
| `base_lr` | `[1e-5, 5e-2]` | log |
| `classifier_lr` | `[1e-5, 5e-2]` | log |
| `momentum` | `[0.85, 0.95]` | uniform |

Fixed unless explicitly overridden:

- `weight_decay=1e-5`
- `nesterov=False`
- `batch_size=96`
- ImageNet normalization

### Waterbirds-95 Vanilla MobileNet

Create:

- `run_waterbirds95_vanilla_mobilenet_sweep.sh`

Behavior:

- Trials: 50.
- Epochs: 200.
- Objective: validation balanced group accuracy.
- Best-config rerun seeds 0-4.

### Waterbirds-100 Vanilla MobileNet

Create:

- `run_waterbirds100_vanilla_mobilenet_sweep.sh`

Behavior:

- Trials: 50.
- Epochs: 200.
- Objective: validation balanced group accuracy.
- Best-config rerun seeds 0-4.

### RedMeat Vanilla MobileNet

Create:

- `RedMeat_Runs/run_redmeat_vanilla_mobilenet_sweep_optuna.sh`

Behavior:

- Trials: 50.
- Epochs: 150.
- Objective: validation balanced class accuracy, reported as balanced group for consistency with existing RedMeat tables.
- Best-config rerun seeds 0-4.

## DecoyMNIST Handling

Do not include DecoyMNIST in the first MobileNet implementation.

Keep DecoyMNIST as:

- LeNet-style CNN.
- 19 epochs.
- Adam, `lr=1e-3`, `weight_decay=1e-4`.
- Existing guided/vanilla/GALS/ABN/upweight/AFR comparison setup.

If MobileNet DecoyMNIST is added later, put it behind separate scripts and clearly label it as a stress test:

- Convert grayscale PNGs to RGB.
- Resize to 224x224.
- Use ImageNet normalization.
- Do not compare it directly as a drop-in replacement for the LeNet DecoyMNIST table without a note.

## Implementation Order

### Step 1: Introduce The Wrapper

Add a MobileNet-capable CAM wrapper in a shared helper file, for example:

- `mobilenet_cam.py`, or
- a new section in the existing guided model utility file if one is already shared.

Keep this minimal and testable.

Deliverable:

- `MobileNetV3CAM(num_classes, pretrained=True)`
- unit/smoke check that a forward pass returns logits and captures feature maps.

### Step 2: Add MobileNet To Vanilla Training

Modify the vanilla model factories first because they are simpler than guided KL.

Deliverable:

- Waterbirds vanilla MobileNet can run one epoch.
- RedMeat vanilla MobileNet can run one epoch.
- Classifier/base parameter groups are correct.

### Step 3: Add MobileNet To Guided KL Training

Modify the guided runners to consume the wrapper interface.

Deliverable:

- Waterbirds guided MobileNet can run one epoch with teacher maps.
- RedMeat guided MobileNet can run one epoch with teacher maps.
- CAM maps are nonzero and have expected shape after interpolation.

### Step 4: Add Sweep Scripts

Add the six scripts:

- `run_guided_waterbirds95_mobilenet_sweep.sh`
- `run_guided_waterbirds100_mobilenet_sweep.sh`
- `RedMeat_Runs/run_guided_redmeat_mobilenet_sweep.sh`
- `run_waterbirds95_vanilla_mobilenet_sweep.sh`
- `run_waterbirds100_vanilla_mobilenet_sweep.sh`
- `RedMeat_Runs/run_redmeat_vanilla_mobilenet_sweep_optuna.sh`

Each script should:

- default to 50 Optuna trials;
- write sweep CSVs;
- rerun best hyperparameters over seeds 0-4;
- print enough per-trial metric information to debug without dumping full training logs;
- use the existing RC data paths and log directories.

### Step 5: Smoke Test Before Full Submission

Run small jobs before launching the 50-trial sweeps:

- 1 trial.
- 1 epoch.
- `SAVE_CHECKPOINTS=0`.
- Verify CSV row creation.
- Verify no model-loading or CAM-shape errors.

### Step 6: Submit Full Sweeps

Submit guided and vanilla MobileNet sweeps separately so failures are isolated.

Recommended order:

1. Waterbirds-95 vanilla MobileNet.
2. Waterbirds-100 vanilla MobileNet.
3. RedMeat vanilla MobileNet.
4. Waterbirds-95 guided MobileNet.
5. Waterbirds-100 guided MobileNet.
6. RedMeat guided MobileNet.

The vanilla jobs validate the model/data plumbing before the guided KL jobs add teacher-map complexity.

## Validation Checklist

Before calling the implementation complete:

- MobileNet forward pass works on one batch for each dataset.
- Feature maps are captured for MobileNet.
- CAM computation produces nonconstant maps.
- Guided KL loss is finite.
- Vanilla and guided CSVs contain the same core metric fields as ResNet runs.
- Waterbirds objective is validation balanced group accuracy.
- RedMeat objective is validation balanced class accuracy.
- 50-trial defaults are set in sbatch scripts.
- Post-sweep seed reruns use seeds 0-4.
- Checkpoint names include `mobilenet_v3_large`.

## Expected Risk Points

1. **CAM feature shape mismatch**
   MobileNet feature maps must be normalized/interpolated exactly like ResNet CAM maps before KL.

2. **Classifier weight access**
   MobileNet classifier is nested under `model.classifier`, not `model.fc`.

3. **Parameter grouping**
   Base/classifier LR separation must not accidentally place all MobileNet parameters into the base group.

4. **Old `layer4_head` assumptions**
   Any partial fine-tuning mode referencing `layer4` must be generalized or avoided for MobileNet.

5. **Checkpoint loading**
   Existing ResNet checkpoint loaders and saliency scripts will not load MobileNet checkpoints unless explicitly updated.

## Non-Goals For First Pass

- Do not implement MobileNet DecoyMNIST as part of the main table.
- Do not update every saliency/pointing-game script unless needed for the first MobileNet metrics.
- Do not tune new MobileNet-only hyperparameters beyond the agreed existing sweep ranges.
- Do not change teacher maps, data splits, or validation objectives.
