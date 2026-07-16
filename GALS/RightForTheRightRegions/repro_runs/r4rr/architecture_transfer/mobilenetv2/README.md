# MobileNetV2 Architecture Transfer Runners

This folder contains MobileNetV2 student experiments for R4RR architecture
transfer. The goal is to keep the same teacher maps, KL evidence-alignment
loss, training schedule, and evaluation protocol used by the ResNet-50 R4RR
runners while swapping only the student backbone.

Supported datasets:
- Waterbirds95
- Waterbirds100
- RedMeat

Not included:
- DecoyMNIST, which uses the LeNet-style setup in the main DecoyMNIST runners.
- SLURM launchers.

## Implementation

MobileNetV2 is implemented as a CAM-compatible CNN in `common.py`.

The wrapper uses ImageNet-pretrained torchvision MobileNetV2, removes the
stock dropout classifier, and replaces it with a deterministic
GAP + Linear head:

```text
images -> MobileNetV2.features -> feature_maps -> GAP -> Linear -> logits
```

The R4RR CAM map is then computed by the reused canonical training loop from
the class-specific linear weights and the final MobileNetV2 feature maps, just
as in the ResNet-50 CAM setup.

## File Layout

```text
mobilenetv2/
  common.py
  r4rr/
    waterbirds.py              # single-run guided MobileNetV2 runner
    waterbirds_fixed_seeds.py  # fixed-hparam multi-seed guided runner
    waterbirds_sweep.py        # Optuna sweep for guided MobileNetV2
    redmeat.py                 # single-run guided MobileNetV2 runner
    redmeat_fixed_seeds.py     # fixed-hparam multi-seed guided runner
    redmeat_sweep.py           # Optuna sweep for guided MobileNetV2
  baseline/
    waterbirds_fixed_seeds.py  # CE-only multi-seed MobileNetV2 baseline
    waterbirds_sweep.py        # Optuna sweep for CE-only MobileNetV2
    redmeat_fixed_seeds.py     # CE-only multi-seed MobileNetV2 baseline
    redmeat_sweep.py           # Optuna sweep for CE-only MobileNetV2
```

## Example Commands

Run commands from the repository root.

Waterbirds100 guided MobileNetV2 sweep:

```bash
python repro_runs/r4rr/architecture_transfer/mobilenetv2/r4rr/waterbirds_sweep.py \
  "$PWD/data/waterbird_1.0_forest2water2" \
  "$PWD/WeCLIPPlus/results_waterbirds100_r4rr/val/prediction_cmap" \
  --n-trials 50
```

Waterbirds95 guided MobileNetV2 sweep:

```bash
python repro_runs/r4rr/architecture_transfer/mobilenetv2/r4rr/waterbirds_sweep.py \
  "$PWD/data/waterbird_complete95_forest2water2" \
  "$PWD/WeCLIPPlus/results_waterbirds95_r4rr/val/prediction_cmap" \
  --n-trials 50
```

RedMeat guided MobileNetV2 sweep:

```bash
python repro_runs/r4rr/architecture_transfer/mobilenetv2/r4rr/redmeat_sweep.py \
  "$PWD/data/food-101-redmeat" \
  "$PWD/WeCLIPPlus/results_redmeat_r4rr/val/prediction_cmap" \
  --n-trials 50
```

Waterbirds MobileNetV2 CE baseline sweep:

```bash
python repro_runs/r4rr/architecture_transfer/mobilenetv2/baseline/waterbirds_sweep.py \
  "$PWD/data/waterbird_1.0_forest2water2" \
  --n-trials 50
```

RedMeat MobileNetV2 CE baseline sweep:

```bash
python repro_runs/r4rr/architecture_transfer/mobilenetv2/baseline/redmeat_sweep.py \
  "$PWD/data/food-101-redmeat" \
  --n-trials 50
```

## Search Spaces

Guided R4RR sweeps:
- `attention_epoch`: `[0, num_epochs - 1]`
- `kl_lambda`: `[1, 500]`, log scale
- `base_lr`: `[1e-5, 5e-2]`, log scale
- `classifier_lr`: `[1e-5, 5e-2]`, log scale
- `lr2_mult`: `[0.1, 3.0]`, log scale

CE-only baseline sweeps:
- `base_lr`: `[1e-5, 5e-2]`, log scale
- `classifier_lr`: `[1e-5, 5e-2]`, log scale
- `momentum`: `[0.85, 0.95]`

Fixed values:
- Waterbirds epochs: `200`
- RedMeat epochs: `150`
- batch size: `96`
- image size: `224`
- momentum for guided R4RR: `0.9`
- weight decay: `1e-5`
- ImageNet normalization
