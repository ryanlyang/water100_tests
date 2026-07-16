# ViT Architecture Transfer Runners

This folder contains ViT student experiments for R4RR architecture transfer.

Supported datasets:
- Waterbirds95
- Waterbirds100
- RedMeat

Not included:
- DecoyMNIST
- exploratory attribution variants from the source ViT branch
- SLURM launchers

## File Layout

```text
vit/
  r4rr/
    waterbirds.py              # single-run guided ViT runner
    waterbirds_fixed_seeds.py  # fixed-hparam multi-seed guided runner
    waterbirds_sweep.py        # Optuna sweep for guided ViT
    redmeat.py                 # single-run guided ViT runner
    redmeat_fixed_seeds.py     # fixed-hparam multi-seed guided runner
    redmeat_sweep.py           # Optuna sweep for guided ViT
  baseline/
    waterbirds_fixed_seeds.py  # CE-only multi-seed ViT ERM baseline
    waterbirds_sweep.py        # Optuna sweep for CE-only ViT ERM
    redmeat_fixed_seeds.py     # CE-only multi-seed ViT ERM baseline
    redmeat_sweep.py           # Optuna sweep for CE-only ViT ERM
```

## Usage

Run commands from the repository root.

Waterbirds100 guided ViT, fixed seeds:

```bash
python repro_runs/r4rr/architecture_transfer/vit/r4rr/waterbirds_fixed_seeds.py \
  "$PWD/data/waterbird_1.0_forest2water2" \
  "$PWD/WeCLIPPlus/results_waterbirds100_r4rr/val/prediction_cmap" \
  --seed-start 0 \
  --n-seeds 5
```

Waterbirds95 guided ViT, fixed seeds:

```bash
python repro_runs/r4rr/architecture_transfer/vit/r4rr/waterbirds_fixed_seeds.py \
  "$PWD/data/waterbird_complete95_forest2water2" \
  "$PWD/WeCLIPPlus/results_waterbirds95_r4rr/val/prediction_cmap" \
  --seed-start 0 \
  --n-seeds 5
```

Waterbirds ViT ERM baseline, fixed seeds:

```bash
python repro_runs/r4rr/architecture_transfer/vit/baseline/waterbirds_fixed_seeds.py \
  "$PWD/data/waterbird_1.0_forest2water2" \
  "$PWD/WeCLIPPlus/results_waterbirds100_r4rr/val/prediction_cmap" \
  --seed-start 0 \
  --n-seeds 5
```

The second positional argument is unused by the Waterbirds baseline, but is kept for CLI parity with guided runs.

RedMeat guided ViT, fixed seeds:

```bash
python repro_runs/r4rr/architecture_transfer/vit/r4rr/redmeat_fixed_seeds.py \
  "$PWD/data/food-101-redmeat" \
  "$PWD/WeCLIPPlus/results_redmeat_r4rr/val/prediction_cmap" \
  --seed-start 0 \
  --n-seeds 5
```

RedMeat ViT ERM baseline, fixed seeds:

```bash
python repro_runs/r4rr/architecture_transfer/vit/baseline/redmeat_fixed_seeds.py \
  "$PWD/data/food-101-redmeat" \
  --seed-start 0 \
  --n-seeds 5
```

## Sweeps

Waterbirds guided ViT sweep:

```bash
python repro_runs/r4rr/architecture_transfer/vit/r4rr/waterbirds_sweep.py \
  "$PWD/data/waterbird_1.0_forest2water2" \
  "$PWD/WeCLIPPlus/results_waterbirds100_r4rr/val/prediction_cmap" \
  --n-trials 50
```

Waterbirds ViT ERM baseline sweep:

```bash
python repro_runs/r4rr/architecture_transfer/vit/baseline/waterbirds_sweep.py \
  "$PWD/data/waterbird_1.0_forest2water2" \
  "$PWD/WeCLIPPlus/results_waterbirds100_r4rr/val/prediction_cmap" \
  --n-trials 50
```

RedMeat guided ViT sweep:

```bash
python repro_runs/r4rr/architecture_transfer/vit/r4rr/redmeat_sweep.py \
  "$PWD/data/food-101-redmeat" \
  "$PWD/WeCLIPPlus/results_redmeat_r4rr/val/prediction_cmap" \
  --n-trials 50
```

RedMeat ViT ERM baseline sweep:

```bash
python repro_runs/r4rr/architecture_transfer/vit/baseline/redmeat_sweep.py \
  "$PWD/data/food-101-redmeat" \
  --n-trials 50
```

## Notes

- Base ViT means CE-only ViT ERM.
- Guided ViT means the R4RR LGM-style ViT setup using teacher maps and KL alignment.
- Common flags accept both dash and underscore forms, e.g. `--base-lr` and `--base_lr`.
- The Waterbirds guided runner uses ViT-B/16 with 640px input and interpolated ImageNet positional embeddings.
