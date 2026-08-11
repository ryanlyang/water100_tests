# DecoyMNIST Five-Seed Pointing Game

This workflow trains the seven learned DecoyMNIST baselines with their locked
LeNet-style settings and evaluates each validation-selected checkpoint over
seeds 0--4:

- vanilla
- ElRep
- upweight
- ABN
- GALS
- AFR
- R4RR

The submission layout is seven jobs, one per method. Each job trains and
evaluates five seeds sequentially. Completed seed checkpoints and Pointing
Game CSVs are validated and reused when a job is resubmitted.

## Evaluation protocol

- Split: full DecoyMNIST test set (10,000 images) by default.
- Explanation: standard Grad-CAM at the LeNet `conv2` layer.
- Target: ground-truth class by default.
- Ground-truth region: foreground pixels from the corresponding clean
  torchvision MNIST image.
- Primary score: resolution-matched Pointing Game. The clean `28x28` digit
  mask is adaptively max-pooled to the native `8x8` `conv2` Grad-CAM grid. A
  native CAM peak is a hit when its cell overlaps any clean digit pixel.
- Diagnostic score: conventional pixel-level Pointing Game after bilinearly
  upsampling the native Grad-CAM to `28x28`.
- Decoy exclusion: exported PNG filenames preserve the original MNIST index,
  so masks are recovered from the clean source rather than thresholded from
  the decoy image. The synthetic 5x5 corner patch is never part of the mask.
- Empty explanation: an all-zero Grad-CAM is a miss.
- Reporting: native-grid overall, macro-digit, worst-digit, per-digit, and
  random-hit rates, followed by population mean and standard deviation across
  seeds. Pixel-level scores are retained under `pg_*` fields as diagnostics;
  primary resolution-matched fields use the `pg_native_*` prefix.

CLIP-ZS and CLIP-LR are excluded from this seven-job workflow. They do not
require stochastic CNN retraining and need a separate explanation protocol
because their visual backbones do not expose the LeNet `conv2` layer.

## Research-compute paths

Defaults in the worker are:

```text
repo:        /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
PNG data:    /home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png
clean MNIST: /home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data
GALS maps:   <PNG data>/clip_rn50_attention_gradcam
R4RR maps:   /home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist/val/prediction_cmap
outputs:     /home/ryreu/guided_cnn/logsMNIST/pointing_game_5seed_gradcam
```

All paths can be overridden through the corresponding environment variables:
`PNG_ROOT`, `MNIST_ROOT`, `GALS_MAPS`, `R4RR_MAPS`, `RUN_ROOT`, and `LOG_DIR`.
Only GALS requires `GALS_MAPS`; Vanilla trains directly from the DecoyMNIST PNGs.

## Submit

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash submit_decoymnist_pointing_game_5seed_all.sh
```

The jobs appear as:

```text
pg5_decoy_vanilla
pg5_decoy_elrep
pg5_decoy_upweight
pg5_decoy_abn
pg5_decoy_gals
pg5_decoy_afr
pg5_decoy_r4rr
```

## Resume

Run the same submission command again. The output root is stable, so each job
skips valid seed manifests and valid current-protocol Pointing Game summaries.
Summaries from the older pixel-only protocol are automatically reevaluated
from their existing checkpoints; no retraining is needed. A seed whose
training was interrupted before checkpoint creation is retrained from the
start; DecoyMNIST training is only 19 LeNet epochs.

To resume one method only:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
sbatch --job-name=pg5_decoy_r4rr \
  --export=ALL,METHOD=r4rr \
  run_decoymnist_pointing_game_5seed_method.sh
```

## Smoke test

Evaluate a small deterministic subset while checking the complete training
and checkpoint path:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
MAX_SAMPLES=100 SEEDS_CSV=0 DRY_RUN=1 \
  bash submit_decoymnist_pointing_game_5seed_all.sh
```

Remove `DRY_RUN=1` to submit that smoke configuration. Use a separate
`RUN_ROOT` for smoke results so they cannot be mistaken for full results.

## Combined summary

After all seven methods finish:

```bash
python summarize_decoymnist_pointing_game_5seed.py \
  --run-root /home/ryreu/guided_cnn/logsMNIST/pointing_game_5seed_gradcam \
  --seeds 0,1,2,3,4
```

The combined table is written to:

```text
pointing_game_all_methods_5seed_summary.csv
```

## R4RR epochwise diagnostic

The separate epochwise runner trains one R4RR seed, preserves checkpoints for
epochs 1--19, and evaluates every checkpoint with the same test-set protocol.
Test Pointing Game measurements are diagnostic only and never select a
checkpoint. The resulting table marks the epoch selected by validation
classification accuracy.

```bash
sbatch run_decoymnist_r4rr_epochwise_pointing.sh
```

The default stable output directory is:

```text
/home/ryreu/guided_cnn/logsMNIST/decoy_r4rr_epochwise_pointing_seed0
```

The main result is `epochwise_pointing_game.csv`. Resubmission reuses a valid
training trajectory and completed epoch evaluations. Set `SEED`, `RUN_DIR`, or
`FORCE_RETRAIN=1` through `sbatch --export` when a different diagnostic is
required.
