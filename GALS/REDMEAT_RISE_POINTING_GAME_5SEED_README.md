# Five-seed RedMeat RISE Pointing Game

This workflow trains the validation-selected RedMeat baselines and evaluates
all methods on the reviewed 1,250-image test mask package with one shared,
deterministic RISE mask bank.

The submitter creates nine jobs:

- five-seed jobs: `vanilla`, `elrep`, `upweight`, `abn`, `gals`, `afr`, `r4rr`
- deterministic one-run jobs: `clip_lr`, `clip_zs`

Each trained method uses its finalized RedMeat hyperparameters from
`RightForTheRightRegions/configs/redmeat_optimized_hparams.yaml`. Deep models
run for 150 epochs, except AFR, which retains its native two-stage training
protocol. Checkpoints are selected by the same validation criteria used in the
main experiments.

## Protocol

- split: RedMeat test, 250 images per class
- masks: reviewed COCO polygons converted to union foreground masks
- target: ground-truth class (`TARGET_MODE=label`)
- explainer: RISE, with a shared bank of 2,000 masks, 8x8 grid, `p1=0.1`
- primary metric: whether the 224x224 RISE argmax lies in the meat mask
- reporting: overall, macro-class, worst-class, correct-only, random-chance,
  classification, saliency-mass, and per-class metrics
- uncertainty: population standard deviation over seeds (`ddof=0`)

## Prerequisites

Validate the transferred mask package once:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
python3 RedMeat_Runs/validate_redmeat_pointing_masks.py \
  --package-root /home/ryreu/guided_cnn/Food101/data/food-101-redmeat/redmeat_pointing_masks \
  --data-root /home/ryreu/guided_cnn/Food101/data/food-101-redmeat
```

The ABN job requires `weights/resnet50_abn_imagenet.pth.tar`. GALS requires
the RedMeat RN50 attention maps at
`food-101-redmeat/clip_rn50_attention_gradcam`; R4RR requires the OpenCLIP
LAION+DINO teacher maps configured in the worker.

## Submit

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash RedMeat_Runs/submit_redmeat_rise_pointing_game_all.sh
```

Preview all nine submissions without queueing them:

```bash
DRY_RUN=1 bash RedMeat_Runs/submit_redmeat_rise_pointing_game_all.sh
```

Outputs are stable under
`/home/ryreu/guided_cnn/logsRedMeat/pointing_game_5seed_rise`. A repeated
submission skips each valid training manifest and evaluation CSV, allowing an
interrupted job to continue at the first missing seed.

Existing checkpoints can be imported with `EXISTING_CHECKPOINT_CSV`. The CSV
columns are `dataset,method,seed,checkpoint,stage1_checkpoint`; AFR requires
both checkpoint columns.

After every job completes, create the strict combined summary:

```bash
python RedMeat_Runs/summarize_redmeat_rise_pointing_game.py \
  --run-root /home/ryreu/guided_cnn/logsRedMeat/pointing_game_5seed_rise \
  --seeds 0,1,2,3,4 \
  --clip-seeds 0
```

For an in-progress report, add `--allow-partial`. The final paper summary
should not use partial mode.
