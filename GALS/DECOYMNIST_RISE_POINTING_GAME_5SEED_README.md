# Five-seed DecoyMNIST RISE Pointing Game

This workflow reevaluates the existing DecoyMNIST checkpoints with the
model-agnostic RISE explainer used by the original GALS Pointing Game. It does
not retrain any models and does not overwrite the previous Grad-CAM results.

## Protocol

- checkpoints: validation-selected seeds 0--4 from
  `pointing_game_5seed_gradcam`
- evaluation split: DecoyMNIST test set
- explanation target: ground-truth class
- explainer: RISE
- RISE masks: 2,000 masks, 8x8 grid, `p1=0.1`, seed 0
- mask bank: one deterministic bank shared by every method and seed
- Pointing Game mask: nonzero foreground of the corresponding clean MNIST digit
- primary metric: peak of the 28x28 RISE map inside the clean-digit mask
- diagnostics: macro/worst-class hit rate, random-hit baseline, classification
  accuracy, zero maps, and saliency mass inside the digit

Methods: `vanilla`, `elrep`, `upweight`, `abn`, `gals`, `afr`, and `r4rr`.

## Full evaluation

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash submit_decoymnist_rise_pointing_game_5seed_all.sh
```

Outputs are written under:

```text
/home/ryreu/guided_cnn/logsMNIST/pointing_game_5seed_rise/
```

Each method job processes its five checkpoints sequentially. Completed seed
summaries are reused on resubmission.

## Small pilot

To check runtime and behavior on the same 1,000-image subset for R4RR and one
comparison model:

```bash
MAX_SAMPLES=1000 METHODS_CSV=r4rr,vanilla \
RUN_ROOT=/home/ryreu/guided_cnn/logsMNIST/pointing_game_5seed_rise_pilot1000 \
bash submit_decoymnist_rise_pointing_game_5seed_all.sh
```

Do not combine pilot and full-test summaries in the same reported table.

## Combined summary

After all seven full jobs finish:

```bash
python summarize_decoymnist_rise_pointing_game_5seed.py \
  --run-root /home/ryreu/guided_cnn/logsMNIST/pointing_game_5seed_rise \
  --seeds 0,1,2,3,4
```
