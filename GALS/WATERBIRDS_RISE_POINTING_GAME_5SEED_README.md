# Waterbirds Five-Seed RISE Pointing Game

This workflow evaluates the already-trained five-seed Waterbirds checkpoints
with RISE. It does not retrain or modify any model.

## Fixed protocol

- Datasets: Waterbirds-95 and Waterbirds-100
- Methods: Vanilla, ElRep, Upweight, ABN, GALS, AFR, and R4RR
- Seeds: `0,1,2,3,4`
- Evaluation split: test
- Saliency target: ground-truth class
- Foreground masks: CUB-200-2011 bird segmentations
- RISE masks: `N=2000`, grid size `8`, Bernoulli probability `0.1`, seed `0`
- Metric: the RISE argmax is a hit when it lies inside the bird mask

Every method is explained through its final classifier score. For binary
single-logit GALS, Upweight, and ABN checkpoints, the score adapter uses
`[1-sigmoid(z), sigmoid(z)]`. Other checkpoints use softmax probabilities.
ABN therefore uses RISE over its classifier like every other method; its native
attention branch is not substituted for RISE.

The RISE mask bank is generated once and shared by every dataset, method, and
seed. Its SHA-256 digest is recorded in every result. Saliency arrays are
aggregated online and are not saved, avoiding hundreds of gigabytes of output.

## Submit

The source checkpoints must first exist under:

```text
/home/ryreu/guided_cnn/logsWaterbird/pointing_game_5seed_cam/
```

The submission script checks all 70 checkpoint manifests before it queues any
jobs. Then run:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash submit_waterbirds_rise_pointing_game_5seed_all.sh
```

If some five-seed training jobs are still running, queue only the complete
dataset-method pairs with:

```bash
ALLOW_PARTIAL=1 bash submit_waterbirds_rise_pointing_game_5seed_all.sh
```

Run the same command again after the remaining training jobs finish. It skips
incomplete pairs, active RISE jobs, and already-completed RISE evaluations, so
rerunning it does not duplicate work.

This submits 14 three-day A100 jobs, one for each dataset-method pair. Each job
evaluates all five seeds sequentially and resumes from valid per-seed result
CSVs. Results are written to:

```text
/home/ryreu/guided_cnn/logsWaterbird/pointing_game_5seed_rise/
```

Inspect submissions without queuing jobs:

```bash
DRY_RUN=1 bash submit_waterbirds_rise_pointing_game_5seed_all.sh
```

For a short end-to-end diagnostic, override the sample and RISE counts:

```bash
MAX_SAMPLES=20 RISE_NUM_MASKS=64 \
RUN_ROOT=/home/ryreu/guided_cnn/logsWaterbird/pointing_game_5seed_rise_smoke \
bash submit_waterbirds_rise_pointing_game_5seed_all.sh
```

Do not mix reduced-mask diagnostics with the full result root.

## Final summary

After all jobs complete:

```bash
python summarize_waterbirds_rise_pointing_game_5seed.py \
  --run-root /home/ryreu/guided_cnn/logsWaterbird/pointing_game_5seed_rise \
  --seeds 0,1,2,3,4
```

The combined CSV reports mean and population standard deviation for overall,
macro-group, worst-group, random-baseline, classification, and saliency-mass
metrics, along with each Waterbirds group's Pointing Game result.
