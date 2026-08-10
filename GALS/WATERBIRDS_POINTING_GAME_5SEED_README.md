# Five-seed Waterbirds Pointing Game

This workflow retrains the validation-selected fixed-hyperparameter models for
seeds 0--4 and evaluates their localization with the same CAM-style Pointing
Game protocol used by the existing Waterbirds results.

Methods: `vanilla`, `elrep`, `upweight`, `abn`, `gals`, `afr`, and `r4rr`.
Datasets: Waterbirds-95 and Waterbirds-100. The submitter creates 14 jobs, one
per method/dataset pair; each job processes its five seeds sequentially.

## Default protocol

- split: validation
- explanation target: ground-truth class
- explanation: class-specific CAM for GAP+linear ResNet models; ABN's learned
  attention map for ABN
- masks: the same dataset-specific mask roots used by the previous one-seed
  Pointing Game runs (`MASK_PROTOCOL=legacy`)
- optimized hyperparameters: the Waterbirds YAML files under
  `RightForTheRightRegions/configs/`
- summary standard deviation: population standard deviation (`ddof=0`),
  matching the other five-seed runners

To evaluate against the original CUB segmentation masks instead, submit with
`MASK_PROTOCOL=cub`. Do not mix the two protocols in one reported table.

## Reusing existing checkpoints

Copy `waterbirds_pointing_game_existing_checkpoints.example.csv` and add any
existing validation-selected checkpoints. AFR rows require both its stage-2
head (`checkpoint`) and stage-1 backbone (`stage1_checkpoint`). Then export the
CSV path before submission:

```bash
export EXISTING_CHECKPOINT_CSV=/absolute/path/to/existing_checkpoints.csv
```

Rows absent from the CSV are trained normally. Completed training manifests
and Pointing Game CSVs are reused automatically, so resubmitting the same
workflow resumes missing work.

## Submit

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash submit_waterbirds_pointing_game_5seed_all.sh
```

Stable outputs are written under
`/home/ryreu/guided_cnn/logsWaterbird/pointing_game_5seed_cam/` by default.
After all jobs finish, combine all method summaries with:

```bash
python summarize_waterbirds_pointing_game_5seed.py \
  --run-root /home/ryreu/guided_cnn/logsWaterbird/pointing_game_5seed_cam \
  --seeds 0,1,2,3,4
```
