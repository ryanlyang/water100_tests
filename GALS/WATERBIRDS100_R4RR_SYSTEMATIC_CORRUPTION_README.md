# Waterbirds-100 R4RR Systematic Teacher Corruption

This implements the Waterbirds-100 portion of the Round 2 systematic
teacher-corruption study. Because training has perfect class-background
correlation, the systematic unit is an entire bird class rather than one of
the four evaluation groups.

## Locked protocol

- Systematic conditions: all Landbird maps and all Waterbird maps
- One exact count-matched random control for each class
- Training counts: `3684` Landbirds and `1111` Waterbirds
- Training groups: `(3684, 0, 0, 1111)` in canonical group order
- Training seeds: `0,1,2,3,4`; corruption-selection seed: `0`
- The selected subset is fixed and shared across all five training seeds
- Validation and test splits are unchanged and never load teacher maps
- Teacher operation: `1 - M`, nonnegative clamp, then sum normalization
- Student: ImageNet-pretrained ResNet-50 with full fine-tuning and CAM evidence
- Epochs: `200`; batch size: `96`
- Optimizer: SGD, momentum `0.9`, weight decay `1e-5`, no scheduler
- R4RR: `attention_epoch=73`, `kl_lambda=495.61`, `kl_increment=0`
- Learning rates: base `5.72e-5`, classifier `3.57e-3`, phase-2 multiplier `0.123`
- Checkpoint selection: best validation balanced class accuracy after alignment begins
- No hyperparameter retuning

Teacher-map files are never overwritten. Inversion is applied lazily according
to a persisted condition manifest. Each manifest records exact split indices,
stable sample paths, class/group composition, dataset fingerprint, and
checksums. Resubmission validates and reuses the manifest.

## Submit on research compute

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash submit_waterbirds100_r4rr_systematic_corruption_all.sh
```

This submits four jobs. Each job runs all five training seeds sequentially.
Resubmitting skips complete conditions, and workers skip valid completed seeds.

Dry-run first with:

```bash
DRY_RUN=1 bash submit_waterbirds100_r4rr_systematic_corruption_all.sh
```

Run only systematic Waterbird corruption with:

```bash
sbatch --job-name=r4c_w100_water \
  --export=ALL,CONDITION=class_waterbird \
  run_waterbirds100_r4rr_systematic_corruption_condition.sh
```

## Outputs

The default result root is:

```text
/home/ryreu/guided_cnn/logsWaterbird/r4rr_round2_systematic_teacher_corruption/waterbirds100
```

Every condition contains `run_contract.json`, per-seed `metrics.json`,
`per_seed_metrics.csv`, `summary.csv`, and `summary.json`. Corruption manifests
are stored under `corruption_manifests/<condition>/`.

After all jobs finish:

```bash
python summarize_waterbirds100_r4rr_systematic_corruption.py \
  --run-root /home/ryreu/guided_cnn/logsWaterbird/r4rr_round2_systematic_teacher_corruption/waterbirds100
```

The combined outputs include aggregate accuracy, mean/worst group accuracy,
all four test-group accuracies, and paired systematic-minus-random differences.
