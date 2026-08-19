# Waterbirds-95 R4RR Systematic Teacher Corruption

This implements the Waterbirds-95 portion of the Round 2 systematic
teacher-corruption study. It compares inversion of every training teacher map
in one class-background subgroup against inversion of an exactly count-matched
random subset sampled from the full training split.

## Locked protocol

- Systematic groups: Land-on-Land, Land-on-Water, Water-on-Land, Water-on-Water
- Four random controls, each exactly matched to its corresponding group count
- Training group counts: `3498`, `184`, `56`, and `1057`, respectively
- Training seeds: `0,1,2,3,4`; corruption-selection seed: `0`
- The selected subset is fixed and shared across all five training seeds
- Validation and test splits are unchanged and never load teacher maps
- Teacher operation: `1 - M`, nonnegative clamp, then sum normalization
- Student: ImageNet-pretrained ResNet-50 with full fine-tuning and CAM evidence
- Epochs: `200`; batch size: `96`
- Optimizer: SGD, momentum `0.9`, weight decay `1e-5`, no scheduler
- R4RR: `attention_epoch=109`, `kl_lambda=295.30`, `kl_increment=0`
- Learning rates: base `4.82e-5`, classifier `2.93e-3`, phase-2 multiplier `0.409`
- Checkpoint selection: best validation balanced class accuracy after alignment begins
- No hyperparameter retuning

Teacher-map files are never overwritten. Inversion is applied lazily according
to a persisted condition manifest. Each manifest records exact split indices,
stable sample paths, subgroup composition, dataset fingerprint, and checksums.
Resubmission validates and reuses that manifest rather than drawing a new set.

## Submit on research compute

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash submit_waterbirds95_r4rr_systematic_corruption_all.sh
```

This submits eight jobs. Each job runs all five training seeds sequentially.
Resubmitting skips complete conditions, and workers skip valid completed seeds.

Dry-run first with:

```bash
DRY_RUN=1 bash submit_waterbirds95_r4rr_systematic_corruption_all.sh
```

Run only systematic Water-on-Land corruption with:

```bash
sbatch --job-name=r4c_w95_gwl \
  --export=ALL,CONDITION=group_water_on_land \
  run_waterbirds95_r4rr_systematic_corruption_condition.sh
```

## Outputs

The default result root is:

```text
/home/ryreu/guided_cnn/logsWaterbird/r4rr_round2_systematic_teacher_corruption/waterbirds95
```

Every condition contains `run_contract.json`, per-seed `metrics.json`,
`per_seed_metrics.csv`, `summary.csv`, and `summary.json`. Corruption manifests
are stored under `corruption_manifests/<condition>/`.

After all jobs finish:

```bash
python summarize_waterbirds95_r4rr_systematic_corruption.py \
  --run-root /home/ryreu/guided_cnn/logsWaterbird/r4rr_round2_systematic_teacher_corruption/waterbirds95
```

The combined outputs include aggregate accuracy, mean/worst group accuracy,
all four group accuracies, and seed-paired systematic-minus-random differences.
