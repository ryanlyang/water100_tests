# DecoyMNIST R4RR Systematic Teacher Corruption

This implements the DecoyMNIST portion of the Round 2 systematic
teacher-corruption study. It compares one fixed random 10% corruption control
against systematic inversion of all training teacher maps for each digit.

## Locked protocol

- Conditions: `random_10pct`, then `digit_0` through `digit_9`
- Training seeds: `0,1,2,3,4`
- Corruption-selection seed: `0`
- Actual train split: 54,000 examples; random control: exactly 5,400 examples
- Validation split: the unchanged 6,000 examples selected with split seed `0`
- Teacher operation: `1 - M`, followed by sum normalization
- Model: the existing CDEP-style LeNet with Grad-CAM evidence
- Epochs: `19`
- Optimizer: Adam, `lr=1e-3`, `weight_decay=1e-4`
- R4RR: `attention_epoch=7`, `kl_lambda=495.61`, `kl_increment=0`
- Checkpoint selection: best validation accuracy after alignment begins
- No hyperparameter tuning and no Pointing Game

The Decoy LeNet uses one learning rate. The optimized configuration's
`base_lr` and `classifier_lr` are both `1e-3`; `lr2_mult` is recorded for
provenance but is not applicable to this single-LR implementation.

Teacher maps are never overwritten. Inversion is applied lazily according to
the persisted condition manifest. Each manifest contains base dataset indices,
stable relative sample paths, class counts, split and selection hashes, and an
audit of selected and non-selected examples.

## Submit on research compute

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash submit_decoymnist_r4rr_systematic_corruption_all.sh
```

This submits 11 jobs. Each job runs all five training seeds sequentially.
Results use stable condition directories, so resubmitting the command skips
completed conditions and each worker skips completed seeds.

Dry-run the submission first with:

```bash
DRY_RUN=1 bash submit_decoymnist_r4rr_systematic_corruption_all.sh
```

Run a single condition with:

```bash
sbatch --job-name=r4c_dec_d0 \
  --export=ALL,CONDITION=digit_0 \
  run_decoymnist_r4rr_systematic_corruption_condition.sh
```

## Outputs

The default result root is:

```text
/home/ryreu/guided_cnn/logsMNIST/r4rr_round2_systematic_teacher_corruption/decoymnist
```

Every condition contains:

- `run_contract.json`
- `seed_<N>/metrics.json`
- `per_seed_metrics.csv`
- `summary.csv`
- `summary.json`

Corruption selections are under `corruption_manifests/<condition>/` and are
shared across all five training seeds.

After all jobs finish:

```bash
python summarize_decoymnist_r4rr_systematic_corruption.py \
  --run-root /home/ryreu/guided_cnn/logsMNIST/r4rr_round2_systematic_teacher_corruption/decoymnist
```

