# RedMeat R4RR Systematic Teacher Corruption

This implements the RedMeat portion of the Round 2 systematic
teacher-corruption study. It compares one fixed random 20% corruption control
against systematic inversion of all training teacher maps for each food class.

## Locked protocol

- Conditions: `random_20pct` and one condition for each of the five classes
- Training seeds: `0,1,2,3,4`
- Corruption-selection seed: `0`
- Train split: 2,500 examples, exactly 500 per class
- Every condition corrupts exactly 500 training examples (20%)
- Validation and test splits are unchanged and never load teacher maps
- Teacher operation: `1 - M`, followed by sum normalization
- Student: ImageNet-pretrained ResNet-50 with full fine-tuning and CAM evidence
- Epochs: `150`; batch size: `96`
- Optimizer: SGD, momentum `0.9`, weight decay `1e-5`, no scheduler
- R4RR: `attention_epoch=2`, `kl_lambda=11.44`, `kl_increment=0`
- Learning rates: base `2.40e-3`, classifier `2.33e-4`, phase-2 multiplier `1.567`
- Checkpoint selection: best validation balanced class accuracy after alignment begins
- No hyperparameter retuning

Teacher-map files are never overwritten. Inversion is applied lazily according
to a persisted condition manifest. The manifest records the exact selected
training rows, stable sample paths, class composition, dataset fingerprint, and
checksums. The same manifest is reused by all five model-training seeds.

## Submit on research compute

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash RedMeat_Runs/submit_redmeat_r4rr_systematic_corruption_all.sh
```

This submits six jobs. Each job runs all five training seeds sequentially.
Resubmitting skips complete conditions, and workers skip valid completed seeds.

Dry-run first with:

```bash
DRY_RUN=1 bash RedMeat_Runs/submit_redmeat_r4rr_systematic_corruption_all.sh
```

Run only Prime Rib with:

```bash
sbatch --job-name=r4c_meat_prime \
  --export=ALL,CONDITION=class_prime_rib \
  RedMeat_Runs/run_redmeat_r4rr_systematic_corruption_condition.sh
```

## Outputs

The default result root is:

```text
/home/ryreu/guided_cnn/logsRedMeat/r4rr_round2_systematic_teacher_corruption/redmeat
```

Every condition contains `run_contract.json`, per-seed `metrics.json`,
`per_seed_metrics.csv`, `summary.csv`, and `summary.json`. Corruption manifests
are stored under `corruption_manifests/<condition>/`.

After all jobs finish:

```bash
python RedMeat_Runs/summarize_redmeat_r4rr_systematic_corruption.py \
  --run-root /home/ryreu/guided_cnn/logsRedMeat/r4rr_round2_systematic_teacher_corruption/redmeat
```

The combined CSV includes test accuracy, mean-class accuracy, worst-class
accuracy, and each of the five individual class accuracies for every condition.

