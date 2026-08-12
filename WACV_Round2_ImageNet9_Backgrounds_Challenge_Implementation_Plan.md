# WACV Round 2: ImageNet-9 Backgrounds Challenge Implementation Plan

## Purpose of this document

This is an implementation handoff for adding an **ImageNet-9 (IN-9) Backgrounds Challenge** experiment to the Round 2 revision of **Right for the Right Regions (R4RR)**.

The goal is deliberately narrow:

> Add one larger, ImageNet-derived background-shift benchmark that tests whether the main R4RR result transfers beyond DecoyMNIST, Waterbirds, and RedMeat.

This experiment is meant to directly address Reviewer NbgG's concern that the current evaluation is concentrated on relatively small or narrow datasets and does not include an ImageNet-based benchmark. It is **not** meant to become a fourth dataset on which every architecture, teacher, corruption, curriculum, and sensitivity ablation must be repeated.

The main revision already has several other workstreams:

1. alignment-loss ablation beyond forward KL;
2. Pointing Game/localization evaluation on the existing datasets;
3. a stronger teacher-map failure/corruption experiment;
4. a substantial rewrite and clearer positioning relative to GALS;
5. this ImageNet-9 addition.

The IN-9 work should therefore be implemented in the **smallest rigorous form that answers the scale/generalization concern**.

---

# 1. Locked experimental question

The primary question is:

> **Does R4RR reduce reliance on image backgrounds on a larger ImageNet-derived benchmark, compared with standard ERM and the closest prior method, GALS?**

This should be tested using the standard Backgrounds Challenge setup rather than inventing a new main protocol.

## Primary protocol

**Train on Original IN-9.**

Then evaluate the exact same trained checkpoint on:

- Original
- Mixed-Same
- Mixed-Rand
- Mixed-Next

The central robustness metric is:

\[
\text{BG-Gap}
=
\text{Accuracy(Mixed-Same)}
-
\text{Accuracy(Mixed-Rand)}.
\]

Lower BG-Gap means the classifier's accuracy depends less on whether the background is class-consistent.

### Why Original training is the primary protocol

This makes the experiment recognizable as an evaluation on the established ImageNet-9 Backgrounds Challenge.

It also avoids making the paper look as though we constructed a special training distribution specifically to favor R4RR.

The experiment therefore asks:

> If all methods are trained normally on IN-9, which model is least disrupted when the foreground is preserved but the background-label relationship changes?

That is a strong and clean generalization test.

---

# 2. Optional secondary protocol

Only run this if:

- the primary IN-9 experiment is complete;
- all direct reviewer requests are complete;
- compute and time remain comfortable.

## Mixed-Same training stress test

Train on **Mixed-Same**, where each foreground is composited with a background drawn from the same IN-9 class.

Then evaluate on:

- Mixed-Same
- Mixed-Rand
- Mixed-Next

This is a more controlled shortcut-learning experiment because the source class of the background is class-consistent by construction during training.

Conceptually, it is more analogous to Waterbirds100.

However, this is **secondary**, because the main reason for adding IN-9 is to give the paper a standard ImageNet-derived benchmark. Do not delay the primary revision work to run this stress test.

---

# 3. Dataset summary

ImageNet-9 groups ImageNet classes into nine coarse categories:

1. dog
2. bird
3. vehicle
4. reptile
5. carnivore
6. insect
7. instrument
8. primate
9. fish

The benchmark contains approximately:

- **45,405 training images**
- **4,050 test images**
- **9 classes**
- roughly **5,045 training images per class**
- **450 test images per class**

The Backgrounds Challenge provides several foreground/background manipulations.

## Relevant variants

### Original

The normal IN-9 image.

### Mixed-Same

The foreground is placed on a background taken from an image of the **same class**.

The background source is therefore class-consistent.

### Mixed-Rand

The foreground is placed on a background sampled from a **random class**.

This breaks the systematic background-label relationship.

### Mixed-Next

The foreground from class \(y\) is paired with a background from a fixed different class.

This produces a systematically misleading background.

### Only-FG

The foreground is retained while background information is removed.

This is useful as a secondary diagnostic but is not required for the main table.

### Background-only variants

The benchmark also includes variants in which foreground information is removed.

These are useful diagnostics of how predictive background alone is, but they are not required for the main revision claim.

---

# 4. Primary methods

Keep the IN-9 comparison intentionally small.

## Required

1. **ERM / Vanilla**
2. **GALS**
3. **R4RR**

That is the main experiment.

### Why these three

ERM answers:

> How background-dependent is a normal classifier?

GALS answers:

> How does the closest existing VLM/language-guided attention method behave?

R4RR answers:

> Does positive latent evidence alignment plus Classify-then-Align reduce background dependence at larger scale?

This comparison is much more important than reproducing every baseline from the original main table.

## Optional fourth method

If it is essentially free to integrate, one additional strong robustness baseline such as AFR can be included.

Do **not** add it if it complicates the implementation or delays GALS/R4RR.

---

# 5. Student architecture

Use the same default student architecture as the main natural-image experiments:

- **ImageNet-pretrained ResNet-50**
- 9-way classification head
- same CAM-style student evidence construction used by R4RR on Waterbirds/RedMeat

Do not run MobileNetV2 or ViT on IN-9 for Round 2.

The architecture-generalization experiment and the large-scale dataset experiment answer different questions.

The paper already tests whether R4RR transfers across student architectures. IN-9 is being added to test whether the main robustness effect transfers to a larger ImageNet-derived benchmark.

There is no need to evaluate the Cartesian product of every dataset and every architecture.

---

# 6. Training budget: 20 epochs is the default

The current Waterbirds ResNet experiments train for 200 epochs with batch size 96.

Waterbirds contains roughly 4.8k training images. IN-9 contains roughly 45.4k.

This means that 20 epochs on IN-9 is surprisingly close to 200 Waterbirds epochs in terms of the total number of training examples seen.

Approximate sample exposures:

\[
200 \times 4{,}795 \approx 959{,}000
\]

for Waterbirds, versus

\[
20 \times 45{,}405 \approx 908{,}100
\]

for IN-9.

With batch size 96, both are also on the order of \(10^4\) optimizer steps.

Therefore:

> **Use 20 epochs as the default IN-9 training budget.**

This is not an arbitrary 10x reduction. It approximately preserves the amount of optimization work measured in examples/steps.

## Pilot check before launching the sweep

Before launching 20 Optuna trials per method:

1. run one ERM job for 20 epochs;
2. run one R4RR job for 20 epochs using an exposure-scaled Waterbirds configuration;
3. plot or inspect validation accuracy by epoch.

### Keep 20 epochs if

validation performance has clearly flattened by approximately epochs 15-20.

### Increase to 25-30 epochs if

validation performance is still rising materially at epoch 20.

If the budget changes, change it **for all methods** before the full sweep.

Do not allow R4RR 30 epochs while ERM/GALS receive 20 unless there is a method-specific reason that is explicitly documented.

---

# 7. Hyperparameter tuning policy

## Core rule

Use an **equal completed tuning budget** for GALS and R4RR.

Recommended:

> **20 completed Optuna trials per method.**

This is deliberately smaller than the 50-trial budget used for the main ResNet/LeNet comparisons in the current paper, but the paper already uses a smaller 25-trial budget for computationally heavier architecture-transfer experiments.

The IN-9 addition is also a computationally heavier revision experiment, so a 20-trial equal budget is defensible.

## Target sweep counts

| Method | Completed tuning trials | Final seeds |
|---|---:|---:|
| ERM | 20 | 5 |
| GALS | 20 | 5 |
| R4RR | 20 | 5 |

If GALS is the bottleneck, do not compensate by giving R4RR 50 trials.

A fair 20-vs-20 comparison is better than a 50-vs-20 comparison.

If only 18 GALS trials finish successfully, either:

1. finish two additional GALS trials, or
2. use an equal 18-trial effective budget for the comparison.

The paper should report **completed** trials, not merely launched jobs.

---

# 8. Validation and model selection

This is important.

## Never tune on Mixed-Rand, Mixed-Next, or the official test variants

All hyperparameter decisions must be frozen **before** looking at the final background-shift test results.

For primary Original-IN-9 training:

- train on the Original training split;
- tune/select using the Original validation split supplied with the training data;
- after selecting the configuration, freeze it;
- run the 5 final seeds;
- only then evaluate those checkpoints on the official test variants.

This gives a particularly strong interpretation:

> The model was selected using normal in-distribution validation performance, yet R4RR generalized better when the background relationship changed.

## Selection metric

Use validation top-1 accuracy.

Because IN-9 is balanced across the nine coarse classes, ordinary validation accuracy and macro class accuracy should be very similar.

For completeness, log both:

- overall validation accuracy;
- mean per-class validation accuracy.

Use one metric consistently as the Optuna objective.

Recommended:

> **mean per-class validation accuracy**

because it is directly comparable to the balanced/group-style reporting already used elsewhere in the paper.

## R4RR checkpoint rule

Preserve the current R4RR principle:

> only select checkpoints after the alignment phase has started.

Do not allow an R4RR trial to "win" using a checkpoint from the CE-only warmup.

---

# 9. R4RR hyperparameter search

The current Waterbirds R4RR sweep searches:

- attention/alignment start epoch;
- KL weight;
- backbone learning rate;
- classifier learning rate;
- phase-2 LR multiplier.

The existing Waterbirds sweep uses approximately:

- `kl_lambda`: 1 to 500, log scale
- `base_lr`: \(10^{-5}\) to \(5\times10^{-2}\), log scale
- `classifier_lr`: \(10^{-5}\) to \(5\times10^{-2}\), log scale
- `lr2_mult`: 0.1 to 3.0, log scale
- attention epoch across the training schedule

Do not increase the maximum learning rate simply because the epoch count is lower. The total number of optimizer updates is already similar to Waterbirds.

## Recommended IN-9 R4RR search space

For a 20-epoch run:

```text
attention_epoch: 2 to 12
kl_lambda:       1 to 500, log
base_lr:         1e-5 to 5e-2, log
classifier_lr:   1e-5 to 5e-2, log
lr2_mult:        0.1 to 3.0, log
```

### Why restrict attention_epoch

The current Waterbirds100 selected configuration uses alignment start epoch 73 out of 200.

Scaling by dataset size/exposure:

\[
73 \times \frac{4795}{45405} \approx 7.7.
\]

So an IN-9 alignment start around epoch 8 is the natural exposure-matched analogue.

A search range of 2-12:

- includes substantially earlier alignment;
- includes the exposure-matched value;
- includes later alignment;
- still guarantees at least 8 full epochs of Phase 2 in a 20-epoch run.

There is little value in letting Optuna choose epoch 18 or 19, where almost no alignment training would occur.

---

# 10. Seed the R4RR Optuna study with known sensible points

With only 20 trials, use existing Waterbirds results as priors rather than making every trial blind.

The current optimized Waterbirds100 R4RR configuration is approximately:

```text
attention_epoch: 73 / 200
kl_lambda:       495.61
base_lr:         5.72e-5
classifier_lr:   3.57e-3
lr2_mult:        0.123
```

The exposure-scaled IN-9 version is:

```text
attention_epoch: 8 / 20
kl_lambda:       495.61
base_lr:         5.72e-5
classifier_lr:   3.57e-3
lr2_mult:        0.123
```

Enqueue this as one of the first trials.

The Waterbirds95 selected R4RR configuration was approximately:

```text
attention_epoch: 109 / 200
kl_lambda:       295.30
base_lr:         4.82e-5
classifier_lr:   2.93e-3
lr2_mult:        0.409
```

The exposure-scaled alignment epoch is roughly 11-12.

This can be a second seeded trial:

```text
attention_epoch: 11
kl_lambda:       295.30
base_lr:         4.82e-5
classifier_lr:   2.93e-3
lr2_mult:        0.409
```

Then let TPE explore normally.

This is preferable to shrinking the LR search space aggressively and risking excluding a good IN-9 configuration.

---

# 11. ERM hyperparameter tuning

Keep ERM simple and fair.

Use the same optimizer family and parameter-group structure used in the current paper.

Tune the same ERM hyperparameters that are normally tuned in the Waterbirds experiments, for example:

- backbone learning rate;
- classifier learning rate;
- momentum if the current ERM sweep tunes it.

Use 20 completed Optuna trials.

Do not give ERM a hand-selected learning rate while giving R4RR a full search, or vice versa.

---

# 12. GALS hyperparameter tuning

GALS is expected to be the expensive method.

The goal is **not** to reproduce an enormous second-level search over every possible GALS attention mechanism and map source on IN-9.

Use the same GALS formulation that is treated as the principal/closest GALS baseline in the current paper, then tune its continuous hyperparameters on IN-9.

Recommended policy:

- fixed GALS formulation/map type;
- 20 completed Optuna trials;
- same 20-epoch cap;
- same train/validation split;
- same validation selection metric;
- 5 final seeds.

If the existing paper's main GALS result is defined as the strongest variant chosen from several mechanisms/map sources, document exactly which variant is used for IN-9 and why.

A defensible wording is:

> For the larger-scale IN-9 experiment, we use the strongest GALS formulation identified in our existing benchmark study and independently retune its continuous hyperparameters on IN-9 using the same 20-trial budget as R4RR.

This avoids a massive nested sweep while still giving GALS a real dataset-specific optimization.

---

# 13. Teacher-map generation for R4RR

## Generate maps only for training data

For the primary experiment, R4RR requires teacher maps only for:

> **Original IN-9 training images**

Do **not** generate WeCLIP+ teacher maps for:

- Original test
- Mixed-Same test
- Mixed-Rand test
- Mixed-Next test

The teacher is not used at inference.

This substantially reduces map-generation time and storage.

## Prompt construction

Start with the nine official IN-9 coarse class names.

Recommended foreground concepts:

```text
dog
bird
vehicle
reptile
carnivore
insect
musical instrument
primate
fish
```

"musical instrument" is preferable to the more ambiguous standalone word "instrument."

Do not create hand-engineered class-specific background prompts such as:

```text
dog -> grass
fish -> water
vehicle -> road
```

That would inject knowledge about the background correlations and weaken the experiment.

Use the same generic background/context prompt design already used by the R4RR teacher pipeline.

## Prompt fairness

Where GALS requires class/language concepts, use equivalent class semantics.

Do not give R4RR a richer hand-curated set of prompts while giving GALS only the raw class name.

If synonyms are needed for a coarse class, decide them during teacher-map quality control and freeze them **before model results are inspected**.

---

# 14. Teacher-map quality-control stage

Do not immediately generate all 45k maps and launch the sweep.

First generate maps for a small, fixed diagnostic subset.

Recommended:

- 20 images per class;
- 9 classes;
- 180 images total.

Create overlays showing:

- original image;
- WeCLIP+ teacher map;
- teacher map overlaid on image.

Inspect whether the map generally localizes the target object rather than the background.

Pay special attention to potentially ambiguous coarse classes:

- dog vs carnivore;
- vehicle;
- primate;
- instrument.

If the teacher fails badly because a coarse label is linguistically ambiguous, fix the **task-level prompt wording**, regenerate the same diagnostic subset, and then freeze the prompt list.

Do not repeatedly change prompts after seeing downstream R4RR test performance.

---

# 15. Geometric augmentation and map synchronization

This is an implementation-critical detail.

R4RR's image and teacher map must undergo the **same geometric transformation**.

If training applies:

- random crop;
- resize;
- horizontal flip;
- affine transform;

the corresponding teacher map must receive the exact same crop/flip/resize.

Otherwise the alignment loss will compare the student evidence for one region against a teacher map that has moved relative to the image.

Implementation rule:

> Any stochastic spatial transform must be sampled once and applied jointly to image and teacher map.

Photometric transforms such as:

- color jitter;
- normalization;

should apply only to the image.

The GALS spatial maps require the same care.

Before the first full run, create a debugging visualization after augmentation showing:

1. transformed image;
2. transformed teacher map;
3. overlay.

Confirm alignment manually.

---

# 16. Teacher-map storage strategy

Research-compute storage is limited, so avoid unnecessary copies.

## Store only what is required

For the primary experiment:

- Original IN-9 train images
- Original validation images
- official test release
- R4RR train teacher maps
- GALS maps required by the chosen GALS formulation
- current/best checkpoints
- logs and CSV files

Do not store teacher maps for test variants.

## Avoid duplicate archives

After a dataset archive has been successfully extracted and verified:

```bash
rm <archive>.tar.gz
```

unless there is a reason to keep it.

## Intermediate WeCLIP+ outputs

If the teacher pipeline emits:

- raw CAMs;
- refined maps;
- visualizations;
- temporary feature dumps;

keep only the representation actually required by student training once generation has been verified.

If possible without changing the mathematics, store maps in a compact representation such as:

- float16 arrays;
- compressed NumPy files;
- lossless image maps if the existing loader already uses them.

Do not change the teacher normalization semantics purely to save disk space.

---

# 17. Dataset directory layout

Suggested layout:

```text
data/
└── imagenet9/
    ├── train_source/
    │   └── original/
    │       ├── train/
    │       └── val/
    │
    ├── official_test/
    │   ├── original/
    │   ├── mixed_same/
    │   ├── mixed_rand/
    │   ├── mixed_next/
    │   ├── only_fg/
    │   └── ...
    │
    ├── teacher_maps/
    │   └── r4rr_original_train/
    │
    └── gals_maps/
        └── original_train/
```

Keep training data and official test data clearly separated.

This reduces the chance that the training loader accidentally points at a test variant.

---

# 18. Suggested code organization

Do not modify the Waterbirds scripts in-place until the IN-9 loader works.

Create a separate IN-9 path first.

Suggested structure:

```text
repro_runs/
├── r4rr/
│   └── imagenet9/
│       ├── dataset.py
│       ├── train_r4rr.py
│       ├── sweep_r4rr.py
│       ├── eval_backgrounds.py
│       └── prompts.py
│
└── other_models/
    └── imagenet9/
        ├── train_erm.py
        ├── sweep_erm.py
        └── gals/
            ├── train_gals.py
            └── sweep_gals.py
```

If enough code is genuinely shared with Waterbirds, refactor shared logic only after the IN-9 prototype works.

The first goal is correctness, not a beautiful abstraction.

---

# 19. Dataset loader requirements

The IN-9 loader should expose:

```text
image
label
sample_id / relative_path
```

For R4RR training it must also resolve:

```text
teacher_map_path(sample_id)
```

Requirements:

- fixed class-to-index mapping;
- exactly 9 output classes;
- deterministic sample IDs;
- no label remapping between train and test variants;
- identical preprocessing conventions across all evaluation variants.

The class mapping must be saved alongside each run.

Example:

```json
{
  "dog": 0,
  "bird": 1,
  "vehicle": 2,
  "reptile": 3,
  "carnivore": 4,
  "insect": 5,
  "instrument": 6,
  "primate": 7,
  "fish": 8
}
```

Use the official repository's ordering if it differs from this example.

Do not silently impose a new ordering.

---

# 20. Evaluation integration

The official Backgrounds Challenge code accepts a 9-class IN-9 classifier.

Our evaluation code should either:

1. directly adapt the official `in9_eval.py`/challenge logic to the R4RR checkpoint format; or
2. produce a checkpoint wrapper compatible with the official evaluator.

The important part is to preserve the benchmark's class mapping and preprocessing.

## Required final metrics

For each final seed, record:

```text
Original accuracy
Mixed-Same accuracy
Mixed-Rand accuracy
Mixed-Next accuracy
BG-Gap = Mixed-Same - Mixed-Rand
```

Then report:

```text
mean ± standard deviation over 5 seeds
```

For BG-Gap, compute the gap **within each seed first**, then average the five gaps.

Do not compute only:

```text
mean(Mixed-Same) - mean(Mixed-Rand)
```

even though the central value will be similar, because per-seed gaps allow the correct variance to be reported.

---

# 21. Cheap secondary evaluations

Once the final checkpoints exist, evaluation is much cheaper than training.

Therefore it is reasonable to run additional official IN-9 variants and place them in the supplementary material.

Useful secondary metrics:

- Only-FG accuracy
- No-FG accuracy
- Only-BG variants
- adversarial Backgrounds Challenge accuracy

These should **not** expand the tuning loop.

Run them after hyperparameters and checkpoints are frozen.

## Priority order

1. Original
2. Mixed-Same
3. Mixed-Rand
4. BG-Gap
5. Mixed-Next
6. Only-FG
7. adversarial challenge
8. background-only variants

The main paper probably only needs items 1-5.

---

# 22. No IN-9 Pointing Game is required

Do not make IN-9 Pointing Game a dependency for this experiment.

The reviewer requested broader Pointing Game evaluation on the current datasets, and that should be addressed separately.

IN-9 already provides a direct behavioral test:

> Keep the foreground object and change the background. Does the prediction survive?

That is arguably a more natural robustness measurement for this benchmark than adding another explanation metric.

If a clean official foreground-mask representation is later found and using it is trivial, IN-9 localization can be a supplementary bonus.

It is not required.

---

# 23. Preflight implementation tests

Before launching any sweep, pass all of these.

## Test A: dataset counts

Verify expected class and split sizes.

Print:

```text
train count
validation count
test count per variant
count per class
```

Investigate any mismatch.

## Test B: class mapping

Take one known image from every class and verify:

```text
folder/category -> numeric label -> model class name
```

## Test C: official evaluator sanity check

Use an official/pretrained ResNet-50 checkpoint and make sure the evaluation pipeline produces numbers in the vicinity of the official repository's published values.

The MadryLab repository reports approximately:

```text
ResNet-50:
Original:    95.6
Mixed-Same:  86.2
Mixed-Rand:  78.9
BG-Gap:       7.3
```

Exact reproduction can vary slightly with preprocessing/software versions, but a large disagreement means the class mapping or preprocessing is probably wrong.

## Test D: R4RR map lookup

For 100 random training samples:

- load image;
- resolve teacher map;
- verify no missing file;
- verify map is finite;
- verify map has nonzero mass;
- verify correct sample identity.

## Test E: augmentation synchronization

Visualize at least 20 randomly augmented image/map pairs.

## Test F: one-batch forward/backward

Run:

- ERM
- GALS
- R4RR

for a few batches and verify finite losses/gradients.

## Test G: one short overfit

Take a very small subset and verify that each method can substantially overfit it.

This catches label and loader bugs before expensive sweeps.

---

# 24. Pilot phase

Before Optuna:

## ERM pilot

```text
epochs: 20
seed: 0
reasonable LR configuration
```

Inspect:

- train loss;
- train accuracy;
- validation accuracy;
- validation curve by epoch.

## R4RR pilot

Use the exposure-scaled Waterbirds100 configuration:

```text
epochs:            20
attention_epoch:    8
kl_lambda:        495.61
base_lr:            5.72e-5
classifier_lr:      3.57e-3
lr2_mult:           0.123
```

Inspect:

- CE loss;
- alignment loss;
- validation accuracy;
- evidence maps before/after alignment;
- whether alignment causes instability.

## GALS pilot

Run one configuration long enough to estimate:

- minutes per epoch;
- total wall time;
- GPU memory;
- likely sweep throughput.

Use this to schedule jobs, not to tune using test performance.

---

# 25. Full sweep stage

After pilots confirm 20 epochs is sufficient:

## Sweep 1: ERM

```text
20 completed Optuna trials
train seed = fixed
validation objective = macro validation accuracy
epochs = 20
```

Save:

```text
trial id
all hyperparameters
best validation metric
best checkpoint epoch
runtime
checkpoint path
```

## Sweep 2: R4RR

```text
20 completed Optuna trials
same train seed
validation objective = macro validation accuracy
epochs = 20
attention_epoch = 2..12
```

Seed TPE with the two exposure-scaled Waterbirds configurations described above.

## Sweep 3: GALS

```text
20 completed Optuna trials
same train seed
validation objective = macro validation accuracy
epochs = 20
```

Launch GALS early because it is likely to determine the critical path.

---

# 26. Final seeded runs

Once the best configuration for each method is selected:

> Freeze the hyperparameters.

Run five independent seeds.

Recommended:

```text
0
1
2
3
4
```

For each seed:

1. train once;
2. save the validation-selected checkpoint;
3. evaluate the same checkpoint on every test variant;
4. write one result row.

Do not retune per seed.

---

# 27. Result file format

Use a single tidy CSV.

Example:

```csv
method,seed,original,mixed_same,mixed_rand,mixed_next,bg_gap,only_fg,checkpoint
ERM,0,...
ERM,1,...
GALS,0,...
R4RR,0,...
```

Also save a summary CSV:

```csv
method,metric,mean,std
ERM,original,...
ERM,mixed_rand,...
ERM,bg_gap,...
GALS,...
R4RR,...
```

Keep hyperparameter sweep results separate from final test results.

Suggested directory:

```text
results/imagenet9/
├── sweeps/
│   ├── erm.csv
│   ├── gals.csv
│   └── r4rr.csv
│
├── final/
│   ├── per_seed.csv
│   └── summary.csv
│
├── logs/
└── checkpoints/
```

---

# 28. Main paper table

The cleanest main-text table is probably:

| Method | Original ↑ | Mixed-Same ↑ | Mixed-Rand ↑ | BG-Gap ↓ | Mixed-Next ↑ |
|---|---:|---:|---:|---:|---:|
| ERM | | | | | |
| GALS | | | | | |
| R4RR | | | | | |

Report mean ± standard deviation where space allows.

If space is limited, keep the full table in supplementary and put the most important metrics in the main text:

- Original
- Mixed-Rand
- BG-Gap
- Mixed-Next

---

# 29. How to interpret possible outcomes

## Best-case outcome

R4RR:

- retains similar Original accuracy;
- improves Mixed-Rand;
- improves Mixed-Next;
- substantially reduces BG-Gap;
- beats GALS on the robustness metrics.

This is an extremely strong addition.

The main claim becomes:

> R4RR's improvement is not restricted to small shortcut benchmarks; it also reduces background dependence on a substantially larger ImageNet-derived challenge.

## Good outcome

R4RR sacrifices a small amount of Original accuracy but meaningfully improves:

- Mixed-Rand;
- Mixed-Next;
- BG-Gap.

This is still useful.

It suggests the expected robustness/average-accuracy tradeoff rather than failure.

## Neutral outcome

R4RR and GALS/ERM are statistically very close.

Before concluding that the method does not transfer, check:

1. teacher-map localization quality;
2. augmentation synchronization;
3. convergence at 20 epochs;
4. whether the correct checkpoint is being selected;
5. class mapping;
6. whether the teacher maps correspond to the correct samples.

Do not respond by tuning on the official test variants.

## Negative outcome

If the implementation is verified and R4RR genuinely performs worse under background shift, treat that as a scientific result/limitation.

Because the extra dataset was not explicitly required as a condition of resubmission, discuss with the advisor whether it belongs in the final revision.

Do not use repeated test-set inspection to redesign hyperparameters until the result becomes favorable.

---

# 30. Success criteria defined before running

The experiment should be considered successful if R4RR shows a clear improvement on **background robustness**, not merely Original accuracy.

Primary indicators:

1. lower BG-Gap than ERM;
2. lower BG-Gap than GALS;
3. higher Mixed-Rand accuracy than ERM;
4. higher Mixed-Rand accuracy than GALS;
5. higher Mixed-Next accuracy than ERM/GALS.

Secondary:

6. Original accuracy remains competitive;
7. Only-FG accuracy is strong;
8. adversarial Backgrounds Challenge accuracy improves.

No single exact numerical threshold needs to be declared in advance, but robustness gains should be larger than ordinary seed noise to support a strong claim.

---

# 31. Paper positioning

This experiment should **not** be woven into every ablation section.

Give it a clearly defined role.

Possible section title:

> **Large-Scale Background Shift on ImageNet-9**

Suggested narrative:

1. The original experiments establish R4RR on synthetic, natural, and domain-specific shortcut benchmarks.
2. To test whether the behavior scales beyond those datasets, evaluate on ImageNet-9.
3. Train ERM, GALS, and R4RR on Original IN-9.
4. Evaluate how performance changes when the foreground is retained and the background is altered.
5. R4RR's lower BG-Gap / stronger Mixed-Rand and Mixed-Next accuracy indicates reduced background reliance.

This prevents the reader from asking why IN-9 was not included in:

- MobileNet experiments;
- ViT experiments;
- teacher ablations;
- curriculum ablations;
- corruption ablations;
- every Pointing Game analysis.

Those experiments answer different questions.

---

# 32. Suggested reviewer-response framing

The rebuttal can say something close to:

> We appreciate the concern regarding benchmark scale and diversity. In the revision, we add ImageNet-9, a substantially larger ImageNet-derived benchmark designed to measure reliance on image backgrounds. We train ERM, GALS, and R4RR under an equal validation-based tuning budget and evaluate the frozen models on Original, Mixed-Same, Mixed-Rand, and Mixed-Next test variants. This directly tests whether the R4RR robustness effect extends beyond the smaller benchmarks in the original submission.

Then state the actual result.

Do not oversell IN-9 as equivalent to full ImageNet-1K.

Call it:

> **an ImageNet-derived benchmark**

not:

> **full-scale ImageNet evaluation**

---

# 33. Suggested implementation sequence

Follow this order.

## Stage 0: download and disk sanity

- clone/use official MadryLab Backgrounds Challenge assets;
- download Original training data;
- download official test release;
- inspect disk use with `du -sh`;
- delete compressed archives after verified extraction if space is tight.

## Stage 1: dataset loader

- implement IN-9 class mapping;
- train/val loader;
- official test-variant loader;
- verify counts.

## Stage 2: official evaluator sanity check

- reproduce the official pretrained ResNet-50 behavior closely enough to verify preprocessing/mapping.

## Stage 3: teacher-map prototype

- add IN-9 class prompts;
- generate 180 diagnostic maps;
- inspect overlays;
- freeze prompt list.

## Stage 4: full R4RR teacher-map generation

- Original training set only;
- remove unnecessary intermediates after validation.

## Stage 5: GALS map setup

- generate/prepare only the map representation required for the selected GALS formulation.

## Stage 6: augmentation synchronization test

- visualize image/map pairs after train transforms.

## Stage 7: 20-epoch pilots

- ERM;
- R4RR;
- GALS runtime estimate.

## Stage 8: freeze epoch budget

- stay at 20 if converged;
- otherwise move all methods to 25-30 before sweeps.

## Stage 9: launch sweeps

- GALS first because it is slow;
- R4RR and ERM in parallel;
- 20 completed trials each.

## Stage 10: select configurations

- validation data only;
- save final YAML/JSON configs.

## Stage 11: five final seeds

- ERM;
- GALS;
- R4RR.

## Stage 12: frozen evaluation

- Original;
- Mixed-Same;
- Mixed-Rand;
- Mixed-Next;
- calculate BG-Gap.

## Stage 13: cheap secondary evaluation

If useful:

- Only-FG;
- background-only;
- adversarial challenge.

## Stage 14: paper integration

- one main table/figure;
- full details in supplement;
- exact hyperparameters;
- exact search budget;
- direct reviewer response.

## Stage 15: optional Mixed-Same training stress test

Only if everything above is done comfortably.

---

# 34. SLURM/job strategy

Because most of the cost is unattended compute, optimize for parallelism and recoverability.

## Every training job should save

- config;
- git commit/hash if available;
- random seed;
- hostname/GPU;
- start/end time;
- validation history;
- best epoch;
- checkpoint;
- stdout/stderr.

## Sweeps should be resumable

The existing R4RR sweep already supports resume behavior.

Preserve this for IN-9.

A sweep should be able to stop after 11 trials and later continue to 20 without repeating completed runs.

## GALS

Launch GALS earliest.

If there is a wall-time limit, run individual trials as separate jobs or use a resumable launcher so one slow/failed trial does not destroy the whole sweep.

---

# 35. Storage-conscious checkpoint policy

For tuning:

> Keep only the best checkpoint for each trial, or preferably only the current global-best trial if the sweep code supports it.

For final five-seed runs:

> Keep all five selected checkpoints until the paper is finalized.

Do not keep every epoch checkpoint.

If intermediate checkpoints are required for selecting the best epoch, delete non-selected ones at the end of each run.

---

# 36. Reproducibility metadata to save

For every method:

```text
dataset version/source
training variant
validation variant
test release
class mapping
image preprocessing
augmentation
architecture
pretraining source
optimizer
epochs
batch size
all search ranges
number of completed trials
Optuna sampler
sweep seed
training seed
best hyperparameters
final seeds
checkpoint selection rule
```

For R4RR additionally:

```text
teacher model
teacher prompts
teacher-map preprocessing
student evidence layer
alignment loss
alignment direction
alignment start epoch
KL/alignment weight
phase-2 LR multiplier
```

For GALS additionally:

```text
map source/backbone
attention regularization formulation
gradient criterion
regularization weight
```

---

# 37. Important methodological guardrails

## Do not use the official test variants for hyperparameter selection

This is the most important guardrail.

## Do not change prompts based on Mixed-Rand/Mixed-Next results

Teacher prompt refinement should happen only from direct teacher-map inspection on training examples.

## Use equal tuning budgets for R4RR and GALS

This avoids a fairness objection.

## Keep the epoch budget fixed during the actual comparison

Use pilots to choose 20 vs 25/30, then freeze it.

## Do not add unrelated IN-9 ablations

The purpose is scale/generalization.

## Report BG-Gap as a robustness metric, not the only metric

A tiny BG-Gap can be achieved by being bad everywhere.

Always show Mixed-Same/Mixed-Rand and clean/original accuracy alongside it.

---

# 38. Minimal version if time becomes tight

If the revision timeline becomes constrained, the minimum acceptable IN-9 experiment is:

```text
Training:
  Original IN-9

Methods:
  ERM
  GALS
  R4RR

Tuning:
  20 trials each
  equal validation-based budget

Epochs:
  20, assuming pilots show convergence

Final:
  5 seeds each

Evaluation:
  Original
  Mixed-Same
  Mixed-Rand
  Mixed-Next
  BG-Gap

Architecture:
  ResNet-50 only

No:
  architecture ablations
  teacher ablations
  corruption ablations
  Pointing Game
  second training distribution
```

That alone directly answers the reviewer's scale/ImageNet-based benchmark concern.

---

# 39. Expanded version if everything finishes early

Only after the minimum version and the other Round 2 reviewer requests are complete:

1. run Only-FG and background-only evaluation;
2. run the adversarial Backgrounds Challenge evaluator;
3. optionally train on Mixed-Same and evaluate Mixed-Rand/Mixed-Next;
4. optionally add one cheap robustness baseline.

Do not move to this list while direct reviewer requests remain unfinished.

---

# 40. Final decision summary

The intended Round 2 IN-9 experiment is:

> **Train ImageNet-pretrained ResNet-50 ERM, GALS, and R4RR on Original ImageNet-9 for approximately 20 epochs. Independently tune each method on the normal validation set using an equal 20-trial Optuna budget. Freeze the selected hyperparameters, train five final seeds, and evaluate the same checkpoints on Original, Mixed-Same, Mixed-Rand, and Mixed-Next. Report clean accuracy, shifted-background accuracy, and BG-Gap.**

R4RR should generate teacher maps **only for the Original training split**.

The experiment is a **large-scale/background-shift generalization test**, not a new dataset for every existing ablation.

The rest of the Round 2 revision should remain focused on:

- alternative alignment losses;
- broader Pointing Game evaluation;
- stronger systematic teacher-failure analysis;
- clearer novelty positioning against GALS;
- a substantial readability rewrite.

If all of those pieces are completed cleanly, the revised paper will directly address essentially every concrete scientific concern raised in Round 1 while adding a substantial ImageNet-derived generalization result.

---

# Sources / reference points

- Current R4RR submission, *Right for the Right Regions: Vision-Language Guided Evidence Alignment for Robust Classification*.
- Current R4RR supplementary code and optimized Waterbirds hyperparameters.
- Reviewer NbgG, especially Major Weakness #4 regarding lack of an ImageNet-based benchmark.
- Kai Xiao, Logan Engstrom, Andrew Ilyas, and Aleksander Madry, *Noise or Signal: The Role of Image Backgrounds in Object Recognition*, ICLR 2021.
- MadryLab Backgrounds Challenge repository: https://github.com/MadryLab/backgrounds_challenge
