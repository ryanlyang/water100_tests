# Round 2 Systematic Teacher-Map Corruption Study

## Overview

This document specifies the expanded teacher-map corruption study for the Round 2 revision of **Right for the Right Regions (R4RR)**.

The original paper already includes a teacher-map corruption stress test across all four evaluation settings:

- DecoyMNIST
- Waterbirds95
- Waterbirds100
- RedMeat

In that experiment, a fixed **15% random subset of training teacher maps** is corrupted before student training. Two corruption types are considered:

1. aggressive Gaussian blur, representing spatially imprecise or diffuse supervision;
2. inversion, representing an actively misleading teacher signal.

The original result shows that R4RR remains relatively stable when teacher errors are **randomly scattered** through the training set.

The reviewer raised a more specific concern: random corruption may not capture realistic failure modes in which the frozen VLM teacher is **systematically wrong for a particular class, subgroup, or domain**.

The new Round 2 experiment is designed to answer exactly that concern.

The central question is:

> **Is R4RR more sensitive when the same amount of teacher-map error is concentrated systematically within one semantic class or subgroup, rather than distributed randomly across the training set?**

The experiment is intended as a **failure-mode characterization**, not as an attempt to show that R4RR is immune to arbitrarily bad teachers.

---

# 1. Main Scientific Goal

The original corruption study answers:

> Can R4RR tolerate a moderate number of individually incorrect teacher maps when those errors are randomly distributed?

The new study asks:

> What happens when teacher failures are correlated with semantic structure?

Examples include:

- the teacher consistently producing bad localization maps for one digit class;
- the teacher consistently failing on one food category;
- the teacher failing specifically for waterbirds appearing on land;
- the teacher failing on an entire bird class under perfect shortcut correlation.

This distinction matters because a real VLM teacher is unlikely to make perfectly independent errors. Its failures may instead be concentrated around:

- certain semantic categories;
- particular visual contexts;
- specific class-background combinations;
- rare or unusual subgroups.

The goal is therefore to distinguish **random teacher noise** from **systematic teacher failure**.

---

# 2. High-Level Experimental Principle

For each dataset, we will compare:

1. **Systematic corruption**
   - Invert all teacher maps belonging to a particular semantic class or subgroup.

2. **Matched random corruption**
   - Invert approximately the same number of teacher maps, but select those examples randomly from the full training set.

The most important comparison is therefore:

> **Systematic corruption vs. the same corruption budget distributed randomly.**

This lets us determine whether the *structure* of teacher failure matters independently of the total number of corrupted examples.

---

# 3. What Will Stay Fixed

The corruption experiment should change **only the teacher maps**.

For every corruption condition:

- use the same student architecture as the original R4RR experiment;
- use the same dataset split;
- use the same precomputed teacher maps before corruption;
- use the same teacher-map inversion function used in the original corruption study;
- use the same validation-selected R4RR hyperparameters;
- use the same Classify-then-Align schedule;
- use the same alignment start epoch;
- use the same alignment weight;
- use the same optimizer;
- use the same learning-rate schedule;
- use the same Phase-2 learning-rate multiplier;
- use the same total number of epochs;
- use the same checkpoint-selection procedure;
- use the same evaluation protocol;
- run **five independent random training seeds**.

## No retuning

We will **not** rerun Optuna or otherwise retune R4RR separately for each corruption condition.

This is important.

The goal is to measure the sensitivity of the already-selected R4RR configuration to teacher quality. If each corrupted condition were separately retuned, the experiment would instead measure how well the method could adapt after being told in advance what kind of teacher failure it would encounter.

Therefore:

> **Each dataset uses its already optimized R4RR hyperparameters for every systematic and random corruption condition.**

The only experimental variable is which teacher maps are inverted.

---

# 4. Corruption Type

## Use inversion only

The expanded systematic-failure study will use **teacher-map inversion**, matching the inversion operation already used in the original corruption stress test.

We will not repeat the systematic study with Gaussian blur.

The reason is that the new experiment is intended to isolate one specific variable:

> **random error structure vs. systematic error structure**

Changing both the corruption pattern and corruption type would make the experiment harder to interpret.

The original blur condition already tests whether R4RR can tolerate spatially diffuse teacher supervision.

The new study instead focuses on a stronger scenario in which the teacher provides an **actively misleading spatial target**.

---

# 5. Experimental Design by Dataset

---

# 5.1 DecoyMNIST

## Motivation

DecoyMNIST provides a simple and controlled setting for testing class-conditional teacher failure.

The evaluation groups correspond directly to the ten digit classes:

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

Because the digit classes are approximately balanced, corrupting one entire class affects roughly one tenth of the training set.

This gives a natural systematic corruption unit:

> **one digit class**

## Systematic conditions

Run ten systematic corruption conditions:

1. invert all teacher maps for digit 0;
2. invert all teacher maps for digit 1;
3. invert all teacher maps for digit 2;
4. invert all teacher maps for digit 3;
5. invert all teacher maps for digit 4;
6. invert all teacher maps for digit 5;
7. invert all teacher maps for digit 6;
8. invert all teacher maps for digit 7;
9. invert all teacher maps for digit 8;
10. invert all teacher maps for digit 9.

Each condition is trained with five random seeds.

## Random control

Add one matched random corruption condition.

The cleanest implementation is:

- determine the average or representative fraction of the training set occupied by one digit class;
- randomly select approximately that same fraction of training examples;
- invert those teacher maps;
- keep the selected corrupted subset fixed across all five model-training seeds.

Because DecoyMNIST is approximately class-balanced, this should be around **10% random inversion**.

If exact class counts differ slightly, the precise random fraction should be documented. We do not need ten separate random controls unless class imbalance turns out to be large enough to matter.

## Main DecoyMNIST question

> If roughly 10% of teacher maps are wrong, is R4RR more affected when all of those errors belong to the same digit class than when the same amount of error is randomly distributed?

## Reporting

For every condition report:

- overall or standard DecoyMNIST summary metric used in the paper;
- worst-group accuracy;
- per-digit accuracy;
- mean and standard deviation across five seeds.

The per-digit breakdown is important because it may reveal whether corruption produces a targeted failure on the affected class.

For example:

> If only the teacher maps for digit 7 are inverted, does performance on digit 7 degrade much more than performance on the remaining digits?

---

# 5.2 RedMeat

## Motivation

RedMeat contains five target classes and evaluates robustness at the class level.

This gives a particularly clean systematic corruption design because each semantic class can be corrupted independently.

The main advantage of rotating through all five classes is that we do not need to justify why a particular class was selected.

There is no cherry-picking.

Instead, we directly characterize teacher sensitivity across **every RedMeat class**.

## Systematic conditions

Run five systematic corruption conditions:

1. invert all teacher maps for RedMeat class 1;
2. invert all teacher maps for RedMeat class 2;
3. invert all teacher maps for RedMeat class 3;
4. invert all teacher maps for RedMeat class 4;
5. invert all teacher maps for RedMeat class 5.

In the final implementation and paper, use the actual class names rather than generic numbering.

Each condition is trained with five random seeds.

## Random control

If the five classes are balanced as expected, corrupting one complete class corresponds to approximately **20% of the training set**.

Therefore add:

- **20% random inversion**

The random subset should be sampled once using a fixed corruption seed and reused across all five training seeds.

If the class counts are not exactly equal, either:

1. use the exact average class fraction and state it clearly; or
2. use a size-matched random subset for each distinct class count if imbalance is non-negligible.

The preferred simple design is one 20% random condition if the classes are effectively balanced.

## Main RedMeat question

> When roughly 20% of teacher maps are corrupted, is systematic failure on one semantic food class more harmful than randomly distributed corruption at the same rate?

## Why this is better than corrupting two classes

Corrupting two classes would require explaining:

- why those two were selected;
- whether the result depends on the chosen pair;
- whether another pair would behave differently.

Rotating through all five single-class corruption conditions avoids this issue completely.

It also keeps the interpretation simple:

> one semantic class fails at a time.

## Reporting

For every condition report:

- MeanGroup accuracy;
- WorstGroup accuracy;
- accuracy for each of the five RedMeat classes;
- mean and standard deviation across five seeds.

The individual class accuracies may reveal whether teacher failure is localized.

For example:

> When the teacher is corrupted only for Prime Rib, does Prime Rib accuracy fall disproportionately while the other classes remain stable?

That would be a highly interpretable result.

---

# 5.3 Waterbirds95

## Motivation

Waterbirds95 is the most direct response to the reviewer's concern because it contains all four combinations of bird class and background:

1. landbird on land;
2. landbird on water;
3. waterbird on land;
4. waterbird on water.

This allows us to test failure concentrated within a **specific semantic-context subgroup**, rather than only an entire class.

This is particularly valuable because the four groups include both:

- dominant bias-aligned groups;
- rare bias-conflicting groups.

The experiment can therefore show whether teacher quality is especially important for certain class-background combinations.

## Systematic conditions

Run four systematic corruption conditions:

1. invert every teacher map for landbird-on-land training examples;
2. invert every teacher map for landbird-on-water training examples;
3. invert every teacher map for waterbird-on-land training examples;
4. invert every teacher map for waterbird-on-water training examples.

Each condition is trained with five random seeds.

## Why one generic 25% random control is not sufficient

The four Waterbirds95 groups are not equally sized.

Therefore, corrupting all examples in one group may affect a very different fraction of the training set than corrupting all examples in another group.

If we compared every systematic condition only against a fixed 25% random corruption baseline, we would confound:

- the semantic structure of the corruption;
- the number of corrupted maps.

For example, if one systematic condition corrupts only a small rare group and another corrupts a large majority group, their performance differences could simply reflect corruption volume.

## Size-matched random controls

For each Waterbirds95 systematic condition:

1. count the number of training examples in that group, call it `N_group`;
2. sample exactly `N_group` examples uniformly at random from the complete Waterbirds95 training set;
3. invert the teacher maps for those randomly selected examples;
4. save that random subset;
5. use the same corrupted subset for all five model-training seeds.

This produces four matched random controls:

- random subset matched to landbird-on-land count;
- random subset matched to landbird-on-water count;
- random subset matched to waterbird-on-land count;
- random subset matched to waterbird-on-water count.

## Main Waterbirds95 question

> Holding the number of corrupted teacher maps fixed, does concentrating those errors within one class-background subgroup cause more damage than distributing the same errors randomly?

This is arguably the strongest direct answer to the reviewer.

## Additional question

The experiment also lets us ask:

> Are systematic errors on rare bias-conflicting groups disproportionately harmful?

The bias-conflicting groups are especially interesting because they provide evidence that the background shortcut is unreliable.

If corruption of those rare groups causes a larger effect than expected from their size, that would be an informative mechanistic result.

## Reporting

For every condition report:

- MeanGroup accuracy;
- WorstGroup accuracy;
- landbird-on-land test accuracy;
- landbird-on-water test accuracy;
- waterbird-on-land test accuracy;
- waterbird-on-water test accuracy;
- mean and standard deviation across five seeds.

The four individual group accuracies are important.

They let us distinguish:

- targeted degradation in the corrupted group;
- class-wide degradation;
- broader collapse across multiple groups.

---

# 5.4 Waterbirds100

## Motivation

Waterbirds100 is fully biased during training.

The training distribution contains the dominant class-background combinations, while the bias-conflicting combinations are absent.

Therefore, the natural systematic corruption unit is not one of four training groups.

It is:

> **one entire bird class**

This tests a stronger teacher failure mode in the most shortcut-dependent Waterbirds setting.

## Systematic conditions

Run two systematic corruption conditions:

1. invert all landbird teacher maps;
2. invert all waterbird teacher maps.

Each condition is trained with five random seeds.

## Random controls

The two classes may not contain the same number of training samples.

Therefore use a size-matched random control for each class.

For the landbird corruption condition:

1. count the number of landbird training examples;
2. randomly select the same number of Waterbirds100 training examples from the complete training set;
3. invert those maps;
4. keep the selected random subset fixed across all five training seeds.

For the waterbird corruption condition, repeat the same process using the waterbird class count.

This creates:

- all-landbird systematic corruption;
- landbird-count matched random corruption;
- all-waterbird systematic corruption;
- waterbird-count matched random corruption.

## Main Waterbirds100 question

> Under perfect shortcut correlation, is an entire class of systematically wrong teacher maps more harmful than the same number of wrong maps distributed randomly across training examples?

This is a particularly strong stress test because Waterbirds100 provides no counter-bias evidence during training.

---

# 6. Global Inversion Will Not Be Included

We considered adding an extreme condition in which **every teacher map is inverted**.

Scientifically, this could be interesting because it would test what happens when the spatial teacher becomes globally adversarial.

However, we will **not include global inversion in the Round 2 paper**.

## Reason

The reviewer did not ask whether R4RR survives a maximally adversarial teacher.

The reviewer asked whether the existing random corruption study captures **systematic failures on specific classes or domains**.

The class- and subgroup-specific experiments answer that concern directly.

Global inversion introduces a different question:

> What happens when every spatial supervision signal is intentionally wrong?

That condition is less realistic and could distract from the main result.

It could also create an unnecessarily confusing narrative if a large performance collapse is interpreted superficially as general fragility rather than the expected behavior of a teacher-guided alignment method under globally adversarial supervision.

Therefore:

> **Do not include global inversion in the planned Round 2 corruption study.**

It may still be run privately later if useful for mechanistic investigation, but it is outside the scope of the reviewer-response experiment.

---

# 7. Random Corruption Subset Handling

Random controls should be carefully constructed so that the only stochasticity across the five reported runs comes from model training.

For every random-control condition:

1. choose a fixed corruption-selection seed;
2. sample the random training indices once;
3. save the selected indices to disk;
4. reuse the exact same corrupted subset for all five model-training seeds.

Do **not** resample the corrupted examples independently for each training seed.

Otherwise the reported standard deviation would mix together:

- model-training randomness;
- corruption-subset randomness.

Keeping the corruption mask fixed makes the experiment easier to interpret.

---

# 8. Seed Protocol

Every condition should be run with **five random training seeds**, matching the paper's main reporting protocol.

Preferably use the same five seed values already used in the original R4RR evaluations.

For example:

```text
seed_1
seed_2
seed_3
seed_4
seed_5
```

Use the actual existing seed values from the project.

The same five seeds should be reused across:

- clean R4RR;
- systematic corruption;
- matched random corruption.

This permits paired comparisons across conditions if desired.

---

# 9. Hyperparameter Protocol

## Important rule

**No corruption condition is retuned.**

For each dataset, load the hyperparameters already selected for the original optimized R4RR model.

### DecoyMNIST

Use the existing validation-selected DecoyMNIST R4RR configuration.

### RedMeat

Use the existing validation-selected RedMeat R4RR configuration.

### Waterbirds95

Use the existing validation-selected Waterbirds95 R4RR configuration.

### Waterbirds100

Use the existing validation-selected Waterbirds100 R4RR configuration.

This includes all dataset-specific values such as:

- learning rate;
- alignment coefficient;
- alignment-start epoch;
- Phase-2 learning-rate multiplier;
- batch size;
- weight decay;
- optimizer configuration;
- total epochs;
- checkpoint-selection rule;
- any other R4RR-specific training parameters.

The corruption experiment should not alter them.

---

# 10. Evaluation Metrics

The new systematic-corruption experiment will focus entirely on **classification performance**.

We will **not use Pointing Game for this stress test**.

The purpose is to keep the analysis tightly focused on the reviewer concern:

> Does systematic teacher failure affect downstream robustness differently from random teacher noise?

Adding localization metrics would introduce another layer of interpretation that is not necessary for this question.

## Core metrics

Report the same classification metrics used elsewhere in the paper.

For Waterbirds and RedMeat:

- MeanGroup accuracy;
- WorstGroup accuracy.

For DecoyMNIST:

- the corresponding aggregate metrics already used in the paper;
- worst-group accuracy where applicable.

## Per-group / per-class results

Also report the individual group or class accuracies.

### DecoyMNIST

Report accuracy for all ten digit classes.

### RedMeat

Report accuracy for all five target classes.

### Waterbirds95

Report accuracy for:

- landbird on land;
- landbird on water;
- waterbird on land;
- waterbird on water.

### Waterbirds100

Report the four standard Waterbirds test groups even though only the aligned groups exist in the training distribution.

These detailed results are important because they reveal whether systematic teacher corruption causes:

- localized failure;
- class-wide failure;
- background-specific failure;
- broader degradation.

---

# 11. Complete Experimental Matrix

## DecoyMNIST

| Condition | Type | Five seeds? |
|---|---|---:|
| Clean R4RR | Reference | Yes |
| Random ~10% inversion | Random control | Yes |
| Digit 0 inverted | Systematic | Yes |
| Digit 1 inverted | Systematic | Yes |
| Digit 2 inverted | Systematic | Yes |
| Digit 3 inverted | Systematic | Yes |
| Digit 4 inverted | Systematic | Yes |
| Digit 5 inverted | Systematic | Yes |
| Digit 6 inverted | Systematic | Yes |
| Digit 7 inverted | Systematic | Yes |
| Digit 8 inverted | Systematic | Yes |
| Digit 9 inverted | Systematic | Yes |

New corruption conditions excluding clean baseline: **11**

New training runs: **55**

---

## RedMeat

| Condition | Type | Five seeds? |
|---|---|---:|
| Clean R4RR | Reference | Yes |
| Random ~20% inversion | Random control | Yes |
| Class 1 inverted | Systematic | Yes |
| Class 2 inverted | Systematic | Yes |
| Class 3 inverted | Systematic | Yes |
| Class 4 inverted | Systematic | Yes |
| Class 5 inverted | Systematic | Yes |

Replace `Class 1` through `Class 5` with the actual class names in the implementation and paper.

New corruption conditions excluding clean baseline: **6**

New training runs: **30**

---

## Waterbirds95

| Condition | Type | Five seeds? |
|---|---|---:|
| Clean R4RR | Reference | Yes |
| Landbird-on-land inverted | Systematic | Yes |
| Matched random for LB-land count | Random control | Yes |
| Landbird-on-water inverted | Systematic | Yes |
| Matched random for LB-water count | Random control | Yes |
| Waterbird-on-land inverted | Systematic | Yes |
| Matched random for WB-land count | Random control | Yes |
| Waterbird-on-water inverted | Systematic | Yes |
| Matched random for WB-water count | Random control | Yes |

New corruption conditions excluding clean baseline: **8**

New training runs: **40**

---

## Waterbirds100

| Condition | Type | Five seeds? |
|---|---|---:|
| Clean R4RR | Reference | Yes |
| All landbird maps inverted | Systematic | Yes |
| Landbird-count matched random inversion | Random control | Yes |
| All waterbird maps inverted | Systematic | Yes |
| Waterbird-count matched random inversion | Random control | Yes |

New corruption conditions excluding clean baseline: **4**

New training runs: **20**

---

# 12. Approximate Total Compute

Excluding clean baselines that already exist:

- DecoyMNIST: 55 runs
- RedMeat: 30 runs
- Waterbirds95: 40 runs
- Waterbirds100: 20 runs

**Total: 145 new training runs**

Of these:

- 55 are the cheaper DecoyMNIST / LeNet-style runs;
- 90 are Waterbirds or RedMeat ResNet-style runs.

There is **no hyperparameter optimization**, so the compute cost is limited to direct five-seed training runs.

If necessary, existing clean R4RR five-seed results can be reused rather than rerun, provided the training code and evaluation protocol remain identical.

---

# 13. Primary Comparisons

The paper should not present this as dozens of isolated corruption numbers.

The results should be organized around a few clear comparisons.

## Comparison 1: Random vs. systematic class failure

Use:

- DecoyMNIST;
- RedMeat.

Question:

> Is class-conditional teacher corruption more damaging than randomly distributing the same amount of corruption?

---

## Comparison 2: Random vs. systematic subgroup failure

Use:

- Waterbirds95.

Question:

> Is corruption concentrated within a specific class-background subgroup more damaging than an equal number of randomly corrupted maps?

This is the most direct test of the reviewer's concern about systematic failures tied to classes or domains.

---

## Comparison 3: Entire-class teacher failure under perfect bias

Use:

- Waterbirds100.

Question:

> When the training data contains no bias-conflicting evidence, how much more harmful is an entire class of systematically wrong teacher supervision than an equal-sized random corruption?

---

# 14. What Counts as a Good Result?

A good experiment does **not** require R4RR to remain unaffected under every systematic corruption condition.

The experiment is successful if it clearly characterizes the behavior.

Several outcomes are scientifically useful.

---

## Outcome A: Systematic corruption is substantially worse than matched random corruption

This would show:

> R4RR is tolerant to isolated teacher mistakes but more sensitive to correlated teacher failures.

This directly validates the reviewer's concern and allows us to characterize the limitation quantitatively.

---

## Outcome B: Systematic corruption is only slightly worse than matched random corruption

This would be a strong robustness result.

It would suggest that R4RR remains stable even when teacher errors are concentrated within coherent semantic subsets.

---

## Outcome C: Some classes or groups are much more sensitive than others

This would indicate that teacher quality matters unevenly across the data distribution.

For example:

- rare bias-conflicting Waterbirds groups may be especially important;
- certain RedMeat classes may rely more heavily on teacher guidance;
- specific digits may be unusually robust or fragile.

This is an informative characterization rather than a negative result.

---

## Outcome D: The corrupted class or subgroup degrades selectively

This would be especially interpretable.

For example:

- corrupting one RedMeat class primarily damages that class;
- corrupting waterbird-on-land supervision primarily damages waterbird-on-land test accuracy.

This would demonstrate that teacher-map failures have targeted downstream consequences.

---

## Outcome E: Corruption produces broader degradation

This would indicate that local teacher errors can alter the learned representation more globally.

Again, this is useful information.

The purpose is to understand the dependency, not to force a particular outcome.

---

# 15. Interpretation Boundaries

We should be careful not to overclaim.

The experiment does **not** establish that:

- the chosen inversion operation perfectly reproduces natural VLM failure;
- every real-world systematic teacher failure will behave the same way;
- R4RR is robust to arbitrary adversarial teachers.

Instead, it establishes a controlled stress test:

> When actively misleading teacher maps are distributed either randomly or systematically according to semantic structure, how does downstream robustness change?

This is a much stronger and more precise statement.

---

# 16. Suggested Paper Narrative

The new corruption section should be concise in the main paper.

The core story can be:

> Our original analysis corrupted a randomly selected 15% of teacher maps and showed that R4RR was stable to moderate teacher noise. To address whether this robustness extends to structured VLM failures, we additionally corrupt complete semantic classes or class-context groups while keeping the total corruption budget matched to random controls. Across DecoyMNIST and RedMeat we rotate corruption through every class; on Waterbirds95 we corrupt each of the four bird-background groups; and on Waterbirds100 we corrupt each bird class. All experiments use the original validation-selected R4RR hyperparameters without retuning and are averaged over five random seeds.

Then describe the actual result.

The key point should be:

> **The new experiment separates sensitivity to the amount of teacher error from sensitivity to the structure of that error.**

---

# 17. Suggested Supplementary Presentation

The complete results will likely fit best in the supplementary material.

## DecoyMNIST table

Rows:

- clean;
- random control;
- digit 0 corrupted;
- ...
- digit 9 corrupted.

Columns:

- aggregate accuracy;
- worst-group accuracy;
- digit 0 accuracy;
- ...
- digit 9 accuracy.

If that becomes too wide, split the per-digit accuracies into a secondary table.

---

## RedMeat table

Rows:

- clean;
- 20% random;
- each of five class corruption conditions.

Columns:

- MeanGroup;
- WorstGroup;
- five individual class accuracies.

---

## Waterbirds95 table

Rows:

- clean;
- each systematic group corruption;
- each matched random control.

Columns:

- MeanGroup;
- WorstGroup;
- LB-land;
- LB-water;
- WB-land;
- WB-water.

For readability, systematic conditions and their corresponding matched-random controls should be placed adjacent to each other.

---

## Waterbirds100 table

Rows:

- clean;
- landbird systematic;
- landbird-count random;
- waterbird systematic;
- waterbird-count random.

Columns:

- MeanGroup;
- WorstGroup;
- LB-land;
- LB-water;
- WB-land;
- WB-water.

---

# 18. Recommended Visual Summary

If page space permits, a compact figure could summarize the main effect.

For each systematic condition, plot something like:

> **Performance change relative to its matched random control**

For example:

```text
Δ WorstGroup = WorstGroup(systematic) - WorstGroup(matched random)
```

Interpretation:

- near zero: structure of corruption does not matter much;
- negative: systematic corruption is more harmful than random corruption;
- positive: systematic corruption is unexpectedly less harmful.

This would allow many conditions to be summarized compactly while the full absolute results remain in tables.

This figure is optional.

The absolute classification results should still be reported somewhere.

---

# 19. Implementation Details

## Step 1: Reuse existing clean teacher maps

Start from the exact teacher-map files used in the original optimized R4RR runs.

Do not regenerate teacher maps unless necessary for reproducibility.

---

## Step 2: Build corruption index files

Create explicit saved index files for every systematic and random condition.

Example structure:

```text
corruption_indices/
    decoy/
        digit_0.npy
        digit_1.npy
        ...
        digit_9.npy
        random_10pct.npy

    redmeat/
        class_baby_back_ribs.npy
        class_filet_mignon.npy
        class_pork_chop.npy
        class_prime_rib.npy
        class_steak.npy
        random_20pct.npy

    waterbirds95/
        landbird_land.npy
        random_match_landbird_land.npy
        landbird_water.npy
        random_match_landbird_water.npy
        waterbird_land.npy
        random_match_waterbird_land.npy
        waterbird_water.npy
        random_match_waterbird_water.npy

    waterbirds100/
        landbird.npy
        random_match_landbird.npy
        waterbird.npy
        random_match_waterbird.npy
```

Use the actual class labels and project naming conventions.

---

## Step 3: Apply inversion before training

For each condition:

1. load clean teacher maps;
2. load the saved corruption indices;
3. apply the existing inversion function only to those indices;
4. leave all remaining maps unchanged;
5. save or expose the corrupted map set to the training pipeline.

---

## Step 4: Run five seeds

For each corruption configuration:

```text
for seed in FIVE_EXISTING_SEEDS:
    train R4RR
    evaluate best validation-selected checkpoint
    save full per-group/per-class metrics
```

No hyperparameter search.

---

## Step 5: Aggregate results

For every reported metric calculate:

- mean across five seeds;
- standard deviation across five seeds.

Keep the individual seed values as well for debugging and possible significance analysis.

---

# 20. Sanity Checks Before Launching Full Runs

Before submitting all jobs, manually verify several corrupted maps from every dataset.

For each condition confirm:

- only intended examples are corrupted;
- all intended examples are corrupted;
- non-target examples remain unchanged;
- inversion matches the previous corruption implementation exactly;
- teacher-map normalization remains valid after inversion;
- filenames / indices still align with the correct training examples.

For random controls confirm:

- number of corrupted maps exactly matches the intended budget;
- random indices are unique;
- random indices are sampled from the full training set;
- saved indices remain fixed across training seeds.

For Waterbirds confirm:

- subgroup labels are correct;
- class and background metadata have not been swapped.

---

# 21. Reproducibility Information to Save

For every experiment save:

- dataset;
- systematic target class/group;
- total training-set size;
- number of corrupted maps;
- corruption percentage;
- corruption-selection seed;
- exact corruption index file;
- training seed;
- hyperparameter configuration;
- checkpoint selected;
- MeanGroup;
- WorstGroup;
- all per-group or per-class accuracies.

This will make the supplementary table generation straightforward.

---

# 22. Final Experiment Summary

The full Round 2 teacher-map corruption study is:

### DecoyMNIST

- systematically invert each digit class one at a time;
- compare against approximately 10% random inversion;
- five seeds;
- fixed optimized DecoyMNIST R4RR hyperparameters.

### RedMeat

- systematically invert each of the five classes one at a time;
- compare against approximately 20% random inversion;
- five seeds;
- fixed optimized RedMeat R4RR hyperparameters.

### Waterbirds95

- systematically invert each of the four class-background groups one at a time;
- compare each against an exactly size-matched random corruption condition;
- five seeds;
- fixed optimized Waterbirds95 R4RR hyperparameters.

### Waterbirds100

- systematically invert all landbird maps;
- compare against an equal-count random corruption;
- systematically invert all waterbird maps;
- compare against an equal-count random corruption;
- five seeds;
- fixed optimized Waterbirds100 R4RR hyperparameters.

### Explicitly excluded

- no hyperparameter retuning;
- no systematic blur sweep;
- no Pointing Game for this corruption study;
- no global 100% inversion condition in the Round 2 paper.

---

# 23. Core Message

The final experiment should let us answer the reviewer with a much stronger statement than the original 15% random corruption test:

> **R4RR does not require a perfectly accurate teacher for every training example. We further characterize this dependency by comparing randomly distributed teacher errors with systematic failures concentrated within semantic classes and class-context groups. This isolates whether the structure of teacher error, rather than simply its frequency, determines downstream robustness.**

That is the central purpose of the study.

The goal is not to claim that systematic teacher errors never hurt.

The goal is to show **how much they hurt, where they hurt, and whether they are meaningfully more damaging than an equivalent amount of random teacher noise.**
