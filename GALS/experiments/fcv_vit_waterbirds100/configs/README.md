# Configurations

`waterbirds100_vit_s16_first_study.yaml` locks the production online FCV
study. It defines 27 LR/weight-decay/seed training runs, 20 live epoch
candidates per run (540 total), the train-derived 80/20 holdout, FCV/control
interventions, privileged Oracle validation analysis, post-hoc test analysis,
and the 40 GiB storage envelope.

The model-selection boundary is fixed:

```text
train-derived holdout -> Vanilla and FCV selectors
original mixed validation -> privileged Oracle analysis only
test -> post-hoc evaluation/rank analysis only
```

At most three unique checkpoints remain per run, corresponding to the running
winners of biased validation accuracy, primary FCV, and Oracle balanced-group
validation. A replacement transaction may briefly stage one fourth file so an
interruption cannot destroy the previously committed winner; this is included
in the storage budget and pruned after commit. Token banks and ordinary epoch checkpoints are
node-local temporary artifacts. Once global selections are frozen, a hashed,
restartable cleanup keeps only the one to three unique global primary winners;
all other local winners are deleted because every epoch's test evidence has
already been persisted independently.

The 40 GiB hard cap is checked against current allocated bytes plus a locked
5 GiB worst-case allowance covering all eight concurrent writers. A new epoch
is refused if that projected peak would exceed the cap. Before releasing the
array, the real smoke additionally extrapolates its separately measured
evidence, compact-index, checkpoint, resume, and intervention-plan sizes to all
540 candidates with a 2x safety factor; this full-campaign projection must stay
below the 35 GiB launch guard. Its slowest complete online epoch is also
projected across all 20 epochs with a 1.5x factor and must fit the locked
seven-day run limit.

Do not change the split seed, candidate grid, selector formulas, or evaluation
rules after observing test results. A protocol change requires a new
`study.protocol_version` and a new output directory.
