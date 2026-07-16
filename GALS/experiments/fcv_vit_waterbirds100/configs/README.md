# Configurations

`waterbirds100_vit_s16_first_study.yaml` is the locked configuration for the
first ViT-FCV study.

The train-derived validation protocol is intentional:

```text
Waterbirds100 train split
  -> fixed, class-stratified 80% candidate-training subset
  -> fixed, class-stratified 20% biased validation subset
```

Both Vanilla and FCV select from the same candidate pool using only the 20%
holdout. The original mixed-group validation split is reserved for the
explicitly privileged Oracle selector. The test split remains evaluation-only.

Do not alter the holdout seed, primary selector weights, or candidate sweep
after examining test results. Any later protocol revision should increment
`study.protocol_version` and use a new output directory.

The Step 4 candidate pool is also locked here: fully fine-tuned pretrained
ViT-S/16, AdamW, bfloat16 autocast, two warmup epochs followed by cosine decay,
20 total epochs, random-resized-crop plus horizontal flip for training, and
256-resize/224-center-crop evaluation. Every epoch remains available through
successful Step 12 rank/scatter analysis; only then may non-selected
checkpoints be pruned.

Strict mode also locks the FCV construction: evidence/background thresholds
`0.60/0.10`, `keep_target` ambiguity handling, at least 20 safe background
patches, five globally sampled with-replacement opposite-context donors with
self-exclusion, and donor/control seeds `0/1`. Token extraction, FCV/control
chunking, Oracle inference, and final-test batch/worker settings are canonical
execution settings rather than tunable parameters.

Step 9 also locks the selector-analysis protocol. The main FCV score uses
equal 0.5 weights on original and opposite-context accuracy. Accuracy-weight
ablations use lambda values 0.25, 0.5, and 1.0; control normalization uses a
fixed coefficient of 1.0; probability retention uses epsilon `1e-8`; Oracle
validation uses float32 batches of 128; and exact ties use ascending candidate
ID only. These values are analysis definitions, not parameters selected using
Oracle or test performance.

Step 10 locks final selected-checkpoint evaluation to float32 with batch size
128. Each unique checkpoint is evaluated once, selector row order is preserved,
and a complete Step 9 selection table is mandatory before the test manifest can
be opened. These settings do not affect checkpoint selection.

Step 11 locks gap closure to test worst-group accuracy and to the ordinary
biased-validation, equal-weight FCV, and Oracle balanced-group selectors. The
reported ratio is raw rather than clipped. The complete 81-checkpoint test
pool is evaluated only after selection to provide an explicitly unfair upper
bound; those scores are prohibited from changing any selected checkpoint.
Every pool checkpoint must retain the exact path and SHA-256 frozen by Step 9.

Step 12 locks six rank-analysis criteria, including FCV stability as mean
counterfactual true-class probability. Spearman and Kendall tau-b use scores
oriented so higher is better; therefore biased validation loss is negated.
Top-k overlap is fixed at `k={1,5,10,25}`, and all score ties use candidate ID
ascending. Scatter generation and the post-hoc-only designation are also part
of the locked protocol.
