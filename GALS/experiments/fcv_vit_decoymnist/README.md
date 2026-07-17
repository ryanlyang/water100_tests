# DecoyMNIST ViT shortcut-susceptibility pilot

This pilot answers one question before FCV is transferred to DecoyMNIST:
does the same ImageNet-pretrained ViT-S/16 used by the Waterbirds study learn
the **unmodified** DecoyMNIST corner shortcut?

The source benchmark is not recolored, enlarged, or regenerated. Its original
28x28 grayscale PNGs are converted to RGB and resized directly to 224x224 for
ViT. No crop or flip is used because this diagnostic must not intermittently
remove the corner shortcut.

## Protocol

- Model: `vit_small_patch16_224.augreg_in21k_ft_in1k`
- Optimizer: AdamW
- Learning rates: `1e-5`, `3e-5`, `1e-4`
- Weight decay: `0.05`
- Seeds: `0`, `1`, `2`
- Epochs: 10, evaluated online at every epoch
- Candidate training data: 90% of the original biased training split
- Biased validation: one fixed, class-stratified 10% training holdout
- Test: the official reversed-decoy test split
- Checkpoints saved: none

For biased validation and reversed test, the pilot measures:

1. **Original:** unchanged image.
2. **Digit-only:** the detected 5x5 class-coded corner patch is erased.
3. **Patch-only:** every pixel except that corner patch is erased.

The detector validates the published encoding (`255-25*y` in train and
`25*y` in test) against source PNG pixels. It does not infer a more convenient
or strengthened shortcut.

An epoch passes the preregistered susceptibility diagnostic when:

- biased validation accuracy is at least 95%;
- biased-validation minus reversed-test accuracy is at least 10 points; and
- either patch erasure drops biased-validation accuracy by at least 10 points
  or patch-only biased-validation accuracy is at least 80%.

The aggregate recommendation proceeds only if every seed has at least one
passing epoch. All underlying metrics remain available regardless of this
summary gate.

## Tigris launch

From the GALS repository root on Tigris:

```bash
bash experiments/fcv_vit_decoymnist/scripts/submit_susceptibility_pilot.sh
```

The launcher submits a pretrained/data preflight, a nine-task GH200 array, and
an aggregation job with `afterok` dependencies. Outputs are written under:

```text
/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility
```

The concise final result is `pilot_summary.json`; the complete 90-candidate
matrix is `all_epoch_metrics.csv`.

