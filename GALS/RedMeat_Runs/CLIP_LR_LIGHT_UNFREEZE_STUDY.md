# RedMeat CLIP-LR Full Visual-Encoder Fine-Tuning Study

This diagnostic tests whether RedMeat CLIP-LR benefits from preserving the
pretrained OpenAI CLIP RN50 representation.

For each seed, the complete OpenAI CLIP RN50 visual encoder is fine-tuned with
a temporary five-class head. This includes the convolutional stem, all four
residual stages, and attention-pooling module. The text encoder and CLIP's
logit scale remain frozen because the image-classification objective does not
use them. The temporary head is discarded at evaluation time.

At epochs `0,1,2,4,8,16`, the image encoder is frozen and the same
L2-normalized CLIP-LR pipeline is evaluated in two ways:

- `fixed_c`: hold the original optimized `C=1.329346323656201` fixed.
- `retuned_c`: at epochs greater than zero, retune only `C` with 25 Optuna TPE
  trials over `[1e-2,1e2]`, maximizing validation macro-class accuracy. Epoch
  zero uses the original optimized C and performs no new tuning.

Both protocols use L2 logistic regression with `lbfgs` and an intercept. Test
metrics are computed only for the fixed C and the validation-selected C, never
for candidate C values during tuning.

Submit five parallel seed jobs:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash RedMeat_Runs/submit_redmeat_clip_lr_light_unfreeze_all.sh
```

After all jobs finish:

```bash
python RedMeat_Runs/summarize_clip_lr_light_unfreeze_study.py \
  --run-root /home/ryreu/guided_cnn/logsRedMeat/clip_lr_rn50_full_visual_finetune \
  --seeds 0,1,2,3,4
```

For the secondary partial-adaptation variant, submit with
`UNFREEZE_SCOPE=layer4_attnpool` and a distinct `RUN_ROOT`.
