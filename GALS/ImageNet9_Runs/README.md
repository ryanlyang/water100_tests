# ImageNet-9 Preparation

This directory contains the isolated ImageNet-9 Backgrounds Challenge pipeline.
The preparation stage reads RIT's shared, extracted ImageNet-2012 images and a
local copy of the public localization annotations. It does not copy ImageNet
images into the user's home allocation.

## Research-compute preparation

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
mkdir -p /home/ryreu/guided_cnn/logsImageNet9
sbatch ImageNet9_Runs/run_prepare_imagenet9.sbatch
```

The default output is:

```text
/home/ryreu/guided_cnn/data/imagenet9/
├── metadata/reconstructed_original_bbox1_v1/
│   ├── manifest.csv
│   ├── eligible_candidates.csv
│   ├── rejections.csv
│   ├── official_test_manifest.csv
│   ├── audit_summary.json
│   └── summary.json
└── train_source/reconstructed_original_bbox1_v1/
    ├── train/<IN-9 class>/*.JPEG -> /shared/rc/datasets/imagenet2012/...
    └── val/<IN-9 class>/*.JPEG   -> /shared/rc/datasets/imagenet2012/...
```

The official training archive is unavailable at its published Dropbox URL.
This preparation is therefore explicitly recorded as a deterministic
reconstruction. It implements the published WordNet mapping, availability of a
bounding-box annotation, exactly one bounding box, class balancing, and source
ID exclusion against the official test release. It does not claim to reproduce
the unavailable archive's exact filename sample or unpublished random seed.

The builder writes `audit_summary.json` and the complete candidate/rejection
manifests before selecting a split. If the requested 5,045 training plus 450
validation images per class are not available after filtering, the job stops
with per-class deficits recorded in that audit. This intentionally avoids
silently changing the documented training size or using challenge test images
for validation.

The vendored `assets/in_to_in9.json` and `assets/in9_classes.txt` files come
from the official MadryLab Backgrounds Challenge repository.

## Loader and model-selection audit

The experiment uses the reconstructed Original validation split only for
hyperparameter and checkpoint selection. The fixed objective is macro class
accuracy. Official `original`, `mixed_same`, `mixed_rand`, `mixed_next`, and
diagnostic variants are final evaluation data and must not enter Optuna,
pruning, checkpoint selection, or prompt selection. This policy is recorded in
`configs/original_validation_protocol.yaml`.

After preparation, submit the loader audit:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
sbatch ImageNet9_Runs/run_audit_imagenet9_loaders.sbatch
```

The audit verifies all train, validation, and official-variant paths and class
counts, checks train/validation disjointness, and decodes representative images
through the deterministic evaluation transform. The manifest loader preserves
`sample_id` in every item; this is the key used to join training images to
teacher maps in the next stage.

## Non-teacher baseline sweeps

The shared baseline trainer currently covers ERM, Upweight, ABN, and ElRep.
Every method uses an ImageNet-pretrained ResNet-50, batch size 96, 20 epochs,
SGD, weight decay `1e-5`, and checkpoint selection by Original validation macro
class accuracy. The official Backgrounds Challenge variants are not loaded by
the trainer or Optuna driver.

Search spaces preserve the corresponding main-experiment contracts:

| Method | Tuned parameters |
|---|---|
| ERM | `base_lr`, `classifier_lr` in `[1e-5, 5e-2]` log; `momentum` in `[0.85, 0.95]` |
| Upweight | `base_lr`, `classifier_lr` in `[5e-5, 1e-1]` log |
| ABN | Upweight LR space plus `abn_cls_weight` in `[1e-2, 1e2]` log |
| ElRep | ERM LR space plus `theta1` in `[1e-5, 1e-2]` log and `theta2` in `[1e-6, 1e-3]` log |
| CLIP-LR | `C` in `[1e-2, 1e2]` log; all other logistic-regression settings fixed |

Because the reconstructed IN-9 training split has exactly 5,045 examples in
every class, inverse-frequency Upweight produces nine weights equal to one.
Its training loss is therefore mathematically identical to ERM here, although
it is retained as an independently tuned comparator for protocol consistency.

Run one-epoch smoke studies first:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash ImageNet9_Runs/submit_imagenet9_non_teacher.sh smoke
```

After those pass, launch the 50-completed-trial, four-day sweeps:

```bash
bash ImageNet9_Runs/submit_imagenet9_non_teacher.sh sweep
```

The full studies use stable paths under
`/home/ryreu/guided_cnn/logsImageNet9/sweeps/<method>/main/`. Each trial runs in
a fresh Python process. Optuna state is persisted in `optuna.sqlite3`, with
`trials.csv` and `summary.json` refreshed after every attempt. If maintenance
or the wall-time limit interrupts a study, rerun the same `sweep` command; it
continues until 50 trials have completed. The stored contract hash prevents a
study from resuming with changed data, objective, epochs, or search ranges.

CLIP-LR is submitted by the same wrapper but uses its own driver. It caches
frozen OpenAI CLIP RN50 features, then tunes only logistic-regression `C`.
CLIP-ZS is not tuned. AFR is also submitted by the wrapper, but retains its
native procedure: stage 1 trains on a deterministic 80% partition, stage 2
reweights the remaining 20%, and validation macro class accuracy selects among
the 33-by-5 `gamma`/`reg_coeff` grid. Its stage-1 checkpoint, embedding cache,
and each completed stage-2 configuration persist independently across jobs.
GALS and R4RR are launched only after their teacher maps have been generated
and audited.

## GALS ViT map quality control

ImageNet-9 GALS uses OpenAI CLIP ViT-B/32 transformer relevance maps. Each
training image receives two maps using its known coarse class and the templates
`an image of a/an ...` and `a photo of a/an ...`. The nine concepts follow the
benchmark class names, except `instrument` is written as `musical instrument`
to avoid the non-visual meaning. No prompt names a background or context.

Generate the fixed diagnostic subset before producing all maps:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
sbatch ImageNet9_Runs/run_imagenet9_gals_vit_maps.sbatch
```

This deterministically selects 20 training images per class (180 total), writes
maps under
`/home/ryreu/guided_cnn/data/imagenet9/gals_maps/clip_vit_b32_transformer_v1/`,
and creates one QA sheet per class under `qa/`. Inspect those sheets and freeze
the prompt contract before the full generation job.

After quality control passes, the complete 45,405-image run can be submitted as
a resumable 46-task array with four concurrent GPUs:

```bash
sbatch --partition=tier3 --time=4-00:00:00 --array=0-45%4 \
  --export=ALL,MODE=full,CHUNK_SIZE=1000 \
  ImageNet9_Runs/run_imagenet9_gals_vit_maps.sbatch
```

Every map is named by the unique ImageNet source `sample_id`, retains the GALS
`unnormalized_attentions` tensor schema, and is recorded in a per-shard CSV.
Existing valid files are reused after interruption. Only Original training
images are selected; validation and all official variants are excluded.

After the full map audit passes, launch the resumable GALS sweep:

```bash
sbatch ImageNet9_Runs/run_imagenet9_gals_sweep.sbatch
```

The study maximizes Original validation macro-class accuracy and targets 50
completed Optuna trials. It sweeps `base_lr` and `classifier_lr` over
`[1e-5, 5e-2]` (log), `grad_weight` over `[1e3, 1e5]` (log), and
`grad_criterion` over `{L1,L2}`. ResNet-50 initialization, SGD momentum 0.9,
weight decay `1e-5`, 20 epochs, and the GALS `average_nonzero`/
`suppress_outside` mechanics are fixed. The Slurm walltime is three days and
the driver stops admitting trials after 70 hours. Re-submit the same command
to continue the stable `imagenet9_gals_main` SQLite study until 50 trials have
completed; validation and official Backgrounds Challenge variants are never
used as teacher-map inputs or robustness-selection targets.

The corresponding GALS Grad-CAM comparison uses the same map set, student,
training length, validation objective, persistence behavior, and three-day
walltime:

```bash
sbatch ImageNet9_Runs/run_imagenet9_gals_gradcam_sweep.sbatch
```

This separate 50-completed-trial study matches the exhaustive GALS comparison:
it sweeps `base_lr` over `[5e-4,5e-2]`, `classifier_lr` over `[1e-5,1e-3]`,
and `cam_weight` over `[1e-2,1e2]` (all log-scaled). It fixes the student map
to ground-truth-class Grad-CAM from ResNet-50 `layer4`, aggregates teacher
prompts with `average_nonzero`, and uses L1 map matching. Its stable state is
stored under `logsImageNet9/sweeps/gals_gradcam/main/`; re-submit the same
command after interruption to continue to 50 completed trials.

The GALS+ABN comparison is likewise isolated and resumable:

```bash
sbatch ImageNet9_Runs/run_imagenet9_gals_abn_sweep.sbatch
```

It uses the pretrained ABN ResNet-50 and keeps the auxiliary ABN
classification weight fixed at `1.0`. The ABN 14-by-14 spatial attention is
supervised by the same ViT maps with L1 `suppress_outside` loss. The 50-trial
study sweeps `base_lr` over `[1e-3,1e-1]`, `classifier_lr` over
`[1e-4,1e-2]`, and `abn_att_weight` over `[1e-2,1e2]` (all log-scaled), with
a three-day walltime and Original validation macro-class selection. Its stable
state is stored under `logsImageNet9/sweeps/gals_abn/main/`.

## R4RR VOC compatibility workspace

R4RR map generation uses the existing WeCLIP+ CLIP+DINO+CRF pipeline. Prepare
its VOC-compatible view without copying ImageNet images:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
sbatch ImageNet9_Runs/run_prepare_imagenet9_r4rr_voc_workspace.sbatch
```

The workspace places absolute symlinks named `<sample_id>.jpg` under
`VOCdevkit/VOC2012/JPEGImages/`, pointing to the shared ImageNet `.JPEG`
sources. `classes.txt` fixes the foreground order to `dog`, `bird`, `vehicle`,
`reptile`, `carnivore`, `insect`, `instrument`, `primate`, and `fish`. Each
class has complete VOC `+1/-1` files for both `train` and `val`. In this teacher
workspace, `val` intentionally aliases Original training IDs for map inference;
it does not contain the held-out Original validation split or official test
variants. The builder is resumable, refuses unexpected or mismatched links,
and writes a manifest, input contract, and full audit under
`voc_workspace/metadata/`.

The workspace also contains one lightweight VOC XML file per training image.
Each XML records only that image's coarse ImageNet-9 class and a full-image
extent. These are image-level labels, not localization annotations; they ensure
that WeCLIP+ generates CAMs only for the image's actual foreground class rather
than all nine classes.

Train the WeCLIP+ teacher after workspace preparation. Create its isolated
Python 3.8/PyTorch 2.0 environment once on the submit node:

```bash
bash ImageNet9_Runs/setup_imagenet9_r4rr_weclip_env.sh
```

The training runner uses `r4rr-weclip` by default. This is intentionally
separate from `gals_a100`: the vendored DINOv2 code requires PyTorch 2
scaled-dot-product attention, while the legacy experiment environment uses
PyTorch 1.12.

Then run a two-iteration GPU smoke test:

```bash
sbatch --partition=debug --time=04:00:00 --export=ALL,MODE=smoke \
  ImageNet9_Runs/run_imagenet9_r4rr_weclip_train.sbatch
```

Then launch the full training run:

```bash
sbatch ImageNet9_Runs/run_imagenet9_r4rr_weclip_train.sbatch
```

At batch size four, the dataset-scaled schedule is 29,000 iterations. Training
uses OpenCLIP's ViT-B/16 implementation with OpenAI weights and DINOv2
ViT-L/14-register refinement, matching the primary R4RR teacher family. The
full run writes one replaceable continuation state every 5,000 iterations and
a final model checkpoint under
`r4rr_teacher/weclipplus_clip_dino_v1/training/full/`. Re-submit the same full
command after interruption to continue from the saved model, optimizer,
iteration, and RNG state. Smoke and full states are isolated from one another.
