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

## Final five-seed evaluation

After the four non-teacher Optuna studies and the AFR grid are complete, submit
one resumable job per method:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash ImageNet9_Runs/submit_imagenet9_final_non_teacher_5seed.sh
```

This submits ERM, Upweight, ABN, ElRep, AFR, and CLIP-LR as six independent
jobs. For the first four methods, `summary.json` must certify all 50 completed
trials; its best validation-selected parameters are locked before seeds 0--4
are retrained. AFR preserves its native 33-by-5 validation grid independently
within every seed. CLIP-LR locks the selected `C`, extracts official-test RN50
features once, and repeats the fixed logistic-regression fit across the five
seeds to measure any solver variance.

Each seed's frozen checkpoint is evaluated on all eight official variants only
after validation selection. Stable results are written to
`logsImageNet9/final/<method>/main/`: `per_seed.csv` contains the individual
results and `summary.csv` contains their mean and population standard
deviation. BG-Gap is computed as Mixed-Same minus Mixed-Rand within each seed
before aggregation. Re-running the submitter safely resumes incomplete jobs.

After teacher-method sweeps reach 50 completed trials, submit their locked
five-seed evaluations with:

```bash
bash ImageNet9_Runs/submit_imagenet9_final_teacher_5seed.sh
```

The submitter checks every GALS and R4RR sweep summary, skips incomplete
studies, and launches one resumable job per complete variant. Re-running it
later picks up newly completed GALS studies and skips final variants already at
five seeds. Stable results are written under
`logsImageNet9/final/<variant>/main/` using the same per-seed and population
mean/std format as the non-teacher methods. Each job has a 12-hour walltime and
can be resubmitted to continue from completed seed artifacts.

For the explicitly exploratory five-seed evaluation of completed forward-KL
trial 13, use the same runner with an isolated variant name. Its parameters are
loaded and verified directly from the sweep's `trials.csv`, and its results do
not replace the validation-selected R4RR final:

```bash
sbatch --job-name=in9ft_r4rr_t13 \
  --export=ALL,VARIANT=r4rr_trial13 \
  ImageNet9_Runs/run_imagenet9_final_teacher_5seed.sbatch
```

To transfer every ramp-selected R4RR configuration, including trial 13, to a
fixed `kl_increment=0` final retraining schedule, run:

```bash
bash ImageNet9_Runs/submit_imagenet9_final_r4rr_klincr0_5seed.sh
```

These exploratory results use separate `<variant>_klincr0` directories. Their
contracts record both the sweep's `kl_lambda/10` increment policy and the final
zero-increment override; they do not overwrite the schedule-matched finals.

## CLIP ViT zero-shot Backgrounds Challenge evaluation

The following test-only job evaluates ViT-B/16 and ViT-B/32 through the same
OpenCLIP implementation with OpenAI weights on all eight official Backgrounds
Challenge variants. Both architectures use their native CLIP preprocessing and
the same frozen two-template foreground prompt ensemble. No reconstructed
validation images or official test results are used for prompt selection or
tuning. The job uses the `r4rr-weclip` environment because that environment
contains the pinned OpenCLIP implementation used by the WeCLIP+ teacher.

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
sbatch ImageNet9_Runs/run_imagenet9_clip_vit_zeroshot.sbatch
```

Stable outputs are written under
`/home/ryreu/guided_cnn/logsImageNet9/clip_vit_zeroshot_openai/`:

- `variant_results.csv`: overall and macro-class accuracy for each variant.
- `per_class_results.csv`: class-level accuracy and support.
- `robustness_summary.csv`: Original, mixed-variant average, foreground-only,
  background-only, no-foreground, and worst-variant summaries.
- `evaluation_contract.json`: model, weights, prompt, manifest, and split audit.

Each model also has a completed JSON under `models/`, allowing the same job to
resume without repeating a model whose eight variant evaluations are complete.

The corresponding deterministic OpenAI CLIP RN50 zero-shot evaluation uses the
same prompt ensemble and official-test-only protocol, with an independent
output contract:

```bash
sbatch ImageNet9_Runs/run_imagenet9_clip_rn50_zeroshot.sbatch
```

Its stable outputs are written under
`logsImageNet9/clip_rn50_zeroshot_openai/`.

To compare the other frozen teacher-family VLMs used in the paper, run:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash ImageNet9_Runs/setup_imagenet9_openclip_zs_env.sh
sbatch ImageNet9_Runs/run_imagenet9_openclip_siglip2_zeroshot.sbatch
```

This evaluates OpenCLIP `ViT-B-32` with `laion2b_s34b_b79k` weights and
`ViT-B-16-SigLIP2-256` with `webli` weights. The class prompts, official test
variants, metrics, and resumable output format are identical to the OpenAI
CLIP evaluation. Stable outputs are written to
`logsImageNet9/openclip_laion_siglip2_zeroshot/`. The isolated `openclip-zs`
environment pins `open_clip_torch==2.31.0`, the first release containing the
SigLIP2 model registry, without changing the older OpenCLIP version required by
the WeCLIP+ training environment. It also pins a compatible Transformers and
tokenizers stack and verifies the SigLIP2 tokenizer during setup.

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
`suppress_outside` mechanics are fixed. The Slurm walltime is six days and
the driver stops admitting trials after 142 hours. Re-submit the same command
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

The final exhaustive GALS variant replaces the ViT teacher maps with OpenAI
CLIP RN50 Grad-CAM maps while retaining the RRR input-gradient objective. Run a
180-image diagnostic first:

```bash
sbatch ImageNet9_Runs/run_imagenet9_gals_rn50_maps.sbatch
```

After inspecting the per-class QA sheets, generate all 45,405 maps with a
resumable 46-task array:

```bash
sbatch --partition=tier3 --time=3-00:00:00 --array=0-45 \
  --export=ALL,MODE=full,CHUNK_SIZE=1000 \
  ImageNet9_Runs/run_imagenet9_gals_rn50_maps.sbatch
```

The maps use two class-conditioned prompts, OpenAI CLIP RN50, and Grad-CAM at
`layer4.2.relu`. They are stored under
`data/imagenet9/gals_maps/clip_rn50_gradcam_v1/` with a contract distinct from
the ViT maps. After the full map audit passes, launch the separate 50-trial RRR
sweep:

```bash
sbatch ImageNet9_Runs/run_imagenet9_gals_rn50_sweep.sbatch
```

This study uses the same RRR search space and fixed training setup as the ViT
map variant, but persists independently under
`logsImageNet9/sweeps/gals_rn50/main/`. Its walltime is 40 hours; the driver
stops admitting trials after 38 hours, and the same submission command resumes
the SQLite study until it reaches 50 completed trials. Samples whose RN50
prompt maps are all zero remain in training for classification but are excluded
from the RRR auxiliary loss. Epoch logs report both valid and all-zero teacher
map counts.

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

After full training completes, export maps for all 45,405 Original training
images. First run a five-image inference smoke test against the full checkpoint:

```bash
sbatch --partition=debug --time=04:00:00 --array=0 \
  --export=ALL,CHUNK_SIZE=5,OUTPUT_ROOT=/home/ryreu/guided_cnn/data/imagenet9/r4rr_teacher/weclipplus_clip_dino_v1/inference/smoke/val \
  ImageNet9_Runs/run_imagenet9_r4rr_weclip_maps.sbatch
```

After confirming that its final record reaches `[MAP] 5/5` with no traceback,
export and automatically audit the complete set:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash ImageNet9_Runs/submit_imagenet9_r4rr_weclip_maps.sh
```

The first job is a resumable 46-task array with 1,000 manifest-ordered images
per task (405 in the final task). Each task loads the fixed 29,000-iteration
checkpoint and preserves the established WeCLIP+ inference protocol: scales
1.0 and 1.5, horizontal-flip averaging, equal CLIP/DINO logits, and DenseCRF.
The exporter disables WeCLIP+'s auxiliary CAM/PAR label branch because these
labels are not consumed when producing the fused-logit teacher maps; this
avoids allocating full-resolution PAR tensors for large ImageNet source files
without changing the saved logits.
Maps are written as RGB VOC-colormap PNGs under
`r4rr_teacher/weclipplus_clip_dino_v1/inference/full/val/prediction_cmap/`.
Existing readable maps with the correct source dimensions are reused after an
interruption.

The dependent CPU audit runs only after every array task succeeds. It requires
exactly one map per training `sample_id`, rejects missing, extra, malformed,
wrong-sized, and unknown-color maps, and confirms that no validation or
official test data entered the inference contract. It also reports expected
foreground, background, unexpected-class, and empty-map rates globally and by
class under `inference/full/audit/`.

After the full map audit reports `status: ok`, launch the 50-trial R4RR sweep:

```bash
sbatch ImageNet9_Runs/run_imagenet9_r4rr_sweep.sbatch
```

The sweep uses the established R4RR space: `attention_epoch` in `0..19`,
log-scaled `kl_lambda` in `[1,500]`, `base_lr` and `classifier_lr` in
`[1e-5,5e-2]`, and log-scaled `lr2_mult` in `[0.1,3]`. Training uses 20 epochs,
an ImageNet-pretrained ResNet-50 student, forward KL, and Original validation
macro class accuracy for selection. The image crop and horizontal flip are
applied jointly to each teacher mask. Maps with no target-class foreground,
including maps made empty by augmentation, remain classification-only samples.
The 44-hour job persists its Optuna SQLite study and per-trial CSV under
`logsImageNet9/sweeps/r4rr/main/`; resubmitting the same command continues until
50 trials have completed.

To run the four fixed-loss alignment ablations with the same maps, objective,
training setup, and five-dimensional search space, submit:

```bash
bash ImageNet9_Runs/submit_imagenet9_r4rr_alignment_sweeps.sh
```

This creates one independent 50-trial study for each of `reverse_kl`,
`jensen_shannon`, `squared_l2`, and `cosine`. Outputs are isolated under
`logsImageNet9/sweeps/r4rr_<loss>/main/`. Each job has a 44-hour walltime and
is resumable by rerunning the same submitter; completed trials remain in that
loss's SQLite study.

## Waterbirds-95 hyperparameter transfer

The cross-dataset transfer study uses the validation-selected Waterbirds-95
hyperparameters without retuning them on ImageNet-9. Standard 200-epoch
Waterbirds training exposes the model to `200 * 4,795 = 959,000` images; the
nearest ImageNet-9 schedule is 21 epochs, or `21 * 45,405 = 953,505` images.
R4RR therefore transfers attention epoch 109 to epoch 12 and uses its original
constant forward-KL weight (`kl_increment=0`). ImageNet-9 validation macro
accuracy is used only to select a checkpoint within each training run. The
official Backgrounds Challenge variants remain evaluation-only.

Submit one resumable five-seed job for each trained method with:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash ImageNet9_Runs/submit_imagenet9_wb95_transfer_5seed.sh
```

This launches ERM, Upweight, ABN, ElRep, the WB95-selected GALS RRR variant
with ImageNet-9 ViT teacher maps, AFR, CLIP-LR, and R4RR. AFR transfers its
fixed `gamma=11` and `reg_coeff=0`; its 50-epoch source stage-one exposure is
matched by seven epochs over ImageNet-9's 80% stage-one subset. CLIP-LR
transfers `C=30.481669...` and does not have an epoch schedule. The complete
contract is in `configs/waterbirds95_hparam_transfer.yaml`, and results are
written under `logsImageNet9/transfer/waterbirds95/<method>/main/`.

After all eight jobs finish, verify five seeds per method and create one
comparison CSV with:

```bash
python ImageNet9_Runs/summarize_imagenet9_wb95_transfer.py \
  --run-root /home/ryreu/guided_cnn/logsImageNet9/transfer/waterbirds95
```

### Alignment-loss transfer

The four alternative R4RR losses can be transferred under the same protocol.
Each job resolves its completed 50-trial Waterbirds-95 sweep CSV, selects the
WB95 validation winner, scales that trial's attention epoch by cumulative image
exposure, and trains ImageNet-9 for 21 epochs with `kl_increment=0`. The source
CSV path, SHA-256, winning trial, source parameters, and transferred parameters
are frozen in the run contract. Submit all four five-seed studies with:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash ImageNet9_Runs/submit_imagenet9_wb95_alignment_transfer_5seed.sh
```

This launches reverse KL, Jensen--Shannon, squared L2, and cosine as four
independent resumable 18-hour jobs. Results are written under
`logsImageNet9/transfer/waterbirds95_alignment/<loss>/main/`. After completion,
compare them with the existing forward-KL transfer using:

```bash
python ImageNet9_Runs/summarize_imagenet9_wb95_alignment_transfer.py \
  --transfer-root /home/ryreu/guided_cnn/logsImageNet9/transfer
```

## WB95-transfer RISE Pointing Game

The official Backgrounds Challenge release includes 4,050 binary foreground
masks under `fg_mask/val`. The localization protocol applies the model's native
evaluation crop jointly to each image/mask pair, targets the ground-truth class,
and uses one shared deterministic bank of 2,000 GALS-style RISE masks. The
primary evaluation is the official `original` test variant.

Submit one resumable job per method and seed (40 jobs) with:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash ImageNet9_Runs/submit_imagenet9_wb95_transfer_rise.sh
```

The submitter first audits all 4,050 image/mask joins and native dimensions.
Each GPU job appends completed image batches to a contract-locked JSONL file,
so resubmission continues interrupted evaluations. Results are stored under
`logsImageNet9/pointing_game_rise_wb95_transfer/`.

The same foreground masks can be audited and evaluated on the three composited
variants with:

```bash
VARIANTS="mixed_same mixed_rand mixed_next" \
  bash ImageNet9_Runs/submit_imagenet9_wb95_transfer_rise.sh
```

Aggregate the primary five-seed results after all jobs finish:

```bash
python ImageNet9_Runs/summarize_imagenet9_wb95_transfer_rise.py \
  --run-root /home/ryreu/guided_cnn/logsImageNet9/pointing_game_rise_wb95_transfer \
  --variants original
```

For all four variants, pass
`--variants original,mixed_same,mixed_rand,mixed_next`. The summary reports
overall, macro-class, and worst-class Pointing Game accuracy, the foreground
area random baseline, classification accuracy, and saliency mass inside the
foreground. Standard deviations are population standard deviations over the
five training seeds.

## Systematic R4RR teacher corruption

The ImageNet-9 training split is exactly class-balanced at 5,045 examples per
class. This study therefore compares nine class-conditional teacher failures
against one shared, exactly count-matched random control. Every condition
inverts 5,045 of the 45,405 training maps using `1-M` followed by sum
normalization. The selected maps are persisted in a checksummed manifest and
shared across training seeds 0--4; validation and official test data remain
unchanged.

The study locks the validation-selected ImageNet-9 forward-KL R4RR parameters,
uses `kl_increment=0`, and performs no corruption-specific retuning. Each of
the ten Slurm jobs runs five seeds sequentially and resumes at the seed
boundary. Preview or submit all conditions with:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
DRY_RUN=1 bash ImageNet9_Runs/submit_imagenet9_r4rr_systematic_corruption_all.sh
bash ImageNet9_Runs/submit_imagenet9_r4rr_systematic_corruption_all.sh
```

After all jobs finish, aggregate five-seed means, population standard
deviations, and seed-paired class-minus-random differences with:

```bash
python ImageNet9_Runs/summarize_imagenet9_r4rr_systematic_corruption.py \
  --run-root /home/ryreu/guided_cnn/logsImageNet9/r4rr_systematic_teacher_corruption/imagenet9
```
