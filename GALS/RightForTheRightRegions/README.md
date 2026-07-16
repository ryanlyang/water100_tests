# Right for the Right Regions (R4RR)

This repository packages the full experiment stack used for R4RR:
- R4RR training and sweeps
- baseline models (Vanilla, Upweight, ABN, GALS variants, AFR, CLIP baselines)
- teacher map generation (WeCLIP+ and GALS attention pipelines)
- one-command reproduction wrappers for each dataset

The code is organized so you can either:
1. run full paper-facing reproductions from `pipelines/train_CNN`, or
2. run individual methods directly from `repro_runs`.

## Repository Layout

```text
RightForTheRightRegions/
├── README.md
├── READMEGals.md
├── requirements_runs.txt
├── requirements_weclip.txt
├── configs/
│   ├── waterbirds95_optimized_hparams.yaml
│   ├── waterbirds100_optimized_hparams.yaml
│   ├── redmeat_optimized_hparams.yaml
│   ├── decoymnist_hparams.yaml
│   └── r4rr_optimized_hparams.yaml
├── data/
│   ├── make_decoymnist_pngs.py
│   └── generate_waterbirds100.py
├── pipelines/
│   ├── generate_r4rr_maps/
│   ├── generate_gals_maps/
│   └── train_CNN/
├── repro_runs/
│   ├── r4rr/
│   │   ├── train/
│   │   ├── sweeps/
│   │   ├── ablations/
│   │   └── architecture_transfer/
│   ├── other_models/
│   ├── evaluation/
│   └── third_party/
│       ├── GALS/
│       ├── CDEP/
│       ├── afr/
│       └── group_DRO/
└── WeCLIPPlus/
```

## Environment Setup

You will typically use **two environments**:
- `runs` env: model training / sweeps / reproduction wrappers
- `weclip` env: WeCLIP+ teacher-map generation

### 1) Runs environment

```bash
conda create -n r4rr-runs python=3.10 -y
conda activate r4rr-runs

# Install torch/torchvision first for your CUDA, then:
pip install -r requirements_runs.txt

# Confirm the environment is complete:
python pipelines/check_r4rr_runs_env.py
```

### 2) WeCLIP+ environment

`WeCLIPPlus` dependencies are older and are usually most stable in Python 3.8.

```bash
conda create -n r4rr-weclip python=3.8 -y
conda activate r4rr-weclip
python -m pip install -U pip setuptools wheel

# IMPORTANT: install pydensecrf from conda-forge (avoid pip wheel build failures)
conda install -y -c conda-forge pydensecrf

# Install torch/torchvision for your CUDA first (recommended)
# Example (adjust for your CUDA/GPU):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
# On RTX 5090, use a torch build with sm_120 support.

# Then install remaining WeCLIP deps
pip install -r requirements_weclip.txt

# Note: `requirements_weclip.txt` uses `timm>=0.9,<1.1` to avoid `torch._six` errors with torch 2.x.

# Confirm the environment is complete:
python pipelines/check_r4rr_weclip_env.py
```

If `pydensecrf` still gets pulled by pip in your environment, run:

```bash
pip install -r requirements_weclip.txt --no-deps
pip install matplotlib==3.3.3 tqdm==4.46.1 omegaconf==2.0.0 numpy==1.23.5 'timm>=0.9,<1.1' Pillow==8.4.0 scikit_learn==1.0.1 tensorboard ftfy regex ttach lxml tensorflow colour open_clip_torch mmcv==1.3.17 mmcv_full==1.2.7
```

## Dataset Setup

The repo expects datasets under `data/` (or explicit paths passed to scripts).

### Waterbirds-95

Download from the official Group DRO release:
- https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz

Example:

```bash
mkdir -p data
cd data
wget https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
tar -xzf waterbird_complete95_forest2water2.tar.gz
```

Expected folder:

```text
data/waterbird_complete95_forest2water2/
```

### Waterbirds-100

Use the helper script in this repo:
- `data/generate_waterbirds100.py`

This wraps the Group DRO generation logic but takes CLI args (no hardcoded paths).

Required source datasets:
- CUB images (`CUB_200_2011.tgz`): https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
- CUB segmentations (`segmentations.tgz`): https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1
- Places365: http://places2.csail.mit.edu/download.html

Important Places365 note: that link may first show a legacy page with a "sign this form" prompt.
Submit that form to access the Places365 dataset downloads.
For WB100 generation in this repo, download the **high-resolution training images** (not the 256x256 small-image package), matching Group DRO conventions.

After extraction, you can either:
- point `--cub-dir` to the folder containing `images.txt`, `train_test_split.txt`, `images/`, and `segmentations/`, or
- provide segmentations separately with `--segmentations-dir`.

For Places:
- `--places-dir` can point to either the Places root or directly to `data_large/`.
- `categories_places365.txt` is read from this repo at `data/categories_places365.txt`.

Example:

```bash
python data/generate_waterbirds100.py \
  --cub-dir /path/to/CUB_200_2011 \
  --places-dir /path/to/data_large \
  --segmentations-dir /path/to/segmentations \
  --output-dir "$PWD/data"
```

Expected output:

```text
data/waterbird_1.0_forest2water2/
```

Reference docs/script from Group DRO are still included here if you want the original path:
- `repro_runs/third_party/group_DRO/README.md`
- `repro_runs/third_party/group_DRO/dataset_scripts/generate_waterbirds.py`

### RedMeat (Food-101 subset)

Food-101 download (official):
- https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

If you downloaded the tarball, extract it before running RedMeat prep / generation scripts:

```bash
tar -xzf food-101.tar.gz
```

Prepare the RedMeat subset from Food-101 with:

```bash
python data/prepare_redmeat_food101.py \
  --food101-root /path/to/food-101 \
  --output-dir "$PWD/data/food-101-redmeat" \
  --overwrite
```

Notes:
- `--food101-root` can point to either the extracted `food-101/` folder or its parent.
- Default classes are `prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon`.
- Default `--val-fraction` is `1/3`, giving `500 train / 250 val` per class from the official Food-101 train split.
- Use `--mode hardlink|copy|symlink` to control how files are materialized (default: `hardlink`).

This creates `all_images.csv` plus split folders. Typical layout:

```text
data/food-101-redmeat/
├── all_images.csv
├── train/<class>/*.jpg
├── val/<class>/*.jpg
├── test/<class>/*.jpg
└── split_images/
   ├── train/<class>/*.jpg
   └── val/<class>/*.jpg
```

### DecoyMNIST PNG conversion

Generate DecoyMNIST arrays + PNG layout via:

```bash
python data/make_decoymnist_pngs.py
```

This runs CDEP's `00_make_data.py` and writes:

```text
data/DecoyMNIST_png/
├── train/<digit>/*.png
└── test/<digit>/*.png
```

---

## Generate Teacher/Attention Maps

## A) R4RR teacher maps (WeCLIP+)

Scripts:
- `pipelines/generate_r4rr_maps/generate_pseudo_masks_waterbirds.py`
- `pipelines/generate_r4rr_maps/generate_pseudo_masks_redmeat.py`
- `pipelines/generate_r4rr_maps/generate_pseudo_masks_DecoyMNIST.py`

These scripts auto-detect WeCLIPPlus under either `<repo-root>/WeCLIPPlus` or `<repo-root>/code/WeCLIPPlus`.
Each script now uses its own isolated VOC workspace by default under:
`pipelines/generate_r4rr_maps/voc_workspaces/{waterbirds_<src-basename>,redmeat,decoymnist}`
so they can run concurrently without clobbering `JPEGImages`/`ImageSets`.
Use `--voc-workspace-root` to override.

### Waterbirds teacher maps

```bash
conda activate r4rr-weclip
python pipelines/generate_r4rr_maps/generate_pseudo_masks_waterbirds.py \
  --repo-root "$PWD" \
  --src-img-dir "$PWD/data/waterbird_complete95_forest2water2" \
  --class-name bird \
  --setup-data \
  --results-dir results_waterbirds95_r4rr
```

Waterbirds-100 command:

```bash
conda activate r4rr-weclip
python pipelines/generate_r4rr_maps/generate_pseudo_masks_waterbirds.py \
  --repo-root "$PWD" \
  --src-img-dir "$PWD/data/waterbird_1.0_forest2water2" \
  --class-name bird \
  --setup-data \
  --results-dir results_waterbirds100_r4rr
```

This uses a different default VOC workspace automatically.

### RedMeat teacher maps

Prerequisite: extract Food-101 (`tar -xzf food-101.tar.gz`) and run `data/prepare_redmeat_food101.py` first.

```bash
conda activate r4rr-weclip
python pipelines/generate_r4rr_maps/generate_pseudo_masks_redmeat.py \
  --repo-root "$PWD" \
  --split-images-dir "$PWD/data/food-101-redmeat/split_images" \
  --class-name meat \
  --setup-data \
  --results-dir results_redmeat_r4rr
```

### DecoyMNIST teacher maps (WeCLIP+ path)

```bash
conda activate r4rr-weclip
python pipelines/generate_r4rr_maps/generate_pseudo_masks_DecoyMNIST.py \
  --repo-root "$PWD" \
  --src-png-root "$PWD/data/DecoyMNIST_png" \
  --class-name digit \
  --setup-data \
  --results-dir results_decoy_r4rr
```

### Switching CLIP/DINO backbones in WeCLIP+

Use script flags to swap teacher backbones:
- `--clip-backend` (`openai`, `openclip`, `siglip2` depending on script)
- `--clip-model`
- `--clip-pretrained`
- `--dino-model`
- `--dino-fts-dim`
- `--dino-decoder-layers`

Default behavior note:
- Pseudo-mask generation scripts now default to OpenCLIP with OpenAI weights.
- If you explicitly pass `--clip-backend OpenCLIP` (or `openclip`) and omit `--clip-pretrained`, the loader auto-selects a LAION OpenCLIP checkpoint for that model.
- If you want OpenCLIP with OpenAI weights while still setting backend explicitly, add `--clip-pretrained openai`.

Example (RedMeat, explicit LAION OpenCLIP):

```bash
python pipelines/generate_r4rr_maps/generate_pseudo_masks_redmeat.py \
  --repo-root "$PWD" \
  --split-images-dir "$PWD/data/food-101-redmeat/split_images" \
  --class-name meat \
  --setup-data \
  --clip-backend OpenCLIP \
  --clip-model ViT-B-16 \
  --dino-model xcit_medium_24_p16 \
  --dino-fts-dim 512 \
  --dino-decoder-layers 2 \
  --results-dir results_redmeat_openclip_xcit
```

## B) GALS attention maps

Scripts:
- `pipelines/generate_gals_maps/run_generate_waterbirds_rn50_attentions_95_100_debug.py`
- `pipelines/generate_gals_maps/run_generate_waterbirds_vit_attentions_95_100.py`
- `pipelines/generate_gals_maps/run_generate_redmeat_rn50_attentions.py`
- `pipelines/generate_gals_maps/run_generate_redmeat_vit_attentions.py`
- `pipelines/generate_gals_maps/run_generate_decoymnist_rn50_attentions.py`
- `pipelines/generate_gals_maps/run_generate_decoymnist_vit_attentions.py`

### Waterbirds (RN50 + ViT)

```bash
conda activate r4rr-runs
python pipelines/generate_gals_maps/run_generate_waterbirds_rn50_attentions_95_100_debug.py
python pipelines/generate_gals_maps/run_generate_waterbirds_vit_attentions_95_100.py
```

### RedMeat (RN50 + ViT)

```bash
conda activate r4rr-runs
python pipelines/generate_gals_maps/run_generate_redmeat_rn50_attentions.py \
  --dataset-dir food-101-redmeat

python pipelines/generate_gals_maps/run_generate_redmeat_vit_attentions.py \
  --dataset-dir food-101-redmeat
```

### DecoyMNIST (RN50 Grad-CAM + ViT)

```bash
conda activate r4rr-runs
python pipelines/generate_gals_maps/run_generate_decoymnist_vit_attentions.py \
  --png-root "$PWD/data/DecoyMNIST_png" \
  --output-root "$PWD/data/DecoyMNIST_png/clip_vit_attention"
```

All scripts now run in subprocess chunks by default (`--chunk-size 1000`) to avoid long-run CUDA OOM from memory growth.  
Tune with `--chunk-size <N>` or disable chunking with `--chunk-size 0`.

---

## One-Command Reproduction Runs

These wrappers run each method once per dataset, using hyperparameters from `configs/*_optimized_hparams.yaml`, and write consolidated outputs.
For full (non-`recreate_r4rr_runs.py`) wrappers, this includes:
- GALS-family runs (`vanilla`, `upweight`, `abn`, map-based variants)
- `clip_lr`
- `clip_zs` (CLIP zero-shot baseline)
- `afr`
- R4RR variants

Scripts:
- `pipelines/train_CNN/recreate_waterbirds95_runs.py`
- `pipelines/train_CNN/recreate_waterbirds100_runs.py`
- `pipelines/train_CNN/recreate_redmeat_runs.py`
- `pipelines/train_CNN/recreate_decoymnist_runs.py`
- `pipelines/train_CNN/recreate_r4rr_runs.py` (R4RR-only, all datasets once)

All wrappers produce:

```text
logs/recreate/<dataset>_<timestamp>/
├── <method>/stdout.log
├── summary.csv
└── summary.json
```

ABN pretrained weights prerequisite (required by wrappers that include ABN methods):

`repro_runs/third_party/GALS/approaches/abn.py` expects:
`repro_runs/third_party/GALS/weights/resnet50_abn_imagenet.pth.tar`

Download and place it there before running recreate wrappers (this source provides the file as `model_best.pth.tar`, so rename it to the expected name):

```bash
mkdir -p "$PWD/repro_runs/third_party/GALS/weights"
python -m pip install -U gdown
gdown --folder --remaining-ok \
  "https://drive.google.com/drive/folders/1SRtzbnE-IpB5talp7PLNK1mzMV3UPQNV" \
  -O "$PWD/repro_runs/third_party/GALS/weights"
mv "$PWD/repro_runs/third_party/GALS/weights/model_best.pth.tar" \
   "$PWD/repro_runs/third_party/GALS/weights/resnet50_abn_imagenet.pth.tar"
```

If the `gdown` command fails (Google Drive rate limits/permission quirks), open the folder link directly in your browser:
`https://drive.google.com/drive/folders/1SRtzbnE-IpB5talp7PLNK1mzMV3UPQNV`
Then download `model_best.pth.tar` manually and rename/move it to:
`repro_runs/third_party/GALS/weights/resnet50_abn_imagenet.pth.tar`.



### R4RR-only (all datasets, one run each)

`pipelines/train_CNN/recreate_r4rr_runs.py` runs one R4RR training run for:
- Waterbirds-95
- Waterbirds-100
- RedMeat
- DecoyMNIST

It reads `configs/r4rr_optimized_hparams.yaml`, uses the dataset-appropriate training scripts
(DecoyMNIST uses its CDEP-style CNN runner), and writes:

```text
logs/recreate/r4rr_all_<timestamp>/
├── waterbirds95/stdout.log
├── waterbirds100/stdout.log
├── redmeat/stdout.log
├── decoymnist/stdout.log
├── summary.csv
└── summary.json
```

Basic run:

```bash
conda activate r4rr-runs
python pipelines/train_CNN/recreate_r4rr_runs.py
```

With explicit map paths:

```bash
conda activate r4rr-runs
python pipelines/train_CNN/recreate_r4rr_runs.py \
  --wb95-teacher-maps "$PWD/WeCLIPPlus/results_waterbirds95_r4rr/val/prediction_cmap" \
  --wb100-teacher-maps "$PWD/WeCLIPPlus/results_waterbirds100_r4rr/val/prediction_cmap" \
  --redmeat-teacher-maps "$PWD/WeCLIPPlus/results_redmeat_r4rr/val/prediction_cmap" \
  --decoy-teacher-maps "$PWD/WeCLIPPlus/results_decoy_r4rr/val/prediction_cmap"
```

### Waterbirds-95

Path meaning for these wrappers:
- `--r4rr-rn50-teacher-maps-dir`: GALS RN50 attention map directory.
- `--r4rr-vit-teacher-maps-dir`: GALS ViT attention map directory.
- `--weclip-prediction-cmap-dir`: shared WeCLIP+ pseudo-mask directory (`.../prediction_cmap`).
  By default this same path is used for `gals_our_masks`, `r4rr_optimized`, and Waterbirds R4RR ablations.
- `--gals-segmentation-dir`, `--r4rr-optimized-teacher-maps-dir`, and `--r4rr-ablation-teacher-maps-dir` are optional advanced overrides if you want different directories.
Legacy flag names are still accepted as aliases for backward compatibility.
When `--r4rr-rn50-teacher-maps-dir` / `--r4rr-vit-teacher-maps-dir` are provided, wrappers now forward those paths directly into GALS `DATA.ATTENTION_DIR`, so custom directory names are supported (for example `clip_rn50_attention_gradcam` vs `meat_clip_rn50_attention_gradcam`).

```bash
conda activate r4rr-runs
python pipelines/train_CNN/recreate_waterbirds95_runs.py \
  --r4rr-rn50-teacher-maps-dir <PATH_TO_GALS_RN50_ATTENTION_MAPS> \
  --r4rr-vit-teacher-maps-dir <PATH_TO_GALS_VIT_ATTENTION_MAPS> \
  --weclip-prediction-cmap-dir <PATH_TO_WECLIP_PREDICTION_CMAP>
```

### Waterbirds-100

```bash
conda activate r4rr-runs
python pipelines/train_CNN/recreate_waterbirds100_runs.py \
  --r4rr-rn50-teacher-maps-dir <PATH_TO_GALS_RN50_ATTENTION_MAPS> \
  --r4rr-vit-teacher-maps-dir <PATH_TO_GALS_VIT_ATTENTION_MAPS> \
  --weclip-prediction-cmap-dir <PATH_TO_WECLIP_PREDICTION_CMAP>
```

### RedMeat

```bash
conda activate r4rr-runs
python pipelines/train_CNN/recreate_redmeat_runs.py \
  --dataset-path "$PWD/data/food-101-redmeat" \
  --r4rr-rn50-teacher-maps-dir <PATH_TO_GALS_RN50_ATTENTION_MAPS> \
  --r4rr-vit-teacher-maps-dir <PATH_TO_GALS_VIT_ATTENTION_MAPS> \
  --weclip-prediction-cmap-dir <PATH_TO_WECLIP_PREDICTION_CMAP>
```

### DecoyMNIST

DecoyMNIST uses separate map paths:
- `--gals-vit-teacher-maps-dir`: GALS ViT `.pth` attention maps.
- `--decoy-r4rr-teacher-maps-dir`: WeCLIP+ `prediction_cmap` image masks for Decoy R4RR.

```bash
conda activate r4rr-runs
python pipelines/train_CNN/recreate_decoymnist_runs.py \
  --png-root "$PWD/data/DecoyMNIST_png" \
  --gals-vit-teacher-maps-dir "$PWD/data/DecoyMNIST_png/clip_vit_attention" \
  --decoy-r4rr-teacher-maps-dir "$PWD/WeCLIPPlus/results_decoy_r4rr/val/prediction_cmap"
```

Tip: add `--dry-run` to any wrapper to print commands without executing training.

---

## Running Individual Methods Directly

For method-level runs, go to `repro_runs/`.

## R4RR

### Train
- `repro_runs/r4rr/train/r4rr_waterbirds.py`
- `repro_runs/r4rr/train/r4rr_redmeat.py`
- `repro_runs/r4rr/train/r4rr_decoy_fixed.py`

Examples:

```bash
python repro_runs/r4rr/train/r4rr_waterbirds.py \
  "$PWD/data/waterbird_complete95_forest2water2" \
  <TEACHER_MAP_PATH> \
  --attention_epoch 109 --kl_lambda 295.30 \
  --base_lr 4.82e-5 --classifier_lr 2.93e-3 --lr2_mult 0.409
```

```bash
python repro_runs/r4rr/train/r4rr_redmeat.py \
  "$PWD/data/food-101-redmeat" \
  <TEACHER_MAP_PATH> \
  --attention-epoch 2 --kl-lambda 11.44 \
  --base_lr 2.40e-3 --classifier_lr 2.33e-4 --lr2-mult 1.567
```

```bash
python repro_runs/r4rr/train/r4rr_decoy_fixed.py \
  --png-root "$PWD/data/DecoyMNIST_png" \
  --teacher-map-path "$PWD/WeCLIPPlus/results_decoy_r4rr/val/prediction_cmap" \
  --attention-epoch 7 --kl-lambda 495.61 --lr 0.001 --epochs 19
```

### Sweeps / ablations
- `repro_runs/r4rr/sweeps/r4rr_waterbirds_sweep.py`
- `repro_runs/r4rr/sweeps/r4rr_redmeat_sweep.py`
- `repro_runs/r4rr/ablations/r4rr_waterbirds_invert.py`
- `repro_runs/r4rr/ablations/r4rr_waterbirds_joint.py`

### Architecture transfer: ViT students

ViT student architecture-transfer runners live in:

- `repro_runs/r4rr/architecture_transfer/`
- `repro_runs/r4rr/architecture_transfer/vit/`

These scripts cover Waterbirds95, Waterbirds100, and RedMeat. They include:
- base ViT ERM baselines (CE-only)
- R4RR LGM-style ViT guided runners
- fixed-seed evaluation scripts
- Optuna sweep scripts

See the internal docs for the full file map and commands:
- `repro_runs/r4rr/architecture_transfer/README.md`
- `repro_runs/r4rr/architecture_transfer/vit/README.md`

Example Waterbirds100 guided ViT run:

```bash
python repro_runs/r4rr/architecture_transfer/vit/r4rr/waterbirds_fixed_seeds.py \
  "$PWD/data/waterbird_1.0_forest2water2" \
  "$PWD/WeCLIPPlus/results_waterbirds100_r4rr/val/prediction_cmap" \
  --seed-start 0 \
  --n-seeds 5
```

Example RedMeat base ViT ERM run:

```bash
python repro_runs/r4rr/architecture_transfer/vit/baseline/redmeat_fixed_seeds.py \
  "$PWD/data/food-101-redmeat" \
  --seed-start 0 \
  --n-seeds 5
```

### Architecture transfer: MobileNetV2 students

MobileNetV2 student architecture-transfer runners live in:

- `repro_runs/r4rr/architecture_transfer/`
- `repro_runs/r4rr/architecture_transfer/mobilenetv2/`

These scripts mirror the ViT transfer layout for Waterbirds95, Waterbirds100, and RedMeat. They include:
- MobileNetV2 ERM baselines (CE-only)
- MobileNetV2 R4RR guided runners
- fixed-seed evaluation scripts
- Optuna sweep scripts

The MobileNetV2 implementation keeps the ResNet-style CAM interface by using the final convolutional feature map with a dropout-free GAP + linear classification head. DecoyMNIST is intentionally not included here, since its LeNet-style setup is handled separately.

See the internal docs for the full file map, implementation notes, and runnable commands:
- `repro_runs/r4rr/architecture_transfer/README.md`
- `repro_runs/r4rr/architecture_transfer/mobilenetv2/README.md`

## Qualitative Evaluation Utilities

Small CSV-driven utilities for saliency and localization checks live in:

- `repro_runs/evaluation/rise_saliency.py`
- `repro_runs/evaluation/pointing_game.py`

`rise_saliency.py` loads a ResNet-50 or MobileNetV2 checkpoint and writes RISE
maps for a manifest of curated images. `pointing_game.py` evaluates saliency
maps against binary masks and reports overall and per-group hit rates. See
`repro_runs/evaluation/README.md` for the manifest format and example commands.

## Other models

### Waterbirds
- `repro_runs/other_models/waterbirds/sweeps/gals_waterbirds_sweep.py`
- `repro_runs/other_models/waterbirds/sweeps/clip_lr_waterbirds_sweep.py`
- `repro_runs/other_models/waterbirds/sweeps/afr_waterbirds_sweep.py`
- `repro_runs/other_models/waterbirds/sweeps/elrep_waterbirds_sweep.py`
- `repro_runs/other_models/waterbirds/baselines/clip_zeroshot_waterbirds.py`

### RedMeat
- `repro_runs/other_models/redmeat/sweeps/gals_redmeat_sweep.py`
- `repro_runs/other_models/redmeat/sweeps/clip_lr_redmeat_sweep.py`
- `repro_runs/other_models/redmeat/sweeps/afr_redmeat_sweep.py`
- `repro_runs/other_models/redmeat/sweeps/elrep_redmeat_sweep.py`
- `repro_runs/other_models/redmeat/baselines/clip_zeroshot_redmeat.py`

### DecoyMNIST
- `repro_runs/other_models/decoymnist/train/upweight_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/train/abn_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/train/gals_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/train/afr_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/train/elrep_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/baselines/clip_lr_decoy_fixed.py`
- `repro_runs/other_models/decoymnist/baselines/clip_zeroshot_decoy.py`

---

## Hyperparameter Config Files

The canonical hparam files are:
- `configs/waterbirds95_optimized_hparams.yaml`
- `configs/waterbirds100_optimized_hparams.yaml`
- `configs/redmeat_optimized_hparams.yaml`
- `configs/decoymnist_hparams.yaml`
- `configs/r4rr_optimized_hparams.yaml`

The `pipelines/train_CNN/recreate_*` wrappers read these YAMLs directly.

---

## Practical Notes / Troubleshooting

- Use absolute paths for dataset and map folders when possible.
- If a method is missing required map paths in the wrapper scripts, it is marked as `skipped` in summary outputs.
- For CLIP+LR stability on some systems, prefer conservative solver settings (`l2:lbfgs`).
- For DecoyMNIST, avoid very high dataloader worker counts.
- For newer GPUs, verify your PyTorch build supports the GPU architecture before long runs.

---

## Citations / Upstream Dependencies

This repo vendors and builds on:
- GALS: `repro_runs/third_party/GALS`
- Group DRO: `repro_runs/third_party/group_DRO`
- AFR: `repro_runs/third_party/afr`
- CDEP: `repro_runs/third_party/CDEP`
- WeCLIPPlus: `WeCLIPPlus`

Please cite original works when using their components.

## Acknowledgements

We thank the authors and maintainers of [**WeCLIP**](https://github.com/zbf1991/WeCLIP), [**GALS**](https://github.com/spetryk/GALS), [**GroupDRO**](https://github.com/kohpangwei/group_DRO), [**AFR**](https://github.com/AndPotap/afr), and [**CDEP (deep-explanation-penalization)**](https://github.com/laura-rieger/deep-explanation-penalization?tab=readme-ov-file) for their foundational contributions. A substantial portion of this repository builds directly on their open-source code, ideas, and released tooling.
