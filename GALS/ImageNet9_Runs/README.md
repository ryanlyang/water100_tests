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
