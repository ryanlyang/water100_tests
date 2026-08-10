# Tigris Research-Compute Handoff

Last updated: 2026-07-26

This document is the infrastructure and workflow handoff for continuing this
project in a new folder or with a new agent. It records the working Tigris
cluster setup, environment, paths, Slurm conventions, storage constraints,
dataset locations, and launch practices established during the Waterbirds and
DecoyMNIST Feature-Counterfactual Validation (FCV) studies.

This is an operational reference, not a substitute for an experiment's frozen
configuration or protocol document. Before changing or launching a study,
read its README, implementation plan, configuration, launcher, and Slurm files.
Where this document and an experiment's current checked-in configuration
disagree, investigate the difference rather than silently choosing one.

---

## 1. Quick reference

| Item | Current value |
|---|---|
| Login host | `tigris` / `tigris.rc.rit.edu` |
| Slurm account | `reu-aisocial` |
| Slurm partition | `tigris` |
| GPU request | `gpu:gh200:1` |
| Compute GPU | NVIDIA GH200 |
| Compute architecture | `aarch64` |
| Conda distribution | `/home/ryreu/miniforge3-aarch64` |
| Conda environment | `fcv_gh200` |
| Environment path | `/home/ryreu/miniforge3-aarch64/envs/fcv_gh200` |
| Python | `/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python` |
| Tigris repository root | `/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs` |
| Tigris GALS root | `/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS` |
| Waterbirds logs | `/home/ryreu/guided_cnn/logsWaterbird` |
| DecoyMNIST logs | `/home/ryreu/guided_cnn/logsMNIST` |
| RedMeat logs | `/home/ryreu/guided_cnn/logsRedMeat` |
| Zero-shot logs | `/home/ryreu/guided_cnn/logsZeroShot` |
| SpuCo/SpuCoDogs logs | `/home/ryreu/guided_cnn/logsSpuCo` |

Non-negotiable cluster rule: all new Tigris jobs must use
`--account=reu-aisocial`. Do not submit them under an old account.

---

## 2. What changed when the project moved to Tigris

The earlier experiments were run on the SPORC-side A100 cluster. Tigris is a
different execution environment:

- Tigris compute nodes use NVIDIA GH200 GPUs.
- The compute-node CPU architecture is `aarch64`, not x86-64.
- The working Python environment is built with aarch64 Miniforge.
- The Slurm partition is `tigris`, not `tier3`, `debug`, or another old
  SPORC partition.
- The GPU resource is `gpu:gh200:1`, not `gpu:a100:1`.
- Old x86 environments and wheels cannot be assumed to work.

Many historical `.sh` and `.sbatch` files in this repository still contain
SPORC-era directives or activate:

```text
/home/ryreu/miniconda3/envs/gals_a100
```

Those files are useful references, but they are **not automatically
Tigris-ready**. In particular,
`GALS/RedMeat_Runs/common_env.sh` still activates the old `gals_a100`
environment. Do not source it unchanged in a new GH200 job.

Port an old runner by checking all of the following:

1. account;
2. partition;
3. GPU type;
4. environment activation;
5. Python version compatibility;
6. repository and dataset paths;
7. output paths;
8. requested wall time;
9. checkpoint/storage behavior;
10. any compiled or architecture-specific dependency.

---

## 3. Confirmed GH200 environment

The environment that successfully ran the ViT FCV work is:

```text
/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
```

Confirmed versions on a GH200 compute node:

```text
Python       3.10.20
torch        2.11.0+cu130
torchvision  0.26.0+cu130
timm         1.0.28
CUDA built   13.0
cuDNN        91900
GPU compute capability 9.0
```

The Tigris driver reported CUDA 13.2. That is the driver's supported CUDA
level; `torch.version.cuda` is the CUDA version against which PyTorch was
built. The two values do not need to be identical.

Activate the environment on a login or compute node with:

```bash
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/fcv_gh200
```

The shorter interactive form also works:

```bash
source ~/miniforge3-aarch64/bin/activate
conda activate fcv_gh200
```

For production scripts, prefer the explicit absolute paths.

### 3.1 Verify the environment before a new campaign

Run this on an allocated GH200, not only on the login node:

```bash
python --version
which python
uname -m
nvidia-smi
python - <<'PY'
import torch
import torchvision
import timm

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("built CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("cuDNN:", torch.backends.cudnn.version())
PY
```

Expected architecture: `aarch64`. Expected device: NVIDIA GH200.

### 3.2 Dependency rules

- Do not reuse a local x86 `.venv` on Tigris.
- Do not use the old `gals_a100` environment for GH200 work.
- Do not assume a package has an aarch64 wheel. Test installation in the
  existing environment or a separate environment before changing production.
- Do not casually upgrade `torch`, `torchvision`, or `timm` in the shared
  working environment while campaigns are active.
- Record versions in every preflight and job log.
- Set `PYTHONNOUSERSITE=1` in jobs to avoid contamination from user-site
  packages.
- Tigris did not expose a useful `module spider` command in the tested shell,
  and neither Apptainer nor Singularity was found. Current work does not depend
  on either.

If the environment must be rebuilt, first capture the current state:

```bash
conda list -n fcv_gh200
/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python -m pip freeze
```

Do not invent a replacement environment from memory when these records and the
experiment preflight provide a reproducible reference.

---

## 4. Interactive GH200 allocation

Use an interactive allocation for environment checks, one-batch smoke tests,
and GPU debugging:

```bash
srun \
  --account=reu-aisocial \
  --partition=tigris \
  --gres=gpu:gh200:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=00:15:00 \
  --pty bash -l
```

Then activate the environment:

```bash
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/fcv_gh200
```

Do not run training on the login node. A login-node import/path check is fine;
CUDA tests and material compute belong in an allocation.

---

## 5. Repository layout and Git workflow

### 5.1 Tigris checkout

The active checkout is:

```text
/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
```

The GALS directory is:

```text
/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
```

An older checkout was moved to:

```text
/home/ryreu/guided_cnn/waterbirds/old_Waterbird_Runs
```

Do not accidentally edit or launch from `old_Waterbird_Runs`.

The replacement checkout was cloned from:

```text
https://github.com/ryanlyang/water100_tests.git
```

Do not assume that this remote or branch will remain unchanged forever.
Confirm the active repository before work:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
git rev-parse --show-toplevel
git remote -v
git branch --show-current
git status --short
git log -1 --oneline
```

### 5.2 Safe update sequence

Use:

```bash
git status --short
git pull --ff-only
git log -1 --oneline
```

If the worktree is dirty, inspect and preserve the changes. Do not reset or
overwrite them merely to make a pull succeed.

Important Slurm behavior: the submitted batch script is captured by Slurm, but
the Python modules and configuration it imports are normally read from the
checkout when the job actually starts. Pulling or editing the checkout while
queued/running jobs depend on it can create a mixed-code campaign. For a
scientific campaign:

1. reach a clean commit;
2. record `git rev-parse HEAD`;
3. submit;
4. avoid changing that checkout until the campaign is complete, or use a
   separate checkout/worktree for new development.

The FCV launchers write the commit to a submission receipt. New launchers
should do the same.

### 5.3 Never commit generated experiment data

The top-level repository ignores:

```text
a_download/
download_logs/
```

That is not sufficient protection for every possible output directory. Before
committing, always inspect:

```bash
git status --short
git diff --stat
git diff --check
```

Do not commit:

- downloaded datasets;
- `.pt`, `.pth`, `.ckpt`, or optimizer states;
- token banks or embeddings;
- generated `.npy`/`.npz` files unless they are deliberately tiny fixtures;
- run logs;
- downloaded result folders;
- model caches;
- large figures or archives.

GitHub rejects individual files above 100 MB and warns above 50 MB. A file
deleted after being committed still exists in Git history until the commit or
history is corrected.

---

## 6. Canonical dataset and teacher-map paths

Paths in this section were established on Tigris. Check them with `test -d`
or `test -f` before launching; storage can be moved independently of code.

### 6.1 Waterbirds-95

#### Live Tigris paths

Dataset root:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2
```

Metadata:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2/metadata.csv
```

Current OpenCLIP-LAION + DINOvIT teacher evidence maps used by the recent
R4RR runs:

```text
/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap
```

**This `prediction_cmap` directory is the Waterbirds-95 VLM-mask root on
Tigris.** It is intentionally outside the Waterbirds-95 image directory. A
metadata row is joined to its VLM mask by `img_filename`, not by dataframe row
number, class label, split-local position, or dataset iteration order. The
complete dataset-relative image path and image stem are encoded in the PNG
filename as described under “Resolve a teacher map from an image name” below.
The standard producer creates maps for official metadata splits `0` and `1`
(train plus validation), not split `2` (test).

Historical WeCLIP+ teacher-map tree:

```text
/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
```

The current and historical teacher-map roots are different experimental
artifacts. Do not substitute one for the other because a filename happens to
match. Freeze the chosen root in the experiment configuration and receipt.

Waterbirds-95 is the partially biased version: its training split contains the
dominant label/background combinations plus a small number of counter-bias
examples. The metadata file, not a recursive directory scan, defines which
examples belong to train, validation, and test.

### 6.2 Waterbirds-100

#### Live Tigris paths

Dataset root:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
```

Metadata:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/metadata.csv
```

Current OpenCLIP-LAION + DINOvIT teacher evidence maps:

```text
/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
```

**This `prediction_cmap` directory is the Waterbirds-100 VLM-mask root on
Tigris.** As with Waterbirds-95, the join key is the metadata row's
`img_filename`. It is not a positional join. The literal `val` component in
the WeCLIP+ output path names the producer's evaluation output. Its `val.txt`
contains official Waterbirds train plus validation rows (`split in {0, 1}`),
so `val/prediction_cmap` is expected to cover both of those splits. The
standard producer excludes official test rows (`split == 2`). A live directory
may contain stale or separately generated extras, so measured coverage still
must be audited; those extras do not change the producer contract.

Historical WeCLIP+ teacher-map tree:

```text
/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
```

The production Waterbirds-100 FCV config is the source of truth:

```text
GALS/experiments/fcv_vit_waterbirds100/configs/waterbirds100_vit_s16_first_study.yaml
```

Waterbirds-100 is the completely biased training version. In its source
training split, target and context are locked together (`y == place`). The
original validation and test splits contain analysis information that must not
silently become selector-visible. In the first FCV study, both ordinary
validation and FCV selection use one shared, deterministic 80/20,
class-stratified split of the original training split. The original mixed
validation split is Oracle-only and the test split is evaluation-only. Reuse
the manifest generated by the FCV preflight rather than independently
resplitting the data in each run.

#### Shared Waterbirds metadata schema

Both datasets are loaded from `metadata.csv`. The relevant columns are:

| Column | Meaning |
| --- | --- |
| `img_filename` | Image path relative to the dataset root. |
| `y` | Target: `0` = landbird, `1` = waterbird. |
| `place` | Context: `0` = land, `1` = water. |
| `split` | Official split: `0` = train, `1` = validation, `2` = test. |

The four analysis groups use `group = 2 * y + place`:

| Group | Name |
| --- | --- |
| `0` | `Land_on_Land` |
| `1` | `Land_on_Water` |
| `2` | `Water_on_Land` |
| `3` | `Water_on_Water` |

For group-agnostic model selection, keep `place`, `group`, and `group_name` out
of selector-visible manifests. They may be retained in a separately labeled
analysis-only manifest.

#### What “mask” can mean in this repository

There are three spatial-artifact families around the Waterbirds data. They
must not be treated as interchangeable:

1. **R4RR/FCV teacher evidence maps.** The `prediction_cmap` directories listed
   above contain WeCLIP+ outputs refined with DINOvIT. Recent R4RR and FCV
   experiments use these maps. They are generated teacher evidence, not
   automatically human ground-truth segmentations.
2. **CUB segmentation annotations.** The generic GALS Waterbirds loader expects
   the CUB-style segmentation tree at:

   ```text
   /home/ryreu/guided_cnn/waterbirds/CUB_200_2011/segmentations
   ```

   This tree mirrors the CUB species/image hierarchy. Check that the directory
   and the required files exist on the live cluster before using it; merely
   having `CUB_200_2011` is not a coverage guarantee.
3. **Saved CLIP attention tensors.** Directories named
   `clip_vit_attention` or `clip_rn50_attention_gradcam` under a dataset root
   contain method-specific attention artifacts, commonly `.pth` files. They
   are inputs to particular GALS variants and are neither PNG teacher maps nor
   CUB segmentation masks.

Always record which family, exact root, generation method, and checksum or Git
commit an experiment uses. A paper phrase such as “teacher masks” is not
sufficient provenance for code.

#### Resolve images from metadata

Do not infer official splits from folder names or enumerate species folders
with `ImageFolder` when `metadata.csv` is present. A minimal loader should
follow this pattern:

```python
from pathlib import Path
import pandas as pd
from PIL import Image

SPLIT_ID = {"train": 0, "val": 1, "test": 2}

data_root = Path("/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2")
metadata = pd.read_csv(data_root / "metadata.csv")
rows = metadata.loc[metadata["split"] == SPLIT_ID["val"]].copy()

def resolve_image(relative_name: str) -> Path:
    relative = Path(str(relative_name).strip().lstrip("/"))
    candidates = (data_root / relative, data_root / "images" / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing Waterbirds image: {relative_name}")

row = rows.iloc[0]
image_path = resolve_image(row["img_filename"])
with Image.open(image_path) as source:
    image = source.convert("RGB")

label = int(row["y"])
context = int(row["place"])       # analysis-only when the protocol is group-agnostic
group = 2 * label + context       # analysis-only when the protocol is group-agnostic
```

The repository already implements this more defensively in:

```text
GALS/experiments/fcv_vit_waterbirds100/src/fcv/waterbirds_metadata.py
```

That implementation also hashes source images, resolves teacher maps, creates
the Waterbirds-100 train holdout, and emits public versus analysis-only
manifests. Reuse it for FCV-family studies instead of making a second loader.
The generic GALS dataset implementation is:

```text
GALS/datasets/waterbirds.py
```

The standalone R4RR metadata loader is:

```text
GALS/RightForTheRightRegions/repro_runs/r4rr/train/r4rr_waterbirds.py
```

#### Resolve a teacher map from an image name

The authoritative producer is:

```text
GALS/RightForTheRightRegions/pipelines/generate_r4rr_maps/generate_pseudo_masks_waterbirds.py
```

Its `_make_image_id(...)` function computes the image ID from the **complete
path relative to the dataset root**:

1. remove the final image extension;
2. replace every path separator with `_`;
3. replace every run of characters outside `[A-Za-z0-9_-]` with `_`;
4. strip leading/trailing underscores; and
5. save the prediction as `<image_id>.png`.

During workspace construction, the producer appends `_1`, `_2`, and so on if
two source paths collapse to the same sanitized ID. Those suffixes depend on
the producer's recursive traversal order and cannot safely be guessed from a
metadata row afterward. The Waterbirds FCV preflight therefore rejects
flattened-name collisions. If a future dataset has collisions, change the
producer to save an explicit source-path-to-image-ID manifest and consume that
manifest.

WeCLIP's saver is:

```text
GALS/RightForTheRightRegions/WeCLIPPlus/test_msc_flip_voc.py
```

It writes `encode_cmap(pred)` to
`<results-dir>/val/prediction_cmap/<image_id>.png`.

For example:

```text
image:
200.Common_Yellowthroat/Common_Yellowthroat_0071_190665.jpg

flat teacher map:
200_Common_Yellowthroat_Common_Yellowthroat_0071_190665.png
```

If metadata contains an additional prefix such as
`images/200.Common_Yellowthroat/...`, that prefix is also flattened into the
mask name. Using only the immediate parent directory is therefore not the
general producer contract.

The exact producer-compatible mapping is:

```python
import re
from pathlib import Path

def producer_vlm_mask_name(img_filename: str) -> str:
    normalized = str(img_filename).strip().replace("\\", "/")
    if Path(normalized).is_absolute():
        raise ValueError("img_filename must be relative to the dataset root")
    normalized = normalized.lstrip("/")
    relative = Path(normalized)
    relative_without_extension = str(relative.with_suffix("")).replace("\\", "/")
    flattened = relative_without_extension.replace("/", "_")
    image_id = re.sub(r"[^A-Za-z0-9_-]+", "_", flattened).strip("_")
    if not image_id:
        raise ValueError(f"Could not derive WeCLIP image ID from {img_filename!r}")
    return image_id + ".png"

mask_path = vlm_mask_root / producer_vlm_mask_name(row["img_filename"])
```

Do not use `row.name`, a reset dataframe index, the integer metadata index, or
the sample's position in a `DataLoader` to identify the mask.

Historical consumers also supported a one-parent flat name and several nested
layouts. New code should try the exact producer name first, retain those
layouts only as explicit legacy fallbacks, and fail on ambiguity:

```python
import re
from pathlib import Path

def teacher_map_candidates(teacher_root: Path, image_name: str):
    normalized = str(image_name).strip().replace("\\", "/").lstrip("/")
    relative = Path(normalized)
    parent = relative.parent.name
    parent_underscored = parent.replace(".", "_")
    base = relative.stem + ".png"
    producer_flat = producer_vlm_mask_name(normalized)
    legacy_one_parent_flat = f"{parent_underscored}_{relative.stem}.png"
    return (
        teacher_root / producer_flat,
        teacher_root / legacy_one_parent_flat,
        teacher_root / parent_underscored / base,
        teacher_root / parent / base,
        teacher_root / relative.with_suffix(".png"),
    )

def resolve_teacher_map(teacher_root: Path, image_name: str) -> Path:
    matches = [
        candidate.resolve()
        for candidate in teacher_map_candidates(teacher_root, image_name)
        if candidate.is_file()
    ]
    # The flat and mirrored candidates can occasionally describe the same
    # resolved path, so deduplicate before enforcing a one-to-one join.
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No teacher map for {image_name}")
    raise RuntimeError(f"Ambiguous teacher maps for {image_name}: {matches}")
```

The FCV metadata module now implements this producer-first, fail-closed
behavior in:

```text
GALS/experiments/fcv_vit_waterbirds100/src/fcv/waterbirds_metadata.py
```

Verify exactly one resolved path for every required row and preserve missing
and duplicate reports. Training should consume the frozen, preflight-generated
manifest instead of resolving paths again.

#### End-to-end metadata-row to image/VLM-mask loading

The following pattern makes both roots and the join explicit. Select the
Waterbirds-95 or Waterbirds-100 pair of roots at the top; the remaining code is
identical:

```python
from pathlib import Path
import pandas as pd
from PIL import Image

# Waterbirds-95:
data_root = Path(
    "/home/ryreu/guided_cnn/waterbirds/"
    "waterbird_complete95_forest2water2"
)
vlm_mask_root = Path(
    "/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/"
    "results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap"
)

# For Waterbirds-100, replace the two roots with:
# data_root = Path(
#     "/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2"
# )
# vlm_mask_root = Path(
#     "/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/"
#     "results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap"
# )

metadata = pd.read_csv(data_root / "metadata.csv")
row = metadata.iloc[0]

image_path = resolve_image(row["img_filename"])
mask_path = resolve_teacher_map(vlm_mask_root, row["img_filename"])

with Image.open(image_path) as source:
    image = source.convert("RGB")
with Image.open(mask_path) as source:
    # Preserve categorical colors for the VOC decoder. Do not convert the
    # current VLM mask to a normalized image tensor.
    encoded_vlm_mask = source.convert("RGB")

sample = {
    "metadata_index": int(row.name),
    "img_filename": str(row["img_filename"]),
    "image_path": str(image_path),
    "vlm_mask_path": str(mask_path),
    "label": int(row["y"]),
    "context_analysis_only": int(row["place"]),
    "official_split": int(row["split"]),
}
```

In real campaign code, preserve `sample` in a manifest, hash the image and
mask, and load from that manifest afterward. Do not repeatedly perform an
unrecorded best-effort path search inside worker processes.

For FCV, turn the encoded VLM mask into ViT patch occupancy with the tested
decoder:

```python
import sys
from pathlib import Path

fcv_src = Path(
    "/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/"
    "experiments/fcv_vit_waterbirds100/src"
)
sys.path.insert(0, str(fcv_src))

from fcv.patch_masks import teacher_map_to_patch_scores

patch_scores, mask_metadata = teacher_map_to_patch_scores(
    mask_path,
    image_size=224,
    patch_size=16,
    normalize_to_unit_interval=True,
    interpolation="nearest",
    eval_resize_size=256,
    map_format="voc_colormap_class_ids",
    foreground_class_ids=[1],
)

assert patch_scores.shape == (196,)  # 14 x 14 ViT-S/16 patch grid
```

Those geometry values are correct for the locked ViT-S/16 FCV study. Other
models must take image size, evaluation resize, patch size, and normalization
from their own frozen configuration rather than copying `224/256/16`.

#### Decode the current `prediction_cmap` PNGs correctly

The current FCV configuration declares:

```yaml
format: voc_colormap_class_ids
foreground_class_ids: [1]
interpolation: nearest
```

These PNGs are categorical VOC-colormap images, not continuous heat maps.
Class `0` is background and class `1` is the selected foreground/evidence
region; VOC class `1` is encoded as RGB `[128, 0, 0]`. Robust code must decode
RGB colors to class IDs, reject unknown colors, and then construct
`foreground = class_id in foreground_class_ids`.

The important consequence is that a realistic foreground fixture is
`[128, 0, 0]`, **not white `[255, 255, 255]`**. A binary `0/255` test fixture
does not exercise the real producer format and should be rejected when
`map_format: voc_colormap_class_ids` is configured.

Use the tested implementation rather than duplicating the decoder:

```text
GALS/experiments/fcv_vit_waterbirds100/src/fcv/patch_masks.py
```

In particular, `teacher_map_to_patch_scores(...)`:

- decodes the VOC colormap exactly;
- applies the configured evaluation geometry;
- uses nearest-neighbor interpolation for the categorical mask;
- pools foreground occupancy over the ViT patch grid; and
- validates shape, value range, and map coverage.

Do not normalize the RGB PNG as though it were an input image. Do not use
bilinear or bicubic interpolation on categorical labels. Converting to
grayscale and testing `> 0` happens to recognize a simple black/red map, but it
loses class semantics and is not the preferred loader for a new study.

If a legacy binary consumer genuinely needs only foreground versus background,
the correct threshold is **nonzero**, not `127`, `128`, or normalized `0.5`.
VOC red `[128, 0, 0]` becomes a relatively dark value when converted to
grayscale, so a midpoint threshold incorrectly erases the foreground. These
two forms are safe for the current single-foreground-class output:

```python
import numpy as np
from PIL import Image

with Image.open(mask_path) as source:
    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)

# Preferred for this producer: preserve and check the exact class encoding.
foreground_exact = np.all(rgb == np.array([128, 0, 0], dtype=np.uint8), axis=-1)
background_exact = np.all(rgb == 0, axis=-1)
if not np.all(foreground_exact | background_exact):
    colors = np.unique(rgb.reshape(-1, 3), axis=0)
    raise ValueError(f"Unexpected VOC colors in {mask_path}: {colors[:10]}")

# Equivalent binary fallback only after confirming the map contains class 0/1.
foreground_nonzero = np.any(rgb != 0, axis=-1)
assert np.array_equal(foreground_exact, foreground_nonzero)
```

For future multiclass maps, decode the full VOC palette and select explicit
class IDs. `np.any(rgb != 0)` means “any non-background class,” not
specifically class `1`.

For the locked ViT-S/16 FCV study, images are resized on their shorter side to
256 with bicubic interpolation and center-cropped to `224 x 224`. Teacher maps
receive the identical resize-and-crop geometry with nearest-neighbor
interpolation, then are average-pooled into a `14 x 14` patch-occupancy grid.
The study-specific evidence and background thresholds are recorded in the
YAML; they are not universal Waterbirds constants.

#### Which metadata splits should have VLM masks?

The current producer reads `metadata.csv` and constructs:

```text
WeCLIP train.txt = official split 0
WeCLIP val.txt   = official splits 0 + 1
```

The final `prediction_cmap` export evaluates `val.txt`. Therefore its normal
coverage contract is:

| Official Waterbirds split | Expected in `prediction_cmap`? | Why |
| --- | --- | --- |
| Train (`split == 0`) | Yes | Used to train the teacher and included in its evaluation/export list. |
| Validation (`split == 1`) | Yes | Included in the teacher evaluation/export list. |
| Test (`split == 2`) | No | Explicitly omitted by `_load_waterbirds_split_rel_no_exts`. |

This coverage is sufficient for the existing protocols:

- standalone R4RR requests teacher maps only for its training dataset;
- the GALS OurMasks configurations set `SEG_TRAIN_ONLY: True`;
- Waterbirds-100 FCV preprocesses maps only for the train-derived biased
  validation holdout; and
- validation-set Pointing Game evaluation can use the split-1 maps.

Classification evaluation on test images does not need test masks. A study
that performs mask-conditioned test-time inference or localization evaluation
on official test rows is a different requirement and must explicitly generate
and audit test masks first. Never silently borrow a train/validation mask or
derive a mask from a row position.

The live directory should still be audited because old output roots can retain
stale files. The following read-only script checks the actual contract for
either Waterbirds root:

```python
import re
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(
    "/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2"
)
MASK_ROOT = Path(
    "/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/"
    "results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap"
)

def producer_name(img_filename: str) -> str:
    normalized = str(img_filename).strip().replace("\\", "/")
    if Path(normalized).is_absolute():
        raise ValueError("img_filename must be relative to DATA_ROOT")
    normalized = normalized.lstrip("/")
    without_extension = str(Path(normalized).with_suffix("")).replace("\\", "/")
    image_id = re.sub(
        r"[^A-Za-z0-9_-]+", "_", without_extension.replace("/", "_")
    ).strip("_")
    return image_id + ".png"

metadata = pd.read_csv(DATA_ROOT / "metadata.csv")
metadata["expected_mask_name"] = metadata["img_filename"].map(producer_name)

if metadata["expected_mask_name"].duplicated().any():
    collisions = metadata.loc[
        metadata["expected_mask_name"].duplicated(keep=False),
        ["img_filename", "expected_mask_name", "split"],
    ]
    raise RuntimeError(f"Flattened-name collisions:\n{collisions.to_string(index=False)}")

rows = []
for split_id, split_rows in metadata.groupby("split", sort=True):
    expected = set(split_rows["expected_mask_name"])
    present = {name for name in expected if (MASK_ROOT / name).is_file()}
    missing = sorted(expected - present)
    rows.append((int(split_id), len(expected), len(present), len(missing)))
    print(
        f"split={int(split_id)} expected={len(expected)} "
        f"present={len(present)} missing={len(missing)}"
    )
    if missing:
        print("  first missing:", missing[:5])

all_pngs = {path.name for path in MASK_ROOT.glob("*.png")}
known = set(metadata["expected_mask_name"])
extras = sorted(all_pngs - known)
print("flat PNGs not mapped to metadata rows:", len(extras))
print("first extras:", extras[:10])

# Producer-contract assertions. Test files may exist as separately generated
# extras, but train and validation coverage must be complete for this root.
coverage = {split_id: (expected, present, missing) for split_id, expected, present, missing in rows}
assert coverage[0][2] == 0, "Incomplete official-train VLM-mask coverage"
assert coverage[1][2] == 0, "Incomplete official-validation VLM-mask coverage"
```

For Waterbirds-95, replace only `DATA_ROOT` and `MASK_ROOT` with the
Waterbirds-95 paths listed above.

#### Keep image and mask geometry aligned

For deterministic evaluation:

- open the image as RGB;
- resize its shorter side to the configured evaluation size;
- center-crop to the configured image size;
- convert to a tensor and apply the model's expected normalization;
- decode the categorical mask before resizing;
- apply the exact same spatial resize and crop to the mask using nearest
  interpolation; and
- never normalize a mask with ImageNet mean and standard deviation.

For training with random crops or horizontal flips, sample the crop rectangle
and flip decision once and apply those same spatial parameters to both image
and mask. The image may use bicubic interpolation and antialiasing; the
categorical mask must use nearest-neighbor interpolation. Independent random
transforms create a visually plausible but incorrect image-mask pair.

The historical standalone R4RR Waterbirds runner intentionally reproduces its
original protocol with a direct `224 x 224` image resize and a corresponding
mask resize. Preserve that behavior when reproducing published R4RR numbers.
For a new paired-augmentation study, use an explicit paired transform rather
than changing the reproduction runner in place.

#### Required Waterbirds preflight

Before a full campaign, produce and retain a compact audit containing:

- dataset root, metadata SHA-256, teacher-map root, and code commit;
- row counts by official split, class, context, and group;
- proof that every metadata image resolves uniquely;
- proof that every required teacher map resolves uniquely;
- image and map hashes or a deterministic aggregate receipt;
- decoded class-ID counts and unknown-color failures;
- empty/full/implausible foreground-area counts;
- overlays sampled across split, class, context, and foreground-area quantile;
- confirmation that the image and mask use identical spatial geometry; and
- confirmation that selector-visible manifests exclude forbidden group
  information.

Do not weaken a failed coverage, color-decoding, or geometry gate just to start
a campaign. Preserve the diagnostics and fix the path, format, or protocol.

### 6.3 DecoyMNIST

PNG dataset root:

```text
/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png
```

Expected subdirectories:

```text
train/
test/
```

Current OpenCLIP+DINO teacher maps:

```text
/home/ryreu/guided_cnn/MNIST_AGAIN/DecoyGen/LearningToLook/code/WeCLIPPlus/results_decoy_mnist_openclip/val/prediction_cmap
```

The full FCV campaign configuration is:

```text
GALS/experiments/fcv_vit_decoymnist/configs/decoymnist_vit_s16_fcv_full_online.yaml
```

The full campaign uses an authenticated 48,000/6,000/6,000 split of the
60,000 original training samples and preserves the official 10,000-sample test
set. Do not regenerate or strengthen the decoy without defining a new study.

### 6.4 RedMeat

Dataset root:

```text
/home/ryreu/guided_cnn/Food101/data/food-101-redmeat
```

Metadata:

```text
/home/ryreu/guided_cnn/Food101/data/food-101-redmeat/all_images.csv
```

Current OpenCLIP-LAION + DINOvIT teacher maps:

```text
/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_laion_dinovit/val/prediction_cmap
```

Some legacy RedMeat/GALS code expects compatibility links such as
`food-101`, `meta/all_images.csv`, or split directories. The historical
`RedMeat_Runs/common_env.sh` contains the compatibility-layout logic, but its
environment activation is SPORC-specific. If reusing that logic on Tigris,
separate the data-layout function from the old environment activation.

### 6.5 SpuCoDogs

Downloaded data root:

```text
/home/ryreu/guided_cnn/data/spuco
```

Image root:

```text
/home/ryreu/guided_cnn/data/spuco/spuco_dogs
```

Expected splits:

```text
train/
val/
test/
```

Author-provided full SpuCoAnimals mask artifact:

```text
/home/ryreu/guided_cnn/data/spuco/spuco_animals_masks.pkl
```

Integrity receipts:

```text
/home/ryreu/guided_cnn/data/spuco/spuco_animals_masks.sha256
/home/ryreu/guided_cnn/data/spuco/spuco_dogs_archive.sha256
```

At the last verified download:

- `spuco_dogs` contained 24,050 images and used about 2.5 GB;
- the full animals mask pickle used about 9.4 GB.

#### 6.5.1 What SpuCoDogs is and what is currently verified

SpuCoDogs is the dog-only subset of SpuCoAnimals. It is intended to test
shortcut robustness with naturally occurring images and class-context
correlations rather than a synthetic pasted-background construction.
Do not treat its groups, context variable, or split protocol as interchangeable
with Waterbirds merely because both are shortcut-learning benchmarks.

The official `SpuCoDogs` loader—not a local naming guess—defines this hierarchy:

```text
spuco_dogs/
  {train,val,test}/
    {small_dogs,big_dogs}/
      {indoor,outdoor}/
        INTEGER_MASK_ID.<image extension>
```

The dog-size directory is the target:

```text
small_dogs = target label 0
big_dogs   = target label 1
```

The context directory is the spurious/environment label:

```text
indoor  = environment label 0
outdoor = environment label 1
```

The official loader source is:

```text
https://github.com/BigML-CS-UCLA/SpuCo/blob/master/src/spuco/datasets/spuco_dogs.py
```

Its expected live counts are:

| Split | Small/indoor | Small/outdoor | Big/indoor | Big/outdoor | Total |
|---|---:|---:|---:|---:|---:|
| `train` | 10,000 | 500 | 500 | 10,000 | 21,000 |
| `val` | 500 | 25 | 25 | 500 | 1,050 |
| `test` | 500 | 500 | 500 | 500 | 2,000 |

Thus the full image count is 24,050. The training and validation splits have
the intended dog-size/context correlation, while test is balanced over all
four target-environment cells.

Build an explicit manifest and verify the live tree against all 12 official
counts before a study. Group labels may be used for Oracle/post-hoc analysis
only if that is the frozen protocol; they must not silently enter an FCV
selector advertised as group-agnostic.

A read-only mask audit completed successfully on 2026-07-26 as Tigris job
`17169`, using repository commit
`2d2ce1909abe0630cf3403401e324642b5581a79`. It established:

- the mask artifact checksum matches its recorded SHA-256 receipt;
- the pickle root is a dictionary with 48,100 entries;
- every entry is a two-dimensional boolean NumPy array;
- the 48,100 records cover full SpuCoAnimals, while the SpuCoDogs image tree
  contains 24,050 images;
- all 24,050 SpuCoDogs images match exactly one mask;
- no SpuCoDogs image is missing a mask;
- no SpuCoDogs image has an ambiguous mask match;
- the remaining 24,050 masks are the non-dog portion of full SpuCoAnimals;
- masks retain many different native spatial shapes, so no fixed mask
  resolution may be assumed;
- loading and auditing the full pickle peaked at approximately 9.97 GiB RSS
  and took approximately 102 seconds; and
- the generated visual overlays looked usable but not pixel-perfect.

The successful audit outputs are:

```text
/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/audit_17169/mask_audit_report.json
/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/audit_17169/image_mask_alignment.csv
/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/audit_17169/mask_polarity_overlays.jpg
```

The corresponding Slurm logs are:

```text
/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/run_logs/audit_17169.out
/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/run_logs/audit_17169.err
```

The empty `.err` file, successful completion, matching checksum, and complete
one-to-one image coverage together make the artifact mechanically suitable
for developing a SpuCoDogs loader.

A second, all-sample deep audit completed successfully later on 2026-07-26 as
Tigris job `17250`, using repository commit
`0071b076f9ca064f6b156691be447def93e0b911`. It inspected all 24,050 images and
their masks in approximately 573 seconds with 9.91 GiB peak RSS. Its output is:

```text
/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/deep_audit_17250
```

It confirmed:

- all 12 official split/target/environment counts exactly;
- 21,000 train, 1,050 validation, and 2,000 test images;
- successful decoding and a mask for every image;
- exact agreement between every mask and the corresponding raw decoded image
  geometry;
- no need to apply EXIF transposition before the official native mask
  operation;
- a valid mask-pickle SHA-256 receipt;
- enough space for compact conversion and validation; and
- no authenticated dog-identity or source-sequence field suitable for a
  leakage-resistant identity split.

The extracted tree has canonical manifest SHA-256:

```text
0850d352dd0ad6903de9f5b35e9ffe76d5308d86f7bc99824d5706db85cd51c3
```

The original downloaded tar archive is no longer live, so its receipt cannot
be rehashed directly. Preserve the archive receipt, canonical extracted-tree
digest, mask receipt, and audit report together.

#### 6.5.2 Mask lookup and polarity

The official SpuCoAnimals loader is the semantic source of truth:

```text
https://github.com/BigML-CS-UCLA/SpuCo/blob/master/src/spuco/datasets/spuco_animals.py
```

It derives a numeric mask key from the image filename:

```python
mask_index = int(Path(image_path).stem)
spurious_mask = masks[mask_index]
```

This is not positional lookup within the 24,050-image dog subset. Never use a
DataFrame row number, DataLoader index, sorted-file index, or split-local index
as the mask key. Use the integer image filename stem.

The polarity is easy to reverse accidentally:

```text
True / high mask value   = spurious context/background
False / low mask value   = core animal/dog
```

Therefore:

```python
context_mask = np.asarray(spurious_mask, dtype=bool)
core_mask = np.logical_not(context_mask)
```

The audit overlay used red for `True`/background and blue for
`False`/core-animal pixels. The visual inspection was decent rather than
perfect. That is acceptable for development, but code should protect uncertain
object boundaries rather than pretending these are flawless pixel-level
annotations.

Call the masks **author-provided SpuCoAnimals masks**. Artifact inspection
proves their structure, alignment, polarity, and visual behavior; it does not
prove whether they were manually drawn, generated by a model, or refined by a
particular annotation process. Do not call them human-made or ground truth
unless the dataset's source documentation establishes that provenance.

#### 6.5.3 Trusted loading example

Python pickle can execute code while loading. Only load the trusted downloaded
artifact after checking its receipt. Do not load an arbitrary replacement
pickle from an untrusted source.

Shell integrity check:

```bash
cd /home/ryreu/guided_cnn/data/spuco
sha256sum -c spuco_animals_masks.sha256
```

Minimal Python lookup:

```python
from pathlib import Path
import pickle
import numpy as np
from PIL import Image

MASK_PICKLE = Path(
    "/home/ryreu/guided_cnn/data/spuco/spuco_animals_masks.pkl"
)

with MASK_PICKLE.open("rb") as handle:
    all_spuco_masks = pickle.load(handle)  # trusted, checksum-verified file

def load_spucodogs_image_and_masks(image_path: str | Path):
    image_path = Path(image_path)
    mask_index = int(image_path.stem)
    context_mask = np.asarray(all_spuco_masks[mask_index], dtype=bool)
    if context_mask.ndim != 2:
        raise ValueError(
            f"Expected a 2-D mask for {image_path}, got {context_mask.shape}"
        )

    image = Image.open(image_path).convert("RGB")
    expected = (image.height, image.width)
    if context_mask.shape != expected:
        raise ValueError(
            f"Image/mask geometry mismatch for {image_path}: "
            f"image={expected}, mask={context_mask.shape}"
        )

    core_mask = np.logical_not(context_mask)
    return image, core_mask, context_mask
```

Keep the geometry assertion during initial development. Do not silently resize
a mismatched source mask merely to make a run continue; first determine
whether the mismatch is an orientation, decoding, or lookup error.

Do not reload the 9.4 GB pickle inside every `Dataset.__getitem__`, every
batch, or every DataLoader worker. At minimum, load it once per process.
For production arrays, prefer the future compact dog-only artifact described
below so multiple concurrent tasks do not repeatedly impose 10 GiB resident
memory and heavy shared-filesystem reads.

#### 6.5.4 Image and mask transforms must be paired

Any spatial image operation must be applied identically to its mask:

- resize;
- center crop;
- random resized crop;
- horizontal flip;
- padding; and
- any future geometric augmentation.

Sample random crop/flip parameters once and reuse them for both image and mask.
Applying independent random transforms destroys alignment.

Use normal antialiased image interpolation for RGB images, but use
nearest-neighbor interpolation for boolean masks. Bilinear/bicubic mask
resizing invents fractional boundary values and changes the mask semantics.
Convert the mask back to boolean after transformation.

Illustrative paired crop:

```python
import random
import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF

def paired_random_resized_crop(image, context_mask, output_size, scale, ratio):
    mask_image = Image.fromarray(
        np.asarray(context_mask, dtype=np.uint8) * 255,
        mode="L",
    )
    top, left, height, width = RandomResizedCrop.get_params(
        image, scale=scale, ratio=ratio
    )
    image = TF.resized_crop(
        image,
        top,
        left,
        height,
        width,
        output_size,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    mask_image = TF.resized_crop(
        mask_image,
        top,
        left,
        height,
        width,
        output_size,
        interpolation=InterpolationMode.NEAREST,
    )

    if random.random() < 0.5:
        image = TF.hflip(image)
        mask_image = TF.hflip(mask_image)

    context_mask = np.asarray(mask_image) > 127
    core_mask = np.logical_not(context_mask)
    return image, core_mask, context_mask
```

For reproducible experiments, the actual implementation should draw
randomness from the campaign's seeded generator rather than uncontrolled
global `random.random()`. The snippet only demonstrates shared geometry.
Photometric transforms such as normalization or color jitter apply to the
image, not the boolean mask.

#### 6.5.5 Conversion to ViT patch-token masks

FCV intervenes on patch tokens, while the provided masks are pixel masks.
After applying the exact evaluation image geometry, aggregate the background
fraction inside each ViT patch. For a ViT-S/16 model at `224x224`, the patch
grid is `14x14`, but code should derive patch size/grid from the actual model
and transformed input rather than hard-code it.

Conceptually:

```python
# context_mask is HxW boolean after the paired image transform.
context_fraction_per_patch = average_pool(context_mask.float(), patch_size)
```

Because the masks are imperfect near dog boundaries, use a conservative,
predeclared occupancy rule:

- high-confidence context tokens: context fraction at or above a frozen
  context threshold;
- high-confidence core tokens: context fraction at or below a frozen core
  threshold;
- mixed/boundary tokens: excluded from counterfactual replacement.

The exact occupancy thresholds are scientific hyperparameters/protocol
choices. Freeze them before viewing selector/test results, record them in the
study configuration, and test them in preflight. Do not quietly tune them
against test accuracy. A conservative boundary exclusion is preferable to
replacing tokens that may contain part of the dog.

Preserve token ordering:

- never include the CLS token in a spatial patch mask;
- verify the flattened mask length equals the number of patch tokens;
- verify row-major patch ordering matches the model's token ordering;
- record the number/fraction of core, context, and excluded boundary tokens;
- reject or explicitly mark samples with no eligible context tokens; and
- use the target image's positional embeddings after donor-token replacement,
  following the frozen FCV intervention definition.

#### 6.5.6 Audit tooling and rerunning the audit

The checked-in read-only audit is:

```text
GALS/experiments/spucodogs_mask_audit/audit_spucodogs_masks.py
GALS/experiments/spucodogs_mask_audit/slurm/audit_spucodogs_masks.sbatch
GALS/experiments/spucodogs_mask_audit/submit_mask_audit.sh
GALS/experiments/spucodogs_mask_audit/README.md
```

Launch it on Tigris with:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash experiments/spucodogs_mask_audit/submit_mask_audit.sh
```

It is CPU-only and requests 128 GB RAM because it intentionally inspects the
full 9.4 GB trusted pickle. It does not alter, compact, or delete either source
artifact. Important outputs are:

```text
mask_audit_report.json
image_mask_alignment.csv
mask_polarity_overlays.jpg
```

A future audit should additionally make exact image-versus-mask shape
agreement an explicit all-sample acceptance gate, even though current
alignment and visual checks succeeded.

That second-stage audit is now checked in:

```text
GALS/experiments/spucodogs_mask_audit/deep_audit_spucodogs.py
GALS/experiments/spucodogs_mask_audit/slurm/deep_audit_spucodogs.sbatch
GALS/experiments/spucodogs_mask_audit/submit_deep_audit.sh
```

Run it with:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash experiments/spucodogs_mask_audit/submit_deep_audit.sh
```

It is a read-only, CPU-only job using account `reu-aisocial`, partition
`tigris`, 8 CPUs, 128 GiB RAM, the `fcv_gh200` environment, and a 12-hour
ceiling. It verifies:

- all 12 official split/target/environment counts;
- every filename's integer mask key and key uniqueness in the dog tree;
- every image decode and every source image/mask shape pair;
- raw versus EXIF-transposed geometry;
- missing, malformed, empty, full, border-touching, fragmented, and
  broad-area-warning masks;
- exact file and decoded-pixel duplicates across splits;
- perceptual near-duplicate review candidates across splits;
- available EXIF/source identity evidence;
- reusable local loaders/manifests in the checked-out repository;
- mask and archive integrity receipt status;
- a canonical digest for the extracted image tree; and
- free-space safety for compact-artifact construction and validation.

Its principal outputs are:

```text
spucodogs_deep_audit_report.json
official_group_counts.csv
image_mask_inventory.csv
cross_split_exact_duplicates.csv
cross_split_perceptual_duplicate_candidates.csv
mask_quality_review.jpg
repository_reuse_candidates.json
```

The dHash CSV is a candidate screen, not proof that images share identity.
The gallery is required because a mask alone cannot prove that it visually
contains every part of the dog without independent boxes or segmentations.
Inspect both before freezing a leakage claim or mask-quality statement.

The completed job found 27 cross-split dHash review candidates and did not hit
the configured truncation limit. These are not automatically duplicates and
must be visually reviewed.

It initially reported 28 exact-duplicate rows because each underlying pair was
detected by both exact file SHA-256 and decoded-pixel SHA-256. After grouping
the two hash channels, there are 14 distinct exact duplicate pairs:

- 10 train/test pairs;
- 4 train/validation pairs;
- 0 validation/test pairs.

All 14 pairs remain within the same dog-size target and indoor/outdoor
environment. The exact path pairs are preserved in:

```text
/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/deep_audit_17250/cross_split_exact_duplicates.csv
```

This is genuine official-split overlap, although it is small:

- 10 of 2,000 test samples, or 0.50%;
- 4 of 1,050 validation samples, or approximately 0.38%.

Do not silently alter the official benchmark. A defensible study protocol is:

1. report the official-split result as the primary benchmark result for
   comparability;
2. predeclare a duplicate-clean sensitivity analysis;
3. keep the complete official training split fixed;
4. exclude the 4 validation counterparts that duplicate training images from
   the clean validation view;
5. exclude the 10 test counterparts that duplicate training images from the
   clean test view;
6. compute official and clean metrics online from the same model state; and
7. save an immutable exclusion manifest containing paths, reasons, source CSV
   hash, and resulting sizes of 1,046 validation and 1,990 test images.

Removing the evaluation-side counterpart eliminates train/evaluation overlap
without changing training data or requiring a second campaign. Do not use the
27 perceptual candidates as automatic exclusions; review them first and freeze
any additional sensitivity manifest before looking at model-selection
outcomes.

#### 6.5.7 Compact dog-only production artifact

The full pickle should eventually be converted once into a compact,
authenticated, dog-only artifact. The audit reports that filename-based
compaction is mechanically safe, but the converter must still be reviewed and
tested.

A good compact format should:

- retain only the 24,050 mask IDs referenced by SpuCoDogs;
- store the numeric image ID and original `(height, width)`;
- preserve boolean values exactly, preferably with packed bits;
- use deterministic ordering;
- avoid arbitrary-code execution during normal loading if practical;
- include a manifest mapping every relative image path to its numeric mask ID;
- include source-pickle SHA-256, converter commit, conversion command, and
  output SHA-256;
- support exact reconstruction tests against the source pickle;
- be shardable or lazily readable so workers do not load all masks; and
- fail on missing, duplicate, malformed, or geometry-mismatched records.

Do not delete the original pickle after merely producing the compact file.
First verify:

1. all 24,050 images have one reconstructed mask;
2. every reconstructed mask is bit-exact with the source;
3. every reconstructed shape agrees with its corresponding image;
4. polarity remains `True=context`, `False=core`;
5. visual overlays still align;
6. the production DataLoader passes multi-worker tests;
7. the compact artifact and manifest have recorded hashes; and
8. at least one independent copy or retrieval path exists.

Only then should deletion of the 9.4 GB source be considered, and deletion
should be an explicit owner decision rather than automatic cleanup.

#### 6.5.8 Required SpuCoDogs preflight for a future FCV study

No production FCV SpuCoDogs study is frozen merely because the data and masks
exist. Define and review its split, Oracle, context-inference, donor,
counterfactual, and selector protocol before implementing or launching it.

At minimum, preflight must validate:

- dataset root, official split metadata, and expected image count;
- class, split, and analysis-only group counts;
- untouched test isolation;
- mask source checksum and compact-artifact checksum, if used;
- numeric filename stems and unique mask lookup for every included image;
- exact image/mask source geometry;
- paired crop/flip behavior;
- transformed mask dimensions;
- ViT patch-grid dimensions and CLS-token exclusion;
- core/context/boundary token occupancy statistics by split and class;
- eligibility/exclusion counts and reasons;
- fixed donor pools without forbidden group/test information;
- deterministic donor plans and hashes;
- one real online FCV pass on a GH200;
- output/storage projections; and
- a visual overlay sample from every class, split, and relevant context.

The fact that masks are author-provided does not authorize use of hidden group
labels in the primary selector. Keep Vanilla/FCV-visible artifacts separate
from Oracle/test-analysis-only artifacts exactly as in the existing online
Waterbirds and DecoyMNIST FCV studies.

The all-sample mask-quality screen adds these operational facts:

- mean core-mask area: 73.80%;
- median core-mask area: 77.10%;
- 1st--99th percentile core area: approximately 19.48%--99.12%;
- 2 empty core masks, equivalently 2 full-context masks;
- 2,821 fragmented-core warnings, approximately 11.73%;
- 2,051 broad core-area warnings, approximately 8.53%;
- 24,046 core masks touch an image border.

The border count and large core areas show why these artifacts must not be
described as flawless, tight dog segmentations. They are author-provided
spurious/core region cues following the official loader semantics. For FCV,
prefer high-confidence `True` context pixels/tokens and exclude uncertain
mixed boundaries. Do not assume that the entire inverse mask is independently
verified dog foreground.

The two empty-core/full-context samples require an explicit policy before a
campaign. A safe policy is to mark them ineligible for mask-dependent FCV
construction, retain them in ordinary classification evaluation, record the
reason in the manifest, and never silently replace the whole image. Review the
quality gallery before freezing that policy:

```text
/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit/deep_audit_17250/mask_quality_review.jpg
```

#### 6.5.9 Direct answers and remaining evidence boundaries

The following questions are already answered by the official loader and
completed first audit:

- **Hierarchy and filename:** split / dog-size / context / integer filename.
- **Labels:** `small_dogs=0`, `big_dogs=1`; `indoor=0`, `outdoor=1`.
- **Official counts:** the 12 cells shown in Section 6.5.1, totaling 24,050.
- **Pickle schema:** `dict` with 48,100 entries.
- **Mask schema:** every record inspected is a two-dimensional NumPy `bool`
  array, not polygon, RLE, tensor, or encoded image.
- **Lookup:** `masks[int(image filename stem)]`; never row or dataset order.
- **Polarity:** `True=context/background`, `False=core/animal`.
- **Dog coverage:** all 24,050 dog images map uniquely; no missing or ambiguous
  dog mask.
- **Dog-only filtering:** the dog tree's numeric stems select exactly the
  needed half of the full 48,100-mask SpuCoAnimals dictionary.
- **Mask receipt:** the live pickle SHA-256 matches its recorded receipt.
- **Prior free space:** 72 GiB was free during job 17169, but current free
  space must always be rechecked before compaction.

The second audit resolved the remaining storage question:

- dog-only masks occupy approximately 4.91 GB as unpacked booleans;
- exact bit-packed mask payload is approximately 613.7 MB;
- estimated compact artifact including IDs/paths is approximately 616.1 MB;
- conservative two-copy staging plus temporary validation products is
  approximately 3.38 GB;
- approximately 76.39 GB was free in bytes at audit time; and
- compaction is safe while retaining a 10 GiB reserve.

Important boundaries:

- A materialized Python dictionary has unique keys. It can prove that the
  final artifact has 48,100 unique dictionary entries, but cannot reveal
  whether a hypothetical earlier serializer overwrote a duplicate key.
  More relevantly, the deep audit checks that no two dog images reuse the same
  numeric mask ID.
- The integer stem is documented as a mask lookup ID, not dog identity, breed,
  capture sequence, or collection identity. Do not invent a leakage-resistant
  identity split from nearby integers.
- The official loader applies a mask to the natively decoded image before its
  `Resize((256,256))` and `CenterCrop((224,224))`. This strongly implies native
  pre-resize geometry, but the deep audit verifies every live image/mask pair
  and explicitly separates raw from EXIF-transposed matches.
- Exact split duplicates, empty/fragmented/area-warning masks,
  source-identity metadata, the archive receipt's live-verification status,
  and the current storage projection are resolved by job `17250` above.
- The 27 perceptual candidates still require visual review; dHash proximity
  alone is not evidence of shared image or dog identity.
- “Visibly omits part of the dog” is not fully machine-resolvable from these
  masks alone. The deep audit generates a stratified and extreme-case gallery
  for a documented human review; do not present that review as an independent
  ground-truth segmentation benchmark.

### 6.6 UrbanCars

UrbanCars was being generated on another machine and had not yet been
confirmed as uploaded to Tigris at the time of this handoff.

A sensible future location is:

```text
/home/ryreu/guided_cnn/data/urbancars
```

That is a proposed path, not a verified one. A new agent must inspect the
actual transfer, manifests, and dataset-generation provenance before using it.
The local source repository used to investigate generation was:

```text
/home/ryan/ComputerScience/LearnToLook/SwitchVLM/Whac-A-Mole
```

---

## 7. Log and output locations

Use dataset-specific persistent roots:

```text
/home/ryreu/guided_cnn/logsWaterbird
/home/ryreu/guided_cnn/logsMNIST
/home/ryreu/guided_cnn/logsRedMeat
/home/ryreu/guided_cnn/logsZeroShot
/home/ryreu/guided_cnn/logsSpuCo
```

Current FCV campaign roots:

```text
/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility
/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_full_campaign
/home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit
```

Organize a new campaign under one unique root:

```text
CAMPAIGN_ROOT/
  run_logs/
  preflight/
  manifests/
  online_metrics/
  selection/
  reports/
```

The exact names may differ, but keep raw logs, selector-visible metrics,
analysis-only test data, manifests, and final reports visibly separated.

Slurm opens `--output` and `--error` before the job body executes. Therefore,
the parent log directory must exist **before** `sbatch`:

```bash
mkdir -p /home/ryreu/guided_cnn/logsWaterbird/my_campaign/run_logs
sbatch path/to/job.sbatch
```

Creating the directory inside the submitted script is too late for Slurm's own
stdout/stderr files.

---

## 8. Storage constraints and scratch policy

Storage is a real constraint. During the completed SpuCoDogs mask audit on
2026-07-26, the home filesystem reported approximately 72 GB free; earlier
checks were substantially lower. This number is highly time-sensitive and
must always be checked live:

```bash
df -h /home/ryreu
du -sh /home/ryreu/guided_cnn/logsWaterbird 2>/dev/null
du -sh /home/ryreu/guided_cnn/logsMNIST 2>/dev/null
du -sh /home/ryreu/guided_cnn/logsRedMeat 2>/dev/null
du -sh /home/ryreu/guided_cnn/data 2>/dev/null
```

Use `quota -s` as an additional check if it is supported on the login node.

### 8.1 Persistent-output policy

For broad sweeps and FCV studies, prefer online evaluation while the model is
in memory. Persist:

- aggregate CSV/JSON metrics;
- compact manifests and split indices;
- configuration snapshots and hashes;
- provenance/audit receipts;
- concise logs and final plots;
- a final model only when downstream use truly requires one.

Avoid persisting:

- every epoch checkpoint;
- optimizer/resume state for short independent runs;
- per-image logits for every candidate;
- token banks;
- embeddings;
- donor-expanded predictions;
- duplicate pretrained models.

The DecoyMNIST full FCV campaign is the clean reference: it evaluates all
selectors and analysis-only test metrics online, saves no model or optimizer
states, and keeps only compact aggregate evidence.

Some older Waterbirds FCV configuration paths include bounded checkpoint
retention for restart or selected winners. Do not copy that behavior into a
new campaign automatically. The owner's current preference is to avoid model
checkpoint storage when complete online results make checkpoints unnecessary.

### 8.2 Node-local scratch

GH200 compute nodes exposed a large node-local `/tmp` (about 1 TB during the
environment test). Use node-local scratch for temporary token banks, staging,
and transient extraction:

```bash
TASK_TMP=$(mktemp -d "${TMPDIR:-/tmp}/my_campaign.${SLURM_JOB_ID}.XXXXXX")
cleanup() {
  if [[ -n "${TASK_TMP:-}" && -d "$TASK_TMP" ]]; then
    rm -rf -- "$TASK_TMP"
  fi
}
trap cleanup EXIT
```

Only use the exact directory returned by `mktemp`; never clean a broad path or
an unresolved variable. Do not assume `/tmp` survives job completion or is
shared between nodes.

An array task must clean its transient artifacts after each epoch or run, not
only after the full array completes.

### 8.3 Shared model caches

Current jobs use:

```bash
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
```

Cache a pretrained model once in a preflight job and make the training array
depend on that job. Do not let many array tasks race to download the same
weights. Hash or otherwise authenticate the cached model when model
provenance matters.

---

## 9. Recommended Tigris SBATCH template

Use this as a starting point, then right-size CPUs, memory, wall time, and
array concurrency from a real smoke test:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=my_job
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/my_campaign/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/my_campaign/run_logs/%x_%j.err

set -Eeuo pipefail

REPO=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$REPO"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
echo "commit=$(git rev-parse HEAD 2>/dev/null || printf UNKNOWN)"
python --version
python -c "import torch, torchvision, timm; print(torch.__version__, torchvision.__version__, timm.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered python -u path/to/runner.py
```

Notes:

- Use `set -Eeuo pipefail`.
- Use absolute paths for cluster resources.
- Let Slurm assign the GPU. Do not hard-code a physical GPU ID.
- `python -u` and `PYTHONUNBUFFERED=1` make progress visible.
- Put scientific settings in a config/CLI, not only in comments.
- Print the commit, environment versions, host, job ID, and resolved paths.
- Check that the requested wall time is legal for `tigris`.
- A CPU-only aggregation job can omit `--gres`, but still needs the correct
  account and partition.

### 9.1 Arrays

Example:

```bash
#SBATCH --array=0-107%8
```

This creates 108 tasks and limits concurrency to eight. The `%8` cap controls
simultaneous GPU, home-storage, cache, and metadata pressure. Do not remove it
only because more GPUs appear idle.

Inside the script:

```bash
srun --unbuffered python -u path/to/runner.py \
  --run-index "$SLURM_ARRAY_TASK_ID"
```

Map every array index deterministically to a frozen hyperparameter/seed row.
Persist that mapping or make it inspectable before submission.

### 9.2 Dependencies

For a required pipeline, use `afterok`:

```bash
preflight=$(sbatch --parsable preflight.sbatch)
smoke=$(sbatch --parsable --dependency="afterok:${preflight%%;*}" smoke.sbatch)
array=$(sbatch --parsable --dependency="afterok:${smoke%%;*}" array.sbatch)
freeze=$(sbatch --parsable --dependency="afterok:${array%%;*}" freeze.sbatch)
report=$(sbatch --parsable --dependency="afterok:${freeze%%;*}" report.sbatch)
```

`afterok` prevents downstream jobs from running after a failed gate.
`afterany` is inappropriate when correctness requires the parent to succeed.

Recommended chain:

```text
data/model preflight
  -> real one-epoch smoke
  -> production array
  -> selector freeze
  -> post-hoc test/report
```

Write all returned job IDs, the git commit, protocol size, and submission time
to a submission receipt.

### 9.3 How to submit a script

For an SBATCH file:

```bash
sbatch path/to/job.sbatch
```

For a launcher that calls `sbatch` itself:

```bash
bash path/to/submit_campaign.sh
```

Typing `run_something.sh` without `./`, `bash`, or `sbatch` will normally
produce `command not found`, because the current directory is not on `PATH`.

---

## 10. Preflight and smoke requirements

Do not release a large array directly. A useful preflight should validate:

- repository and configuration path;
- clean/recorded Git commit;
- environment and package versions;
- CUDA availability and GPU name;
- every required dataset/map directory;
- metadata columns and expected split counts;
- class/group counts;
- complete teacher-map coverage;
- frozen manifests and hashes;
- pretrained-model cache/provenance;
- output free space;
- projected persistent and concurrent storage;
- model forward pass;
- one real train/evaluate epoch;
- selector visibility boundaries;
- no forbidden checkpoint-like artifacts;
- projected runtime against the requested wall time.

The smoke must execute the real production path on real data. A toy unit test
does not reveal realistic runtime, checkpoint size, token-bank size, or data
lookup failures.

Large launchers should refuse duplicate active campaigns unless an explicit
override is supplied.

---

## 11. Monitoring, accounting, cancellation, and retries

### 11.1 Queue

```bash
squeue --me
squeue --me --start
squeue --me -o '%.18i %.28j %.2t %.10M %.6D %R'
```

### 11.2 Completed and failed jobs

```bash
sacct -X -S today \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End
```

For a specific job or array:

```bash
sacct -j JOB_ID \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
scontrol show job JOB_ID
```

For arrays, inspect both the parent and task IDs. A parent may look completed
while one task failed, or downstream jobs may remain pending because a
dependency failed.

### 11.3 Logs

Search for meaningful failures:

```bash
rg -n 'Traceback|RuntimeError|CUDA out of memory|Disk quota exceeded|No space left|Killed|FAILED' \
  /home/ryreu/guided_cnn/logsWaterbird/my_campaign/run_logs
```

A nonempty `.err` file is not automatically a failed job; warnings and
download progress often go to stderr. Use the traceback plus `sacct` state and
exit code.

### 11.4 Cancel

Cancel exact known jobs:

```bash
scancel JOB_ID_1 JOB_ID_2 JOB_ID_3
```

Do not broadly cancel every user job unless that scope is intentional.
After an upstream failure, cancel stale dependent jobs if they remain queued.

### 11.5 Retry

Retry behavior is experiment-specific:

- Some tasks are idempotent and reuse authenticated completed output.
- The DecoyMNIST full FCV task restarts an interrupted partial run from epoch
  1 and reuses only a fully authenticated completed run.
- Some older sweeps write CSVs but do **not** restore Optuna sampler state.
- A CSV alone does not prove that a sweep can resume faithfully.
- True Optuna continuation normally needs a persistent study database or
  explicit code that imports completed trials.

Before resubmitting, read the runner's restart logic and inspect existing
receipts. Do not assume `--restart`, `--resume`, or a CSV path means the same
thing across experiments.

---

## 12. Current FCV reference implementations

### 12.1 Waterbirds-100

Directory:

```text
GALS/experiments/fcv_vit_waterbirds100
```

Primary documents:

```text
GALS/experiments/fcv_vit_waterbirds100/README.md
GALS/experiments/fcv_vit_waterbirds100/configs/waterbirds100_vit_s16_first_study.yaml
```

Current full online launcher:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash experiments/fcv_vit_waterbirds100/scripts/submit_full_online_540_study.sh
```

The launcher submits:

```text
pretrained cache
  -> interruption/resume smoke
  -> 27-run array (maximum 8 concurrent)
  -> validation-selection freeze
  -> post-hoc test analysis
```

The candidate pool is 27 training runs (3 learning rates, 3 weight decays, 3
seeds) times 20 online epochs, for 540 candidate states. The study creates an
80/20 class-stratified split from Waterbirds-100 training data so Vanilla and
FCV do not receive privileged validation data.

Read the current config before reusing this launcher: its historical storage
and metric choices evolved during development.

### 12.2 DecoyMNIST susceptibility pilot

Directory:

```text
GALS/experiments/fcv_vit_decoymnist
```

Launcher:

```bash
bash experiments/fcv_vit_decoymnist/scripts/submit_susceptibility_pilot.sh
```

Output:

```text
/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility
```

### 12.3 DecoyMNIST full online FCV campaign

Read:

```text
GALS/experiments/fcv_vit_decoymnist/FCV_DECOYMNIST_FULL_CAMPAIGN_IMPLEMENTATION_PLAN.md
GALS/experiments/fcv_vit_decoymnist/configs/decoymnist_vit_s16_fcv_full_online.yaml
GALS/experiments/fcv_vit_decoymnist/README.md
```

Launch:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash experiments/fcv_vit_decoymnist/scripts/submit_full_campaign.sh
```

The production design uses:

- 108 training runs;
- 10 online epochs each;
- 1,080 candidate states;
- a real one-epoch GH200 smoke;
- no model checkpoints;
- no optimizer/resume states;
- no persisted token banks or per-image predictions;
- harmonic original/counterfactual FCV score;
- separate Vanilla, FCV, Oracle-analysis-only, and test-analysis-only
  namespaces;
- `afterok` dependencies through final reporting.

This is the best current example of a low-storage online-selection campaign.

---

## 13. Scientific information-boundary rules

Infrastructure code must preserve the experimental protocol:

1. Vanilla and FCV may only read their declared non-privileged validation
   artifacts.
2. Oracle data must live in an explicitly analysis-only namespace.
3. Official test metrics must not affect training, hyperparameter choices,
   selector choices, aggregation weights, or control design.
4. Freeze selections before joining test results.
5. Use the exact same candidate pool for Vanilla, FCV, and Oracle comparisons.
6. Bind manifests, configurations, and candidate grids with hashes/receipts.
7. Keep controls warning-only if that is the frozen protocol.
8. Do not change thresholds, eligibility rules, masks, or donor rules after
   inspecting results unless defining and documenting a new study.
9. Record all exclusions and missing inputs rather than silently dropping
   them.
10. Distinguish author-provided masks, VLM-generated teacher maps, and
    human-annotated masks accurately.

When creating a new dataset study, write and review the protocol before coding:

- candidate training split;
- non-privileged validation split;
- Oracle construction;
- untouched test split;
- teacher/mask source;
- target and donor eligibility;
- intervention location;
- primary selector equation;
- controls;
- candidate grid;
- seeds;
- online metrics and storage policy;
- preflight acceptance gates.

---

## 14. Transferring code, data, and logs

SSH access may require Duo MFA. Never put passwords or MFA codes in scripts.

### 14.1 Prefer Git for code

Push a clean commit locally, then pull it on Tigris:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
git status --short
git pull --ff-only
```

Use `rsync`/`scp` for data and results, not for replacing a tracked code tree
without review.

### 14.2 Upload a dataset or untracked artifact

From the local machine:

```bash
rsync -avP LOCAL_SOURCE/ \
  ryreu@tigris.rc.rit.edu:/home/ryreu/guided_cnn/data/DESTINATION/
```

After transfer, compare counts, sizes, and checksums. Use a staging directory
and rename into the canonical path only after validation.

### 14.3 Download compact results

From the local machine:

```bash
rsync -avP \
  ryreu@tigris.rc.rit.edu:/home/ryreu/guided_cnn/logsMNIST/CAMPAIGN/ \
  /home/ryan/ComputerScience/LearnToLook/SwitchVLM/download_logs/logsMNIST/CAMPAIGN/
```

Exclude bulky artifacts when they are not needed:

```bash
rsync -avP \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='token_banks/' \
  SOURCE/ DESTINATION/
```

Do not use the old `sporcsubmit.rc.rit.edu` host for a Tigris-only result
unless the files are genuinely still on the old system.

---

## 15. Checklist for a new agent or new experiment folder

Before implementation:

- [ ] Read this handoff.
- [ ] Run `git rev-parse --show-toplevel`, `git remote -v`, and
      `git status --short`.
- [ ] Confirm whether work is local or on Tigris.
- [ ] Read the relevant study README, plan, config, launcher, and tests.
- [ ] Verify every dataset and teacher-map path on Tigris.
- [ ] Identify which paths are current and which are historical.
- [ ] Define selector visibility and test isolation.
- [ ] Estimate persistent and concurrent disk use.

Before submission:

- [ ] Use account `reu-aisocial`.
- [ ] Use partition `tigris`.
- [ ] Request `gpu:gh200:1` for GPU jobs.
- [ ] Activate `fcv_gh200` from aarch64 Miniforge.
- [ ] Set `PYTHONNOUSERSITE=1`.
- [ ] Create Slurm log directories before `sbatch`.
- [ ] Record package versions and GPU in the log.
- [ ] Record the Git commit in a submission receipt.
- [ ] Run syntax checks, unit tests, preflight, and a real GH200 smoke.
- [ ] Use `afterok` dependencies.
- [ ] Throttle arrays.
- [ ] Verify no unintended checkpoint or token-bank persistence.
- [ ] Check live free space.

During execution:

- [ ] Avoid changing the active checkout.
- [ ] Monitor both `squeue` and `sacct`.
- [ ] Inspect exact failed array tasks and tracebacks.
- [ ] Confirm scratch cleanup.
- [ ] Cancel stale dependents after a failed gate.

After execution:

- [ ] Authenticate expected run/candidate counts.
- [ ] Freeze selector choices before post-hoc test joins.
- [ ] Preserve compact reports, configs, manifests, hashes, and receipts.
- [ ] Remove only exact, verified transient artifacts.
- [ ] Download compact results locally.
- [ ] Keep generated outputs out of Git.

---

## 16. First commands a new agent should request or run

On Tigris:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
git rev-parse --show-toplevel
git remote -v
git branch --show-current
git status --short
git log -1 --oneline

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/fcv_gh200
which python
python --version

df -h /home/ryreu
squeue --me -o '%.18i %.28j %.2t %.10M %.6D %R'
```

Path audit:

```bash
for p in \
  /home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2 \
  /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2 \
  /home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap \
  /home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap \
  /home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png \
  /home/ryreu/guided_cnn/Food101/data/food-101-redmeat \
  /home/ryreu/guided_cnn/data/spuco/spuco_dogs; do
  if [[ -d "$p" ]]; then
    echo "OK $p"
  else
    echo "MISSING $p"
  fi
done
```

Waterbirds file audit:

```bash
for p in \
  /home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2/metadata.csv \
  /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/metadata.csv; do
  if [[ -f "$p" ]]; then
    echo "OK $p"
    sha256sum "$p"
  else
    echo "MISSING $p"
  fi
done

if [[ -d /home/ryreu/guided_cnn/waterbirds/CUB_200_2011/segmentations ]]; then
  echo "OK CUB segmentation tree"
else
  echo "MISSING OR NOT INSTALLED: CUB segmentation tree"
fi
```

SpuCoDogs mask inputs:

```bash
for p in \
  /home/ryreu/guided_cnn/data/spuco/spuco_animals_masks.pkl \
  /home/ryreu/guided_cnn/data/spuco/spuco_animals_masks.sha256 \
  /home/ryreu/guided_cnn/data/spuco/spuco_dogs_archive.sha256; do
  if [[ -f "$p" ]]; then
    echo "OK $p"
  else
    echo "MISSING $p"
  fi
done

cd /home/ryreu/guided_cnn/data/spuco
sha256sum -c spuco_animals_masks.sha256
```

Do not launch a large campaign until these basics and the experiment-specific
preflight agree.

---

## 17. Final cautions

- “It worked on SPORC” does not prove it is Tigris-ready.
- “The CSV exists” does not prove a sweep can resume.
- “The `.err` file is nonempty” does not prove a job failed.
- “The path exists” does not prove it is the correct version of a dataset or
  teacher map.
- “The GPU has large memory” does not remove home-quota or I/O constraints.
- “The job was submitted” does not prove its dependency chain will run.
- “The test metric was computed online” does not make it selector-visible.
- Do not silently weaken preflight acceptance gates to make a campaign start.
- Do not save large checkpoint collections unless the study genuinely needs
  them and the owner explicitly accepts the storage cost.
- When uncertain, preserve provenance and fail with an exact diagnostic.
