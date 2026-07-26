# Tigris Research-Compute Handoff

Last updated: 2026-07-25

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

Dataset:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2
```

Metadata is normally:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2/metadata.csv
```

Current OpenCLIP-LAION + DINOvIT teacher maps used by the recent R4RR runs:

```text
/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap
```

An older teacher-map tree also exists in historical scripts:

```text
/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
```

Do not substitute one for the other without checking the experiment's frozen
configuration and map provenance.

### 6.2 Waterbirds-100

Dataset:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
```

Metadata:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/metadata.csv
```

Current OpenCLIP-LAION + DINOvIT teacher maps:

```text
/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
```

An older map tree referenced by historical runners is:

```text
/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap
```

The production Waterbirds-100 FCV config is the source of truth:

```text
GALS/experiments/fcv_vit_waterbirds100/configs/waterbirds100_vit_s16_first_study.yaml
```

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

The full mask pickle should eventually be compacted to a verified dog-only
artifact. Do not delete the original until the compact artifact has been
checked for sample coverage, alignment, decoding, and hash/provenance.

Call these masks **author-provided** unless their annotation provenance has
been separately verified; do not automatically describe them as manually
drawn or human-made.

No production FCV SpuCoDogs study is frozen merely because the data exists.
Define and review its split, oracle, counterfactual, and selector protocol
before implementing it.

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
```

Current FCV campaign roots:

```text
/home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study
/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_susceptibility
/home/ryreu/guided_cnn/logsMNIST/fcv_vit_decoymnist_full_campaign
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

Storage is a real constraint. At the last SpuCoDogs check, the home filesystem
had roughly 31 GB free. That number is time-sensitive; always check live:

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

