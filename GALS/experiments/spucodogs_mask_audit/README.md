# SpuCoDogs audits

This directory contains two read-only audits of the downloaded SpuCoDogs
images and author-provided `spuco_animals_masks.pkl`.

The first-stage mask audit answers the mechanical questions required before
building a mask loader:

- What Python container and mask representation are stored?
- How many mask records exist?
- What shapes, dtypes, and value ranges do the masks use?
- Can mask records be matched uniquely to all 24,050 downloaded SpuCoDogs
  images using the official integer-filename indexing rule?
- Do high mask values or low mask values visually cover the dog?
- What was peak memory usage when loading the 9.4 GB pickle?

The official `SpuCoAnimals` loader computes
`mask_index = int(filename_stem)` and retrieves `masks[mask_index]`. It calls
this value the `spurious_mask`: high/true values retain the spurious
background, while the inverted low/false values retain the core animal. The
audit checks that numeric-ID alignment and still produces both polarity
overlays as an independent visual verification.

The audit does **not** modify, compact, or delete the original pickle. It also
cannot establish whether masks were manually annotated or model-generated;
that provenance must come from the dataset's source documentation.

Pickle loading can execute code. Run this only on the trusted artifact
downloaded for this project and verified against its recorded checksum.

## First-stage Tigris launch

After pulling the latest repository commit:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash experiments/spucodogs_mask_audit/submit_mask_audit.sh
```

The CPU-only job uses account `reu-aisocial`, partition `tigris`, 128 GB RAM,
and the aarch64 `fcv_gh200` environment. No GPU is requested.

Monitor it with:

```bash
squeue --me
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

The submission command prints exact stdout and stderr paths. After completion,
find the run directory with:

```bash
find /home/ryreu/guided_cnn/logsSpuCo/spucodogs_mask_audit \
  -maxdepth 1 -type d -name 'audit_*' -printf '%T@ %p\n' \
  | sort -n \
  | tail -n 1
```

Important outputs:

```text
mask_audit_report.json
image_mask_alignment.csv
mask_polarity_overlays.jpg
```

Interpretation:

- Open `mask_polarity_overlays.jpg`.
- The official loader predicts that red/high values cover the background.
- The official loader predicts that blue/low values cover the dog.
- Confirm both expectations visually before writing the compact extractor.
- Check `all_spucodogs_images_uniquely_matched` in the JSON.
- Do not delete the original pickle after this audit. First create and
  independently verify a compact dog-only artifact and loader.

## What the official SpuCoDogs loader specifies

The source of truth is:

```text
https://github.com/BigML-CS-UCLA/SpuCo/blob/master/src/spuco/datasets/spuco_dogs.py
```

The hierarchy is:

```text
spuco_dogs/
  {train,val,test}/
    {small_dogs,big_dogs}/
      {indoor,outdoor}/
        INTEGER_MASK_ID.<image extension>
```

The class label is encoded by the dog-size directory:

```text
small_dogs = target 0
big_dogs   = target 1
```

The spurious/environment label is encoded by the context directory:

```text
indoor  = environment 0
outdoor = environment 1
```

Official expected counts:

| Split | Small/indoor | Small/outdoor | Big/indoor | Big/outdoor | Total |
|---|---:|---:|---:|---:|---:|
| train | 10,000 | 500 | 500 | 10,000 | 21,000 |
| val | 500 | 25 | 25 | 500 | 1,050 |
| test | 500 | 500 | 500 | 500 | 2,000 |

The second-stage audit verifies every one of these counts against the live
Tigris tree instead of trusting the specification alone.

## Second-stage deep audit

`deep_audit_spucodogs.py` answers the remaining mechanically resolvable
questions:

- exact live hierarchy and all 12 split/target/environment counts;
- integer filename validity and globally reused dog image IDs;
- image decode failures and every image-versus-mask shape comparison;
- whether geometry matches the raw decoded image or only an EXIF-transposed
  image;
- missing, malformed, empty, full, border-touching, fragmented, and
  broad-area-warning masks;
- exact file and exact decoded-pixel duplicates across official splits;
- dHash near-duplicate candidates across splits for manual review;
- EXIF/source identity fields and whether a leakage-resistant identity key is
  actually available;
- local repository loaders and metadata/manifest candidates worth reusing;
- mask and archive integrity-receipt status;
- a canonical digest of the extracted image tree; and
- a conservative storage projection for a bit-packed dog-only mask artifact,
  a second verification copy, temporary products, and a 10 GiB free-space
  reserve.

It deliberately does not claim that an automated shape statistic can prove
that a segmentation visibly contains the whole dog. It writes a
`mask_quality_review.jpg` gallery containing stratified and extreme cases for
human inspection. Likewise, dHash pairs are review candidates, not proof of
duplicate identity.

Launch it after pulling the latest commit:

```bash
cd /home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
bash experiments/spucodogs_mask_audit/submit_deep_audit.sh
```

The CPU-only job uses `reu-aisocial`, the `tigris` partition, 8 CPUs, 128 GiB
RAM, the aarch64 `fcv_gh200` environment, and a 12-hour ceiling. It requests no
GPU and never changes the source data.

Primary outputs:

```text
spucodogs_deep_audit_report.json
official_group_counts.csv
image_mask_inventory.csv
cross_split_exact_duplicates.csv
cross_split_perceptual_duplicate_candidates.csv
mask_quality_review.jpg
repository_reuse_candidates.json
```

Inspect the JSON acceptance summary, both duplicate CSVs, and the quality
gallery before freezing an FCV protocol or writing a compact artifact.
