# SpuCoDogs mask audit

This is a read-only audit of the downloaded author-provided
`spuco_animals_masks.pkl`. It answers the mechanical questions required before
building an FCV SpuCoDogs mask loader:

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

## Tigris launch

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
