# Evaluation Utilities

Lightweight scripts for qualitative saliency and Pointing Game checks.

These utilities are intentionally CSV-driven so they can be reused across
Waterbirds, RedMeat, DecoyMNIST, or custom curated subsets without adding
dataset-specific path logic.

## Manifest Format

Use a CSV with at least:

```csv
sample_id,image_path,label
example_000,/abs/path/image.jpg,1
```

For Pointing Game, also include either `mask_path` or pass masks separately in
your own generated `saliency_path` manifest:

```csv
sample_id,image_path,label,mask_path,group
example_000,/abs/path/image.jpg,1,/abs/path/mask.png,Water_on_Land
```

## Generate RISE Maps

```bash
python repro_runs/evaluation/rise_saliency.py \
  --manifest curated_examples.csv \
  --arch resnet50 \
  --num-classes 2 \
  --checkpoint /path/to/model.pth \
  --output-dir outputs/rise_maps \
  --device cuda:0
```

Supported built-in model loaders:

- `resnet50`
- `resnet50_cam` for R4RR-style checkpoints with keys under `base.*`
- `mobilenet_v2` for R4RR MobileNetV2 CAM-compatible checkpoints

The script writes one folder per sample with:

- `rise.npy`
- `rise.png`
- `overlay.png`

It also writes `saliency_manifest.csv`, which can be passed directly to the
Pointing Game script.

## Run Pointing Game

```bash
python repro_runs/evaluation/pointing_game.py \
  --manifest outputs/rise_maps/saliency_manifest.csv \
  --output-dir outputs/pointing_game
```

If your manifest does not include `saliency_path`, pass a directory containing
per-sample folders from `rise_saliency.py`:

```bash
python repro_runs/evaluation/pointing_game.py \
  --manifest curated_examples.csv \
  --saliency-dir outputs/rise_maps \
  --output-dir outputs/pointing_game
```

