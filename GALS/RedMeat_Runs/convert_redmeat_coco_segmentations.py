#!/usr/bin/env python3
"""Convert a Roboflow COCO export into RedMeat masks and review overlays.

All polygon annotations belonging to an image are unioned into one foreground
mask. The output includes binary masks for evaluation, color masks, side-by-side
review images, an audit manifest, and a self-contained HTML gallery.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


IMAGE_ID_RE = re.compile(
    r"^(?P<image_id>\d+)_jpg(?:\.rf\.[^.]+)?\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FOREGROUND_COLOR = np.asarray([235, 55, 65], dtype=np.float32)
BACKGROUND_COLOR = np.asarray([35, 90, 170], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=1250)
    parser.add_argument("--review-quality", type=int, default=88)
    return parser.parse_args()


def source_index(source_root: Path) -> Dict[str, Path]:
    paths = sorted(
        path
        for path in source_root.glob("*/*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    result: Dict[str, Path] = {}
    for path in paths:
        if path.stem in result:
            raise RuntimeError(
                f"Duplicate source image ID {path.stem}: {result[path.stem]} and {path}"
            )
        result[path.stem] = path
    return result


def parse_image_id(filename: str) -> str:
    match = IMAGE_ID_RE.match(Path(filename).name)
    if match is None:
        raise RuntimeError(f"Cannot recover RedMeat image ID from {filename!r}")
    return match.group("image_id")


def discover_coco_jsons(names: Iterable[str]) -> List[str]:
    result = sorted(
        name for name in names if name.lower().endswith("/_annotations.coco.json")
    )
    if not result:
        raise RuntimeError("Archive contains no */_annotations.coco.json files.")
    return result


def polygon_mask(
    width: int,
    height: int,
    annotations: Sequence[Dict[str, object]],
) -> Tuple[Image.Image, int]:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    polygon_count = 0
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            raise RuntimeError(
                "This converter expects nonempty COCO polygon segmentations; "
                f"annotation {annotation.get('id')} is unsupported."
            )
        for coordinates in segmentation:
            if not isinstance(coordinates, list) or len(coordinates) < 6:
                raise RuntimeError(
                    f"Invalid polygon in annotation {annotation.get('id')}: "
                    f"{coordinates!r}"
                )
            if len(coordinates) % 2:
                raise RuntimeError(
                    f"Odd coordinate count in annotation {annotation.get('id')}"
                )
            points = [
                (float(coordinates[index]), float(coordinates[index + 1]))
                for index in range(0, len(coordinates), 2)
            ]
            draw.polygon(points, fill=255)
            polygon_count += 1
    return mask, polygon_count


def color_mask(mask_array: np.ndarray) -> Image.Image:
    foreground = mask_array > 0
    colored = np.empty((*mask_array.shape, 3), dtype=np.uint8)
    colored[foreground] = FOREGROUND_COLOR.astype(np.uint8)
    colored[~foreground] = BACKGROUND_COLOR.astype(np.uint8)
    return Image.fromarray(colored, mode="RGB")


def review_panel(source: Image.Image, mask: Image.Image) -> Image.Image:
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32)
    mask_array = np.asarray(mask) > 0
    tint = np.where(mask_array[..., None], FOREGROUND_COLOR, BACKGROUND_COLOR)
    alpha = np.where(mask_array[..., None], 0.55, 0.32).astype(np.float32)
    overlay = np.clip(source_array * (1.0 - alpha) + tint * alpha, 0, 255).astype(
        np.uint8
    )

    # A thin white contour makes missed edges visible without changing the mask.
    expanded = np.asarray(mask.filter(ImageFilter.MaxFilter(3))) > 0
    contracted = np.asarray(mask.filter(ImageFilter.MinFilter(3))) > 0
    boundary = expanded != contracted
    overlay[boundary] = np.asarray([255, 255, 255], dtype=np.uint8)

    original = source.convert("RGB")
    overlay_image = Image.fromarray(overlay, mode="RGB")
    divider = 4
    panel = Image.new("RGB", (original.width * 2 + divider, original.height), "white")
    panel.paste(original, (0, 0))
    panel.paste(overlay_image, (original.width + divider, 0))
    return panel


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gallery_html(rows: Sequence[Dict[str, object]]) -> str:
    cards = []
    for row in rows:
        class_name = str(row["class_name"])
        image_id = str(row["image_id"])
        review_path = str(row["review_path"])
        binary_path = str(row["binary_mask_path"])
        color_path = str(row["color_mask_path"])
        fraction = float(row["foreground_fraction"])
        annotation_count = int(row["annotation_count"])
        polygon_count = int(row["polygon_count"])
        extreme = fraction < 0.01 or fraction > 0.95
        cards.append(
            f"""
            <article class="card{' extreme' if extreme else ''}"
                     data-class="{html.escape(class_name)}"
                     data-fraction="{fraction:.10f}"
                     data-multi="{int(annotation_count > 1 or polygon_count > 1)}">
              <a href="{html.escape(review_path)}" target="_blank">
                <img loading="lazy" src="{html.escape(review_path)}" alt="{image_id}">
              </a>
              <div class="meta">
                <strong>{image_id}</strong><span>{html.escape(class_name)}</span>
                <span>foreground: {100.0 * fraction:.2f}%</span>
                <span>merged: {annotation_count} ann. / {polygon_count} poly.</span>
                <span><a href="{html.escape(binary_path)}" target="_blank">binary</a>
                · <a href="{html.escape(color_path)}" target="_blank">color mask</a></span>
              </div>
            </article>
            """
        )

    class_options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
        for name in sorted({str(row["class_name"]) for row in rows})
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RedMeat segmentation review</title>
  <style>
    body {{ margin: 0; font: 14px system-ui, sans-serif; background: #f4f5f7; color: #16181d; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 12px 18px; background: #fff; border-bottom: 1px solid #ccd0d8; }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    .controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }}
    select, button {{ padding: 6px 9px; }}
    .legend {{ margin-left: auto; color: #555; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 12px; padding: 14px; }}
    .card {{ overflow: hidden; background: white; border: 1px solid #d4d7de; border-radius: 6px; }}
    .card.extreme {{ border: 3px solid #e7a600; }}
    .card img {{ display: block; width: 100%; aspect-ratio: 2 / 1; object-fit: contain; background: #20242b; }}
    .meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 8px; padding: 8px; font-size: 12px; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <header>
    <h1>RedMeat segmentation review · {len(rows)} images</h1>
    <div class="controls">
      <label>Class <select id="classFilter"><option value="">All</option>{class_options}</select></label>
      <label><input id="multiFilter" type="checkbox"> Multiple merged polygons only</label>
      <label>Sort <select id="sortOrder"><option value="id">Image ID</option><option value="small">Smallest foreground</option><option value="large">Largest foreground</option></select></label>
      <button id="apply">Apply</button>
      <span id="visibleCount"></span>
      <span class="legend">left: original · right: red foreground / blue background · white: boundary</span>
    </div>
  </header>
  <main id="gallery">{''.join(cards)}</main>
  <script>
    const gallery = document.getElementById('gallery');
    const cards = Array.from(gallery.children);
    function apply() {{
      const cls = document.getElementById('classFilter').value;
      const multi = document.getElementById('multiFilter').checked;
      const order = document.getElementById('sortOrder').value;
      const sorted = [...cards].sort((a, b) => {{
        if (order === 'small') return +a.dataset.fraction - +b.dataset.fraction;
        if (order === 'large') return +b.dataset.fraction - +a.dataset.fraction;
        return a.querySelector('strong').textContent.localeCompare(b.querySelector('strong').textContent, undefined, {{numeric: true}});
      }});
      let visible = 0;
      sorted.forEach(card => {{
        gallery.appendChild(card);
        const show = (!cls || card.dataset.class === cls) && (!multi || card.dataset.multi === '1');
        card.classList.toggle('hidden', !show);
        visible += show ? 1 : 0;
      }});
      document.getElementById('visibleCount').textContent = `${{visible}} visible`;
    }}
    document.getElementById('apply').addEventListener('click', apply);
    apply();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)

    sources = source_index(source_root)
    if len(sources) != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} source images, found {len(sources)}"
        )

    records: List[Tuple[str, Dict[str, object], List[Dict[str, object]]]] = []
    observed_ids = set()
    category_names = set()
    with zipfile.ZipFile(archive) as handle:
        json_paths = discover_coco_jsons(handle.namelist())
        for json_path in json_paths:
            split = json_path.split("/", 1)[0]
            payload = json.loads(handle.read(json_path))
            images = {int(image["id"]): image for image in payload.get("images", [])}
            annotations: Dict[int, List[Dict[str, object]]] = defaultdict(list)
            for annotation in payload.get("annotations", []):
                annotations[int(annotation["image_id"])].append(annotation)
            category_names.update(
                str(category.get("name")) for category in payload.get("categories", [])
            )
            for coco_id, image in images.items():
                image_id = parse_image_id(str(image["file_name"]))
                if image_id in observed_ids:
                    raise RuntimeError(f"Duplicate COCO image ID after merging splits: {image_id}")
                observed_ids.add(image_id)
                image_annotations = annotations.get(coco_id, [])
                if not image_annotations:
                    raise RuntimeError(f"Image {image_id} has no segmentation annotations.")
                records.append((split, image, image_annotations))

    if len(records) != args.expected_images or observed_ids != set(sources):
        raise RuntimeError(
            "COCO/source coverage mismatch: "
            f"records={len(records)}, source={len(sources)}, "
            f"missing={sorted(set(sources) - observed_ids)[:10]}, "
            f"extra={sorted(observed_ids - set(sources))[:10]}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for index, (roboflow_split, image_record, annotations) in enumerate(
        sorted(records, key=lambda item: parse_image_id(str(item[1]["file_name"])))
    ):
        image_id = parse_image_id(str(image_record["file_name"]))
        source_path = sources[image_id]
        class_name = source_path.parent.name
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
        coco_size = (int(image_record["width"]), int(image_record["height"]))
        if source.size != coco_size:
            raise RuntimeError(
                f"Dimension mismatch for {image_id}: source={source.size}, COCO={coco_size}"
            )

        mask, polygon_count = polygon_mask(source.width, source.height, annotations)
        mask_array = np.asarray(mask)
        foreground_pixels = int(np.count_nonzero(mask_array))
        if foreground_pixels == 0:
            raise RuntimeError(f"Rasterized mask is empty for {image_id}")
        foreground_fraction = foreground_pixels / float(source.width * source.height)

        binary_rel = Path("binary_masks") / class_name / f"{image_id}.png"
        color_rel = Path("color_masks") / class_name / f"{image_id}.png"
        review_rel = Path("review") / class_name / f"{image_id}.jpg"
        for relative in (binary_rel, color_rel, review_rel):
            (output_root / relative).parent.mkdir(parents=True, exist_ok=True)
        mask.save(output_root / binary_rel, format="PNG", optimize=True)
        color_mask(mask_array).save(output_root / color_rel, format="PNG", optimize=True)
        review_panel(source, mask).save(
            output_root / review_rel,
            format="JPEG",
            quality=int(args.review_quality),
            optimize=True,
        )

        rows.append(
            {
                "image_id": image_id,
                "class_name": class_name,
                "source_path": str(source_path),
                "roboflow_split": roboflow_split,
                "roboflow_filename": str(image_record["file_name"]),
                "width": source.width,
                "height": source.height,
                "annotation_count": len(annotations),
                "polygon_count": polygon_count,
                "foreground_pixels": foreground_pixels,
                "foreground_fraction": foreground_fraction,
                "binary_mask_path": binary_rel.as_posix(),
                "color_mask_path": color_rel.as_posix(),
                "review_path": review_rel.as_posix(),
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"[PROGRESS] {index + 1}/{len(records)}", flush=True)

    write_csv(output_root / "review_manifest.csv", rows)
    (output_root / "index.html").write_text(gallery_html(rows), encoding="utf-8")
    fractions = [float(row["foreground_fraction"]) for row in rows]
    summary = {
        "archive": str(archive),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "images": len(rows),
        "classes": {
            name: sum(row["class_name"] == name for row in rows)
            for name in sorted({str(row["class_name"]) for row in rows})
        },
        "category_names_in_export": sorted(category_names),
        "total_annotations": sum(int(row["annotation_count"]) for row in rows),
        "total_polygons": sum(int(row["polygon_count"]) for row in rows),
        "foreground_fraction_min": min(fractions),
        "foreground_fraction_median": float(np.median(fractions)),
        "foreground_fraction_max": max(fractions),
        "empty_masks": 0,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] binary masks: {output_root / 'binary_masks'}")
    print(f"[DONE] color masks:  {output_root / 'color_masks'}")
    print(f"[DONE] review images: {output_root / 'review'}")
    print(f"[DONE] gallery:       {output_root / 'index.html'}")


if __name__ == "__main__":
    main()
