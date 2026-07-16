#!/usr/bin/env python3
"""Regenerate paper-style saliency grids with an added ElRep column.

This reads the old multimethod saliency artifacts from the March grid workflow
and the newer ElRep artifacts downloaded under SwitchVLM/a_download.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


SWITCHVLM_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path("/home/ryan/ComputerScience/LearnToLook/MNIST_again/deep-explanation-penalization")
DOWNLOAD_ROOT = SWITCHVLM_ROOT / "a_download"

OLD_DECOY = OLD_ROOT / "decoy_multimodel_saliency_nonzero_20260304_022211" / "samples"
ELREP_DECOY = DOWNLOAD_ROOT / "decoy_elrep_saliency_from_index_21373744" / "samples"

OLD_WB95 = OLD_ROOT / "new_maps/waterbirds_curated_saliency_up_abn_afr_21073834/waterbirds_95/samples"
OLD_WB100 = OLD_ROOT / "new_maps/waterbirds_curated_saliency_up_abn_afr_21073834/waterbirds_100/samples"
ELREP_WB95 = DOWNLOAD_ROOT / "waterbirds_elrep_curated_rise_21373734/wb95/samples"
ELREP_WB100 = DOWNLOAD_ROOT / "waterbirds_elrep_curated_rise_21373734/wb100/samples"

REDMEAT_GUIDED_ROOT = OLD_ROOT / "Best_Five_Each"
REDMEAT_CURATED_ROOT = OLD_ROOT / "newMeat/redmeat_curated_saliency_up_abn_afr_20260304_012026/samples"
ELREP_REDMEAT = DOWNLOAD_ROOT / "redmeat_elrep_curated_rise_21373716/samples"

DEFAULT_OUT = DOWNLOAD_ROOT / "grids_with_elrep"


@dataclass(frozen=True)
class GridRow:
    old_dir: Path
    elrep_dir: Optional[Path]
    curated_dir: Optional[Path] = None
    source_dir: Optional[Path] = None
    abn_replacement_heatmap: Optional[Path] = None
    group: Optional[str] = None


@dataclass(frozen=True)
class GridColumn:
    source: str
    filename: str
    label: str


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def tune_gt_overlay(img: Image.Image) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * 1.24 + 16.0, 0.0, 255.0)
    arr[..., 2] = np.clip(arr[..., 2] * 1.20 + 12.0, 0.0, 255.0)
    tuned = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return ImageEnhance.Contrast(tuned).enhance(1.20)


def safe_open_resize(path: Path, size: Tuple[int, int], is_gt_overlay: bool = False) -> Image.Image:
    if not path.exists():
        img = Image.new("RGB", size, (40, 40, 40))
        d = ImageDraw.Draw(img)
        d.text((12, 12), f"Missing:\n{path.name}", fill=(255, 90, 90))
        return img
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.Resampling.BICUBIC)
    if is_gt_overlay:
        img = tune_gt_overlay(img)
    return img


def open_fit_nearest(path: Path, size: Tuple[int, int]) -> Image.Image:
    if not path.exists():
        img = Image.new("RGB", size, (40, 40, 40))
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Missing:\n{path.name}", fill=(255, 90, 90))
        return img
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    return img


def zoom_in(img: Image.Image, factor: float) -> Image.Image:
    if factor <= 1.0:
        return img
    w, h = img.size
    zw = int(round(w * factor))
    zh = int(round(h * factor))
    up = img.resize((zw, zh), Image.Resampling.BICUBIC)
    left = max(0, (zw - w) // 2)
    top = max(0, (zh - h) // 2)
    return up.crop((left, top, left + w, top + h))


def build_overlay_from_heatmap(orig_path: Path, heatmap_path: Path, size: Tuple[int, int]) -> Image.Image:
    base = safe_open_resize(orig_path, size, is_gt_overlay=False)
    heat = safe_open_resize(heatmap_path, size, is_gt_overlay=False)
    return Image.blend(base, heat, alpha=0.52)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    cur = words[0]
    for word in words[1:]:
        trial = f"{cur} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def canonical_sample_name(name: str) -> str:
    """Return a name key shared by old and ElRep sample folder conventions."""
    rest = name.split("_", 1)[1] if "_" in name and name.split("_", 1)[0].isdigit() else name
    if "__" in rest:
        rest = rest.split("__")[-1]
    for prefix in ("Waterbird_", "Landbird_"):
        if rest.startswith(prefix):
            rest = rest[len(prefix) :]
    return rest.lower()


def index_by_key(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    out: Dict[str, Path] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        out[canonical_sample_name(d.name)] = d
    return out


def find_sample_dir(parent_dir: Path, sample_name: str) -> Path:
    exact = parent_dir / sample_name
    if exact.is_dir():
        return exact
    for d in parent_dir.iterdir():
        if d.is_dir() and d.name.endswith(sample_name):
            return d
    return exact


def sample_name_after_source(folder_name: str) -> Tuple[str, str]:
    after_idx = folder_name.split("_", 1)[1]
    source_tag, sample_name = after_idx.split("__", 1)
    return source_tag, sample_name


def waterbird_source_dir(source_tag: str, sample_name: str) -> Path:
    if source_tag == "wb95_water_Best_Picks":
        return find_sample_dir(OLD_ROOT / "wb95_water/Best_Picks", sample_name)
    if source_tag == "wb95_guided_vanilla_gals_saliency_21065978_New_Best":
        return find_sample_dir(OLD_ROOT / "wb95_guided_vanilla_gals_saliency_21065978/New_Best", sample_name)
    if source_tag == "Waterbird_Best_100":
        return find_sample_dir(OLD_ROOT / "Waterbird_Best_100", sample_name)
    if source_tag == "Landbird_Best_100":
        return find_sample_dir(OLD_ROOT / "Landbird_Best_100", sample_name)
    return Path("__missing_source__") / sample_name


def select_waterbird_rows(
    old_root: Path,
    elrep_root: Path,
    source_tag: str,
    selected_names: Sequence[str],
    abn_replacements: Optional[Dict[str, Path]] = None,
) -> List[GridRow]:
    old_lookup: Dict[str, Path] = {}
    for d in sorted(p for p in old_root.iterdir() if p.is_dir()):
        tag, sample_name = sample_name_after_source(d.name)
        if tag == source_tag:
            old_lookup[sample_name] = d
    elrep_lookup = index_by_key(elrep_root)

    rows: List[GridRow] = []
    missing_old: List[str] = []
    missing_elrep: List[str] = []
    for sample_name in selected_names:
        old_dir = old_lookup.get(sample_name)
        if old_dir is None:
            missing_old.append(sample_name)
            continue
        key = canonical_sample_name(sample_name)
        elrep_dir = elrep_lookup.get(key)
        if elrep_dir is None:
            missing_elrep.append(sample_name)
        rows.append(
            GridRow(
                old_dir=old_dir,
                elrep_dir=elrep_dir,
                source_dir=waterbird_source_dir(source_tag, sample_name),
                abn_replacement_heatmap=(abn_replacements or {}).get(sample_name),
            )
        )
    if missing_old or missing_elrep:
        raise RuntimeError(f"Waterbirds missing old={missing_old} elrep={missing_elrep}")
    return rows


def draw_grid(
    rows: Sequence[GridRow],
    columns: Sequence[GridColumn],
    out_path: Path,
    title: str,
    first_image_name: str,
    tile_size: Optional[Tuple[int, int]] = None,
    font_title_size: int = 64,
    font_header_size: int = 42,
    cell_pad: int = 14,
    gap: int = 28,
    title_h: int = 148,
    col_header_h: int = 132,
    nearest: bool = False,
    include_title: bool = True,
) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {out_path}")
    first_img = rows[0].old_dir / first_image_name
    if tile_size is None:
        tile_size = Image.open(first_img).convert("RGB").size
    tile_w, tile_h = tile_size

    cell_w = tile_w + 2 * cell_pad
    cell_h = tile_h + 2 * cell_pad
    n_rows = len(rows)
    n_cols = len(columns)

    margin = 48 if tile_w > 320 else 56
    top_margin = margin if include_title else 10
    effective_title_h = title_h if include_title else 0
    canvas_w = margin + n_cols * cell_w + (n_cols - 1) * gap + margin
    canvas_h = top_margin + effective_title_h + col_header_h + gap + n_rows * cell_h + (n_rows - 1) * gap + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    font_title = load_font(font_title_size)
    font_header = load_font(font_header_size)

    if include_title:
        draw.rectangle([0, 0, canvas_w, margin + title_h], fill=(24, 33, 48))
        draw.text((margin, margin + 14), title, fill=(245, 248, 255), font=font_title)
    draw.rectangle(
        [0, top_margin + effective_title_h, canvas_w, top_margin + effective_title_h + col_header_h + gap],
        fill=(233, 237, 243),
    )

    start_x = margin
    col_y = top_margin + effective_title_h
    header_top_pad = 14 if not include_title else 22
    for j, col in enumerate(columns):
        x = start_x + j * (cell_w + gap)
        lines = wrap_text(draw, col.label, font_header, cell_w - 24)
        heights = [draw.textbbox((0, 0), ln, font=font_header)[3] for ln in lines]
        line_gap = 4
        block_h = sum(heights) + line_gap * (len(lines) - 1)
        y = col_y + header_top_pad + max(0, (col_header_h - header_top_pad - block_h) // 2)
        for ln, h in zip(lines, heights):
            tw = draw.textbbox((0, 0), ln, font=font_header)[2]
            draw.text(
                (x + (cell_w - tw) // 2, y),
                ln,
                fill=(22, 30, 44),
                font=font_header,
                stroke_width=1,
                stroke_fill=(246, 248, 252),
            )
            y += h + line_gap

    top_y = col_y + col_header_h + gap
    for i, row in enumerate(rows):
        row_y = top_y + i * (cell_h + gap)
        for j, col in enumerate(columns):
            x = start_x + j * (cell_w + gap)
            if col.source == "old":
                base = row.old_dir
            elif col.source == "curated":
                base = row.curated_dir or row.old_dir
            elif col.source == "source":
                base = row.source_dir or row.old_dir
            elif col.source == "elrep":
                base = row.elrep_dir
            else:
                raise ValueError(col.source)
            img_path = (base / col.filename) if base is not None else Path("__missing__") / col.filename
            if (
                col.label == "ABN"
                and col.filename == "abn_saliency_overlay_blue_red.png"
                and row.abn_replacement_heatmap is not None
            ):
                original = row.old_dir / "image_rgb.png"
                if not original.exists():
                    original = row.old_dir / "original_image.png"
                img = build_overlay_from_heatmap(original, row.abn_replacement_heatmap, tile_size)
            elif nearest:
                img = open_fit_nearest(img_path, tile_size)
            else:
                img = safe_open_resize(
                    img_path,
                    tile_size,
                    is_gt_overlay=col.filename in {"gt_mask_overlay_blue_red.png", "gt_mask_prediction_cmap.png"},
                )
            if row.group == "Steak":
                img = zoom_in(img, factor=1.12)
            draw.rectangle(
                [x, row_y, x + cell_w, row_y + cell_h],
                fill=(255, 255, 255),
                outline=(188, 194, 204),
                width=3 if not nearest else 15,
            )
            canvas.paste(img, (x + cell_pad, row_y + cell_pad))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", compress_level=0)
    print(f"Wrote {out_path}")


WB_COLUMNS = [
    GridColumn("old", "image_rgb.png", "Image"),
    GridColumn("old", "gt_mask_overlay_blue_red.png", "Teacher maps"),
    GridColumn("source", "guided_saliency_overlay_blue_red.png", "(Ours) Region-Guided"),
    GridColumn("source", "gals_vit_saliency_overlay_blue_red.png", "GALS"),
    GridColumn("source", "vanilla_saliency_overlay_blue_red.png", "Vanilla"),
    GridColumn("elrep", "elrep_saliency_overlay_blue_red.png", "ElRep"),
    GridColumn("old", "abn_saliency_overlay_blue_red.png", "ABN"),
    GridColumn("old", "afr_saliency_overlay_blue_red.png", "AFR"),
    GridColumn("old", "upweight_saliency_overlay_blue_red.png", "Upweight"),
]

DECOY_COLUMNS = [
    GridColumn("old", "original_image.png", "Image"),
    GridColumn("old", "gt_mask_prediction_cmap.png", "Teacher maps"),
    GridColumn("old", "guided_saliency_heatmap_blue_red.png", "(Ours) Region-Guided"),
    GridColumn("old", "gals_saliency_heatmap_blue_red.png", "GALS"),
    GridColumn("old", "vanilla_saliency_heatmap_blue_red.png", "Vanilla"),
    GridColumn("elrep", "elrep_saliency_heatmap_blue_red.png", "ElRep"),
    GridColumn("old", "abn_saliency_heatmap_blue_red.png", "ABN"),
    GridColumn("old", "afr_saliency_heatmap_blue_red.png", "AFR"),
    GridColumn("old", "upweight_saliency_heatmap_blue_red.png", "Upweight"),
]

REDMEAT_COLUMNS = [
    GridColumn("old", "original_image.png", "Image"),
    GridColumn("old", "gt_mask_overlay_blue_red.png", "Teacher maps"),
    GridColumn("old", "guided_saliency_overlay_blue_red.png", "(Ours) Region-Guided"),
    GridColumn("old", "gals_saliency_overlay_blue_red.png", "GALS"),
    GridColumn("old", "vanilla_saliency_overlay_blue_red.png", "Vanilla"),
    GridColumn("elrep", "elrep_saliency_overlay_blue_red.png", "ElRep"),
    GridColumn("curated", "abn_saliency_overlay_blue_red.png", "ABN"),
    GridColumn("curated", "afr_saliency_overlay_blue_red.png", "AFR"),
    GridColumn("curated", "upweight_saliency_overlay_blue_red.png", "Upweight"),
]


WB95_WATER = [
    "106_Horned_Puffin__Horned_Puffin_0024_100620_jpg",
    "144_Common_Tern__Common_Tern_0117_148944_jpg",
    "060_Glaucous_winged_Gull__Glaucous_Winged_Gull_0110_44377_jpg",
    "146_Forsters_Tern__Forsters_Tern_0127_150418_jpg",
    "021_Eastern_Towhee__Eastern_Towhee_0101_22559_jpg",
    "084_Red_legged_Kittiwake__Red_Legged_Kittiwake_0036_73814_jpg",
    "147_Least_Tern__Least_Tern_0082_154396_jpg",
    "101_White_Pelican__White_Pelican_0010_96876_jpg",
]

WB95_LAND = [
    "082_Ringed_Kingfisher__Ringed_Kingfisher_0050_73002_jpg",
    "011_Rusty_Blackbird__Rusty_Blackbird_0113_6664_jpg",
    "038_Great_Crested_Flycatcher__Great_Crested_Flycatcher_0009_29831_jpg",
    "160_Black_throated_Blue_Warbler__Black_Throated_Blue_Warbler_0081_161427_jpg",
    "171_Myrtle_Warbler__Myrtle_Warbler_0037_166690_jpg",
    "069_Rufous_Hummingbird__Rufous_Hummingbird_0095_60360_jpg",
    "019_Gray_Catbird__Gray_Catbird_0094_21303_jpg",
    "198_Rock_Wren__Rock_Wren_0019_188968_jpg",
]

WB100_WATER = [
    "060_Glaucous_winged_Gull__Glaucous_Winged_Gull_0012_44264_jpg",
    "084_Red_legged_Kittiwake__Red_Legged_Kittiwake_0068_795430_jpg",
    "100_Brown_Pelican__Brown_Pelican_0077_93464_jpg",
    "005_Crested_Auklet__Crested_Auklet_0071_785255_jpg",
    "087_Mallard__Mallard_0052_76946_jpg",
    "106_Horned_Puffin__Horned_Puffin_0056_101030_jpg",
    "072_Pomarine_Jaeger__Pomarine_Jaeger_0078_795758_jpg",
    "046_Gadwall__Gadwall_0035_30985_jpg",
]

WB100_LAND = [
    "097_Orchard_Oriole__Orchard_Oriole_0006_91724_jpg",
    "057_Rose_breasted_Grosbeak__Rose_Breasted_Grosbeak_0114_39770_jpg",
    "009_Brewer_Blackbird__Brewer_Blackbird_0140_2586_jpg",
    "018_Spotted_Catbird__Spotted_Catbird_0010_19436_jpg",
    "136_Barn_Swallow__Barn_Swallow_0045_130244_jpg",
    "080_Green_Kingfisher__Green_Kingfisher_0004_71076_jpg",
    "165_Chestnut_sided_Warbler__Chestnut_Sided_Warbler_0014_163801_jpg",
    "178_Swainson_Warbler__Swainson_Warbler_0011_174680_jpg",
]

WB95_LAND_ABN_REPLACEMENTS = {
    "011_Rusty_Blackbird__Rusty_Blackbird_0113_6664_jpg": OLD_ROOT
    / "abn_replace/samples/011_183_Northern_Waterthrush__Northern_Waterthrush_0090_177283/abn_saliency_heatmap_blue_red.png",
    "038_Great_Crested_Flycatcher__Great_Crested_Flycatcher_0009_29831_jpg": OLD_ROOT
    / "abn_replace/samples/012_053_Western_Grebe__Western_Grebe_0025_36251/abn_saliency_heatmap_blue_red.png",
    "171_Myrtle_Warbler__Myrtle_Warbler_0037_166690_jpg": OLD_ROOT
    / "abn_replace/samples/017_035_Purple_Finch__Purple_Finch_0071_27443/abn_saliency_heatmap_blue_red.png",
    "069_Rufous_Hummingbird__Rufous_Hummingbird_0095_60360_jpg": OLD_ROOT
    / "abn_replace/samples/021_166_Golden_winged_Warbler__Golden_Winged_Warbler_0060_164368/abn_saliency_heatmap_blue_red.png",
}

WB100_WATER_ABN_REPLACEMENTS = {
    "005_Crested_Auklet__Crested_Auklet_0071_785255_jpg": OLD_ROOT
    / "abn_replace/samples/026_129_Song_Sparrow__Song_Sparrow_0044_121931/abn_saliency_heatmap_blue_red.png",
}

WB100_LAND_ABN_REPLACEMENTS = {
    "009_Brewer_Blackbird__Brewer_Blackbird_0140_2586_jpg": OLD_ROOT
    / "abn_replace/samples/027_035_Purple_Finch__Purple_Finch_0014_27322/abn_saliency_heatmap_blue_red.png",
    "018_Spotted_Catbird__Spotted_Catbird_0010_19436_jpg": OLD_ROOT
    / "abn_replace/samples/028_147_Least_Tern__Least_Tern_0133_153816/abn_saliency_heatmap_blue_red.png",
    "136_Barn_Swallow__Barn_Swallow_0045_130244_jpg": OLD_ROOT
    / "abn_replace/samples/030_001_Black_footed_Albatross__Black_Footed_Albatross_0032_796115/abn_saliency_heatmap_blue_red.png",
    "165_Chestnut_sided_Warbler__Chestnut_Sided_Warbler_0014_163801_jpg": OLD_ROOT
    / "abn_replace/samples/034_101_White_Pelican__White_Pelican_0050_97913/abn_saliency_heatmap_blue_red.png",
}


def decoy_rows() -> List[GridRow]:
    elrep_lookup = {d.name: d for d in sorted(ELREP_DECOY.iterdir()) if d.is_dir()}
    rows: List[GridRow] = []
    for d in sorted(p for p in OLD_DECOY.iterdir() if p.is_dir()):
        rows.append(GridRow(old_dir=d, elrep_dir=elrep_lookup.get(d.name)))
    return rows


def decoy_first_per_class_rows() -> List[GridRow]:
    overrides = {5: "059_digit5_057470_y5", 7: "075_digit7_053372_y7"}
    rows = decoy_rows()
    by_name = {r.old_dir.name: r for r in rows}
    chosen: List[GridRow] = []
    for label in range(10):
        if label in overrides:
            chosen.append(by_name[overrides[label]])
            continue
        candidates = [r for r in rows if r.old_dir.name.endswith(f"_y{label}")]
        if not candidates:
            raise RuntimeError(f"No DecoyMNIST samples for class {label}")
        chosen.append(candidates[0])
    return chosen


def decoy_class_rows(label: int, count: int = 5) -> List[GridRow]:
    rows = [r for r in decoy_rows() if r.old_dir.name.endswith(f"_y{label}")]
    if len(rows) < count:
        raise RuntimeError(f"Need {count} DecoyMNIST class {label} samples, found {len(rows)}")
    return rows[:count]


def build_decoy(out_root: Path, include_title: bool = True) -> None:
    title_map = {
        0: "Zeroes in DecoyMNIST",
        1: "Ones in DecoyMNIST",
        2: "Twos in DecoyMNIST",
        3: "Threes in DecoyMNIST",
        4: "Fours in DecoyMNIST",
        5: "Fives in DecoyMNIST",
        6: "Sixes in DecoyMNIST",
        7: "Sevens in DecoyMNIST",
        8: "Eights in DecoyMNIST",
        9: "Nines in DecoyMNIST",
    }
    header_font = 47 if not include_title else 40
    header_h = 166 if not include_title else 126
    draw_grid(
        decoy_first_per_class_rows(),
        DECOY_COLUMNS,
        out_root / "decoy" / "DecoyMNIST_FirstPerClass_ElRep_Grid.png",
        "Saliency on DecoyMNIST",
        "original_image.png",
        tile_size=(320, 320),
        font_title_size=58,
        font_header_size=header_font,
        cell_pad=0,
        gap=34,
        title_h=148,
        col_header_h=header_h,
        nearest=True,
        include_title=include_title,
    )
    for label in range(10):
        draw_grid(
            decoy_class_rows(label),
            DECOY_COLUMNS,
            out_root / "decoy" / f"DecoyMNIST_Class{label}_Heatmap_ElRep_Grid.png",
            title_map[label],
            "original_image.png",
            tile_size=(320, 320),
            font_title_size=58,
            font_header_size=header_font,
            cell_pad=0,
            gap=34,
            title_h=148,
            col_header_h=header_h,
            nearest=True,
            include_title=include_title,
        )


def build_waterbirds(out_root: Path, include_title: bool = True) -> None:
    tasks = [
        (
            OLD_WB95,
            ELREP_WB95,
            "wb95_water_Best_Picks",
            WB95_WATER,
            {},
            "Waterbirds from Waterbirds95 Dataset",
            "Waterbirds95_Waterbirds_AllMethods_ElRep_grid.png",
        ),
        (
            OLD_WB95,
            ELREP_WB95,
            "wb95_guided_vanilla_gals_saliency_21065978_New_Best",
            WB95_LAND,
            WB95_LAND_ABN_REPLACEMENTS,
            "Landbirds from the Waterbirds95 Dataset",
            "Waterbirds95_Landbirds_AllMethods_ElRep_grid.png",
        ),
        (
            OLD_WB100,
            ELREP_WB100,
            "Waterbird_Best_100",
            WB100_WATER,
            WB100_WATER_ABN_REPLACEMENTS,
            "Waterbirds from Waterbirds100 Dataset",
            "Waterbirds100_Waterbirds_AllMethods_ElRep_grid.png",
        ),
        (
            OLD_WB100,
            ELREP_WB100,
            "Landbird_Best_100",
            WB100_LAND,
            WB100_LAND_ABN_REPLACEMENTS,
            "Landbirds from the Waterbirds100 Dataset",
            "Waterbirds100_Landbirds_AllMethods_ElRep_grid.png",
        ),
    ]
    header_font = 56 if not include_title else 38
    header_h = 174 if not include_title else 132
    for old_root, elrep_root, source_tag, selected, replacements, title, fname in tasks:
        rows = select_waterbird_rows(old_root, elrep_root, source_tag, selected, replacements)
        draw_grid(
            rows,
            WB_COLUMNS,
            out_root / "waterbirds" / fname,
            title,
            "image_rgb.png",
            font_title_size=64,
            font_header_size=header_font,
            col_header_h=header_h,
            include_title=include_title,
        )


def redmeat_curated_lookup() -> Dict[str, Path]:
    return index_by_key(REDMEAT_CURATED_ROOT)


def redmeat_elrep_lookup() -> Dict[str, Path]:
    return index_by_key(ELREP_REDMEAT)


def redmeat_rows_for_group(group: str) -> List[GridRow]:
    group_dir = REDMEAT_GUIDED_ROOT / group
    curated = redmeat_curated_lookup()
    elrep = redmeat_elrep_lookup()
    rows: List[GridRow] = []
    for d in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        key = canonical_sample_name(d.name)
        rows.append(GridRow(old_dir=d, curated_dir=curated.get(key), elrep_dir=elrep.get(key), group=group))
    return rows


def build_redmeat(out_root: Path, include_title: bool = True) -> None:
    groups = ["Baby_Back_Ribs", "Filet_Mignon", "Pork_Chop", "Prime_Rib", "Steak"]
    header_font = 56 if not include_title else 38
    header_h = 174 if not include_title else 132
    for group in groups:
        draw_grid(
            redmeat_rows_for_group(group),
            REDMEAT_COLUMNS,
            out_root / "redmeat" / f"{group}_ElRep_grid.png",
            group.replace("_", " "),
            "original_image.png",
            font_title_size=64,
            font_header_size=header_font,
            col_header_h=header_h,
            include_title=include_title,
        )

    picks = [
        ("Baby_Back_Ribs", 2),
        ("Filet_Mignon", 5),
        ("Pork_Chop", 4),
        ("Prime_Rib", 1),
        ("Steak", 3),
    ]
    selected_rows: List[GridRow] = []
    for group, pick_1based in picks:
        rows = redmeat_rows_for_group(group)
        selected_rows.append(rows[pick_1based - 1])
    draw_grid(
        selected_rows,
        REDMEAT_COLUMNS,
        out_root / "redmeat" / "One_Per_Category_ElRep_grid.png",
        "Red Meat Dataset",
        "original_image.png",
        font_title_size=64,
        font_header_size=header_font,
        col_header_h=header_h,
        include_title=include_title,
    )


def validate_rows(rows: Iterable[GridRow], columns: Sequence[GridColumn]) -> List[str]:
    missing: List[str] = []
    for row in rows:
        for col in columns:
            if col.source == "old":
                base = row.old_dir
            elif col.source == "curated":
                base = row.curated_dir
            elif col.source == "source":
                base = row.source_dir
            elif col.source == "elrep":
                base = row.elrep_dir
            else:
                base = None
            if base is None or not (base / col.filename).exists():
                missing.append(f"{row.old_dir.name}: {col.label} -> {col.filename}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--which",
        choices=("all", "decoy", "waterbirds", "redmeat"),
        default="all",
    )
    parser.add_argument("--no-title", action="store_true", help="Remove the top dark title banner.")
    args = parser.parse_args()
    include_title = not args.no_title

    if args.which in {"all", "decoy"}:
        missing = validate_rows(decoy_first_per_class_rows(), DECOY_COLUMNS)
        if missing:
            raise RuntimeError(f"Missing Decoy files: {missing[:10]}")
        build_decoy(args.out_root, include_title=include_title)
    if args.which in {"all", "waterbirds"}:
        build_waterbirds(args.out_root, include_title=include_title)
    if args.which in {"all", "redmeat"}:
        build_redmeat(args.out_root, include_title=include_title)
    print(f"[DONE] grids written under {args.out_root}")


if __name__ == "__main__":
    main()
