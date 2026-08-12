#!/usr/bin/env python3
"""Build a deterministic, leakage-audited ImageNet-9 reconstruction.

The official ImageNet-9 training archive is no longer available from its
published Dropbox URL. This tool reconstructs the documented core protocol
from ImageNet-2012 images and localization annotations without copying the
shared ImageNet image files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


CLASS_NAMES = (
    "dog",
    "bird",
    "vehicle",
    "reptile",
    "carnivore",
    "insect",
    "instrument",
    "primate",
    "fish",
)
CLASS_DIRS = tuple(f"{index:02d}_{name}" for index, name in enumerate(CLASS_NAMES))
EXPECTED_SUBCLASS_COUNTS = (116, 52, 42, 36, 35, 27, 26, 20, 16)
OFFICIAL_VARIANTS = (
    "original",
    "mixed_same",
    "mixed_rand",
    "mixed_next",
    "only_fg",
    "only_bg_b",
    "only_bg_t",
    "no_fg",
)
IMAGE_EXTENSIONS = (".JPEG", ".jpeg", ".JPG", ".jpg", ".PNG", ".png")
IMAGENET_ID_RE = re.compile(r"n\d{8}_\d+")


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    label: int
    class_name: str
    class_dir: str
    imagenet_index: int
    synset: str
    source_path: str
    annotation_path: str
    image_width: int
    image_height: int
    bbox_xmin: int
    bbox_ymin: int
    bbox_xmax: int
    bbox_ymax: int


@dataclass(frozen=True)
class Rejection:
    sample_id: str
    imagenet_index: int
    synset: str
    label: int
    reason: str
    annotation_path: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=Path("/shared/rc/datasets/imagenet2012"),
        help="ImageNet root containing extracted train/<synset> directories.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        required=True,
        help="Extracted ILSVRC2012_bbox_train_v2 annotation directory.",
    )
    parser.add_argument(
        "--official-test-root",
        type=Path,
        required=True,
        help="Official bg_challenge directory containing test variants.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="ImageNet-9 root where metadata and optional link trees are written.",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=here / "assets" / "in_to_in9.json",
        help="Official MadryLab ImageNet-index to IN-9-label mapping.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-per-class", type=int, default=5045)
    parser.add_argument("--val-per-class", type=int, default=450)
    parser.add_argument(
        "--protocol-name",
        default="reconstructed_original_bbox1_v1",
        help="Stable name used for metadata and link output directories.",
    )
    parser.add_argument(
        "--materialize-links",
        action="store_true",
        help="Create an ImageFolder-compatible tree of symlinks to shared images.",
    )
    parser.add_argument(
        "--overwrite-links",
        action="store_true",
        help="Replace an existing link tree for this protocol.",
    )
    parser.add_argument(
        "--allow-nonstandard-test-counts",
        action="store_true",
        help="Do not fail when official variants are not 450 images per class.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(path: Path) -> Dict[int, int]:
    with path.open() as handle:
        raw = json.load(handle)
    mapping = {int(index): int(label) for index, label in raw.items()}
    if set(mapping) != set(range(1000)):
        raise RuntimeError(f"Mapping must contain ImageNet indices 0..999: {path}")
    if not set(mapping.values()).issubset(set(range(9)) | {-1}):
        raise RuntimeError(f"Mapping contains invalid IN-9 labels: {path}")
    counts = Counter(label for label in mapping.values() if label >= 0)
    observed = tuple(counts[index] for index in range(9))
    if observed != EXPECTED_SUBCLASS_COUNTS:
        raise RuntimeError(
            "Official mapping subclass counts do not match the paper: "
            f"observed={observed} expected={EXPECTED_SUBCLASS_COUNTS}"
        )
    return mapping


def discover_synsets(train_root: Path) -> List[str]:
    synsets = sorted(
        path.name
        for path in train_root.iterdir()
        if path.is_dir() and re.fullmatch(r"n\d{8}", path.name)
    )
    if len(synsets) != 1000:
        raise RuntimeError(
            f"Expected 1000 extracted ImageNet train synsets under {train_root}; "
            f"found {len(synsets)}"
        )
    return synsets


def discover_annotation_dirs(annotation_root: Path) -> Dict[str, Path]:
    found: MutableMapping[str, List[Path]] = defaultdict(list)
    for root, directories, _files in os.walk(annotation_root):
        for directory in directories:
            if re.fullmatch(r"n\d{8}", directory):
                found[directory].append(Path(root) / directory)

    duplicates = {key: value for key, value in found.items() if len(value) > 1}
    if duplicates:
        preview = {key: [str(path) for path in value] for key, value in list(duplicates.items())[:5]}
        raise RuntimeError(f"Duplicate annotation directories found: {preview}")
    return {key: value[0] for key, value in found.items()}


def imagenet_ids_in_name(name: str) -> Tuple[str, ...]:
    return tuple(IMAGENET_ID_RE.findall(name))


def collect_official_test_ids(official_test_root: Path) -> Set[str]:
    identifiers: Set[str] = set()
    for root, _directories, files in os.walk(official_test_root):
        for filename in files:
            if Path(filename).suffix in IMAGE_EXTENSIONS:
                identifiers.update(imagenet_ids_in_name(filename))
    return identifiers


def class_label_from_dir(path: Path) -> int:
    match = re.match(r"(\d{2})_", path.name)
    if not match:
        raise RuntimeError(f"Expected official class directory like 00_dog, got: {path}")
    label = int(match.group(1))
    if not 0 <= label < 9:
        raise RuntimeError(f"Invalid IN-9 class directory: {path}")
    return label


def build_official_test_manifest(
    official_test_root: Path,
    allow_nonstandard_counts: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, int]]]:
    rows: List[Dict[str, object]] = []
    counts: Dict[str, Dict[str, int]] = {}
    for variant in OFFICIAL_VARIANTS:
        split_root = official_test_root / variant / "val"
        if not split_root.is_dir():
            raise RuntimeError(f"Missing official test variant: {split_root}")
        class_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
        if len(class_dirs) != 9:
            raise RuntimeError(f"Expected 9 classes under {split_root}; found {len(class_dirs)}")

        variant_counts: Dict[str, int] = {}
        for class_dir in class_dirs:
            label = class_label_from_dir(class_dir)
            images = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix in IMAGE_EXTENSIONS
            )
            variant_counts[str(label)] = len(images)
            if not allow_nonstandard_counts and len(images) != 450:
                raise RuntimeError(
                    f"Expected 450 images in {variant}/{class_dir.name}; found {len(images)}"
                )
            for image_path in images:
                identifiers = imagenet_ids_in_name(image_path.name)
                rows.append(
                    {
                        "variant": variant,
                        "label": label,
                        "class_name": CLASS_NAMES[label],
                        "class_dir": class_dir.name,
                        "relative_path": str(image_path.relative_to(official_test_root)),
                        "source_ids": ";".join(identifiers),
                    }
                )
        counts[variant] = variant_counts
    return rows, counts


def annotation_image_id(root: ET.Element, annotation_path: Path) -> str:
    filename = (root.findtext("filename") or annotation_path.stem).strip()
    return Path(filename).stem


def find_source_image(train_root: Path, synset: str, sample_id: str) -> Optional[Path]:
    synset_root = train_root / synset
    for extension in IMAGE_EXTENSIONS:
        candidate = synset_root / f"{sample_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def parse_annotation(
    annotation_path: Path,
    imagenet_index: int,
    synset: str,
    label: int,
    train_root: Path,
    official_test_ids: Set[str],
) -> Tuple[Optional[Candidate], Optional[Rejection]]:
    try:
        root = ET.parse(annotation_path).getroot()
    except (ET.ParseError, OSError) as error:
        return None, Rejection(
            annotation_path.stem,
            imagenet_index,
            synset,
            label,
            f"xml_error:{type(error).__name__}",
            str(annotation_path),
        )

    sample_id = annotation_image_id(root, annotation_path)
    boxes = [obj.find("bndbox") for obj in root.findall("object")]
    boxes = [box for box in boxes if box is not None]
    if len(boxes) != 1:
        return None, Rejection(
            sample_id,
            imagenet_index,
            synset,
            label,
            f"bbox_count_{len(boxes)}",
            str(annotation_path),
        )
    if sample_id in official_test_ids:
        return None, Rejection(
            sample_id,
            imagenet_index,
            synset,
            label,
            "official_test_overlap",
            str(annotation_path),
        )

    source_path = find_source_image(train_root, synset, sample_id)
    if source_path is None:
        return None, Rejection(
            sample_id,
            imagenet_index,
            synset,
            label,
            "source_image_missing",
            str(annotation_path),
        )

    size = root.find("size")
    box = boxes[0]
    try:
        width = int(float(size.findtext("width"))) if size is not None else 0
        height = int(float(size.findtext("height"))) if size is not None else 0
        xmin = int(float(box.findtext("xmin")))
        ymin = int(float(box.findtext("ymin")))
        xmax = int(float(box.findtext("xmax")))
        ymax = int(float(box.findtext("ymax")))
    except (TypeError, ValueError):
        return None, Rejection(
            sample_id,
            imagenet_index,
            synset,
            label,
            "invalid_bbox_values",
            str(annotation_path),
        )
    if width <= 0 or height <= 0 or not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        return None, Rejection(
            sample_id,
            imagenet_index,
            synset,
            label,
            "invalid_bbox_geometry",
            str(annotation_path),
        )

    return Candidate(
        sample_id=sample_id,
        label=label,
        class_name=CLASS_NAMES[label],
        class_dir=CLASS_DIRS[label],
        imagenet_index=imagenet_index,
        synset=synset,
        source_path=str(source_path),
        annotation_path=str(annotation_path),
        image_width=width,
        image_height=height,
        bbox_xmin=xmin,
        bbox_ymin=ymin,
        bbox_xmax=xmax,
        bbox_ymax=ymax,
    ), None


def build_candidates(
    train_root: Path,
    annotation_dirs: Mapping[str, Path],
    synsets: Sequence[str],
    mapping: Mapping[int, int],
    official_test_ids: Set[str],
) -> Tuple[List[Candidate], List[Rejection], Dict[str, int]]:
    candidates: List[Candidate] = []
    rejections: List[Rejection] = []
    counters: Counter[str] = Counter()

    selected_indices = [index for index, label in mapping.items() if label >= 0]
    for position, imagenet_index in enumerate(sorted(selected_indices), start=1):
        synset = synsets[imagenet_index]
        label = mapping[imagenet_index]
        annotation_dir = annotation_dirs.get(synset)
        if annotation_dir is None:
            counters["mapped_synsets_without_annotations"] += 1
            continue
        annotation_paths = sorted(annotation_dir.glob("*.xml"))
        counters["annotations_scanned"] += len(annotation_paths)
        for annotation_path in annotation_paths:
            candidate, rejection = parse_annotation(
                annotation_path,
                imagenet_index,
                synset,
                label,
                train_root,
                official_test_ids,
            )
            if candidate is not None:
                candidates.append(candidate)
                counters["eligible_candidates"] += 1
            elif rejection is not None:
                rejections.append(rejection)
                counters[f"rejected:{rejection.reason}"] += 1
        if position % 25 == 0 or position == len(selected_indices):
            print(
                f"[SCAN] synsets={position}/{len(selected_indices)} "
                f"annotations={counters['annotations_scanned']} "
                f"eligible={counters['eligible_candidates']}",
                flush=True,
            )
    return candidates, rejections, dict(sorted(counters.items()))


def select_splits(
    candidates: Sequence[Candidate],
    seed: int,
    train_per_class: int,
    val_per_class: int,
) -> Tuple[List[Tuple[str, int, Candidate]], Dict[str, Dict[str, int]]]:
    by_label: MutableMapping[int, List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_label[candidate.label].append(candidate)

    selected: List[Tuple[str, int, Candidate]] = []
    counts: Dict[str, Dict[str, int]] = {}
    required = train_per_class + val_per_class
    for label in range(9):
        pool = sorted(by_label[label], key=lambda item: item.sample_id)
        if len(pool) < required:
            raise RuntimeError(
                f"Class {label} ({CLASS_NAMES[label]}) has {len(pool)} eligible candidates; "
                f"need {required} ({train_per_class} train + {val_per_class} val)."
            )
        rng = random.Random(seed + label * 1_000_003)
        rng.shuffle(pool)
        train_rows = pool[:train_per_class]
        val_rows = pool[train_per_class:required]
        selected.extend(("train", rank, item) for rank, item in enumerate(train_rows))
        selected.extend(("val", rank, item) for rank, item in enumerate(val_rows))
        counts[str(label)] = {
            "eligible": len(pool),
            "train": len(train_rows),
            "val": len(val_rows),
            "unused": len(pool) - required,
        }
    selected.sort(key=lambda item: (item[0], item[2].label, item[1]))
    return selected, counts


def candidate_counts_by_class(candidates: Sequence[Candidate]) -> Dict[str, Dict[str, object]]:
    counts = Counter(candidate.label for candidate in candidates)
    return {
        str(label): {
            "class_name": CLASS_NAMES[label],
            "eligible": counts[label],
        }
        for label in range(9)
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def selected_rows(selected: Sequence[Tuple[str, int, Candidate]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split, rank, candidate in selected:
        row = {"split": split, "selection_rank": rank, **asdict(candidate)}
        rows.append(row)
    return rows


def materialize_links(
    link_root: Path,
    selected: Sequence[Tuple[str, int, Candidate]],
    overwrite: bool,
) -> None:
    if link_root.exists() and overwrite:
        shutil.rmtree(link_root)
    if link_root.exists() and any(link_root.iterdir()):
        raise RuntimeError(
            f"Link root already exists and is non-empty: {link_root}. "
            "Use --overwrite-links to rebuild it."
        )

    for split, _rank, candidate in selected:
        destination_dir = link_root / split / candidate.class_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        source = Path(candidate.source_path)
        destination = destination_dir / source.name
        destination.symlink_to(source)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.train_per_class <= 0 or args.val_per_class <= 0:
        raise ValueError("--train-per-class and --val-per-class must be positive")

    train_root = args.imagenet_root / "train"
    for required_path in (
        train_root,
        args.annotation_root,
        args.official_test_root,
        args.mapping_json,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    print(f"[INFO] ImageNet train: {train_root}")
    print(f"[INFO] annotations: {args.annotation_root}")
    print(f"[INFO] official test: {args.official_test_root}")
    print(f"[INFO] output root: {args.output_root}")
    print(
        f"[INFO] requested split: train={args.train_per_class}/class "
        f"val={args.val_per_class}/class seed={args.seed}"
    )

    mapping = load_mapping(args.mapping_json)
    synsets = discover_synsets(train_root)
    annotation_dirs = discover_annotation_dirs(args.annotation_root)
    print(f"[INFO] annotation synset directories: {len(annotation_dirs)}")

    test_rows, test_counts = build_official_test_manifest(
        args.official_test_root,
        args.allow_nonstandard_test_counts,
    )
    official_test_ids = collect_official_test_ids(args.official_test_root)
    print(
        f"[INFO] official test rows={len(test_rows)} "
        f"unique ImageNet source IDs={len(official_test_ids)}"
    )

    candidates, rejections, audit_counts = build_candidates(
        train_root,
        annotation_dirs,
        synsets,
        mapping,
        official_test_ids,
    )

    metadata_root = args.output_root / "metadata" / args.protocol_name
    metadata_root.mkdir(parents=True, exist_ok=True)
    selected_manifest = metadata_root / "manifest.csv"
    candidate_manifest = metadata_root / "eligible_candidates.csv"
    rejection_manifest = metadata_root / "rejections.csv"
    test_manifest = metadata_root / "official_test_manifest.csv"
    audit_summary_path = metadata_root / "audit_summary.json"
    summary_path = metadata_root / "summary.json"

    # A failed rebuild must not leave a stale successful split declaration.
    selected_manifest.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)

    candidate_fields = list(Candidate.__dataclass_fields__)
    rejection_fields = list(Rejection.__dataclass_fields__)
    write_csv(candidate_manifest, (asdict(row) for row in candidates), candidate_fields)
    write_csv(rejection_manifest, (asdict(row) for row in rejections), rejection_fields)
    write_csv(
        test_manifest,
        test_rows,
        ["variant", "label", "class_name", "class_dir", "relative_path", "source_ids"],
    )

    required_per_class = args.train_per_class + args.val_per_class
    candidate_class_counts = candidate_counts_by_class(candidates)
    deficits = {
        label: required_per_class - int(values["eligible"])
        for label, values in candidate_class_counts.items()
        if int(values["eligible"]) < required_per_class
    }
    audit_summary = {
        "protocol_name": args.protocol_name,
        "mapping_path": str(args.mapping_json.resolve()),
        "mapping_sha256": sha256_file(args.mapping_json),
        "requested_train_per_class": args.train_per_class,
        "requested_val_per_class": args.val_per_class,
        "required_per_class": required_per_class,
        "split_feasible": not deficits,
        "deficits": deficits,
        "candidate_counts": candidate_class_counts,
        "official_test_unique_source_ids": len(official_test_ids),
        "official_test_counts": test_counts,
        "audit_counts": audit_counts,
        "manifests": {
            "eligible_candidates": str(candidate_manifest.resolve()),
            "rejections": str(rejection_manifest.resolve()),
            "official_test": str(test_manifest.resolve()),
        },
    }
    audit_summary_path.write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n")
    print(f"[AUDIT] summary: {audit_summary_path}")
    for label, values in candidate_class_counts.items():
        print(
            f"[AUDIT] class={label} name={values['class_name']} "
            f"eligible={values['eligible']} required={required_per_class}",
            flush=True,
        )
    if deficits:
        raise RuntimeError(
            "Requested split is infeasible after protocol filtering. "
            f"Per-class deficits={deficits}. See {audit_summary_path} before choosing "
            "a documented validation allocation."
        )

    selected, split_counts = select_splits(
        candidates,
        args.seed,
        args.train_per_class,
        args.val_per_class,
    )
    write_csv(
        selected_manifest,
        selected_rows(selected),
        ["split", "selection_rank", *candidate_fields],
    )

    link_root: Optional[Path] = None
    if args.materialize_links:
        link_root = args.output_root / "train_source" / args.protocol_name
        materialize_links(link_root, selected, args.overwrite_links)

    selected_ids = {candidate.sample_id for _split, _rank, candidate in selected}
    leakage = sorted(selected_ids & official_test_ids)
    if leakage:
        raise RuntimeError(f"Official test leakage detected after selection: {leakage[:10]}")

    summary = {
        "protocol_name": args.protocol_name,
        "protocol_status": "deterministic_reconstruction_not_filename_identical_to_unavailable_archive",
        "reconstruction_note": (
            "Implements the published WordNet mapping, bounding-box availability, exactly-one-box "
            "filter, class balancing, and official-test source exclusion. The unavailable official "
            "archive also balanced after synthetic-map filtering; its random seed and filename "
            "manifest were not published."
        ),
        "seed": args.seed,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "train_total": args.train_per_class * 9,
        "val_total": args.val_per_class * 9,
        "mapping_path": str(args.mapping_json.resolve()),
        "mapping_sha256": sha256_file(args.mapping_json),
        "imagenet_root": str(args.imagenet_root.resolve()),
        "annotation_root": str(args.annotation_root.resolve()),
        "official_test_root": str(args.official_test_root.resolve()),
        "link_root": str(link_root.resolve()) if link_root is not None else None,
        "official_test_unique_source_ids": len(official_test_ids),
        "official_test_counts": test_counts,
        "audit_counts": audit_counts,
        "split_counts": split_counts,
        "selected_test_overlap_count": 0,
        "manifests": {
            "selected": str(selected_manifest.resolve()),
            "eligible_candidates": str(candidate_manifest.resolve()),
            "rejections": str(rejection_manifest.resolve()),
            "official_test": str(test_manifest.resolve()),
            "audit_summary": str(audit_summary_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"[DONE] train={summary['train_total']} val={summary['val_total']}")
    print(f"[DONE] manifest: {selected_manifest}")
    print(f"[DONE] summary: {summary_path}")
    if link_root is not None:
        print(f"[DONE] ImageFolder links: {link_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise
