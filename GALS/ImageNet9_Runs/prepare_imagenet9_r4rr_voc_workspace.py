#!/usr/bin/env python3
"""Build a symlink-backed VOC compatibility workspace for IN-9 R4RR maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

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
FOREGROUND_PROMPTS: Mapping[str, str] = {
    "dog": "dog",
    "bird": "bird",
    "vehicle": "vehicle",
    "reptile": "reptile",
    "carnivore": "carnivore",
    "insect": "insect",
    "instrument": "musical instrument",
    "primate": "primate",
    "fish": "fish",
}
WORKSPACE_SCHEMA_VERSION = 1
SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    path: Path
    label: int
    class_name: str


def load_training_samples(manifest: Path) -> List[ManifestSample]:
    samples: List[ManifestSample] = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "source_path", "label", "class_name", "split"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Manifest is missing required columns {sorted(missing)}: {manifest}")
        for row in reader:
            if row["split"] != "train":
                continue
            label = int(row["label"])
            if not 0 <= label < len(CLASS_NAMES):
                raise RuntimeError(f"Invalid label {label} in {manifest}")
            class_name = row["class_name"].strip().lower()
            if class_name != CLASS_NAMES[label]:
                raise RuntimeError(
                    f"Label/name mismatch for {row['sample_id']}: "
                    f"label {label} expects {CLASS_NAMES[label]}, got {class_name}"
                )
            source = Path(row["source_path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            samples.append(
                ManifestSample(
                    sample_id=row["sample_id"].strip(),
                    path=source,
                    label=label,
                    class_name=class_name,
                )
            )
    if not samples:
        raise RuntimeError(f"No Original training samples found in {manifest}")
    return samples


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=(
            here.parent
            / "RightForTheRightRegions"
            / "WeCLIPPlus"
            / "clip"
            / "clip_texts"
            / "clip_text_imagenet9.py"
        ),
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == text:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        for line in lines:
            handle.write(line)
    temporary.replace(path)


def validate_samples(samples: Sequence[ManifestSample]) -> None:
    expected_counts = {name: 5045 for name in CLASS_NAMES}
    observed = Counter(sample.class_name for sample in samples)
    if dict(observed) != expected_counts:
        raise RuntimeError(
            f"Expected balanced Original training counts {expected_counts}, got {dict(observed)}"
        )

    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        duplicates = [name for name, count in Counter(ids).items() if count > 1]
        raise RuntimeError(f"Duplicate sample IDs: {duplicates[:10]}")

    unsafe = [sample.sample_id for sample in samples if not SAFE_SAMPLE_ID.fullmatch(sample.sample_id)]
    if unsafe:
        raise RuntimeError(f"Sample IDs are not safe VOC basenames: {unsafe[:10]}")

    if tuple(FOREGROUND_PROMPTS) != tuple(CLASS_NAMES):
        raise RuntimeError(
            "Foreground prompt order does not match ImageNet-9 class order: "
            f"{tuple(FOREGROUND_PROMPTS)} != {tuple(CLASS_NAMES)}"
        )


def ensure_image_links(samples: Sequence[ManifestSample], image_dir: Path) -> Dict[str, int]:
    image_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{sample.sample_id}.jpg" for sample in samples}
    existing_names = {entry.name for entry in image_dir.iterdir()}
    extras = sorted(existing_names - expected_names)
    if extras:
        raise RuntimeError(
            f"VOC JPEGImages contains {len(extras)} unexpected entries; refusing to mix datasets: "
            f"{extras[:10]}"
        )

    created = 0
    reused = 0
    for index, sample in enumerate(samples, start=1):
        source = sample.path.resolve(strict=True)
        destination = image_dir / f"{sample.sample_id}.jpg"
        if destination.is_symlink():
            if destination.resolve(strict=False) != source:
                raise RuntimeError(
                    f"Existing link points to the wrong source: {destination} -> "
                    f"{destination.resolve(strict=False)} (expected {source})"
                )
            reused += 1
        elif destination.exists():
            raise RuntimeError(
                f"Expected a symlink but found a regular filesystem entry: {destination}"
            )
        else:
            destination.symlink_to(source)
            created += 1

        if index % 5000 == 0:
            print(f"[LINKS] checked={index}/{len(samples)} created={created} reused={reused}")

    return {"created": created, "reused": reused}


def class_label_lines(
    samples: Sequence[ManifestSample],
    positive_class: str,
) -> Iterable[str]:
    for sample in samples:
        label = 1 if sample.class_name == positive_class else -1
        yield f"{sample.sample_id} {label}\n"


def write_image_sets(samples: Sequence[ManifestSample], set_dir: Path) -> None:
    set_dir.mkdir(parents=True, exist_ok=True)
    ids = [sample.sample_id for sample in samples]
    split_text = "".join(f"{sample_id}\n" for sample_id in ids)

    atomic_write_text(set_dir / "classes.txt", "".join(f"{name}\n" for name in CLASS_NAMES))
    atomic_write_text(set_dir / "train.txt", split_text)

    # WeCLIP inference defaults to the VOC name "val". For teacher generation,
    # this compatibility split deliberately aliases Original training images.
    atomic_write_text(set_dir / "val.txt", split_text)

    for class_name in CLASS_NAMES:
        atomic_write_lines(
            set_dir / f"{class_name}_train.txt",
            class_label_lines(samples, class_name),
        )
        atomic_write_lines(
            set_dir / f"{class_name}_val.txt",
            class_label_lines(samples, class_name),
        )


def write_workspace_manifest(
    samples: Sequence[ManifestSample],
    image_dir: Path,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = (
        "sample_id",
        "label",
        "class_name",
        "foreground_prompt",
        "source_path",
        "voc_link_path",
    )
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "class_name": sample.class_name,
                    "foreground_prompt": FOREGROUND_PROMPTS[sample.class_name],
                    "source_path": str(sample.path.resolve()),
                    "voc_link_path": str(image_dir / f"{sample.sample_id}.jpg"),
                }
            )
    temporary.replace(path)


def audit_workspace(
    samples: Sequence[ManifestSample],
    image_dir: Path,
    set_dir: Path,
) -> Dict[str, object]:
    ids = [sample.sample_id for sample in samples]
    expected_id_set = set(ids)
    classes = [line.strip() for line in (set_dir / "classes.txt").read_text().splitlines() if line.strip()]
    if classes != list(CLASS_NAMES):
        raise RuntimeError(f"classes.txt order mismatch: {classes}")

    for split in ("train", "val"):
        split_ids = [line.strip() for line in (set_dir / f"{split}.txt").read_text().splitlines() if line.strip()]
        if split_ids != ids:
            raise RuntimeError(f"{split}.txt does not exactly match the Original training IDs")

        positives_by_id = Counter()
        for class_name in CLASS_NAMES:
            path = set_dir / f"{class_name}_{split}.txt"
            rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != len(samples):
                raise RuntimeError(f"Wrong row count in {path}: {len(rows)}")
            row_ids = [row[0] for row in rows]
            if row_ids != ids:
                raise RuntimeError(f"ID order mismatch in {path}")
            invalid = [row for row in rows if len(row) != 2 or row[1] not in {"1", "-1"}]
            if invalid:
                raise RuntimeError(f"Invalid VOC labels in {path}: {invalid[:5]}")
            for sample_id, label in rows:
                if label == "1":
                    positives_by_id[sample_id] += 1
        if set(positives_by_id) != expected_id_set or set(positives_by_id.values()) != {1}:
            raise RuntimeError(f"Every {split} image must have exactly one positive class")

    links = list(image_dir.iterdir())
    if len(links) != len(samples):
        raise RuntimeError(f"Expected {len(samples)} image links, found {len(links)}")
    broken = [path for path in links if not path.is_symlink() or not path.resolve(strict=False).is_file()]
    if broken:
        raise RuntimeError(f"Invalid or broken image links: {broken[:10]}")

    return {
        "status": "ok",
        "num_images": len(samples),
        "class_names": list(CLASS_NAMES),
        "class_counts": dict(Counter(sample.class_name for sample in samples)),
        "num_image_symlinks": len(links),
        "num_class_label_files": 2 * len(CLASS_NAMES),
        "positive_labels_per_image_per_split": 1,
        "source_split": "reconstructed_original_train",
        "voc_train_semantics": "WeCLIP training images",
        "voc_val_semantics": "R4RR teacher-map inference over the same training images",
        "held_out_validation_included": False,
        "official_variants_included": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = args.manifest.resolve(strict=True)
    prompt_config = args.prompt_config.resolve(strict=True)
    workspace_root = args.workspace_root.resolve()
    voc_root = workspace_root / "VOCdevkit" / "VOC2012"
    image_dir = voc_root / "JPEGImages"
    set_dir = voc_root / "ImageSets" / "Main"
    metadata_dir = workspace_root / "metadata"

    samples = sorted(
        load_training_samples(manifest),
        key=lambda sample: (sample.label, sample.sample_id),
    )
    validate_samples(samples)

    print(f"[INFO] manifest={manifest}")
    print(f"[INFO] workspace={workspace_root}")
    print(f"[INFO] source images={len(samples)} (Original train only)")
    print(f"[INFO] class order={list(CLASS_NAMES)}")

    link_stats = ensure_image_links(samples, image_dir)
    write_image_sets(samples, set_dir)
    write_workspace_manifest(samples, image_dir, metadata_dir / "workspace_manifest.csv")
    audit = audit_workspace(samples, image_dir, set_dir)
    audit["links_created"] = link_stats["created"]
    audit["links_reused"] = link_stats["reused"]

    contract = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "dataset": "imagenet9_backgrounds_challenge",
        "protocol": "reconstructed_original_bbox1_v1",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "prompt_config": str(prompt_config),
        "prompt_config_sha256": sha256_file(prompt_config),
        "foreground_class_order": list(CLASS_NAMES),
        "foreground_prompts": dict(FOREGROUND_PROMPTS),
        "image_storage": "absolute_symlinks_to_shared_imagenet",
        "voc_val_is_training_map_inference_alias": True,
        "held_out_validation_included": False,
        "official_variants_included": False,
    }
    atomic_write_text(metadata_dir / "workspace_contract.json", json.dumps(contract, indent=2, sort_keys=True) + "\n")
    atomic_write_text(metadata_dir / "workspace_audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")

    print(f"[DONE] links created={link_stats['created']} reused={link_stats['reused']}")
    print(f"[DONE] VOC root: {voc_root}")
    print(f"[DONE] class label files: {set_dir}")
    print(f"[DONE] audit: {metadata_dir / 'workspace_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
