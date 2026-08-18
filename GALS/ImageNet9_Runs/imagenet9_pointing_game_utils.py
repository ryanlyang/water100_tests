#!/usr/bin/env python3
"""Dependency-light utilities for ImageNet-9 Pointing Game jobs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


METHODS = ("erm", "upweight", "abn", "elrep", "gals", "afr", "clip_lr", "r4rr")
PRIMARY_VARIANTS = ("original", "mixed_same", "mixed_rand", "mixed_next")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields: List[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_progress_jsonl(path: Path) -> List[Dict[str, object]]:
    """Load durable progress, discarding only an interrupted trailing write."""
    if not path.is_file():
        return []
    rows: List[Dict[str, object]] = []
    valid_bytes = 0
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                valid_bytes = handle.tell()
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if handle.tell() == file_size:
                    break
                raise
            rows.append(row)
            valid_bytes = handle.tell()
    if valid_bytes != file_size:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
    keys = [str(row["sample_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate samples in progress file: {path}")
    return rows


def read_manifest(path: Path, variant: str) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["variant"] == variant]
    if len(rows) != 4050:
        raise RuntimeError(f"Expected 4,050 {variant} rows in {path}, found {len(rows)}")
    return rows


def index_foreground_masks(mask_root: Path) -> Dict[Tuple[str, str], Path]:
    result: Dict[Tuple[str, str], Path] = {}
    for path in sorted(mask_root.glob("*/*.npy")):
        key = (path.parent.name, path.stem)
        if key in result:
            raise RuntimeError(f"Duplicate foreground mask key: {key}")
        result[key] = path
    if len(result) != 4050:
        raise RuntimeError(f"Expected 4,050 official foreground masks, found {len(result)}")
    return result


def resolve_foreground_mask(
    row: Mapping[str, str], mask_index: Mapping[Tuple[str, str], Path]
) -> Tuple[Path, str]:
    """The first source ID in a composite filename is its retained foreground."""
    class_dir = row["class_dir"]
    source_ids = [value for value in row.get("source_ids", "").split(";") if value]
    if not source_ids:
        raise RuntimeError(f"Official row has no source ID: {row}")
    foreground_id = source_ids[0]
    path = mask_index.get((class_dir, foreground_id))
    if path is None:
        raise RuntimeError(
            f"No foreground mask for variant={row['variant']} class={class_dir} "
            f"foreground_id={foreground_id} relative_path={row['relative_path']}"
        )
    return path, foreground_id
