#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


SPLIT_ID_TO_NAME = {0: "Training", 1: "Validation", 2: "Test"}


def _read_metadata(path: Path):
    meta_path = path / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv at: {meta_path}")

    with meta_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"metadata.csv has no header: {meta_path}")

        required = {"y", "place", "split"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"metadata.csv missing columns {sorted(missing)}: {meta_path}")

        rows = []
        for row in reader:
            try:
                y = int(row["y"])
                place = int(row["place"])
                split = int(row["split"])
            except Exception as exc:
                raise ValueError(f"Bad row in {meta_path}: {row}") from exc
            rows.append((y, place, split))
    return rows


def _count_splits(rows) -> Dict[str, Dict[Tuple[int, int], int]]:
    counts: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    for y, place, split in rows:
        split_name = SPLIT_ID_TO_NAME.get(split, f"Split{split}")
        counts[split_name][(y, place)] += 1
    return counts


def _format_markdown_table(title: str, counts: Dict[str, Dict[Tuple[int, int], int]]) -> str:
    def c(split: str, y: int, place: int) -> int:
        return int(counts.get(split, {}).get((y, place), 0))

    lines = []
    lines.append(f"{title}")
    lines.append("")
    lines.append("| Split | Landbirds, land | Landbirds, water | Waterbirds, land | Waterbirds, water |")
    lines.append("|---|---:|---:|---:|---:|")
    for split in ("Training", "Validation", "Test"):
        lines.append(
            f"| {split} | {c(split, 0, 0)} | {c(split, 0, 1)} | {c(split, 1, 0)} | {c(split, 1, 1)} |"
        )
    return "\n".join(lines)


def _format_latex_table(title: str, counts: Dict[str, Dict[Tuple[int, int], int]]) -> str:
    def c(split: str, y: int, place: int) -> int:
        return int(counts.get(split, {}).get((y, place), 0))

    lines = []
    lines.append(f"% {title}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Split & Landbirds, land & Landbirds, water & Waterbirds, land & Waterbirds, water \\")
    lines.append(r"\midrule")
    for split in ("Training", "Validation", "Test"):
        lines.append(
            f"{split} & {c(split, 0, 0)} & {c(split, 0, 1)} & {c(split, 1, 0)} & {c(split, 1, 1)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _one_dataset(dataset_path: Path, name: str, latex: bool) -> str:
    rows = _read_metadata(dataset_path)
    counts = _count_splits(rows)
    title = f"{name} ({dataset_path})"
    return _format_latex_table(title, counts) if latex else _format_markdown_table(title, counts)


def main():
    p = argparse.ArgumentParser(description="Print Waterbirds split/group count tables from metadata.csv.")
    p.add_argument(
        "--wb95",
        type=Path,
        default=Path("/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2"),
        help="Path to Waterbirds-95 dataset root (expects metadata.csv).",
    )
    p.add_argument(
        "--wb100",
        type=Path,
        default=Path("/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2"),
        help="Path to Waterbirds-100 dataset root (expects metadata.csv).",
    )
    p.add_argument("--latex", action="store_true", help="Print LaTeX tabular instead of Markdown.")
    args = p.parse_args()

    out = []
    out.append(_one_dataset(args.wb95, "Waterbirds-95%", latex=args.latex))
    out.append("")
    out.append(_one_dataset(args.wb100, "Waterbirds-100%", latex=args.latex))
    print("\n".join(out))


if __name__ == "__main__":
    main()

