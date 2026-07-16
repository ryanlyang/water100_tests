#!/usr/bin/env python3
"""Validate and index all Step 6 model-specific background-token banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXPERIMENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.token_banks import (  # noqa: E402
    aggregate_token_bank_summaries,
    prepare_token_bank_source,
)


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the Step 6 token-bank pool.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patch-masks", type=Path, default=None)
    parser.add_argument("--token-bank-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(args.config)
    output_root = Path(config["paths"]["output_root"])
    manifest = args.manifest or (
        output_root / config["outputs"]["split_manifests"] / "metadata_val.csv"
    )
    patch_masks = args.patch_masks or (
        output_root / config["outputs"]["patch_masks"] / "patch_masks_val.pt"
    )
    token_bank_dir = args.token_bank_dir or (
        output_root / config["outputs"]["token_banks"]
    )
    output_csv = args.output_csv or token_bank_dir / "token_bank_index.csv"
    summary = args.summary or token_bank_dir / "token_bank_pool_summary.json"
    if not Path(manifest).is_file() or not Path(patch_masks).is_file():
        raise FileNotFoundError("Step 6 aggregation requires the manifest and patch masks.")
    source = prepare_token_bank_source(
        config,
        manifest,
        patch_masks,
        check_images=False,
    )
    result = aggregate_token_bank_summaries(
        config,
        token_bank_dir,
        output_csv,
        summary,
        manifest_sha256=sha256_file(Path(manifest)),
        patch_mask_sha256=sha256_file(Path(patch_masks)),
        source=source,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
