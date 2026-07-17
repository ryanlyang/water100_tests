#!/usr/bin/env python3
"""Report the study footprint and enforce the locked pre-launch guard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.storage import assert_storage_budget  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", default="manual_preflight")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optionally persist the storage snapshot atomically for a later smoke check.",
    )
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    result = assert_storage_budget(
        config,
        config["paths"]["output_root"],
        stage=args.stage,
    )
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(output_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
