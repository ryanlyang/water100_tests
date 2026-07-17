#!/usr/bin/env python3
"""Validate or recover the campaign-bound online smoke gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.config import load_and_validate_config  # noqa: E402
from fcv.campaign_provenance import CampaignProvenanceError  # noqa: E402
from fcv.online_analysis import (  # noqa: E402
    OnlineAnalysisError,
    ensure_online_smoke_gate,
    validate_reusable_online_smoke_receipt,
)


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument(
        "--receipt-epoch",
        type=int,
        choices=(1, 2),
        default=None,
        help="Validate and reuse one completed smoke stage instead of the aggregate gate.",
    )
    parser.add_argument(
        "--quiet-missing",
        action="store_true",
        help="Exit 1 without a traceback when no reusable gate exists yet.",
    )
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    try:
        if args.receipt_epoch is None:
            result = ensure_online_smoke_gate(
                config, config["paths"]["output_root"], run_index=args.run_index
            )
        else:
            result = validate_reusable_online_smoke_receipt(
                config,
                config["paths"]["output_root"],
                run_index=args.run_index,
                expected_epoch=args.receipt_epoch,
            )
    except (FileNotFoundError, CampaignProvenanceError, OnlineAnalysisError) as exc:
        if args.quiet_missing:
            print(f"[SMOKE GATE] no reusable gate: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
