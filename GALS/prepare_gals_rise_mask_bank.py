#!/usr/bin/env python3
"""Create and validate one deterministic GALS-style RISE mask bank."""

from __future__ import annotations

import argparse
from pathlib import Path

from gals_rise_utils import load_or_create_mask_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-masks", type=int, default=2000)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--p1", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    masks, digest = load_or_create_mask_bank(
        path=args.output.expanduser().resolve(),
        num_masks=args.num_masks,
        grid_size=args.grid_size,
        height=args.height,
        width=args.width,
        p1=args.p1,
        seed=args.seed,
    )
    print(
        f"[DONE] RISE mask bank={args.output.expanduser().resolve()} "
        f"shape={masks.shape} sha256={digest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
