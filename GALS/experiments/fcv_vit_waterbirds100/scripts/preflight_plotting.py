#!/usr/bin/env python3
"""Render a real noninteractive plot before the expensive campaign launches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import pyparsing  # noqa: E402,F401

from fcv.config import load_and_validate_config  # noqa: E402


DEFAULT_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "waterbirds100_vit_s16_first_study.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    output = (
        Path(config["paths"]["output_root"]).expanduser().resolve()
        / "preflight"
        / "plotting_stack_smoke.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(2.0, 1.5))
    axis.plot([0, 1], [0, 1])
    axis.set_title("FCV plotting preflight")
    figure.tight_layout()
    figure.savefig(output, dpi=80)
    plt.close(figure)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("Matplotlib preflight did not produce a plot.")
    print(output)


if __name__ == "__main__":
    main()
