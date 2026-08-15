#!/usr/bin/env python3
"""Fail fast unless every requested final-evaluation source sweep is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()
    for method in args.methods:
        root = args.log_root / "sweeps" / method / "main"
        summary_path = root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        if method == "afr":
            completed = int(summary.get("completed_stage2_configurations", 0))
            target = int(summary.get("target_stage2_configurations", 165))
            if completed != 165 or target != 165 or not (root / "contract.json").is_file():
                raise RuntimeError(f"AFR source is incomplete: {completed}/{target}")
        else:
            completed = int(summary.get("complete_trials", 0))
            target = int(summary.get("target_complete_trials", 50))
            if completed < target or target != 50:
                raise RuntimeError(f"{method} source is incomplete: {completed}/{target}")
            if method == "clip_lr" and not (root / "openai_clip_rn50_features.npz").is_file():
                raise FileNotFoundError(root / "openai_clip_rn50_features.npz")
        print(f"[READY] method={method} source={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
