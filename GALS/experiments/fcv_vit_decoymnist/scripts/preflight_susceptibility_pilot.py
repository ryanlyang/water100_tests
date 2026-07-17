#!/usr/bin/env python3
"""Validate DecoyMNIST encoding and cache/test the pretrained ViT on Tigris."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_susceptibility import (  # noqa: E402
    MODEL_NAME,
    build_model,
    discover_samples,
    enumerate_runs,
    stratified_train_holdout,
    verify_encoding,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png",
    )
    args = parser.parse_args()
    train = discover_samples(args.data_root, "train")
    test = discover_samples(args.data_root, "test")
    verify_encoding(train, "train", per_class=100)
    verify_encoding(test, "test", per_class=100)
    candidate_train, biased_val = stratified_train_holdout(train)
    if len(candidate_train) != 54000 or len(biased_val) != 6000:
        raise RuntimeError(
            f"Expected 54,000/6,000 train holdout, found "
            f"{len(candidate_train)}/{len(biased_val)}"
        )
    if sum(len(paths) for paths in test.values()) != 10000:
        raise RuntimeError("Expected exactly 10,000 official reversed-test PNGs")
    if len(enumerate_runs()) != 9:
        raise RuntimeError("Pilot must contain exactly nine training runs")
    if not torch.cuda.is_available():
        raise RuntimeError("Preflight requires the allocated GH200")
    model = build_model(pretrained=True).cuda().eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        output = model(torch.zeros(2, 3, 224, 224, device="cuda"))
    if tuple(output.shape) != (2, 10):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")
    print(f"[PASS] model={MODEL_NAME} output={tuple(output.shape)}")
    print("[PASS] unmodified DecoyMNIST encoding validated on 2,000 PNGs")
    print("[PASS] deterministic split train=54000 biased_val=6000")
    print("[PASS] pilot grid=3 learning rates x 3 seeds = 9 runs")


if __name__ == "__main__":
    main()
