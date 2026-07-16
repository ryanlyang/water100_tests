#!/usr/bin/env python3
"""Minimal MobileNetV2 CAM sanity check for R4RR guided training."""

from __future__ import annotations

import argparse
import json

import torch

from models.cam_backbones import MobileNetV2CAM, make_cam_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--min-cam-range", type=float, default=1e-12)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = MobileNetV2CAM(num_classes=args.num_classes, pretrained=args.pretrained).to(device)
    alias_model = make_cam_backbone(
        num_classes=args.num_classes,
        model_name="mobilenetv2",
        pretrained=False,
    )
    require(isinstance(alias_model, MobileNetV2CAM), "mobilenetv2 alias did not construct MobileNetV2CAM")

    model.eval()
    inputs = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )
    labels = torch.arange(args.batch_size, device=device) % args.num_classes

    logits, feats = model(inputs)
    weight = model.classifier.weight
    cams = torch.einsum("bc,bchw->bhw", weight[labels], feats)

    expected_logits = (args.batch_size, args.num_classes)
    require(tuple(logits.shape) == expected_logits, f"bad logits shape: {tuple(logits.shape)}")
    require(feats.ndim == 4, f"feature maps should be 4D, got {tuple(feats.shape)}")
    require(feats.shape[0] == args.batch_size, f"bad feature batch: {tuple(feats.shape)}")
    require(feats.shape[1] == 1280, f"MobileNetV2 feature channels should be 1280, got {feats.shape[1]}")
    if args.image_size == 224:
        require(tuple(feats.shape[2:]) == (7, 7), f"224 input should produce 7x7 CAM features, got {tuple(feats.shape[2:])}")
    require(tuple(weight.shape) == (args.num_classes, feats.shape[1]), f"bad classifier weight shape: {tuple(weight.shape)}")
    require(tuple(cams.shape) == (args.batch_size, feats.shape[2], feats.shape[3]), f"bad CAM shape: {tuple(cams.shape)}")

    for name, tensor in (("logits", logits), ("features", feats), ("cams", cams)):
        require(torch.isfinite(tensor).all().item(), f"{name} contains non-finite values")

    flat_cams = cams.flatten(1)
    cam_ranges = flat_cams.max(dim=1).values - flat_cams.min(dim=1).values
    cam_std = flat_cams.std(dim=1, unbiased=False)
    nonflat = cam_ranges > args.min_cam_range
    require(nonflat.any().item(), f"all CAMs were flat: ranges={cam_ranges.detach().cpu().tolist()}")

    summary = {
        "status": "PASS",
        "model": type(model).__name__,
        "pretrained": bool(args.pretrained),
        "input_shape": list(inputs.shape),
        "logits_shape": list(logits.shape),
        "features_shape": list(feats.shape),
        "classifier_weight_shape": list(weight.shape),
        "cams_shape": list(cams.shape),
        "cam_range_min": float(cam_ranges.min().item()),
        "cam_range_max": float(cam_ranges.max().item()),
        "cam_std_min": float(cam_std.min().item()),
        "cam_std_max": float(cam_std.max().item()),
        "nonflat_count": int(nonflat.sum().item()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
