#!/usr/bin/env python3
"""Small model-loading helpers for evaluation-only scripts."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
from torchvision import models


def _resnet50(pretrained: bool) -> nn.Module:
    if hasattr(models, "ResNet50_Weights"):
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights)
    return models.resnet50(pretrained=pretrained)


def _mobilenet_v2(pretrained: bool) -> nn.Module:
    if hasattr(models, "MobileNet_V2_Weights"):
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v2(weights=weights)
    return models.mobilenet_v2(pretrained=pretrained)


class ResNet50CAM(nn.Module):
    """R4RR-style ResNet wrapper whose state dict uses `base.*` keys."""

    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.base = _resnet50(pretrained=pretrained)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


class MobileNetV2CAM(nn.Module):
    """R4RR MobileNetV2 wrapper with dropout-free GAP + Linear head."""

    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.base = _mobilenet_v2(pretrained=pretrained)
        in_features = int(self.base.classifier[-1].in_features)
        self.classifier = nn.Linear(in_features, num_classes)
        self.base.classifier = self.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = self.base.features(x)
        pooled = nn.functional.adaptive_avg_pool2d(feature_maps, 1).flatten(1)
        return self.classifier(pooled)


def _plain_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    model = _resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _plain_mobilenet_v2(num_classes: int, pretrained: bool) -> nn.Module:
    model = _mobilenet_v2(pretrained=pretrained)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _unwrap_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "net", "checkpoint"):
            value = obj.get(key)
            if isinstance(value, Mapping):
                return _unwrap_state_dict(value)
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise TypeError("Checkpoint does not look like a state dict or wrapped state dict.")


def _strip_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> OrderedDict:
    out = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
        else:
            out[key] = value
    return out


def _load_with_fallbacks(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    attempts = [
        state_dict,
        _strip_prefix(state_dict, "module."),
        _strip_prefix(state_dict, "model."),
    ]
    errors = []
    for candidate in attempts:
        try:
            missing, unexpected = model.load_state_dict(candidate, strict=False)
            overlap = len(set(model.state_dict()).intersection(candidate))
            if overlap > 0:
                return {
                    "loaded_key_overlap": overlap,
                    "missing_key_count": len(missing),
                    "unexpected_key_count": len(unexpected),
                    "missing_keys_preview": list(missing)[:8],
                    "unexpected_keys_preview": list(unexpected)[:8],
                }
        except RuntimeError as exc:
            errors.append(str(exc).splitlines()[0])
    raise RuntimeError(f"Could not load checkpoint into model. Errors: {errors[:3]}")


def build_model(
    arch: str,
    num_classes: int,
    checkpoint: str | None = None,
    pretrained: bool = False,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    arch = arch.lower()
    if arch == "resnet50":
        model = _plain_resnet50(num_classes=num_classes, pretrained=pretrained)
    elif arch == "resnet50_cam":
        model = ResNet50CAM(num_classes=num_classes, pretrained=pretrained)
    elif arch == "mobilenet_v2":
        model = MobileNetV2CAM(num_classes=num_classes, pretrained=pretrained)
    elif arch == "mobilenet_v2_plain":
        model = _plain_mobilenet_v2(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unsupported --arch '{arch}'. Expected one of: "
            "resnet50, resnet50_cam, mobilenet_v2, mobilenet_v2_plain."
        )

    meta: Dict[str, Any] = {"arch": arch, "num_classes": int(num_classes)}
    if checkpoint:
        ckpt_path = Path(checkpoint).expanduser()
        payload = torch.load(ckpt_path, map_location="cpu")
        state_dict = _unwrap_state_dict(payload)
        meta.update(_load_with_fallbacks(model, state_dict))
        meta["checkpoint"] = str(ckpt_path)

    model.to(device)
    model.eval()
    return model, meta


def logits_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

