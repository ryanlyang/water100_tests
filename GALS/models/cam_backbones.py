"""Backbone wrappers for CAM-style guided training.

These wrappers standardize the small interface that R4RR/LearnToLook needs:
forward returns ``(logits, feature_maps)``, ``.classifier`` points to the final
linear head, and helper methods expose classifier/base parameter groups.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


def _mobilenet_v3_large(pretrained: bool) -> nn.Module:
    """Build MobileNetV3-Large across old and new torchvision APIs."""
    if hasattr(models, "MobileNet_V3_Large_Weights"):
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v3_large(weights=weights)
    return models.mobilenet_v3_large(pretrained=pretrained)


def _mobilenet_v2(pretrained: bool) -> nn.Module:
    """Build MobileNetV2 across old and new torchvision APIs."""
    if hasattr(models, "MobileNet_V2_Weights"):
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v2(weights=weights)
    return models.mobilenet_v2(pretrained=pretrained)


def _resnet50(pretrained: bool) -> nn.Module:
    """Build ResNet-50 across old and new torchvision APIs."""
    if hasattr(models, "ResNet50_Weights"):
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights)
    return models.resnet50(pretrained=pretrained)


class CAMBackbone(nn.Module):
    """Common protocol for CNN backbones used with CAM-style supervision."""

    classifier: nn.Linear
    features: Optional[torch.Tensor]

    def classifier_weight(self) -> torch.Tensor:
        return self.classifier.weight

    def classifier_parameters(self) -> Iterable[nn.Parameter]:
        return self.classifier.parameters()

    def base_parameters(self) -> Iterable[nn.Parameter]:
        classifier_param_ids = {id(param) for param in self.classifier.parameters()}
        for param in self.parameters():
            if id(param) not in classifier_param_ids:
                yield param

    @property
    def feature_maps(self) -> Optional[torch.Tensor]:
        return self.features


class ResNet50CAM(CAMBackbone):
    """Torchvision ResNet-50 wrapper matching the existing guided interface."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.base = _resnet50(pretrained=pretrained)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        self.classifier = self.base.fc
        self.features = None
        self.base.layer4.register_forward_hook(self._capture_features)

    def _capture_features(self, _module, _inputs, output: torch.Tensor) -> None:
        self.features = output

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.base(images)
        if self.features is None:
            raise RuntimeError("ResNet50CAM did not capture layer4 features.")
        return logits, self.features


class MobileNetV3CAM(CAMBackbone):
    """Torchvision MobileNetV3-Large wrapper for CAM-style supervision."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.base = _mobilenet_v3_large(pretrained=pretrained)
        if not isinstance(self.base.classifier, nn.Sequential):
            raise TypeError("Expected torchvision MobileNetV3 classifier to be nn.Sequential.")
        if not isinstance(self.base.classifier[-1], nn.Linear):
            raise TypeError("Expected final MobileNetV3 classifier module to be nn.Linear.")

        in_features = self.base.classifier[-1].in_features
        self.base.classifier[-1] = nn.Linear(in_features, num_classes)
        self.classifier = self.base.classifier[-1]
        self.conv_features = None
        self.features = None

    def classifier_parameters(self) -> Iterable[nn.Parameter]:
        return self.base.classifier.parameters()

    def base_parameters(self) -> Iterable[nn.Parameter]:
        classifier_param_ids = {id(param) for param in self.base.classifier.parameters()}
        for param in self.parameters():
            if id(param) not in classifier_param_ids:
                yield param

    def _classifier_prefix_spatial(self, feature_maps: torch.Tensor) -> torch.Tensor:
        # Apply MobileNet's pointwise classifier prefix independently at each spatial
        # location, skipping dropout so CAM features are deterministic.
        spatial = feature_maps.permute(0, 2, 3, 1)
        for module in self.base.classifier[:-1]:
            if isinstance(module, nn.Dropout):
                continue
            spatial = module(spatial)
        return spatial.permute(0, 3, 1, 2).contiguous()

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_features = self.base.features(images)
        self.conv_features = conv_features
        cam_features = self._classifier_prefix_spatial(conv_features)
        self.features = cam_features
        pooled = self.base.avgpool(conv_features)
        flattened = torch.flatten(pooled, 1)
        logits = self.base.classifier(flattened)
        return logits, cam_features


class MobileNetV2CAM(CAMBackbone):
    """Torchvision MobileNetV2 wrapper with an exact GAP + linear CAM head."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.base = _mobilenet_v2(pretrained=pretrained)
        if not hasattr(self.base, "features"):
            raise TypeError("Expected torchvision MobileNetV2 to expose `.features`.")
        if not isinstance(self.base.classifier, nn.Sequential):
            raise TypeError("Expected torchvision MobileNetV2 classifier to be nn.Sequential.")
        if not isinstance(self.base.classifier[-1], nn.Linear):
            raise TypeError("Expected final MobileNetV2 classifier module to be nn.Linear.")

        in_features = self.base.classifier[-1].in_features
        self.classifier = nn.Linear(in_features, num_classes)
        # Keep classifier simple and dropout-free so logits and CAMs use the same
        # spatial feature channels: features -> GAP -> Linear.
        self.base.classifier = self.classifier
        self.features = None

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.base.features(images)
        self.features = feature_maps
        pooled = nn.functional.adaptive_avg_pool2d(feature_maps, 1).flatten(1)
        logits = self.classifier(pooled)
        return logits, feature_maps


def make_cam_backbone(
    num_classes: int,
    model_name: str = "resnet50",
    pretrained: bool = True,
) -> CAMBackbone:
    name = str(model_name).strip().lower()
    if name == "resnet50":
        return ResNet50CAM(num_classes=num_classes, pretrained=pretrained)
    if name in {"mobilenet_v2", "mobilenetv2"}:
        return MobileNetV2CAM(num_classes=num_classes, pretrained=pretrained)
    if name in {"mobilenet_v3_large", "mobilenetv3_large", "mobilenet"}:
        return MobileNetV3CAM(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported CAM backbone: {model_name}")


def get_cam_features(model: nn.Module) -> torch.Tensor:
    features = getattr(model, "feature_maps", None)
    if features is None:
        features = getattr(model, "features", None)
    if features is None:
        raise RuntimeError("Model does not expose CAM feature maps.")
    return features


def get_classifier_weight(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "classifier_weight"):
        return model.classifier_weight()
    classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "weight"):
        raise RuntimeError("Model does not expose classifier weights.")
    return classifier.weight
