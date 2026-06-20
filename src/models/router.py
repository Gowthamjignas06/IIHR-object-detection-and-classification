"""Binary leaf-vs-fruit router model (MobileNetV3-Small)."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights


def build_router(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def set_router_train_phase(model: nn.Module, phase: int, unfreeze_blocks: int = 2) -> None:
    """Phase 1: train classifier head only. Phase 2: unfreeze last feature blocks."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    if phase >= 2:
        feature_blocks = list(model.features.children())
        for block in feature_blocks[-unfreeze_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
