"""10-class leaf disease classifier (EfficientNet-B0)."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def build_leaf_classifier(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_leaf_train_phase(
    model: nn.Module,
    phase: int,
    unfreeze_blocks: int = 3,
) -> None:
    """Phase 1: classifier head only. Phase 2: unfreeze last feature blocks."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    if phase >= 2:
        feature_blocks = list(model.features.children())
        for block in feature_blocks[-unfreeze_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
