import torch.nn as nn
from torchvision import models
from typing import Optional


def get_mobilenet_v2(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_model(name: str = "mobilenet_v2", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if name == "mobilenet_v2":
        return get_mobilenet_v2(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model architecture: {name}")