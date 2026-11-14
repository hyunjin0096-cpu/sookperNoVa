"""ResNet18 thermal-image encoder + sensor MLP + feature fusion.

Training may opt in to publicly available ImageNet pretrained backbone weights.
Inference always constructs weights=None and loads an OWN trained checkpoint.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class FusionNet(nn.Module):
    def __init__(self, sensor_dim: int = 149, pretrained_backbone: bool = False,
                 sensor_dropout: float = 0.2, fusion_dropout: float = 0.2):
        super().__init__()
        if not isinstance(sensor_dim, int) or sensor_dim <= 0:
            raise ValueError("sensor_dim must be a positive integer")
        self.sensor_dim = sensor_dim
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        self.image_backbone = resnet18(weights=weights)
        self.image_backbone.fc = nn.Identity()  # 512-d vector
        self.image_project = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True),
        )
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 128), nn.LayerNorm(128), nn.ReLU(inplace=True),
            nn.Dropout(sensor_dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(192, 128), nn.LayerNorm(128), nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout), nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )

    def forward(self, image: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must be [B, 3, H, W]")
        if sensor.ndim != 2 or sensor.shape[1] != self.sensor_dim:
            raise ValueError(f"sensor must be [B, {self.sensor_dim}]")
        if image.shape[0] != sensor.shape[0]:
            raise ValueError("Batch dimensions of image and sensor do not match")
        image_vec = self.image_project(self.image_backbone(image))
        sensor_vec = self.sensor_encoder(sensor)
        return self.classifier(torch.cat([image_vec, sensor_vec], dim=1))
