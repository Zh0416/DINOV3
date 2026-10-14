from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SegmentationDecoder(nn.Module):
    def __init__(self, input_channels: int = 1024, hidden_channels: int = 256, num_classes: int = 2):
        super().__init__()
        mid_channels = max(hidden_channels // 2, 32)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        hidden = self.stem(features)
        intermediate_size = (min(output_size[0], features.shape[-2] * 4), min(output_size[1], features.shape[-1] * 4))
        hidden = F.interpolate(hidden, size=intermediate_size, mode="bilinear", align_corners=False)
        logits = self.refine(hidden)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

