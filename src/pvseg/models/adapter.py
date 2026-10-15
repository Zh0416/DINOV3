from __future__ import annotations

import torch
from torch import nn


class LegacyPVAwareAdapter(nn.Module):
    """Residual adapter operating on attention-weighted DINO patch features."""

    def __init__(self, channels: int = 1024, bottleneck: int = 256):
        super().__init__()
        self.input_projection = nn.Conv2d(channels, bottleneck, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.GroupNorm(1, bottleneck),
            nn.Conv2d(bottleneck, bottleneck * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(bottleneck * 2, channels, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        if attention.ndim == 3:
            attention = attention.unsqueeze(1)
        if attention.shape[-2:] != features.shape[-2:]:
            raise ValueError(
                f"Attention shape {attention.shape[-2:]} does not match feature shape {features.shape[-2:]}"
            )
        attended = features * attention.to(dtype=features.dtype)
        update = self.mlp(self.input_projection(attended))
        return features + self.residual_scale * update


class PVAwareAdapter(LegacyPVAwareAdapter):
    """Bounded residual, gated AFTER normalization; zero hint gives exact identity."""

    def __init__(self, channels: int = 1024, bottleneck: int = 256):
        super().__init__(channels, bottleneck)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        if attention.ndim == 3:
            attention = attention.unsqueeze(1)
        if attention.shape[-2:] != features.shape[-2:]:
            raise ValueError("Attention and feature spatial sizes must match")
        gate = attention.to(dtype=features.dtype).clamp(0, 1)
        update = self.mlp(self.input_projection(features))
        return features + .2 * self.residual_scale.tanh() * gate * update


class SoftSemanticPVAwareAdapter(nn.Module):
    """Zero-initialized residual adapter for non-blocking semantic priors."""

    def __init__(self, channels: int = 1024, bottleneck: int = 256):
        super().__init__()
        groups = next((candidate for candidate in (16, 8, 4, 2, 1) if bottleneck % candidate == 0), 1)
        self.attention_encoder = nn.Sequential(
            nn.Conv2d(1, bottleneck, kernel_size=3, padding=1),
            nn.GroupNorm(groups, bottleneck),
            nn.GELU(),
            nn.Conv2d(bottleneck, bottleneck, kernel_size=1),
        )
        self.feature_projection = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1),
            nn.GroupNorm(groups, bottleneck),
        )
        self.output_projection = nn.Conv2d(bottleneck, channels, kernel_size=1)
        # gamma=0 gives exact DINO identity while leaving a gradient for gamma.
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        if attention.ndim == 3:
            attention = attention.unsqueeze(1)
        if attention.shape[-2:] != features.shape[-2:]:
            raise ValueError("Attention and feature spatial sizes must match")
        context = 0.1 * torch.tanh(self.attention_encoder(attention.to(features.dtype)))
        reduced = self.feature_projection(features)
        fused = reduced * (1.0 + context)
        update = self.output_projection(fused)
        return features + self.gamma.tanh() * update

