from __future__ import annotations

import torch
from torch import nn

from .adapter import PVAwareAdapter, LegacyPVAwareAdapter
from .decoder import SegmentationDecoder


class CachedFeatureSegmenter(nn.Module):
    def __init__(
        self,
        feature_channels: int = 1024,
        decoder_channels: int = 256,
        adapter_bottleneck: int = 256,
        use_adapter: bool = False,
        adapter_mode: str = "gated",
    ):
        super().__init__()
        self.use_adapter = use_adapter
        self.adapter = (
            (LegacyPVAwareAdapter if adapter_mode == "legacy" else PVAwareAdapter)(
                feature_channels, adapter_bottleneck
            ) if use_adapter else None
        )
        self.decoder = SegmentationDecoder(feature_channels, decoder_channels, num_classes=2)

    def forward(
        self,
        features: torch.Tensor,
        attention: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if self.adapter is not None:
            features = self.adapter(features, attention)
        return self.decoder(features, output_size)

