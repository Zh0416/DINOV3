from __future__ import annotations

import logging
import inspect
from typing import Any

import torch
from torch import nn

from pvseg.utils import model_reference


def _torch_dtype(name: str) -> torch.dtype:
    if name.lower() == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class FrozenDINOv3(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        mean: list[float],
        std: list[float],
        image_size: int,
        patch_size: int,
        expected_hidden_dim: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_size = image_size // patch_size
        self.expected_hidden_dim = expected_hidden_dim
        self.supports_position_interpolation = (
            "interpolate_pos_encoding" in inspect.signature(self.backbone.forward).parameters
        )
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        image_size: int,
        logger: logging.Logger,
    ) -> "FrozenDINOv3":
        from transformers import AutoImageProcessor, AutoModel

        reference, local_only = model_reference(config)
        dtype = _torch_dtype(str(config.get("dtype", "float16")))
        token = None if local_only else __import__("os").getenv("HF_TOKEN")
        common = {
            "local_files_only": local_only,
            "token": token,
            "trust_remote_code": True,
        }
        logger.info("Loading frozen DINOv3 from %s (local_only=%s)", reference, local_only)
        processor = AutoImageProcessor.from_pretrained(reference, **common)
        model_kwargs = {
            **common,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        attention_implementation = config.get("attention_implementation")
        if attention_implementation:
            model_kwargs["attn_implementation"] = attention_implementation
        backbone = AutoModel.from_pretrained(reference, **model_kwargs)
        mean = list(getattr(processor, "image_mean", [0.485, 0.456, 0.406]))
        std = list(getattr(processor, "image_std", [0.229, 0.224, 0.225]))
        return cls(
            backbone,
            mean,
            std,
            image_size,
            int(config["patch_size"]),
            int(config["expected_hidden_dim"]),
        )

    def train(self, mode: bool = True) -> "FrozenDINOv3":
        super().train(False)
        self.backbone.eval()
        return self

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size}, got {images.shape[-2:]}")
        normalized = (images - self.mean.to(images.dtype)) / self.std.to(images.dtype)
        model_kwargs = {"pixel_values": normalized}
        if self.supports_position_interpolation:
            model_kwargs["interpolate_pos_encoding"] = True
        outputs = self.backbone(**model_kwargs)
        tokens = outputs.last_hidden_state
        patch_count = self.feature_size * self.feature_size
        if tokens.shape[1] < patch_count:
            raise RuntimeError(f"Backbone returned {tokens.shape[1]} tokens, expected at least {patch_count}")
        patch_tokens = tokens[:, -patch_count:, :]
        if patch_tokens.shape[-1] != self.expected_hidden_dim:
            raise RuntimeError(
                f"DINO hidden dimension is {patch_tokens.shape[-1]}, expected {self.expected_hidden_dim}"
            )
        batch = patch_tokens.shape[0]
        return patch_tokens.transpose(1, 2).reshape(
            batch, self.expected_hidden_dim, self.feature_size, self.feature_size
        )

