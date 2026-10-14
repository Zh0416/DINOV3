from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
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
        backend: str = "transformers",
    ):
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_size = image_size // patch_size
        self.expected_hidden_dim = expected_hidden_dim
        self.backend = backend
        self.supports_position_interpolation = backend == "transformers" and (
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
        reference, local_only = model_reference(config)
        dtype = _torch_dtype(str(config.get("dtype", "float16")))
        reference_path = Path(reference)
        if local_only and reference_path.is_file() and reference_path.suffix == ".safetensors":
            return cls._from_timm_checkpoint(config, reference_path, image_size, dtype, logger)

        from transformers import AutoImageProcessor, AutoModel

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

    @classmethod
    def _from_timm_checkpoint(
        cls,
        config: dict[str, Any],
        checkpoint: Path,
        image_size: int,
        dtype: torch.dtype,
        logger: logging.Logger,
    ) -> "FrozenDINOv3":
        import timm

        metadata_path = checkpoint.with_name("config.json")
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        architecture = str(
            metadata.get("architecture", config.get("timm_architecture", "vit_large_patch16_dinov3"))
        )
        pretrained = metadata.get("pretrained_cfg", {})
        mean = list(pretrained.get("mean", [0.43, 0.411, 0.296]))
        std = list(pretrained.get("std", [0.213, 0.156, 0.143]))
        logger.info("Loading frozen DINOv3 timm checkpoint from %s (%s)", checkpoint, architecture)
        backbone = timm.create_model(
            architecture,
            pretrained=True,
            num_classes=0,
            img_size=image_size,
            dynamic_img_size=True,
            pretrained_cfg_overlay={"file": str(checkpoint)},
        ).to(dtype=dtype)
        return cls(
            backbone,
            mean,
            std,
            image_size,
            int(config["patch_size"]),
            int(config["expected_hidden_dim"]),
            backend="timm",
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
        if self.backend == "timm":
            outputs = self.backbone.forward_features(normalized)
            if isinstance(outputs, dict):
                tokens = outputs.get("x_norm_patchtokens")
                if tokens is None:
                    tokens = outputs.get("last_hidden_state")
                if tokens is None:
                    raise RuntimeError(f"Unsupported timm feature keys: {sorted(outputs)}")
            else:
                tokens = outputs
        else:
            model_kwargs = {"pixel_values": normalized}
            if self.supports_position_interpolation:
                model_kwargs["interpolate_pos_encoding"] = True
            outputs = self.backbone(**model_kwargs)
            tokens = outputs.last_hidden_state
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise RuntimeError(f"DINOv3 must return [B, tokens, channels], got {type(tokens).__name__}")
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

