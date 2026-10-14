from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import yaml


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cloud_cache(config: dict[str, Any]) -> Path:
    cache_dir = Path(config["runtime"]["model_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.setdefault(name, str(cache_dir))
    return cache_dir


def output_dir(config: dict[str, Any]) -> Path:
    path = Path(config["runtime"]["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path

