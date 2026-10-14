from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pvseg.data import create_split_manifest, discover_samples, find_dataset_root
from pvseg.utils import atomic_write_json


def inspect_dataset(config: dict[str, Any], output_root: Path, logger) -> dict[str, Any]:
    data_config = config["data"]
    root = find_dataset_root(
        data_config["input_root"], data_config["dataset_slug"], data_config["categories"]
    )
    samples = discover_samples(root, data_config)
    image_sizes: Counter[str] = Counter()
    mask_sizes: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    mask_modes: Counter[str] = Counter()
    for sample in samples:
        with Image.open(sample.image_path) as image:
            image_sizes[f"{image.width}x{image.height}"] += 1
            image_modes[image.mode] += 1
        with Image.open(sample.mask_path) as mask:
            mask_sizes[f"{mask.width}x{mask.height}"] += 1
            mask_modes[mask.mode] += 1
    manifest_path, selected = create_split_manifest(config, output_root)
    report = {
        "dataset_root": str(root),
        "image_count": len(samples),
        "label_count": len(samples),
        "category_counts": dict(Counter(sample.category for sample in samples)),
        "image_sizes": dict(image_sizes),
        "label_sizes": dict(mask_sizes),
        "image_modes": dict(image_modes),
        "label_modes": dict(mask_modes),
        "selected_count": len(selected),
        "split_manifest": str(manifest_path),
    }
    atomic_write_json(output_root / "reports" / "data_report.json", report)
    logger.info("Data report: %s", report)
    _save_random_visualization(samples, output_root / "visualizations" / "data_samples.png", config["seed"])
    return report


def _save_random_visualization(samples, destination: Path, seed: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    chosen = rng.sample(samples, k=min(4, len(samples)))
    figure, axes = plt.subplots(len(chosen), 3, figsize=(12, 4 * len(chosen)), squeeze=False)
    for row, sample in enumerate(chosen):
        image = np.asarray(Image.open(sample.image_path).convert("RGB"))
        mask = np.asarray(Image.open(sample.mask_path).convert("L")) > 0
        overlay = image.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"RGB | {sample.category}")
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Mask")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)

