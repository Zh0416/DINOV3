from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pvseg.config import configure_cloud_cache
from pvseg.data import ImageMaskDataset, load_split_manifest
from pvseg.label_free_data import RGBDataset, load_rgb_manifest
from pvseg.models import FrozenDINOv3
from pvseg.utils import atomic_write_json


def cache_dino_features(config: dict[str, Any], output_root: Path, logger) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("DINOv3 feature extraction requires a CUDA GPU")
    configure_cloud_cache(config)
    image_size = int(config["data"]["image_size"])
    if config.get("protocol", {}).get("label_usage") == "final_evaluation_only":
        samples = load_rgb_manifest(output_root / "rgb_manifest.json")
        dataset = RGBDataset(samples, image_size)
    else:
        samples = load_split_manifest(output_root / "split_manifest.json")
        dataset = ImageMaskDataset(samples, image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    feature_dir = output_root / "feature_cache"
    feature_dir.mkdir(parents=True, exist_ok=True)
    model = FrozenDINOv3.from_config(config["dino"], image_size, logger).to("cuda")
    started = time.perf_counter()
    generated = 0
    skipped = 0
    shapes: set[tuple[int, ...]] = set()
    for batch in tqdm(loader, desc="DINOv3 features"):
        sample_id = batch["sample_id"][0]
        destination = feature_dir / f"{sample_id}.pt"
        if destination.is_file() and config["runtime"].get("resume", True):
            skipped += 1
            continue
        images = batch["image"].to("cuda", non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features = model(images)
        feature = features[0].detach().to(device="cpu", dtype=torch.float16).contiguous()
        shapes.add(tuple(feature.shape))
        torch.save(feature, destination)
        generated += 1
    report = {
        "model_id": config["dino"]["model_id"],
        "samples": len(samples),
        "generated": generated,
        "skipped": skipped,
        "feature_shapes": [list(shape) for shape in sorted(shapes)],
        "elapsed_seconds": time.perf_counter() - started,
        "feature_dir": str(feature_dir),
    }
    atomic_write_json(output_root / "reports" / "dino_feature_report.json", report)
    logger.info("DINO feature report: %s", report)
    return report

