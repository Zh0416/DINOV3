from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def sanitize_boxes(value: Any, image_size: int = 1024) -> list[list[int]]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, (tuple, list)) and len(value) == 4 and all(
        isinstance(item, (int, float)) for item in value
    ):
        value = [value]
    boxes: list[list[int]] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            item = [item.get("x_min"), item.get("y_min"), item.get("x_max"), item.get("y_max")]
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            continue
        try:
            x1, y1, x2, y2 = (int(round(float(number))) for number in item)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((max(0, min(image_size - 1, x1)), max(0, min(image_size, x2))))
        y1, y2 = sorted((max(0, min(image_size - 1, y1)), max(0, min(image_size, y2))))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def gaussian_attention_map(
    boxes: Any,
    image_size: int = 1024,
    feature_size: int = 64,
    sigma: float = 48.0,
    confidence: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    full = np.zeros((image_size, image_size), dtype=np.float32)
    for x1, y1, x2, y2 in sanitize_boxes(boxes, image_size):
        full[y1:y2, x1:x2] = 1.0
    if full.max() > 0:
        full = gaussian_filter(full, sigma=float(sigma))
        full /= max(float(full.max()), 1e-8)
        full *= float(np.clip(confidence, 0.0, 1.0))
    reduced = np.asarray(
        Image.fromarray(full, mode="F").resize((feature_size, feature_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    return full, reduced


def save_attention_cache(
    records: list[dict[str, Any]],
    destination: Union[str, Path],
    image_size: int,
    feature_size: int,
    sigma: float,
) -> dict[str, int]:
    output_dir = Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_output_dir = output_dir.parent / "attention_1024"
    full_output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    empty = 0
    for record in records:
        full, reduced = gaussian_attention_map(
            record.get("bbox", []),
            image_size=image_size,
            feature_size=feature_size,
            sigma=sigma,
            confidence=float(record.get("confidence", 0.0)),
        )
        np.save(output_dir / f"{record['sample_id']}.npy", reduced)
        Image.fromarray(np.clip(full * 255.0, 0, 255).astype(np.uint8)).save(
            full_output_dir / f"{record['sample_id']}.png"
        )
        generated += 1
        empty += int(not np.any(reduced))
    return {
        "generated": generated,
        "empty": empty,
        "full_resolution_dir": str(full_output_dir),
        "feature_resolution_dir": str(output_dir),
    }

