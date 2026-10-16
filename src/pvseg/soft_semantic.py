"""Soft grid-teacher fusion and label-free three-way pseudo targets."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from pvseg.label_free_data import load_rgb_manifest
from pvseg.pseudo_labels import fit_prototypes, normalized_features, rgb_seeds
from pvseg.utils import atomic_write_json, read_json

IGNORE = 255


def _grid_array(view: dict[str, Any], grid_size: int) -> np.ndarray:
    result = np.full((grid_size, grid_size), np.nan, np.float32)
    for item in view.get("grid", []) if isinstance(view.get("grid", []), list) else []:
        try:
            row, col = int(item["row"]), int(item["col"])
            probability = float(item.get("pv_probability", item.get("confidence", 0.0)))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= row < grid_size and 0 <= col < grid_size and np.isfinite(probability):
            status = str(item.get("status", "unknown")).lower()
            # Unknown cells abstain even if their numeric probability is high.
            # Other views can still provide independent evidence at this location.
            if status not in {"positive", "negative"}:
                continue
            if status == "negative":
                probability = 0.0
            result[row, col] = np.clip(probability, 0.0, 1.0)
    return result


def _view_heatmap(view: dict[str, Any], feature_size: int, image_size: int,
                  crop_box: list[int] | None = None, flipped: bool = False,
                  grid_size: int = 4) -> np.ndarray:
    """Rasterize relative grid probabilities into the original feature frame."""
    probabilities = _grid_array(view, grid_size)
    if flipped:
        probabilities = np.fliplr(probabilities)
    heatmap = np.zeros((feature_size, feature_size), np.float32)
    if crop_box is None:
        left, top, right, bottom = 0, 0, image_size, image_size
    else:
        left, top, right, bottom = crop_box
    x0 = int(np.floor(left / image_size * feature_size))
    y0 = int(np.floor(top / image_size * feature_size))
    x1 = int(np.ceil(right / image_size * feature_size))
    y1 = int(np.ceil(bottom / image_size * feature_size))
    x0, x1 = max(0, min(feature_size, x0)), max(0, min(feature_size, x1))
    y0, y1 = max(0, min(feature_size, y0)), max(0, min(feature_size, y1))
    if x1 <= x0 or y1 <= y0:
        return heatmap
    for row in range(grid_size):
        for col in range(grid_size):
            value = probabilities[row, col]
            if not np.isfinite(value):
                continue
            xa = x0 + int(np.floor((x1 - x0) * col / grid_size))
            xb = x0 + int(np.ceil((x1 - x0) * (col + 1) / grid_size))
            ya = y0 + int(np.floor((y1 - y0) * row / grid_size))
            yb = y0 + int(np.ceil((y1 - y0) * (row + 1) / grid_size))
            heatmap[ya:yb, xa:xb] = np.maximum(heatmap[ya:yb, xa:xb], value)
    return heatmap


def fuse_soft_attention(record: dict[str, Any], settings: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    size = int(settings["feature_size"])
    image_size = 1024
    grid_size = int(settings.get("grid_size", 4))
    maps = []
    view_meta = record.get("views", {})
    for name, view in view_meta.items():
        # A recovered/truncated JSON response can still contain complete grid
        # cells.  Use those cells as weak evidence; only abstain when the view
        # is not successful or has no usable grid at all.
        if view.get("status") != "ok" or not isinstance(view.get("grid"), list) or not view.get("grid"):
            continue
        if not np.isfinite(_grid_array(view, grid_size)).any():
            continue
        crop = view.get("crop_box")
        maps.append(_view_heatmap(view, size, image_size, crop, bool(view.get("flipped")), grid_size))
    if not maps:
        return np.full((size, size), float(settings.get("base_attention", 0.2)), np.float32), {
            "teacher_views_used": 0, "teacher_consistency": 0.0, "teacher_abstained": True}
    stack = np.stack(maps)
    # Local crops are sparse observations.  A mean would dilute one valid
    # positive crop with several unrelated crops; max pooling preserves the
    # strongest weak evidence and the consistency diagnostic still records
    # disagreement across views.
    fused = stack.max(0)
    if stack.shape[0] > 1:
        consistency = float(np.clip(1.0 - np.mean(np.std(stack, axis=0)), 0.0, 1.0))
    else:
        consistency = 0.0
    if not np.isfinite(fused).any():
        fused[:] = 0.0
    base = float(settings.get("base_attention", 0.2))
    # Never renormalize a uniformly low/unknown grid to one.  Diffuse only the
    # amount above the nonzero base floor, so no positive semantic evidence
    # leaves the map at the safe base level instead of creating a false gate.
    signal = np.maximum(np.nan_to_num(fused, nan=0.0) - base, 0.0)
    signal = gaussian_filter(signal, sigma=float(settings.get("gaussian_sigma_patches", 2.0)))
    attention = np.clip(base + signal, base, 1.0).astype(np.float32)
    return attention, {"teacher_views_used": len(maps), "teacher_consistency": consistency,
                       "teacher_abstained": False, "attention_max": float(attention.max())}


def _resize_feature_mask(mask: np.ndarray, feature_size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((feature_size, feature_size), Image.Resampling.NEAREST), bool)


def build_soft_pseudo_targets(feature: np.ndarray, image_path: str, prototypes: np.ndarray,
                              record: dict[str, Any], settings: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    size = int(settings["feature_size"])
    attention, attention_diag = fuse_soft_attention(record, settings)
    normalized = normalized_features(feature)
    similarity = np.einsum("kc,chw->khw", prototypes, normalized)
    margin = similarity[1] - similarity[0]
    fg_seed, bg_seed = rgb_seeds(image_path, size)
    feature_margin = float(settings.get("feature_margin", 0.04))
    dino_fg = margin >= feature_margin
    dino_bg = margin <= -feature_margin
    positive_probability = float(settings.get("positive_probability", 0.55))
    negative_probability = float(settings.get("negative_probability", 0.18))
    semantic_positive = attention >= positive_probability + 0.15
    raw_candidate = ((attention >= positive_probability) & (fg_seed | dino_fg)) | semantic_positive
    # A strong, multi-view semantic signal can seed a region even when RGB/DINO
    # appearance heuristics miss it; it remains weighted and never becomes a hard gate.
    strict_positive = raw_candidate
    # Erode only the machine core; the transition ring is unknown rather than background.
    boundary_iterations = int(settings.get("boundary_pixels", 1))
    core = (strict_positive.copy() if boundary_iterations <= 0
            else binary_erosion(strict_positive, iterations=boundary_iterations))
    if strict_positive.any() and not core.any():
        core = strict_positive.copy()
    boundary = (np.zeros_like(strict_positive) if boundary_iterations <= 0 else
                binary_dilation(strict_positive, iterations=1) & ~binary_erosion(strict_positive, iterations=1))
    # Attention has a nonzero base level by design, so background is measured
    # relative to that floor rather than compared with an unreachable raw zero.
    base = float(settings.get("base_attention", 0.2))
    background_limit = max(base + 0.03, negative_probability)
    explicit_background = (attention <= background_limit) & bg_seed & dino_bg & ~binary_dilation(strict_positive, iterations=1)
    target = np.full((size, size), IGNORE, np.uint8)
    target[explicit_background] = 0
    target[core] = 1
    target[boundary] = IGNORE
    raw_count = int(raw_candidate.sum())
    retained = int(core.sum()) / max(raw_count, 1)
    min_retention = float(settings.get("min_pseudo_retention", 0.5))
    fallback = bool(raw_count and retained < min_retention)
    if fallback:
        # Keep the broader machine candidate with reduced weights; never invent background.
        target[:] = IGNORE
        target[explicit_background] = 0
        target[strict_positive] = 1
        target[boundary] = IGNORE
    weight = np.zeros((size, size), np.float32)
    weight[target == 0] = 0.5 + 0.5 * (1.0 - attention[target == 0])
    weight[target == 1] = np.maximum(float(settings.get("positive_weight_floor", 0.35)), attention[target == 1])
    weight[boundary] = float(settings.get("unknown_weight", 0.15))
    arrays = {
        "target_soft": target,
        "weight_soft": weight,
        "attention_soft": attention,
        "teacher_candidate": raw_candidate.astype(np.uint8),
        "teacher_core": core.astype(np.uint8),
    }
    diagnostics = {**attention_diag, "raw_candidate_pixels": raw_count, "positive_pixels": int((target == 1).sum()),
                   "negative_pixels": int((target == 0).sum()), "ignored_pixels": int((target == IGNORE).sum()),
                   "pseudo_retention": retained, "retention_fallback": fallback,
                   "pseudo_counts": {str(k): int((target == k).sum()) for k in (0, 1, IGNORE)}}
    return arrays, diagnostics


def generate_soft_pseudo_cache(config: dict[str, Any], output_root: Path, logger) -> dict[str, Any]:
    samples = load_rgb_manifest(output_root / "rgb_manifest.json")
    settings = config["reliability"]
    size = int(settings["feature_size"])
    prototypes, counts, train_ids = fit_prototypes(samples, output_root, size, int(settings["min_seed_pixels"]))
    np.save(output_root / "rgb_dino_prototypes.npy", prototypes)
    atomic_write_json(output_root / "reports" / "prototype_provenance.json", {
        "train_ids": train_ids, "images_per_class": counts, "manual_labels_used": False,
        "semantic_prior": "fixed RGB/DINO appearance seeds only; not PV truth"})
    records = {r["sample_id"]: r for r in read_json(output_root / "semantic_prompt.json")["records"]}
    cache_dir = output_root / "pseudo_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for sample in samples:
        feature = torch.load(output_root / "feature_cache" / f"{sample.sample_id}.pt", map_location="cpu", weights_only=True).numpy()
        arrays, diagnostic = build_soft_pseudo_targets(feature, sample.image_path, prototypes,
                                                       records[sample.sample_id], settings)
        np.savez_compressed(cache_dir / f"{sample.sample_id}.npz", **arrays)
        diagnostics.append({"sample_id": sample.sample_id, "split": sample.split, **diagnostic})
    report = {"manual_labels_used": False, "mode": "soft_semantic", "settings": settings,
              "retention_fallback_images": sum(d["retention_fallback"] for d in diagnostics),
              "rejection_counts": dict(Counter("fallback" if d["retention_fallback"] else "retained" for d in diagnostics)),
              "per_image": diagnostics}
    atomic_write_json(output_root / "reports" / "pseudo_reliability.json", report)
    logger.info("Soft pseudo-label diagnostics: %s", report["rejection_counts"])
    if not any(d["positive_pixels"] for d in diagnostics if d["split"] == "train"):
        raise RuntimeError("No positive soft pseudo-labels in training split; abstain instead of inventing labels")
    return report
