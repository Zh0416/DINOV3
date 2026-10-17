"""RGB heuristic + train-only DINO prototypes + cross-view teacher agreement.

Targets here are machine-generated weak supervision, never manual PV annotations.
The fixed RGB rule can confuse water/shadows with PV and miss atypical panels.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter

from pvseg.label_free_data import load_rgb_manifest
from pvseg.utils import atomic_write_json, read_json


IGNORE = 255


def rgb_seeds(image_path, size):
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BOX), np.float32)/255
    red, green, blue = np.moveaxis(rgb, -1, 0)
    luminance = .299*red + .587*green + .114*blue
    foreground = ((luminance >= .05) & (luminance <= .5) & (blue >= .9*red)
                  & (green <= 1.15*blue))
    background = (luminance >= .65) | (green > 1.25*np.maximum(red, blue))
    return foreground, background


def normalized_features(feature):
    feature = np.asarray(feature, np.float32)
    return feature / np.maximum(np.linalg.norm(feature, axis=0, keepdims=True), 1e-8)


def fit_prototypes(samples, root, size, min_pixels, seed_masks=None):
    """Test RGB/features must never contribute to learned prototypes."""
    sums, counts = None, np.zeros(2, np.int64)
    used = []
    for sample in samples:
        if sample.split != "train":
            continue
        feature = normalized_features(torch.load(root / "feature_cache" / f"{sample.sample_id}.pt",
                                                 map_location="cpu", weights_only=True).numpy())
        if feature.shape[1:] != (size, size) or not np.isfinite(feature).all():
            raise ValueError(f"Invalid prototype features: {sample.sample_id}")
        fg, bg = (rgb_seeds(sample.image_path, size) if seed_masks is None
                  else seed_masks[sample.sample_id])
        if sums is None:
            sums = np.zeros((2, feature.shape[0]), np.float64)
        # Equal weight per usable image avoids a large dark scene dominating.
        for label, seed in ((0, bg), (1, fg)):
            if int(seed.sum()) >= min_pixels:
                vector = feature[:, seed].mean(1)
                sums[label] += vector / max(float(np.linalg.norm(vector)), 1e-8)
                counts[label] += 1
        used.append(sample.sample_id)
    if sums is None or np.any(counts == 0):
        raise RuntimeError("No usable RGB seeds for both prototypes; abstain instead of inventing labels")
    prototypes = sums / counts[:, None]
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if seed_masks is not None and (np.any(norms < 1e-8) or not np.isfinite(norms).all()):
        raise RuntimeError("Degenerate teacher-constrained prototypes; no usable appearance separation")
    prototypes /= np.maximum(norms, 1e-8)
    if seed_masks is not None and np.allclose(prototypes[0], prototypes[1], atol=1e-5):
        raise RuntimeError("Foreground/background prototypes are indistinguishable; abstain")
    return prototypes.astype(np.float32), counts.tolist(), used


def box_raster(boxes, size):
    result = np.zeros((size, size), bool)
    for x1, y1, x2, y2 in boxes:
        left, top = int(x1*size/1024), int(y1*size/1024)
        right, bottom = int(np.ceil(x2*size/1024)), int(np.ceil(y2*size/1024))
        result[max(0, top):min(size, bottom), max(0, left):min(size, right)] = True
    return result


def valid_view(view):
    return (view.get("status") == "ok" and not view.get("parse_warning")
            and bool(view.get("pv_exist")) and bool(view.get("bbox_original_coordinates")))


def view_agreement(record, settings):
    size = int(settings["feature_size"])
    original, flipped = record["views"]["original"], record["views"]["hflip"]
    raw = box_raster(original.get("bbox_original_coordinates", []), size) if valid_view(original) else np.zeros((size, size), bool)
    other = box_raster(flipped.get("bbox_original_coordinates", []), size) if valid_view(flipped) else np.zeros_like(raw)
    union, intersection = raw | other, raw & other
    iou = int(intersection.sum()) / max(int(union.sum()), 1)
    # Area checks use continuous coordinates; coarse raster IoU remains a proxy.
    area = float(union.mean())
    accepted = (valid_view(original) and valid_view(flipped) and iou >= settings["view_iou_min"]
                and settings["box_area_min"] <= area <= settings["box_area_max"])
    reason = "accepted" if accepted else (
        "invalid_or_absent_view" if not (valid_view(original) and valid_view(flipped)) else
        "view_disagreement" if iou < settings["view_iou_min"] else "implausible_area")
    return raw, union, intersection if accepted else np.zeros_like(raw), iou, reason


def build_pseudo_targets(feature, fg_seed, bg_seed, prototypes, record, settings):
    size = int(settings["feature_size"])
    normalized = normalized_features(feature)
    similarity = np.einsum("kc,chw->khw", prototypes, normalized)
    margin = similarity[1] - similarity[0]
    limit = float(settings["feature_margin"])
    foreground = fg_seed & (margin >= limit)
    background = bg_seed & (margin <= -limit)
    heuristic = np.full((size, size), IGNORE, np.uint8)
    heuristic[background], heuristic[foreground] = 0, 1
    raw, union, consensus, iou, reason = view_agreement(record, settings)
    # Raw coarse-box pseudo labels deliberately expose the noisy-box failure mode.
    raw_target = np.full_like(heuristic, IGNORE)
    raw_target[background & ~binary_dilation(raw, iterations=2)] = 0
    raw_target[raw] = 1
    # Agreement alone is insufficient: intersect semantic evidence with appearance
    # seeds and train-only DINO margin. Rejection means unknown, never PV absence.
    candidate = consensus & foreground
    if int(candidate.sum()) < int(settings["min_seed_pixels"]):
        candidate[:] = False
        if reason == "accepted":
            reason = "insufficient_rgb_dino_support"
    filtered = np.full_like(heuristic, IGNORE)
    filtered[background & ~binary_dilation(union, iterations=2)] = 0
    filtered[candidate] = 1
    def attention(mask, strength):
        value = gaussian_filter(mask.astype(np.float32), float(settings["gaussian_sigma_patches"]))
        return value / max(float(value.max()), 1e-8) * strength
    arrays = {"target_heuristic": heuristic, "target_raw": raw_target, "target_filtered": filtered,
              "attention_raw": attention(raw, 1.), "attention_filtered": attention(candidate, iou),
              "teacher_raw": raw.astype(np.uint8), "teacher_consensus": consensus.astype(np.uint8)}
    diagnostic = {"accepted": bool(candidate.any()), "reason": reason, "view_iou": iou,
                  "raw_box_fraction": float(raw.mean()),
                  "pseudo_counts": {name: {str(k): int((value == k).sum()) for k in (0, 1, IGNORE)}
                                    for name, value in arrays.items() if name.startswith("target_")}}
    return arrays, diagnostic


def generate_pseudo_cache(config, output_root, logger):
    if config.get("protocol", {}).get("mode") == "soft_semantic":
        from pvseg.soft_semantic import generate_soft_pseudo_cache
        return generate_soft_pseudo_cache(config, output_root, logger)
    samples = load_rgb_manifest(output_root / "rgb_manifest.json")
    settings = config["reliability"]
    size = int(settings["feature_size"])
    prototypes, counts, train_ids = fit_prototypes(samples, output_root, size, int(settings["min_seed_pixels"]))
    np.save(output_root / "rgb_dino_prototypes.npy", prototypes)
    atomic_write_json(output_root / "reports" / "prototype_provenance.json", {
        "train_ids": train_ids, "images_per_class": counts, "manual_labels_used": False,
        "semantic_prior": "fixed RGB dark/blue foreground; bright/vegetated background, not PV truth"})
    records = {r["sample_id"]: r for r in read_json(output_root / "semantic_prompt.json")["records"]}
    cache_dir = output_root / "pseudo_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for sample in samples:
        feature = torch.load(output_root / "feature_cache" / f"{sample.sample_id}.pt",
                             map_location="cpu", weights_only=True).numpy()
        foreground, background = rgb_seeds(sample.image_path, size)
        arrays, diagnostic = build_pseudo_targets(feature, foreground, background, prototypes,
                                                  records[sample.sample_id], settings)
        np.savez_compressed(cache_dir / f"{sample.sample_id}.npz", **arrays)
        diagnostics.append({"sample_id": sample.sample_id, "split": sample.split, **diagnostic})
    report = {"manual_labels_used": False, "settings": settings,
              "rejection_counts": dict(Counter(d["reason"] for d in diagnostics)), "per_image": diagnostics}
    atomic_write_json(output_root / "reports" / "pseudo_reliability.json", report)
    logger.info("Label-free reliability diagnostics: %s", report["rejection_counts"])
    # Fail clearly if a method has no positive training signal; no silent all-BG
    # fallback or threshold loosening based on evaluation annotations.
    for kind in ("heuristic", "raw", "filtered"):
        positives = sum(d["pseudo_counts"][f"target_{kind}"]["1"] for d in diagnostics if d["split"] == "train")
        if not positives:
            raise RuntimeError(f"No positive training pseudo-labels for {kind}; inspect label-free diagnostics")
    return report
