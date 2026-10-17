"""Read-only, train-RGB-only coverage dry run using existing DINO features."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from pvseg.label_free_data import load_rgb_manifest, manual_label_firewall, rgb_hash
from pvseg.pseudo_labels import fit_prototypes, rgb_seeds
from pvseg.soft_semantic import IGNORE, _fuse_soft_evidence, build_soft_pseudo_targets
from pvseg.teacher_probe import diagnose_grid_pair
from pvseg.utils import read_json


def validate_probe_samples(samples):
    if not 1 <= len(samples) <= 2 or any(s.split != "train" for s in samples):
        raise ValueError("Coverage probe accepts only one or two fixed training RGBs")
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("Duplicate probe sample IDs")
    for sample in samples:
        if Path(sample.sample_id).name != sample.sample_id:
            raise ValueError("Unsafe sample ID")
        if rgb_hash(sample.image_path) != sample.rgb_sha256:
            raise ValueError(f"Probe RGB hash changed: {sample.sample_id}")


def validate_feature_source(samples, feature_root, config):
    """Check available provenance before opening any cached feature tensor."""
    feature_root = Path(feature_root)
    source_config = read_json(feature_root / "protocol_config.json")
    if (source_config.get("dino") != config.get("dino")
            or source_config.get("data", {}).get("image_size") != config["data"]["image_size"]):
        raise ValueError("Cached DINO configuration/image size differs from the probe")
    rows = load_rgb_manifest(feature_root / "rgb_manifest.json")
    indexed = {s.sample_id: s for s in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate feature-source manifest sample IDs")
    for sample in samples:
        source = indexed.get(sample.sample_id)
        if source is None or source.split != "train" or source.rgb_sha256 != sample.rgb_sha256:
            raise ValueError(f"Feature-source RGB hash/split mismatch: {sample.sample_id}")
    return {
        "feature_root": str(feature_root.resolve()),
        "manifest_rgb_hash_and_train_split_verified": True,
        "dino_config_and_image_size_verified": True,
        "checkpoint_bytes_verified": False,
        "limitation": "Historical feature caches do not seal checkpoint bytes or environment model-path overrides; config matching cannot verify those bytes.",
    }


def probe_pseudo_coverage(samples, feature_root, records, config):
    """Reuse the active fusion/target builder, without writing training artifacts."""
    with manual_label_firewall():
        return _probe_pseudo_coverage(samples, Path(feature_root), records, config)


def _probe_pseudo_coverage(samples, feature_root, records, config):
    started = time.perf_counter()
    validate_probe_samples(samples)
    provenance = validate_feature_source(samples, feature_root, config)
    settings = config["reliability"]
    size = int(settings["feature_size"])
    if int(settings.get("grid_size", 4)) != 4 or size < 4:
        raise ValueError("Protocol probe requires a 4x4 grid and feature_size >= 4")
    indexed = {r["sample_id"]: {"views": r["protocols"]["grid"]} for r in records}
    seed_masks, per_image, blocked = {}, [], []
    minimum = int(settings["min_seed_pixels"])
    for sample in samples:
        record = indexed.get(sample.sample_id, {"views": {}})
        pair = diagnose_grid_pair(record["views"], settings.get("positive_probability", .55),
                                  settings.get("negative_probability", .18))
        if (not pair["complete_pair"] or pair["probability_status_violations"]
                or pair["conflict_fraction"] > config["teacher"].get("max_global_pair_conflict", .25)):
            blocked.append(sample.sample_id)
        attention, fusion_diag, evidence = _fuse_soft_evidence(record, settings)
        rgb_fg, rgb_bg = rgb_seeds(sample.image_path, size)
        teacher_fg = evidence["positive_support"] & ~evidence["uncertain"]
        teacher_bg = evidence["negative_support"] & ~evidence["uncertain"]
        rgb_supported_fg = teacher_fg & rgb_fg
        fg = rgb_supported_fg & (attention >= settings.get("positive_probability", .55))
        bg = teacher_bg & rgb_bg
        seed_masks[sample.sample_id] = (fg, bg)
        per_image.append({
            "sample_id": sample.sample_id, "feature_pixels": size * size,
            "teacher_positive_pixels": int(teacher_fg.sum()),
            "teacher_negative_pixels": int(teacher_bg.sum()),
            "teacher_conflict_pixels": fusion_diag["teacher_conflict_pixels"],
            "rgb_foreground_pixels": int(rgb_fg.sum()), "rgb_background_pixels": int(rgb_bg.sum()),
            "teacher_rgb_foreground_pixels": int(rgb_supported_fg.sum()),
            "teacher_rgb_background_pixels": int(bg.sum()),
            "prototype_foreground_pixels_after_smoothing": int(fg.sum()),
            "prototype_background_pixels": int(bg.sum()),
            "foreground_meets_min_seed_pixels": int(fg.sum()) >= minimum,
            "background_meets_min_seed_pixels": int(bg.sum()) >= minimum,
        })
    report = {
        "status": "pending", "manual_labels_used": False, "accuracy_evaluated": False,
        "training_artifacts_written": False, "provenance": provenance,
        "scope": "Two global views only; prototypes use only these selected train RGBs. Not equivalent to six-view pipeline coverage.",
        "effective_supervision_coverage": None, "min_seed_pixels": minimum, "per_image": per_image,
    }

    def finish(status, reason=None):
        report["status"] = status
        if reason:
            report["reason"] = reason
        report["elapsed_seconds"] = time.perf_counter() - started
        return report

    if blocked:
        return finish("blocked_teacher_protocol", f"Invalid/inconsistent grid pair: {blocked}")
    if not all(any(row[key] for row in per_image) for key in (
            "foreground_meets_min_seed_pixels", "background_meets_min_seed_pixels")):
        return finish("blocked_prototype_seed_support", "No train-only seed support for both classes; final coverage not measured")
    # Check all shapes/channels before fitting; never mix feature spaces.
    channels = None
    for sample in samples:
        feature = torch.load(feature_root / "feature_cache" / f"{sample.sample_id}.pt",
                             map_location="cpu", weights_only=True)
        if (not isinstance(feature, torch.Tensor) or feature.ndim != 3
                or feature.shape[1:] != (size, size) or feature.shape[0] < 1
                or not torch.isfinite(feature).all() or not feature.is_floating_point()):
            raise ValueError(f"Invalid cached probe features: {sample.sample_id}")
        if channels is not None and feature.shape[0] != channels:
            raise ValueError("Incompatible cached DINO feature channels")
        channels = feature.shape[0]
    try:
        prototypes, counts, train_ids = fit_prototypes(samples, feature_root, size, minimum, seed_masks)
    except RuntimeError as error:
        return finish("blocked_prototype_appearance", str(error))
    report["prototype_train_ids"] = train_ids
    report["prototype_images_per_class"] = counts
    for sample, row in zip(samples, per_image):
        feature = torch.load(feature_root / "feature_cache" / f"{sample.sample_id}.pt",
                             map_location="cpu", weights_only=True).float().numpy()
        arrays, diagnostic = build_soft_pseudo_targets(
            feature, sample.image_path, prototypes, indexed[sample.sample_id], settings)
        target = arrays["target_soft"]
        row.update({
            "positive_after_dino_margin_pixels": diagnostic["raw_candidate_pixels"],
            "positive_final_pixels": diagnostic["positive_pixels"],
            "negative_final_pixels": diagnostic["negative_pixels"],
            "ignored_pixels": diagnostic["ignored_pixels"],
            "effective_supervision_coverage": float(np.mean(target != IGNORE)),
            "positive_coverage": float(np.mean(target == 1)),
            "negative_coverage": float(np.mean(target == 0)),
            "all_ignore": bool(np.all(target == IGNORE)),
            "all_positive_anomaly": bool(np.all(target == 1)),
            "all_background_anomaly": bool(np.all(target == 0)),
            "retention_fallback": diagnostic["retention_fallback"],
        })
    report["effective_supervision_coverage"] = float(np.mean([
        row["effective_supervision_coverage"] for row in per_image]))
    report["train_class_pixels"] = {
        "positive": sum(row["positive_final_pixels"] for row in per_image),
        "negative": sum(row["negative_final_pixels"] for row in per_image),
    }
    return finish("measured_diagnostic_only")
