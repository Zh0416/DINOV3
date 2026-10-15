"""Frozen teacher views. Agreement is a reliability proxy, not ground truth."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from pvseg.config import configure_cloud_cache
from pvseg.label_free_data import load_rgb_manifest
from pvseg.teacher import InternVLTeacher
from pvseg.utils import atomic_write_json


def restore_hflip_boxes(boxes, size=1024):
    return [[size-x2, y1, size-x1, y2] for x1, y1, x2, y2 in boxes]


def generate_multiview_teacher(config, output_root, logger):
    if config.get("protocol", {}).get("mode") == "soft_semantic":
        return generate_grid_teacher(config, output_root, logger)
    if not torch.cuda.is_available():
        raise RuntimeError("InternVL inference must run on a cloud GPU")
    configure_cloud_cache(config)
    samples = load_rgb_manifest(output_root / "rgb_manifest.json")
    teacher = InternVLTeacher.load(config["teacher"], logger)
    records, started = [], time.perf_counter()
    for index, sample in enumerate(samples):
        views = {}
        for view in ("original", "hflip"):
            try:
                record = teacher.infer(sample.image_path, view=view)
                record["status"] = "ok"
            except Exception as error:
                logger.exception("Teacher failed: image=%s view=%s", sample.sample_id, view)
                record = {"status": "error", "pv_exist": False, "bbox": [],
                          "confidence": 0., "reason": "", "error": str(error)}
            record["bbox_original_coordinates"] = (
                restore_hflip_boxes(record["bbox"]) if view == "hflip" else record["bbox"])
            views[view] = record
        records.append({"sample_id": sample.sample_id, "views": views})
        atomic_write_json(output_root / "semantic_prompt.json", {
            "teacher_model": config["teacher"]["model_id"], "prompt": config["teacher"]["prompt"],
            "coordinates": "xyxy on 1024x1024; hflip boxes restored separately",
            "records": records})
        logger.info("Teacher image %d/%d original=%s flipped=%s elapsed=%.1fs", index+1, len(samples),
                    views["original"].get("pv_exist"), views["hflip"].get("pv_exist"),
                    time.perf_counter()-started)
    report = {"images": len(samples), "inferences": 2*len(samples),
              "elapsed_seconds": time.perf_counter()-started, "manual_labels_used": False,
              "self_confidence_is_calibrated": False,
              "patch_counts": sorted({v.get("patch_count", 0) for r in records for v in r["views"].values()})}
    atomic_write_json(output_root / "reports" / "multiview_teacher_report.json", report)
    return report


def _overlapping_quadrants(size: int = 1024, overlap: float = 0.10) -> list[list[int]]:
    span = int(round(size / 2 * (1.0 + overlap)))
    starts = [0, size - span]
    return [[x, y, min(size, x + span), min(size, y + span)]
            for y in starts for x in starts]


def _error_view(crop_box=None, flipped=False, error=""):
    return {"status": "error", "pv_exist": False, "confidence": 0.0, "grid": [],
            "reason": "", "crop_box": crop_box, "flipped": flipped, "error": str(error)}


def generate_grid_teacher(config, output_root, logger):
    """Generate full-image and overlapping local grid evidence, never hard boxes."""
    if not torch.cuda.is_available():
        raise RuntimeError("InternVL inference must run on a cloud GPU")
    configure_cloud_cache(config)
    samples = load_rgb_manifest(output_root / "rgb_manifest.json")
    destination = output_root / "semantic_prompt.json"
    existing = {}
    if config["runtime"].get("resume", True) and destination.is_file():
        try:
            existing = {row["sample_id"]: row for row in __import__("pvseg.utils", fromlist=["read_json"]).read_json(destination).get("records", [])}
        except (OSError, KeyError, TypeError, ValueError):
            existing = {}
    teacher = InternVLTeacher.load(config["teacher"], logger)
    records, started = [], time.perf_counter()
    prompt = config["teacher"]["prompt"]
    local_boxes = _overlapping_quadrants()
    for index, sample in enumerate(samples):
        if sample.sample_id in existing:
            record = existing[sample.sample_id]
            records.append(record)
            logger.info("Soft teacher image %d/%d resumed=%s", index + 1, len(samples), sample.sample_id)
            continue
        views = {}
        jobs = [("original", None, "original"), ("hflip", None, "hflip")]
        if config["teacher"].get("local_views", True):
            for local_index, box in enumerate(local_boxes):
                jobs.append((f"local_{local_index}", box, "original"))
        for name, crop_box, view in jobs:
            try:
                result = teacher.infer(sample.image_path, view=view, crop_box=crop_box, prompt=prompt)
                result["status"] = "ok"
                result["crop_box"] = crop_box
                result["flipped"] = view == "hflip"
            except Exception as error:
                logger.exception("Soft teacher failed: image=%s view=%s", sample.sample_id, name)
                result = _error_view(crop_box, view == "hflip", error)
            views[name] = result
        record = {"sample_id": sample.sample_id, "views": views,
                  "grid_size": int(config["reliability"].get("grid_size", 4)),
                  "teacher_schema": "grid_v2"}
        records.append(record)
        atomic_write_json(destination, {
            "teacher_model": config["teacher"]["model_id"], "prompt": prompt,
            "coordinates": "4x4 grid coordinates are relative to each view; crop_box is in original 1024 frame",
            "schema": "grid_v2", "records": records})
        logger.info("Soft teacher image %d/%d views=%d elapsed=%.1fs", index + 1, len(samples),
                    len(views), time.perf_counter() - started)
    report = {
        "images": len(samples),
        "inferences": sum(len(r["views"]) for r in records),
        "global_views": 2,
        "local_views_per_image": len(local_boxes) if config["teacher"].get("local_views", True) else 0,
        "elapsed_seconds": time.perf_counter() - started,
        "manual_labels_used": False,
        "self_confidence_is_calibrated": False,
        "schema": "grid_v2",
    }
    atomic_write_json(output_root / "reports" / "multiview_teacher_report.json", report)
    return report
