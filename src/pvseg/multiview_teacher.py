"""Frozen teacher views. Agreement is a reliability proxy, not ground truth."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from pvseg.config import configure_cloud_cache
from pvseg.label_free_data import load_rgb_manifest
from pvseg.teacher import InternVLTeacher, parse_teacher_response
from pvseg.teacher.internvl import GRID_PARSER_VERSION
from pvseg.utils import atomic_write_json, read_json


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


def reparse_grid_record(record):
    """Migrate saved raw responses without inference or changes to view geometry."""
    views = {}
    for name, view in record.get("views", {}).items():
        migrated = dict(view)
        raw = view.get("raw_response")
        if view.get("status") == "ok" and isinstance(raw, str) and raw.strip():
            migrated.pop("parse_warning", None)
            migrated.update(parse_teacher_response(raw))
            migrated["grid_parser_version"] = GRID_PARSER_VERSION
        views[name] = migrated
    return {**record, "views": views}


def generate_grid_teacher(config, output_root, logger):
    """Generate full-image and overlapping local grid evidence, never hard boxes."""
    samples = load_rgb_manifest(output_root / "rgb_manifest.json")
    destination = output_root / "semantic_prompt.json"
    existing, payload = {}, {}
    if config["runtime"].get("resume", True) and destination.is_file():
        # A broken cache must not silently trigger another full teacher run.
        payload = read_json(destination)
        if payload.get("teacher_model", config["teacher"]["model_id"]) != config["teacher"]["model_id"]:
            raise ValueError("Cached teacher model differs; use a separate output directory")
        existing = {row["sample_id"]: row for row in payload.get("records", [])}
    teacher, started = None, time.perf_counter()
    records, reused_views, reparsed_views, new_inferences = {}, 0, 0, 0
    prompt = config["teacher"]["prompt"]
    local_boxes = _overlapping_quadrants()
    jobs = [("original", None, "original"), ("hflip", None, "hflip")]
    if config["teacher"].get("local_views", True):
        jobs.extend((f"local_{index}", box, "original") for index, box in enumerate(local_boxes))

    def save_records():
        atomic_write_json(destination, {
            "teacher_model": config["teacher"]["model_id"], "prompt": prompt,
            "coordinates": "4x4 grid coordinates are relative to each view; crop_box is in original 1024 frame",
            "schema": "grid_v2", "grid_parser_version": GRID_PARSER_VERSION,
            "generation_provenance": "Each view retains its generation_prompt and teacher_model",
            "records": [records[s.sample_id] for s in samples if s.sample_id in records]})

    for sample in samples:
        if sample.sample_id not in existing:
            continue
        record = reparse_grid_record(existing[sample.sample_id])
        for view in record["views"].values():
            view.setdefault("generation_prompt", payload.get("prompt"))
            view.setdefault("teacher_model", payload.get("teacher_model"))
            if view.get("status") == "ok" and view.get("raw_response"):
                reparsed_views += 1
        records[sample.sample_id] = record
    if records:
        # Persist migration even if all views are reused or a later GPU job fails.
        save_records()

    for index, sample in enumerate(samples):
        record = records.get(sample.sample_id, {"sample_id": sample.sample_id, "views": {}})
        views = record["views"]
        for name, crop_box, view in jobs:
            cached = views.get(name, {})
            if cached.get("status") == "ok" and (cached.get("raw_response") or cached.get("grid")):
                reused_views += 1
                continue
            if teacher is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("Uncached InternVL inference must run on a cloud GPU")
                configure_cloud_cache(config)
                teacher = InternVLTeacher.load(config["teacher"], logger)
            new_inferences += 1
            try:
                result = teacher.infer(sample.image_path, view=view, crop_box=crop_box, prompt=prompt)
                result["status"] = "ok"
                result["crop_box"] = crop_box
                result["flipped"] = view == "hflip"
            except Exception as error:
                logger.exception("Soft teacher failed: image=%s view=%s", sample.sample_id, name)
                result = _error_view(crop_box, view == "hflip", error)
            result.update({"grid_parser_version": GRID_PARSER_VERSION,
                           "generation_prompt": prompt, "teacher_model": config["teacher"]["model_id"]})
            views[name] = result
        record.update({"grid_size": int(config["reliability"].get("grid_size", 4)), "teacher_schema": "grid_v2"})
        records[sample.sample_id] = record
        save_records()
        logger.info("Soft teacher image %d/%d views=%d elapsed=%.1fs", index + 1, len(samples),
                    len(views), time.perf_counter() - started)
    report = {
        "images": len(samples),
        "inferences": sum(len(r["views"]) for r in records.values()),
        "new_inferences": new_inferences,
        "reused_views": reused_views,
        "reparsed_views": reparsed_views,
        "global_views": 2,
        "local_views_per_image": len(local_boxes) if config["teacher"].get("local_views", True) else 0,
        "elapsed_seconds": time.perf_counter() - started,
        "manual_labels_used": False,
        "self_confidence_is_calibrated": False,
        "schema": "grid_v2",
    }
    atomic_write_json(output_root / "reports" / "multiview_teacher_report.json", report)
    return report
