"""Frozen teacher views. Agreement is a reliability proxy, not ground truth."""
from __future__ import annotations

import time
import os
from pathlib import Path

import numpy as np
import torch

from pvseg.config import configure_cloud_cache
from pvseg.label_free_data import load_rgb_manifest
from pvseg.teacher import InternVLTeacher, parse_teacher_response
from pvseg.teacher.internvl import GRID_PARSER_VERSION, IMAGE_PREPROCESS_VERSION
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


def validate_complete_grid(view, grid_size=4):
    """Require every location, including explicit unknowns; never pad missing cells."""
    expected = {(row, col) for row in range(grid_size) for col in range(grid_size)}
    cells = view.get("grid", [])
    if not isinstance(cells, list):
        cells = []
    try:
        locations = {(cell["row"], cell["col"]) for cell in cells}
    except (KeyError, TypeError):
        locations = set()
    complete = view.get("status") == "ok" and len(cells) == len(expected) and locations == expected
    view.update({"grid_cell_count": len(cells), "grid_complete": complete})
    return complete


def grid_generation_spec(config):
    teacher = config["teacher"]
    return {"model_id": teacher["model_id"], "prompt": teacher["prompt"],
            "schema": teacher.get("schema", "grid_v2"), "dtype": teacher.get("dtype", "float16"),
            "quantization": teacher.get("quantization", "auto"),
            "max_new_tokens": int(teacher.get("max_new_tokens", 256)),
            "max_dynamic_patches": int(teacher.get("max_dynamic_patches", 6)),
            "use_thumbnail": bool(teacher.get("use_thumbnail", False)),
            "local_model_path": os.getenv(teacher.get("local_path_env", "")) or None,
            "preprocess_version": IMAGE_PREPROCESS_VERSION, "do_sample": False}


def validate_grid_cache_provenance(payload, config):
    """Enforce the same generation contract even when the teacher stage is skipped."""
    spec = grid_generation_spec(config)
    if (payload.get("cache_only_replay") or payload.get("generation_spec") != spec
            or payload.get("teacher_model") != config["teacher"]["model_id"]):
        raise ValueError("Cached teacher generation settings differ or lack provenance; use a new output directory. "
                         "Historical raw responses may only be migrated with cache_only_replay")
    for record in payload.get("records", []):
        for view in record.get("views", {}).values():
            if (view.get("generation_spec") != spec or view.get("generation_prompt") != spec["prompt"]
                    or view.get("teacher_model") != spec["model_id"]):
                raise ValueError("Cached view generation settings differ; refusing mixed teacher evidence")
    runtime_specs = [v["runtime_spec"] for r in payload.get("records", []) for v in r.get("views", {}).values()
                     if isinstance(v.get("runtime_spec"), dict)]
    if runtime_specs and any(item != runtime_specs[0] for item in runtime_specs[1:]):
        raise ValueError("Cached effective teacher precision/model differs between views")


def global_pair_diagnostic(views, positive_probability=.55):
    """Compare explicit decisions only, restoring the hflip to original coordinates."""
    def decisions(view):
        result = {}
        for cell in view.get("grid", []):
            status = cell.get("status")
            if status == "positive" and float(cell.get("pv_probability", 0)) < positive_probability:
                continue
            if status in {"positive", "negative"}:
                row, col = int(cell["row"]), int(cell["col"])
                result[(row, 3-col if view.get("flipped") else col)] = status
        return result
    original, flipped = (decisions(views.get(name, {})) for name in ("original", "hflip"))
    common = original.keys() & flipped.keys()
    conflicts = [list(key) for key in sorted(common) if original[key] != flipped[key]]
    return {"observed_pairs": len(common), "conflicting_cells": conflicts,
            "conflict_fraction": len(conflicts) / 16,
            "self_consistency_is_correctness": False}


def generate_grid_teacher(config, output_root, logger):
    """Generate full-image and overlapping local grid evidence, never hard boxes."""
    samples = load_rgb_manifest(output_root / "rgb_manifest.json")
    destination = output_root / "semantic_prompt.json"
    spec = grid_generation_spec(config)
    replay_only = bool(config["teacher"].get("cache_only_replay", False))
    existing, payload = {}, {}
    if config["runtime"].get("resume", True) and destination.is_file():
        # A broken cache must not silently trigger another full teacher run.
        payload = read_json(destination)
        if payload.get("teacher_model", config["teacher"]["model_id"]) != config["teacher"]["model_id"]:
            raise ValueError("Cached teacher model differs; use a separate output directory")
        if not replay_only:
            validate_grid_cache_provenance(payload, config)
        existing = {row["sample_id"]: row for row in payload.get("records", [])}
    teacher, started = None, time.perf_counter()
    records, reused_views, reparsed_views, new_inferences = {}, 0, 0, 0
    pair_checks = []
    max_pair_conflict = float(config["teacher"].get("max_global_pair_conflict", .25))
    if not 0 <= max_pair_conflict < 1:
        raise ValueError("max_global_pair_conflict must be in [0, 1)")
    prompt = config["teacher"]["prompt"]
    require_complete = config["teacher"].get("require_complete_grid", False)
    grid_size = int(config["reliability"].get("grid_size", 4))
    if grid_size != 4:
        raise ValueError("The grid teacher prompt/parser requires grid_size=4")
    local_boxes = _overlapping_quadrants()
    jobs = [("original", None, "original"), ("hflip", None, "hflip")]
    if config["teacher"].get("local_views", True):
        jobs.extend((f"local_{index}", box, "original") for index, box in enumerate(local_boxes))

    def save_records():
        atomic_write_json(destination, {
            "teacher_model": config["teacher"]["model_id"],
            "prompt": payload.get("prompt") if replay_only else prompt,
            "generation_spec": payload.get("generation_spec") if replay_only else spec,
            "cache_only_replay": replay_only,
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
        records[sample.sample_id] = record
        views = record["views"]
        def check_global_pair():
            diagnostic = {"sample_id": sample.sample_id,
                          **global_pair_diagnostic(views, config["reliability"].get("positive_probability", .55))}
            failed = diagnostic["conflict_fraction"] > max_pair_conflict
            diagnostic["status"] = "failed" if failed else "passed"
            pair_checks.append(diagnostic)
            atomic_write_json(output_root / "reports" / "teacher_pair_checks.json", {
                "status": "failed" if failed else "checked_so_far", "max_conflict_fraction": max_pair_conflict,
                "manual_labels_used": False, "per_image": pair_checks})
            if failed:
                save_records()
                raise RuntimeError(f"Teacher original/hflip conflict on {len(diagnostic['conflicting_cells'])}/16 cells "
                                   f"for {sample.sample_id}; stop before local inference. See teacher_pair_checks.json")
        for name, crop_box, view in jobs:
            cached = views.get(name, {})
            if cached.get("status") == "ok" and (cached.get("raw_response") or cached.get("grid")):
                complete = validate_complete_grid(cached, grid_size)
                if require_complete and not complete:
                    save_records()
                    raise RuntimeError(f"Incomplete cached teacher grid: {sample.sample_id}/{name} "
                                       f"({cached['grid_cell_count']}/16). Use a new output directory to regenerate with the compact prompt.")
                reused_views += 1
                if name == "hflip" and not replay_only:
                    check_global_pair()
                continue
            if replay_only:
                raise RuntimeError(f"Cache-only replay cannot infer missing/failed view: {sample.sample_id}/{name}")
            if teacher is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("Uncached InternVL inference must run on a cloud GPU")
                configure_cloud_cache(config)
                teacher = InternVLTeacher.load(config["teacher"], logger)
                runtime = teacher.runtime_spec
                if isinstance(runtime, dict):
                    previous = [v["runtime_spec"] for r in records.values() for v in r["views"].values()
                                if isinstance(v.get("runtime_spec"), dict)]
                    if any(old != runtime for old in previous):
                        raise RuntimeError("Teacher effective precision/model differs from cached views; use a new output directory")
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
                           "generation_prompt": prompt, "teacher_model": config["teacher"]["model_id"],
                           "generation_spec": spec})
            views[name] = result
            save_records()  # Preserve each expensive inference, including later failures.
            complete = validate_complete_grid(result, grid_size)
            if require_complete and not complete:
                save_records()
                raise RuntimeError(f"Incomplete teacher grid: {sample.sample_id}/{name} "
                                   f"({result['grid_cell_count']}/16). Raw response saved; check the prompt/token budget before continuing.")
            if name == "hflip":
                check_global_pair()
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
        "require_complete_grid": require_complete,
        "grid_cells": sum(len(v.get("grid", [])) for r in records.values() for v in r["views"].values()),
        "complete_grid_views": sum(bool(v.get("grid_complete")) for r in records.values() for v in r["views"].values()),
        "incomplete_grid_views": sum(not v.get("grid_complete", False) for r in records.values() for v in r["views"].values()),
    }
    atomic_write_json(output_root / "reports" / "multiview_teacher_report.json", report)
    if not replay_only:
        atomic_write_json(output_root / "reports" / "teacher_pair_checks.json", {
            "status": "passed", "max_conflict_fraction": max_pair_conflict,
            "manual_labels_used": False, "per_image": pair_checks})
    return report
