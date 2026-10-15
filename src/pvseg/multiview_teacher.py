"""Frozen teacher views. Agreement is a reliability proxy, not ground truth."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from pvseg.config import configure_cloud_cache
from pvseg.label_free_data import load_rgb_manifest
from pvseg.teacher import InternVLTeacher
from pvseg.utils import atomic_write_json


def restore_hflip_boxes(boxes, size=1024):
    return [[size-x2, y1, size-x1, y2] for x1, y1, x2, y2 in boxes]


def generate_multiview_teacher(config, output_root, logger):
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
