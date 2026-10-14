from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from pvseg.config import configure_cloud_cache
from pvseg.data import load_split_manifest
from pvseg.teacher import InternVLTeacher
from pvseg.utils import atomic_write_json, read_json


def generate_teacher_records(config: dict[str, Any], output_root: Path, logger) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("InternVL inference requires a Kaggle GPU")
    configure_cloud_cache(config)
    samples = load_split_manifest(output_root / "split_manifest.json")
    combined_path = output_root / "semantic_prompt.json"
    record_dir = output_root / "teacher_json"
    record_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if combined_path.is_file() and config["runtime"].get("resume", True):
        payload = read_json(combined_path)
        existing = {record["sample_id"]: record for record in payload.get("records", [])}
    missing = [sample for sample in samples if sample.sample_id not in existing]
    teacher = InternVLTeacher.load(config["teacher"], logger) if missing else None
    started = time.perf_counter()
    for sample in tqdm(missing, desc="InternVL teacher"):
        item_started = time.perf_counter()
        try:
            result = teacher.infer(sample.image_path) if teacher else {}
            status = "ok"
        except Exception as error:
            logger.exception("Teacher inference failed for %s", sample.sample_id)
            result = {
                "pv_exist": False,
                "confidence": 0.0,
                "bbox": [],
                "reason": "",
                "error": f"{type(error).__name__}: {error}",
            }
            status = "error"
        record = {
            **sample.to_dict(),
            **result,
            "status": status,
            "image_size": [1024, 1024],
            "elapsed_seconds": time.perf_counter() - item_started,
        }
        existing[sample.sample_id] = record
        atomic_write_json(record_dir / f"{sample.sample_id}.json", record)
        _write_combined(combined_path, config, samples, existing)
    payload = _write_combined(combined_path, config, samples, existing)
    counts = Counter(record.get("status", "unknown") for record in payload["records"])
    bbox_coverage = sum(bool(record.get("bbox")) for record in payload["records"])
    report = {
        "records": len(payload["records"]),
        "status_counts": dict(counts),
        "records_with_bbox": bbox_coverage,
        "elapsed_seconds": time.perf_counter() - started,
        "semantic_prompt": str(combined_path),
    }
    atomic_write_json(output_root / "reports" / "teacher_report.json", report)
    logger.info("Teacher report: %s", report)
    return report


def _write_combined(path, config, samples, records):
    ordered = [records[sample.sample_id] for sample in samples if sample.sample_id in records]
    payload = {
        "teacher_model": config["teacher"]["model_id"],
        "prompt": config["teacher"]["prompt"],
        "coordinate_system": "xyxy pixels on 1024x1024 resized image",
        "records": ordered,
    }
    atomic_write_json(path, payload)
    return payload

