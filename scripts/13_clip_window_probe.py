"""Bounded CLIP region-semantic diagnostic on the same frozen train RGBs."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image, ImageOps

from pvseg.clip_probe import (FrozenCLIPWindowTeacher, bounded_decision, diagnose_pair, diagnose_window_pair,
                              save_region_preview, validate_settings, window_boxes)
from pvseg.config import configure_cloud_cache, load_config, seed_everything
from pvseg.label_free_data import load_rgb_manifest, manual_label_firewall
from pvseg.probe_coverage import validate_probe_samples
from pvseg.utils import atomic_write_json, read_json, setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/clip_window_probe_v1.yaml")
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Use previous protocol probe's frozen rgb_manifest.json")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--compare-report", type=Path,
                        help="Optional saved InternVL report; verify same RGB IDs/hashes, never rerun it")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("CLIP probe output must be new or empty")
    config = load_config(args.config)
    settings = config["clip_probe"]
    validate_settings(settings)
    root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root, "13_clip_window_probe")
    started = time.perf_counter()
    report = {
        "status": "running", "manual_labels_used": False, "accuracy_evaluated": False,
        "training_artifacts_written": False, "effective_supervision_coverage": None,
        "scope": "CLIP crop region evidence, not pixel occupancy; DINO/InternVL/SAM not loaded",
        "state_encoding": {"positive": 1, "negative": 0, "unknown": -1},
        "records": [],
    }

    def save():
        atomic_write_json(root / "clip_window_probe.json", report)

    try:
        with manual_label_firewall():
            # Consume the complete small manifest; never select by directory/category.
            samples = load_rgb_manifest(args.manifest)
            validate_probe_samples(samples)
            expected = [(s.sample_id, s.rgb_sha256) for s in samples]
            if args.compare_report:
                previous = read_json(args.compare_report)
                if previous.get("manual_labels_used") is not False or previous.get("status") != "diagnostic_complete":
                    raise ValueError("Comparison requires a completed label-free InternVL diagnostic")
                actual = [(r["sample_id"], r["rgb_sha256"]) for r in previous["records"]]
                if expected != actual:
                    raise ValueError("Comparison RGB IDs/order/hashes differ")
                report["internvl_reference"] = {
                    "path": str(args.compare_report.resolve()),
                    "conclusion": previous.get("conclusion"),
                    "diagnostics": previous.get("diagnostics"),
                    "runtime_spec": previous.get("runtime_spec"),
                    "model_load_seconds": previous.get("model_load_seconds"),
                    "elapsed_seconds": previous.get("elapsed_seconds"),
                    "pseudo_coverage_dry_run": previous.get("pseudo_coverage_dry_run"),
                    "limitation": "Different output granularity and score meanings; neither measures accuracy",
                }
            for sample in samples:
                with Image.open(sample.image_path) as image:
                    boxes = window_boxes(*image.size, settings)
                    logger.info("Fixed train RGB %s: %s, %d crops/view", sample.sample_id[:8], image.size, len(boxes))
            atomic_write_json(root / "rgb_manifest.json", {
                "manual_labels_used": False, "samples": [s.to_dict() for s in samples]})
            atomic_write_json(root / "protocol_config.json", config)
            seed_everything(config["seed"])
            cache = configure_cloud_cache(config)
            save()
            logger.info("Loading one frozen CLIP; no generative teacher calls")
            load_started = time.perf_counter()
            teacher = FrozenCLIPWindowTeacher.load(settings, cache)
            report["model_load_seconds"] = time.perf_counter() - load_started
            report["runtime_spec"] = teacher.runtime_spec
            save()
            for sample in samples:
                row = {"sample_id": sample.sample_id, "rgb_sha256": sample.rgb_sha256, "views": {}}
                report["records"].append(row)
                maps = {}
                with Image.open(sample.image_path) as source:
                    image = source.convert("RGB")
                for name in ("original", "hflip"):
                    logger.info("Encoding %s %s", sample.sample_id[:8], name)
                    view = image if name == "original" else ImageOps.mirror(image)
                    row["views"][name], maps[name] = teacher.infer(view)
                    save()
                row["pair"], consensus = diagnose_pair(maps["original"], maps["hflip"])
                row["window_pair"] = diagnose_window_pair(row["views"]["original"], row["views"]["hflip"])
                preview = root / f"{sample.sample_id}_regions.png"
                save_region_preview(image, maps["original"], maps["hflip"], consensus, preview)
                row["diagnostic_preview"] = preview.name
                logger.info("%s consensus region coverage %.3f; conflict %.3f", sample.sample_id[:8],
                            row["pair"]["consensus"]["valid_region_evidence_coverage"], row["pair"]["conflict_fraction"])
                logger.info("%s same-location windows: %d/%d jointly valid, %d repeat PV, %d repeat background",
                            sample.sample_id[:8], row["window_pair"]["joint_valid_windows"],
                            row["window_pair"]["matched_windows"], row["window_pair"]["repeat_positive_windows"],
                            row["window_pair"]["repeat_negative_windows"])
                save()
            report["conclusion"] = bounded_decision(report["records"], settings["max_pair_conflict"])
            report["inference_seconds"] = sum(v["elapsed_seconds"] for r in report["records"] for v in r["views"].values())
            report["total_crops"] = sum(len(v["windows"]) for r in report["records"] for v in r["views"].values())
            report["status"] = "diagnostic_complete"
            logger.info("%s: %s", report["conclusion"]["decision"], report["conclusion"]["reasons"])
            logger.info("Report: %s (region diagnostics only; accuracy and training coverage unmeasured)", root / "clip_window_probe.json")
    except Exception as error:
        report.update(status="technical_failure", error=str(error))
        logger.exception("CLIP diagnostic failed; partial report retained")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        try:
            save()
        finally:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
    return 0 if report["conclusion"]["eligible_for_bounded_followup"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
