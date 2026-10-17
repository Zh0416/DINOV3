"""Compare InternVL whole-image presence and 4x4 grids on fixed training RGBs."""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch

from pvseg.config import configure_cloud_cache, load_config, seed_everything
from pvseg.label_free_data import load_rgb_manifest, manual_label_firewall
from pvseg.multiview_teacher import grid_generation_spec
from pvseg.probe_coverage import (probe_pseudo_coverage, validate_feature_source,
                                  validate_probe_samples)
from pvseg.teacher import InternVLTeacher
from pvseg.teacher.internvl import PRESENCE_PARSER_VERSION
from pvseg.teacher_probe import (DEFAULT_PRESENCE_PROMPT, classify_probe,
                                 diagnose_grid_pair, diagnose_presence_pair)
from pvseg.utils import atomic_write_json, setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Existing image-only rgb_manifest.json")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=2, choices=(1, 2))
    parser.add_argument("--feature-root", type=Path,
                        help="Optional existing run containing DINO features; read-only coverage dry run")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Protocol-probe output must be new or empty")
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher protocol probe requires a CUDA GPU")
    config = copy.deepcopy(load_config(args.config))
    if config.get("protocol", {}).get("mode") != "soft_semantic":
        raise ValueError("Teacher protocol probe requires soft_semantic mode")
    config["teacher"].setdefault("presence_prompt", DEFAULT_PRESENCE_PROMPT)
    config["teacher"].setdefault("presence_max_new_tokens", 64)
    root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root, "12_teacher_protocol_probe")
    report_path = root / "teacher_protocol_probe.json"
    started = time.perf_counter()
    report = {
        "status": "running", "manual_labels_used": False, "accuracy_evaluated": False,
        "purpose": "separate whole-image recognition stability from 4x4 grid elicitation; not accuracy",
        "records": [],
    }

    def save():
        atomic_write_json(report_path, report)

    try:
        with manual_label_firewall():
            samples = [sample for sample in load_rgb_manifest(args.manifest)
                       if sample.split == "train"][:args.samples]
            if len(samples) != args.samples:
                raise ValueError(f"Protocol probe requires {args.samples} existing training RGBs")
            validate_probe_samples(samples)
            if args.feature_root:
                report["feature_source"] = validate_feature_source(samples, args.feature_root, config)
            atomic_write_json(root / "rgb_manifest.json", {
                "selection": "first fixed train records from the supplied image-only manifest",
                "manual_labels_used": False, "samples": [sample.to_dict() for sample in samples],
            })
            atomic_write_json(root / "protocol_config.json", config)
            seed_everything(config["seed"])
            configure_cloud_cache(config)
            load_started = time.perf_counter()
            teacher = InternVLTeacher.load(config["teacher"], logger)
            report["model_load_seconds"] = time.perf_counter() - load_started
            report["runtime_spec"] = teacher.runtime_spec
            save()
            prompts = {"presence": config["teacher"]["presence_prompt"],
                       "grid": config["teacher"]["prompt"]}
            specs = {"grid": grid_generation_spec(config)}
            specs["presence"] = {
                **specs["grid"], "prompt": prompts["presence"], "schema": "presence_v1",
                "max_new_tokens": int(config["teacher"]["presence_max_new_tokens"]),
                "presence_parser_version": PRESENCE_PARSER_VERSION,
            }
            for sample in samples:
                record = {"sample_id": sample.sample_id, "rgb_sha256": sample.rgb_sha256,
                          "protocols": {"presence": {}, "grid": {}}}
                report["records"].append(record)
                for protocol in ("presence", "grid"):
                    for view in ("original", "hflip"):
                        inference_started = time.perf_counter()
                        try:
                            if protocol == "presence":
                                result = teacher.infer_presence(sample.image_path, view=view,
                                                                prompt=prompts[protocol])
                            else:
                                result = teacher.infer(sample.image_path, view=view,
                                                       prompt=prompts[protocol])
                            result["status"] = "ok"
                        except Exception as error:
                            logger.exception("Probe failed: sample=%s protocol=%s view=%s",
                                             sample.sample_id, protocol, view)
                            result = {"status": "error", "error": str(error),
                                      "flipped": view == "hflip"}
                        result.update({"elapsed_seconds": time.perf_counter() - inference_started,
                                       "generation_prompt": prompts[protocol],
                                       "generation_spec": specs[protocol]})
                        record["protocols"][protocol][view] = result
                        save()
            positive = float(config["reliability"].get("positive_probability", .55))
            negative = float(config["reliability"].get("negative_probability", .18))
            per_image = []
            for record in report["records"]:
                per_image.append({
                    "sample_id": record["sample_id"],
                    "presence": diagnose_presence_pair(
                        record["protocols"]["presence"], positive, negative),
                    "grid": diagnose_grid_pair(record["protocols"]["grid"], positive, negative),
                })
            max_conflict = float(config["teacher"].get("max_global_pair_conflict", .25))
            report["diagnostics"] = {
                "max_grid_conflict_fraction": max_conflict,
                "per_image": per_image,
                "total_inferences": len(samples) * 4,
                "presence_seconds": sum(
                    view.get("elapsed_seconds", 0.0) for row in report["records"]
                    for view in row["protocols"]["presence"].values()),
                "grid_seconds": sum(
                    view.get("elapsed_seconds", 0.0) for row in report["records"]
                    for view in row["protocols"]["grid"].values()),
            }
            report["conclusion"] = classify_probe(per_image, max_conflict)
            save()
            if args.feature_root:
                report["pseudo_coverage_dry_run"] = probe_pseudo_coverage(
                    samples, args.feature_root, report["records"], config)
            coverage = report.get("pseudo_coverage_dry_run", {})
            coverage_ok = not args.feature_root or (
                coverage.get("status") == "measured_diagnostic_only"
                and all(coverage.get("train_class_pixels", {}).values()))
            report["eligible_for_bounded_followup"] = (
                report["conclusion"]["diagnosis"] == "small_probe_passed" and coverage_ok)
            report["status"] = "diagnostic_complete"
    except Exception as error:
        report.update({"status": "technical_failure", "error": str(error)})
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        try:
            save()
        finally:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
    return 0 if report["eligible_for_bounded_followup"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
