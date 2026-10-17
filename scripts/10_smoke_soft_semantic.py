"""Bounded cache replay or four-image cloud smoke; never run final evaluation."""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

from pvseg.config import load_config
from pvseg.label_free_data import manual_label_firewall
from pvseg.multiview_teacher import generate_grid_teacher
from pvseg.prediction_seal import verify_seal
from pvseg.soft_semantic import fuse_soft_attention
from pvseg.utils import atomic_write_json, read_json, setup_logging


def replay(config, source, root, limit, logger):
    payload = read_json(source)
    records = payload["records"][:limit]
    if len(records) != limit:
        raise ValueError("Insufficient cached images for requested smoke size")
    expected_views = {"original", "hflip"}
    if config["teacher"].get("local_views", True):
        expected_views.update(f"local_{i}" for i in range(4))
    for record in records:
        if not expected_views <= set(record["views"]) or any(
            record["views"][name].get("status") != "ok" or not record["views"][name].get("raw_response")
            for name in expected_views
        ):
            raise ValueError("Replay requires successful raw responses for every requested view; inference is disabled")
    before = {r["sample_id"]: float(fuse_soft_attention(r, config["reliability"])[0].max()) for r in records}
    atomic_write_json(root / "semantic_prompt.json", {**payload, "records": records})
    atomic_write_json(root / "rgb_manifest.json", {"samples": [
        {"sample_id": r["sample_id"], "rgb_sha256": r["sample_id"],
         "image_path": "not-opened-cache-replay", "split": "replay"} for r in records]})
    config = copy.deepcopy(config)
    config["runtime"]["resume"] = True
    # Historical truncations are intentional inputs in recovery-only replay.
    config["teacher"]["require_complete_grid"] = False
    teacher = generate_grid_teacher(config, root, logger)
    if teacher["new_inferences"]:
        raise RuntimeError("Cache-only replay unexpectedly performed inference")
    migrated = read_json(root / "semantic_prompt.json")["records"]
    details = []
    for record in migrated:
        attention, _ = fuse_soft_attention(record, config["reliability"])
        details.append({"sample_id": record["sample_id"], "attention_max_before": before[record["sample_id"]],
                        "attention_max_after": float(attention.max()),
                        "candidate_patches": int((attention >= config["reliability"]["positive_probability"]).sum()),
                        "grid_views": sum(bool(v.get("grid")) for v in record["views"].values()),
                        "grid_cells": sum(len(v.get("grid", [])) for v in record["views"].values())})
    return {"mode": "cached_response_replay", "teacher": teacher, "per_image": details}


def cloud_smoke(config, root, limit, timeout, report):
    # Keep the normal model sizes, six teacher views and compact grid prompt. Only
    # image count, training duration and the number of controls are reduced.
    config = copy.deepcopy(config)
    original_path = Path(config.pop("_config_path"))
    exclusions = config["data"].get("exclude_rgb_hash_file")
    if exclusions:
        config["data"]["exclude_rgb_hash_file"] = str((original_path.parent / exclusions).resolve())
    config["protocol"]["name"] += "_smoke"
    config["data"].update({"max_samples": limit, "train_fraction": .5,
                           "rgb_staging_dir": str(root / "rgb")})
    config["runtime"].update({"output_dir": str(root), "resume": True})
    config["training"].update({"epochs": 2, "num_workers": 0})
    config["teacher"]["require_complete_grid"] = True
    config["ablation"] = {"seeds": [42], "variants": ["baseline_decoder", "semantic_adapter"]}
    config_path = root / "smoke_config.json"
    atomic_write_json(config_path, config)
    script = Path(__file__).with_name("08_label_free_pipeline.py")
    deadline = time.monotonic() + timeout
    report.update({"mode": "cloud_pipeline", "stages_completed": []})
    for stage in ("prepare", "features", "teacher", "pseudo", "train", "seal"):
        subprocess.run([sys.executable, "-u", str(script), "--config", str(config_path), "--stage", stage],
                       check=True, timeout=max(1., deadline - time.monotonic()))
        report["stages_completed"].append(stage)
        atomic_write_json(root / "smoke_report.json", report)
    seal = verify_seal(root)
    teacher = read_json(root / "reports" / "multiview_teacher_report.json")
    if teacher["incomplete_grid_views"]:
        raise RuntimeError("Cloud smoke requires complete 4x4 grids for every view")
    training = read_json(root / "reports" / "label_free_training.json")
    if any(sum(epoch["optimizer_steps"] for epoch in run["history"]) < 2 for run in training["runs"]):
        raise RuntimeError("Smoke requires at least two successful optimizer steps per variant")
    responses = read_json(root / "semantic_prompt.json")["records"]
    report.update({"teacher": teacher, "seal_verified": True, "training_runs": seal["runs"],
                   "grid_views": sum(bool(v.get("grid")) for r in responses for v in r["views"].values()),
                   "grid_cells": sum(len(v.get("grid", [])) for r in responses for v in r["views"].values()),
                   "pseudo": read_json(root / "reports" / "pseudo_reliability.json")["per_image"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples", type=int, default=4, choices=range(4, 7))
    parser.add_argument("--replay-cache", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=2700, choices=range(1, 3601))
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Smoke output must be new or empty; preserve existing experiment artifacts")
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    logger = setup_logging(root, "10_smoke_soft_semantic")
    report = {"status": "running", "images": args.samples, "manual_labels_used": False,
              "accuracy_evaluated": False}
    started = time.perf_counter()
    try:
        with manual_label_firewall():
            if args.replay_cache:
                report.update(replay(config, args.replay_cache, root, args.samples, logger))
            else:
                cloud_smoke(config, root, args.samples, args.timeout_seconds, report)
        report["status"] = "passed"
    except Exception as error:
        report.update({"status": "failed", "error": str(error)})
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_json(root / "smoke_report.json", report)
        logger.info("Smoke report: %s", report)


if __name__ == "__main__":
    main()
