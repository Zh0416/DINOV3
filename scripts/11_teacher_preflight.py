"""Check two training RGBs with global teacher pairs before extracting DINO features."""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

from pvseg.config import load_config, seed_everything
from pvseg.label_free_data import load_rgb_manifest, manual_label_firewall
from pvseg.multiview_teacher import generate_grid_teacher
from pvseg.utils import atomic_write_json, setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="An existing image-only rgb_manifest.json from prepare or a previous smoke")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Preflight output must be new or empty; preserve previous diagnostics")
    config = copy.deepcopy(load_config(args.config))
    if config["protocol"].get("mode") != "soft_semantic":
        raise ValueError("Teacher preflight requires the soft_semantic protocol")
    config["teacher"].update({"local_views": False, "require_complete_grid": True, "cache_only_replay": False})
    config["runtime"].update({"output_dir": str(root), "resume": True})
    root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root, "11_teacher_preflight")
    started = time.perf_counter()
    report = {"status": "running", "manual_labels_used": False, "accuracy_evaluated": False,
              "purpose": "global grid completeness and original/hflip consistency only; not semantic correctness"}
    try:
        with manual_label_firewall():
            samples = [s for s in load_rgb_manifest(args.manifest) if s.split == "train"][:2]
            if len(samples) != 2:
                raise ValueError("Preflight requires at least two existing training RGBs")
            atomic_write_json(root / "rgb_manifest.json", {"samples": [s.to_dict() for s in samples]})
            atomic_write_json(root / "protocol_config.json", config)
            seed_everything(config["seed"])
            report["teacher"] = generate_grid_teacher(config, root, logger)
            report["status"] = "passed_consistency_check"
    except Exception as error:
        report.update({"status": "failed", "error": str(error)})
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_json(root / "preflight_report.json", report)


if __name__ == "__main__":
    main()
