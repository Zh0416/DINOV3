"""Each pre-evaluation stage runs with an annotation-read guard."""
from __future__ import annotations

import argparse

from pvseg.config import load_config, output_dir, seed_everything
from pvseg.label_free_data import manual_label_firewall, prepare_rgb_manifest
from pvseg.utils import read_json, setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["prepare", "features", "teacher", "pseudo", "train", "seal"], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("protocol", {}).get("label_usage") != "final_evaluation_only":
        raise ValueError("Expected final_evaluation_only protocol")
    root = output_dir(config)
    if (root / "prediction_seal.json").exists():
        raise RuntimeError("This output is sealed; do not mutate it")
    frozen_config = root / "protocol_config.json"
    if args.stage != "prepare" and read_json(frozen_config) != config:
        raise RuntimeError("Config changed after pilot selection; use a new protocol/output directory")
    logger = setup_logging(root, f"08_label_free_{args.stage}")
    seed_everything(config["seed"])
    with manual_label_firewall():
        logger.info("Manual-label read firewall enabled for stage=%s", args.stage)
        if args.stage == "prepare":
            prepare_rgb_manifest(config, root, logger)
        elif args.stage == "features":
            from pvseg.feature_cache import cache_dino_features
            cache_dino_features(config, root, logger)
        elif args.stage == "teacher":
            from pvseg.multiview_teacher import generate_multiview_teacher
            generate_multiview_teacher(config, root, logger)
        elif args.stage == "pseudo":
            from pvseg.pseudo_labels import generate_pseudo_cache
            generate_pseudo_cache(config, root, logger)
        elif args.stage == "train":
            from pvseg.label_free_training import train_label_free
            train_label_free(config, root, logger)
        elif args.stage == "seal":
            from pvseg.prediction_seal import seal_predictions
            seal_predictions(config, root)
    logger.info("Stage %s complete", args.stage)


if __name__ == "__main__":
    main()
