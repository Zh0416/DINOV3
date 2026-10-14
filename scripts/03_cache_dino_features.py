from __future__ import annotations

import argparse

from pvseg.config import load_config, output_dir, seed_everything
from pvseg.feature_cache import cache_dino_features
from pvseg.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    root = output_dir(config)
    logger = setup_logging(root, "03_cache_dino_features")
    cache_dino_features(config, root, logger)


if __name__ == "__main__":
    main()

