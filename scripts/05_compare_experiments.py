from __future__ import annotations

import argparse

from pvseg.comparison import compare_experiments
from pvseg.config import load_config, output_dir
from pvseg.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = output_dir(config)
    logger = setup_logging(root, "05_compare_experiments")
    compare_experiments(root, logger)


if __name__ == "__main__":
    main()

