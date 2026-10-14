from __future__ import annotations

import argparse

from pvseg.config import load_config, output_dir, seed_everything
from pvseg.teacher_generation import generate_teacher_records
from pvseg.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    root = output_dir(config)
    logger = setup_logging(root, "02_generate_teacher")
    generate_teacher_records(config, root, logger)


if __name__ == "__main__":
    main()

