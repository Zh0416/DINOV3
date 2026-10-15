from __future__ import annotations

import argparse

from pvseg.ablation import run_ablations
from pvseg.config import load_config, output_dir
from pvseg.utils import setup_logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = output_dir(config)
    run_ablations(config, root, setup_logging(root, "06_audit_and_ablate"))
