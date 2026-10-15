"""Run ONLY after stage=seal; this process is the first to open manual masks."""
import argparse

from pvseg.final_evaluation import evaluate_sealed
from pvseg.utils import read_json, setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    from pathlib import Path
    root = Path(args.output_root)
    # Evaluate exactly the frozen protocol, not an edited YAML file.
    config = read_json(root / "protocol_config.json")
    if Path(config["runtime"]["output_dir"]).resolve() != root.resolve():
        raise ValueError("Output root disagrees with frozen config")
    evaluate_sealed(config, root, setup_logging(root, "09_final_evaluation"))


if __name__ == "__main__":
    main()
