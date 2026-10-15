from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pvseg.ablation_analysis import paired_bootstrap
from pvseg.utils import atomic_write_json, read_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    runs = read_json(args.reports / "ablation_runs.json")
    summary = read_json(args.reports / "ablation_summary.json")
    args.destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.destination / "paired_bootstrap.json", paired_bootstrap(runs))
    atomic_write_json(args.destination / "summary.json", summary)
    atomic_write_json(args.destination / "teacher_audit.json", read_json(args.reports / "teacher_audit.json"))
    variants = list(summary["summary"])
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, (group, metric, title) in zip(axes, [("overall", "miou", "Overall mIoU"), ("困难正样本", "precision", "Hard-positive precision")]):
        values = [summary["summary"][v]["test"][group][metric] for v in variants]
        axis.bar(np.arange(len(variants)), [v["mean"] for v in values],
                 yerr=[v["std"] for v in values], capsize=3)
        axis.set_xticks(np.arange(len(variants)), variants, rotation=40, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title + " (mean +/- seed std)")
    figure.tight_layout()
    figure.savefig(args.destination / "ablation_comparison.png", dpi=140)
    plt.close(figure)
