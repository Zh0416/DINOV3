from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from pvseg.utils import atomic_write_json, read_json


METRICS = ("miou", "pv_iou", "f1", "precision", "recall")


def compare_experiments(output_root: Path, logger) -> dict[str, Any]:
    baseline = read_json(output_root / "reports" / "baseline_results.json")
    adapter = read_json(output_root / "reports" / "internvl_adapter_results.json")
    baseline_overall = baseline["test"]["overall"]
    adapter_overall = adapter["test"]["overall"]
    hard_name = "困难正样本"
    baseline_hard = baseline["test"]["by_category"].get(hard_name, {})
    adapter_hard = adapter["test"]["by_category"].get(hard_name, {})
    comparison = {
        "experiment_a": {metric: baseline_overall[metric] for metric in METRICS},
        "experiment_b": {metric: adapter_overall[metric] for metric in METRICS},
        "delta_b_minus_a": {
            metric: adapter_overall[metric] - baseline_overall[metric] for metric in METRICS
        },
        "hard_positive_analysis": {
            "baseline": baseline_hard,
            "internvl_adapter": adapter_hard,
            "recall_delta": adapter_hard.get("recall", 0.0) - baseline_hard.get("recall", 0.0),
            "image_miss_rate_delta": adapter_hard.get("image_miss_rate", 0.0)
            - baseline_hard.get("image_miss_rate", 0.0),
            "pixel_false_negative_rate_delta": adapter_hard.get("pixel_false_negative_rate", 0.0)
            - baseline_hard.get("pixel_false_negative_rate", 0.0),
        },
    }
    hard = comparison["hard_positive_analysis"]
    comparison["conclusion"] = (
        "InternVL guidance reduced hard-positive misses in this pilot."
        if hard["image_miss_rate_delta"] < 0
        else "This pilot did not demonstrate a reduction in hard-positive image misses."
    )
    reports = output_root / "reports"
    atomic_write_json(reports / "comparison.json", comparison)
    with (reports / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "baseline", "internvl_adapter", "delta"])
        for metric in METRICS:
            writer.writerow(
                [metric, baseline_overall[metric], adapter_overall[metric], comparison["delta_b_minus_a"][metric]]
            )
    _save_metric_plot(comparison, output_root / "visualizations" / "experiment_comparison.png")
    logger.info("Final comparison: %s", comparison)
    return comparison


def _save_metric_plot(comparison, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    baseline = [comparison["experiment_a"][metric] for metric in METRICS]
    adapter = [comparison["experiment_b"][metric] for metric in METRICS]
    positions = list(range(len(METRICS)))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar([position - 0.2 for position in positions], baseline, width=0.4, label="Baseline")
    axis.bar([position + 0.2 for position in positions], adapter, width=0.4, label="InternVL adapter")
    axis.set_xticks(positions, METRICS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)

