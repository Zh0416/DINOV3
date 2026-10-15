"""Paired image bootstrap for the small, already inspected pilot test."""
from __future__ import annotations

import numpy as np


def _metrics(confusions):
    tp, fp, fn, tn = np.moveaxis(confusions, -1, 0)
    pv_iou = tp / np.maximum(tp+fp+fn, 1)
    bg_iou = tn / np.maximum(tn+fp+fn, 1)
    return {"miou": (pv_iou+bg_iou)/2, "f1": 2*tp/np.maximum(2*tp+fp+fn, 1),
            "precision": tp/np.maximum(tp+fp, 1), "recall": tp/np.maximum(tp+fn, 1)}


def paired_bootstrap(runs, reference="baseline", repeats=2000, seed=20261015):
    """Resample images within category; seeds share draws, not independent images."""
    seeds = sorted({r["seed"] for r in runs})
    variants = sorted({r["variant"] for r in runs})
    lookup = {(r["variant"], r["seed"]): r for r in runs}
    template = lookup[(reference, seeds[0])]["test"]["per_image"]
    ids = [r["sample_id"] for r in template]
    categories = np.array([r["category"] for r in template])
    rng = np.random.default_rng(seed)
    result = {"repeats": repeats,
              "interpretation": "Paired, category-stratified image bootstrap on the historical pilot test. Seeds are averaged within each draw. Does not account for spatial dependence, training-set uncertainty or test reuse.",
              "reference": reference, "contrasts": {}}
    for group in ("all", "困难正样本"):
        selected = np.arange(len(ids)) if group == "all" else np.flatnonzero(categories == group)
        if not len(selected):
            result["contrasts"][group] = {"images": 0, "interpretation": "No images for this subgroup"}
            continue
        parts = [selected[categories[selected] == cat] for cat in sorted(set(categories[selected]))]
        draws = np.concatenate([rng.choice(part, size=(repeats, len(part)), replace=True) for part in parts], axis=1)
        by_variant = {}
        for variant in variants:
            seed_metrics = []
            for item_seed in seeds:
                rows = {r["sample_id"]: r for r in lookup[(variant, item_seed)]["test"]["per_image"]}
                if set(rows) != set(ids):
                    raise ValueError("All paired runs must evaluate identical test images")
                confusion = np.array([[rows[i][key] for key in ("tp", "fp", "fn", "tn")] for i in ids])
                seed_metrics.append(_metrics(confusion[draws].sum(1)))
            by_variant[variant] = {key: np.mean([m[key] for m in seed_metrics], axis=0) for key in seed_metrics[0]}
        result["contrasts"][group] = {
            variant: {metric: {"mean_delta": float(np.mean(values-by_variant[reference][metric])),
                                "ci95": np.quantile(values-by_variant[reference][metric], [.025, .975]).tolist()}
                      for metric, values in metrics.items()}
            for variant, metrics in by_variant.items() if variant != reference}
    return result
