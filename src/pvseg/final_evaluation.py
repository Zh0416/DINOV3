"""Annotation access is confined to this final, sealed-prediction evaluation."""
from __future__ import annotations

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from pvseg.ablation_analysis import paired_bootstrap
from pvseg.data import discover_samples, find_dataset_root
from pvseg.label_free_data import load_rgb_manifest, rgb_hash
from pvseg.metrics import Confusion
from pvseg.prediction_seal import file_hash, verify_seal
from pvseg.utils import atomic_write_json, read_json


def _to_native(value, size):
    return np.asarray(Image.fromarray(value.astype(np.uint8)).resize(size, Image.Resampling.NEAREST))


def _sparse_quality(target, truth):
    valid, foreground = target != 255, target == 1
    true_positive = int((foreground & truth).sum())
    return {"valid_pixels": int(valid.sum()), "positive_pixels": int(foreground.sum()),
            "positive_correct": true_positive,
            "background_pixels": int((target == 0).sum()),
            "background_wrong": int(((target == 0) & truth).sum()),
            "gt_pixels": int(truth.sum())}


def _summarize_quality(rows):
    totals = {key: sum(row[key] for row in rows) for key in rows[0]} if rows else {}
    if not totals:
        return {}
    return {**totals, "pseudo_positive_precision": totals["positive_correct"]/max(totals["positive_pixels"], 1),
            "foreground_coverage": totals["positive_correct"]/max(totals["gt_pixels"], 1),
            "background_contamination": totals["background_wrong"]/max(totals["background_pixels"], 1)}


def _load_diagnostics(config, root, samples):
    """Validate the protocol-specific report before any annotation is accessed."""
    soft = config.get("protocol", {}).get("mode") == "soft_semantic"
    report = read_json(root / "reports" / "pseudo_reliability.json")
    fields = (("teacher_abstained", "teacher_views_used", "teacher_consistency", "positive_pixels")
              if soft else ("accepted", "reason", "view_iou"))
    rows = report.get("per_image", [])
    diagnostics = {}
    for row in rows:
        if not isinstance(row, dict) or "sample_id" not in row or any(k not in row for k in fields):
            raise ValueError(f"Invalid pseudo report for {'soft_semantic' if soft else 'bbox'} protocol: required {fields}")
        if row["sample_id"] in diagnostics:
            raise ValueError("Duplicate sample in pseudo reliability report")
        if soft:
            if (not isinstance(row["teacher_abstained"], bool)
                    or any(not isinstance(row[k], (int, float)) or not np.isfinite(row[k])
                           for k in fields[1:])):
                raise ValueError("Invalid soft-semantic diagnostic values")
        diagnostics[row["sample_id"]] = row
    if set(diagnostics) != {s.sample_id for s in samples}:
        raise ValueError("Pseudo reliability report does not match the frozen RGB manifest")
    return diagnostics


def evaluate_sealed(config, root, logger):
    seal = verify_seal(root)  # MUST happen before enumerating/opening annotations.
    samples = load_rgb_manifest(root / "rgb_manifest.json")
    diagnostics = _load_diagnostics(config, root, samples)
    soft = config.get("protocol", {}).get("mode") == "soft_semantic"
    data = config["data"]
    dataset_root = find_dataset_root(data["input_root"], data["dataset_slug"], data["categories"])
    annotated = discover_samples(dataset_root, data)
    requested = {s.rgb_sha256 for s in samples}
    mapping = {}
    for original in annotated:
        digest = rgb_hash(original.image_path)
        if digest in requested:
            if digest in mapping:
                # Refuse an ambiguous GT association, rather than silently choosing.
                raise RuntimeError("Duplicate selected RGB has multiple annotation files")
            mapping[digest] = original
    if set(mapping) != requested:
        raise RuntimeError("Cannot associate every frozen RGB hash with a final evaluation annotation")
    logger.info("Seal verified (%d runs). Opening manual masks for FINAL evaluation only.", len(seal["runs"]))
    targets, sizes = {}, {}
    quality = defaultdict(list)
    teacher_rows = []
    for sample in samples:
        original = mapping[sample.sample_id]
        with Image.open(original.mask_path) as image:
            truth = np.asarray(image.convert("L")) > 0
            sizes[sample.sample_id] = image.size
        targets[sample.sample_id] = truth
        with np.load(root / "pseudo_cache" / f"{sample.sample_id}.npz") as cache:
            kinds = ("soft",) if soft else ("heuristic", "raw", "filtered")
            for kind in kinds:
                target = _to_native(cache[f"target_{kind}"], sizes[sample.sample_id])
                quality[(sample.split, kind)].append(_sparse_quality(target, truth))
            if soft:
                raw = _to_native(cache["teacher_candidate"], sizes[sample.sample_id]).astype(bool)
                consensus = _to_native(cache["teacher_core"], sizes[sample.sample_id]).astype(bool)
                pseudo = _to_native(cache["target_soft"], sizes[sample.sample_id]) == 1
            else:
                raw = _to_native(cache["teacher_raw"], sizes[sample.sample_id]).astype(bool)
                consensus = _to_native(cache["teacher_consensus"], sizes[sample.sample_id]).astype(bool)
                pseudo = _to_native(cache["target_filtered"], sizes[sample.sample_id]) == 1
        diagnostic = diagnostics[sample.sample_id]
        row = {"sample_id": sample.sample_id, "split": sample.split, "category": original.category,
               "has_pv_truth": bool(truth.any())}
        if soft:
            row.update({k: diagnostic[k] for k in
                        ("teacher_abstained", "teacher_views_used", "teacher_consistency", "positive_pixels")})
            row["has_positive_pseudo"] = bool(pseudo.any())
        else:
            row.update({"accepted_proxy": diagnostic["accepted"], "proxy_reason": diagnostic["reason"],
                        "view_iou": diagnostic["view_iou"]})
        row_regions = (("teacher_candidate", raw), ("teacher_core", consensus), ("accepted_seed", pseudo))
        for name, region in row_regions:
            meter = Confusion()
            meter.update(torch.from_numpy(region), torch.from_numpy(truth))
            row[name] = meter.compute()
        teacher_rows.append(row)
    teacher_report = {"evaluated_after_prediction_seal": True,
                      "interpretation": "Agreement/reasons/confidence never establish truth. These are post-hoc GT audits, not thresholds to refit on this test.",
                      "pseudo_quality": {f"{split}/{kind}": _summarize_quality(rows)
                                         for (split, kind), rows in quality.items()},
                      "per_image": teacher_rows}
    if soft:
        teacher_report.update({
            "teacher_abstained_images": sum(r["teacher_abstained"] for r in teacher_rows),
            "positive_pseudo_images": sum(r["has_positive_pseudo"] for r in teacher_rows),
            "positive_pseudo_without_any_gt_pv": sum(r["has_positive_pseudo"] and not r["has_pv_truth"] for r in teacher_rows),
            "no_positive_pseudo_with_gt_pv": sum(not r["has_positive_pseudo"] and r["has_pv_truth"] for r in teacher_rows)})
    else:
        teacher_report.update({
            "accepted_images": sum(r["accepted_proxy"] for r in teacher_rows),
            "accepted_without_any_gt_pv": sum(r["accepted_proxy"] and not r["has_pv_truth"] for r in teacher_rows),
            "rejected_images_with_gt_pv": sum(not r["accepted_proxy"] and r["has_pv_truth"] for r in teacher_rows)})
    atomic_write_json(root / "reports" / "final_teacher_audit.json", teacher_report)
    runs = []
    for name in seal["runs"]:
        variant, seed_text = name.rsplit("_seed", 1)
        overall, categories, per_image = Confusion(), {}, []
        saved = 0
        for sample in samples:
            if sample.split != "test":
                continue
            original, truth = mapping[sample.sample_id], targets[sample.sample_id]
            with Image.open(root / "predictions" / name / f"{sample.sample_id}.png") as image:
                prediction = np.asarray(image.resize(sizes[sample.sample_id], Image.Resampling.NEAREST)) > 0
            pred_t, truth_t = torch.from_numpy(prediction), torch.from_numpy(truth)
            overall.update(pred_t, truth_t)
            categories.setdefault(original.category, Confusion()).update(pred_t, truth_t)
            individual = Confusion()
            individual.update(pred_t, truth_t)
            per_image.append({"sample_id": sample.sample_id, "category": original.category, **individual.compute()})
            # Fixed hash order, never cherry-pick the best examples.
            if seed_text == str(config["ablation"]["seeds"][0]) and saved < config["training"]["visualization_samples"]:
                with Image.open(original.image_path) as image:
                    rgb = np.asarray(image.convert("RGB"))
                _visualization(rgb, truth, prediction, root / "visualizations" / name / f"{sample.sample_id}.png")
                saved += 1
        test = {"overall": overall.compute(), "by_category": {k: v.compute() for k, v in categories.items()},
                "per_image": per_image}
        runs.append({"variant": variant, "seed": int(seed_text), "test": test})
        logger.info("FINAL %s mIoU=%.5f F1=%.5f Precision=%.5f Recall=%.5f", name,
                    test["overall"]["miou"], test["overall"]["f1"], test["overall"]["precision"], test["overall"]["recall"])
    summary = {}
    keys = ("miou", "pv_iou", "f1", "precision", "recall", "fp", "overlap_miss_rate", "image_false_alarm_rate")
    for variant in sorted({r["variant"] for r in runs}):
        grouped = [r for r in runs if r["variant"] == variant]
        summary[variant] = {}
        for group in ("all", "困难正样本", "负样本"):
            metrics = [r["test"]["overall"] if group == "all" else r["test"]["by_category"].get(group)
                       for r in grouped]
            metrics = [m for m in metrics if m is not None]
            summary[variant][group] = {
                key: {"mean": float(np.mean([m[key] for m in metrics])),
                      "std": float(np.std([m[key] for m in metrics], ddof=1)) if len(metrics) > 1 else None}
                for key in keys} if metrics else {"images": 0, "interpretation": "No such images in hash-selected holdout"}
    references = (("baseline_decoder", "semantic_adapter", "semantic_capacity", "semantic_no_adapter", "shuffled_semantic")
                  if config.get("protocol", {}).get("mode") == "soft_semantic"
                  else ("heuristic_decoder", "filtered_decoder", "filtered_capacity", "filtered_shuffled"))
    bootstrap = {reference: paired_bootstrap(runs, reference=reference)
                 for reference in references if any(r["variant"] == reference for r in runs)}
    for result in bootstrap.values():
        result["interpretation"] = ("Paired category-stratified image bootstrap on a fresh hash-selected pilot. "
                                    "Seeds share image draws. Does not account for spatial dependence or training-set uncertainty.")
    final = {"seal_sha256": file_hash(root / "prediction_seal.json"), "protocol": config["protocol"],
             "evaluation_resolution": "native annotation resolution; nearest resize frozen binary predictions",
             "selected_images": len(samples), "train_images": sum(s.split == "train" for s in samples),
             "test_images": sum(s.split == "test" for s in samples),
             "test_category_counts": {k: v.images for k, v in categories.items()},
             "manual_labels_used_for_training_or_selection": False, "summary": summary, "runs": runs,
             "limitations": ["RGB prior is imperfect weak supervision; this is not supervised segmentation.",
                             "View consistency can preserve correlated teacher errors.",
                             "Small image holdout; spatial scene independence has not been established.",
                             "Do not retune on these final labels and claim an untouched test."]}
    atomic_write_json(root / "reports" / "label_free_final.json", final)
    atomic_write_json(root / "reports" / "label_free_bootstrap.json", bootstrap)
    _comparison_plot(summary, root / "visualizations" / "label_free_comparison.png")
    return final


def _visualization(rgb, truth, prediction, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay = rgb.copy()
    overlay[truth] = (.45*overlay[truth] + .55*np.array([255, 0, 0])).astype(np.uint8)
    overlay[prediction] = (.45*overlay[prediction] + .55*np.array([0, 255, 0])).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for axis, value, title in zip(axes, (rgb, truth, prediction, overlay),
                                  ("RGB", "Final GT", "Frozen prediction", "GT red / prediction green")):
        axis.imshow(value, cmap="gray" if value.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def _comparison_plot(summary, destination):
    names = list(summary)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for axis, key in zip(axes, ("miou", "f1", "precision", "recall")):
        axis.bar(np.arange(len(names)), [summary[n]["all"][key]["mean"] for n in names],
                 yerr=[summary[n]["all"][key]["std"] or 0 for n in names], capsize=3)
        axis.set_xticks(np.arange(len(names)), names, rotation=70, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(key)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
