"""Teacher diagnostics. Ground truth is used for audit, never to build test attention."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pvseg.attention import sanitize_boxes
from pvseg.utils import atomic_write_json


def box_mask(record, size=512):
    mask = np.zeros((size, size), dtype=bool)
    for box in sanitize_boxes(record.get("bbox", []), 1024):
        x1, y1, x2, y2 = np.rint(np.asarray(box) * size / 1024).astype(int)
        mask[y1:y2, x1:x2] = True
    return mask


def audit_teacher(samples, records, output_root, logger):
    records_by_id = {r["sample_id"]: r for r in records}
    rows, groups, hashes = [], defaultdict(list), defaultdict(list)
    for sample in samples:
        record = records_by_id[sample.sample_id]
        truth_image = Image.open(sample.mask_path).convert("L")
        truth = np.asarray(truth_image.resize((512, 512), Image.Resampling.NEAREST)) > 0
        boxes = box_mask(record)
        overlap = int((truth & boxes).sum())
        exists = bool(truth.any())
        said_exists = bool(record.get("pv_exist"))
        confidence = float(record.get("confidence", 0))
        precision = overlap / max(int(boxes.sum()), 1)
        coverage = overlap / max(int(truth.sum()), 1)
        row = {
            "sample_id": sample.sample_id, "split": sample.split, "category": sample.category,
            "gt_exists": exists, "teacher_exists": said_exists, "confidence": confidence,
            "p_pv": confidence if said_exists else 1 - confidence,
            "correct_detection": said_exists == exists,
            "box_area_fraction": float(boxes.mean()), "foreground_fraction": float(truth.mean()),
            "box_precision": precision, "foreground_coverage": coverage,
            "box_union_iou": overlap / max(int((truth | boxes).sum()), 1),
            "useful_localization": bool(exists and precision >= .15 and coverage >= .10),
            "has_box": bool(boxes.any()), "parse_warning": record.get("parse_warning"),
            "patch_count": record.get("patch_count"),
        }
        rows.append(row)
        for key in ("all", f"split/{sample.split}", f"category/{sample.category}",
                    f"{sample.split}/{sample.category}"):
            groups[key].append(row)
        digest = hashlib.sha256(Path(sample.image_path).read_bytes()).hexdigest()
        hashes[digest].append({"sample_id": sample.sample_id, "split": sample.split})

    def summarize(items):
        positive = [r for r in items if r["gt_exists"]]
        negative = [r for r in items if not r["gt_exists"]]
        boxed = [r for r in items if r["has_box"]]
        bins, ece = [], 0.
        for lo, hi in zip(np.arange(0, 1, .1), np.arange(.1, 1.1, .1)):
            subset = [r for r in items if lo <= r["confidence"] < hi + (1e-8 if hi >= 1 else 0)]
            if subset:
                acc = float(np.mean([r["correct_detection"] for r in subset]))
                conf = float(np.mean([r["confidence"] for r in subset]))
                ece += len(subset) / len(items) * abs(conf - acc)
                bins.append({"lower": float(lo), "count": len(subset), "confidence": conf, "accuracy": acc})
        return {
            "images": len(items), "positives": len(positive), "negatives": len(negative),
            "detection_accuracy": float(np.mean([r["correct_detection"] for r in items])),
            "positive_recall": sum(r["teacher_exists"] for r in positive) / max(len(positive), 1),
            "negative_false_alarm_rate": sum(r["teacher_exists"] for r in negative) / max(len(negative), 1),
            "boxed_images": len(boxed),
            "zero_overlap_boxes": sum(r["box_precision"] == 0 for r in boxed),
            "mean_box_precision": float(np.mean([r["box_precision"] for r in boxed])) if boxed else 0.,
            "mean_positive_coverage": float(np.mean([r["foreground_coverage"] for r in positive])) if positive else 0.,
            "useful_box_fraction": sum(r["useful_localization"] for r in boxed) / max(len(boxed), 1),
            "brier_pv": float(np.mean([(r["p_pv"] - r["gt_exists"]) ** 2 for r in items])),
            "decision_confidence_ece": ece, "calibration_bins": bins,
            "foreground_fraction_quantiles": np.quantile([r["foreground_fraction"] for r in items], [0, .25, .5, .75, 1]).tolist(),
        }
    report = {
        "protocol": "Diagnostic audit only. Useful localization = box precision >= .15 and GT coverage >= .10; coarse boxes are not pixel masks.",
        "groups": {k: summarize(v) for k, v in groups.items()},
        "patch_counts": dict(Counter(str(r["patch_count"]) for r in rows)),
        "confidence_counts": dict(Counter(str(r["confidence"]) for r in rows)),
        "parse_warning_count": sum(bool(r["parse_warning"]) for r in rows),
        "cross_split_exact_duplicates": [v for v in hashes.values() if len({x["split"] for x in v}) > 1],
    }
    atomic_write_json(output_root / "reports/teacher_audit.json", report)
    atomic_write_json(output_root / "reports/teacher_audit_rows.json", rows)
    # Deterministic extremes for inspection; no test-driven tuning.
    selected = sorted([r for r in rows if r["has_box"] and r["split"] != "test"], key=lambda r: r["box_precision"])
    sample_lookup = {s.sample_id: s for s in samples}
    for label, subset in (("worst", selected[:4]), ("best", selected[-4:])):
        fig, axes = plt.subplots(len(subset), 2, figsize=(9, 4 * len(subset)), squeeze=False)
        for axes_row, row in zip(axes, subset):
            sample = sample_lookup[row["sample_id"]]
            rgb = np.asarray(Image.open(sample.image_path).convert("RGB").resize((512, 512)))
            truth = np.asarray(Image.open(sample.mask_path).convert("L").resize((512, 512), Image.Resampling.NEAREST)) > 0
            axes_row[0].imshow(rgb)
            axes_row[1].imshow(rgb)
            axes_row[1].imshow(np.ma.masked_where(~truth, truth), cmap="autumn", alpha=.6, vmin=0, vmax=1)
            for x1, y1, x2, y2 in sanitize_boxes(records_by_id[sample.sample_id].get("bbox"), 1024):
                axes_row[1].add_patch(plt.Rectangle((x1 / 2, y1 / 2), (x2-x1)/2, (y2-y1)/2, fill=False, color="cyan"))
            axes_row[0].set_title(f"{row['split']} / {sample.sample_id.split('__')[-1]}")
            axes_row[1].set_title(f"GT red, teacher cyan: precision={row['box_precision']:.3f}, coverage={row['foreground_coverage']:.3f}")
            for ax in axes_row:
                ax.axis("off")
        destination = output_root / f"visualizations/teacher_{label}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(destination, dpi=120)
        plt.close(fig)
    logger.info("Teacher audit: %s", report)
    return rows, report
