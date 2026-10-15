"""Paired ablations separating capacity, fusion, teacher quality, and location."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pvseg.attention import gaussian_attention_map
from pvseg.config import seed_everything
from pvseg.data import load_split_manifest
from pvseg.losses import DiceCrossEntropyLoss
from pvseg.models import CachedFeatureSegmenter
from pvseg.reliability import TeacherReliability, filter_attention
from pvseg.teacher_audit import audit_teacher
from pvseg.training import _build_loaders, _fit_experiment
from pvseg.utils import atomic_write_json, read_json


@torch.inference_mode()
def _baseline_probabilities(model, loaders, device, feature_size):
    model.eval()
    result = {}
    for loader in loaders.values():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch["feature"].to(device), batch["attention"].to(device),
                               tuple(batch["mask"].shape[-2:]))
            probabilities = torch.nn.functional.interpolate(
                logits.softmax(1)[:, 1:2].float(), size=(feature_size, feature_size), mode="area")[:, 0]
            for sample_id, probability in zip(batch["sample_id"], probabilities.cpu().numpy()):
                result[sample_id] = probability
    return result


def _write_maps(destination, maps):
    destination.mkdir(parents=True, exist_ok=True)
    for sample_id, attention in maps.items():
        np.save(destination / f"{sample_id}.npy", attention.astype(np.float32))


def run_ablations(config, output_root, logger):
    if not torch.cuda.is_available():
        raise RuntimeError("Run ablations on the cloud GPU; base weights stay in Kaggle")
    device = torch.device("cuda")
    samples = load_split_manifest(output_root / "split_manifest.json")
    records = read_json(output_root / "semantic_prompt.json")["records"]
    by_id = {r["sample_id"]: r for r in records}
    if set(by_id) != {s.sample_id for s in samples}:
        raise ValueError("Teacher cache and manifest IDs must match exactly")
    rows, audit = audit_teacher(samples, records, output_root, logger)
    calibrator = TeacherReliability().fit(records, rows)
    atomic_write_json(output_root / "reports/teacher_reliability.json", calibrator.to_dict())
    fsize = int(config["attention"]["feature_size"])
    raw = {s.sample_id: gaussian_attention_map(
        by_id[s.sample_id].get("bbox"), int(config["data"]["image_size"]), fsize,
        float(config["attention"]["gaussian_sigma"]), float(by_id[s.sample_id].get("confidence", 0))
    )[1] for s in samples}
    raw_dir = output_root / "attention_ablation/raw"
    _write_maps(raw_dir, raw)
    ones = {s.sample_id: np.ones((fsize, fsize), np.float32) for s in samples}
    ones_dir = output_root / "attention_ablation/ones"
    _write_maps(ones_dir, ones)
    loss_fn = DiceCrossEntropyLoss(float(config["training"]["dice_weight"]),
                                  float(config["training"]["cross_entropy_weight"]))
    runs, filtering_reports = [], []
    variants = ("baseline", "legacy_raw", "capacity_only", "gated_raw", "gated_filtered", "gated_shuffled", "filtered_shuffled")
    for seed in config["ablation"]["seeds"]:
        paired_config = copy.deepcopy(config)
        paired_config["seed"] = int(seed)
        seed_everything(seed)
        baseline = CachedFeatureSegmenter(1024, int(config["training"]["decoder_channels"]),
                                         int(config["training"]["adapter_bottleneck"]), False)
        decoder_initialization = copy.deepcopy(baseline.decoder.state_dict())
        loaders = _build_loaders(paired_config, output_root, raw_dir)
        # Model construction can consume RNG; reset the training RNG for every variant.
        seed_everything(seed)
        baseline_result = _fit_experiment(f"baseline_s{seed}", baseline, loaders, loss_fn,
                                          paired_config, output_root, device, logger)
        baseline_result.update(variant="baseline", seed=seed)
        runs.append(baseline_result)
        probability = _baseline_probabilities(baseline, loaders, device, fsize)
        del baseline
        filtered = {}
        rejected = []
        for sample in samples:
            record = by_id[sample.sample_id]
            filtered[sample.sample_id] = filter_attention(
                raw[sample.sample_id], probability[sample.sample_id], calibrator.predict(record),
                float(record.get("confidence", 0)), float(config["teacher"]["confidence_threshold"]))
            if not filtered[sample.sample_id].any():
                rejected.append(sample.sample_id)
        filtered_dir = output_root / f"attention_ablation/filtered_s{seed}"
        _write_maps(filtered_dir, filtered)
        filtering_reports.append({"seed": seed, "rejected_ids": rejected, "nonempty": len(samples)-len(rejected)})
        # Cyclic permutation guarantees a different image's map, within the same split.
        # No category or label is used in this negative control.
        shuffled, shuffled_filtered, assignments = {}, {}, {}
        rng = np.random.default_rng(seed + 10000)
        for split in ("train", "val", "test"):
            ids = [s.sample_id for s in samples if s.split == split]
            rng.shuffle(ids)
            for sample_id, other_id in zip(ids, ids[1:] + ids[:1]):
                shuffled[sample_id] = raw[other_id]
                shuffled_filtered[sample_id] = filtered[other_id]
                assignments[sample_id] = other_id
        shuffled_dir = output_root / f"attention_ablation/shuffled_s{seed}"
        shuffled_filtered_dir = output_root / f"attention_ablation/filtered_shuffled_s{seed}"
        _write_maps(shuffled_dir, shuffled)
        _write_maps(shuffled_filtered_dir, shuffled_filtered)
        atomic_write_json(output_root / f"reports/shuffle_s{seed}.json", assignments)
        destinations = {"legacy_raw": raw_dir, "capacity_only": ones_dir, "gated_raw": raw_dir,
                        "gated_filtered": filtered_dir, "gated_shuffled": shuffled_dir,
                        "filtered_shuffled": shuffled_filtered_dir}
        for variant in variants[1:]:
            seed_everything(seed)
            model = CachedFeatureSegmenter(1024, int(config["training"]["decoder_channels"]),
                                           int(config["training"]["adapter_bottleneck"]), True,
                                           "legacy" if variant == "legacy_raw" else "gated")
            model.decoder.load_state_dict(decoder_initialization)
            loaders = _build_loaders(paired_config, output_root, destinations[variant])
            seed_everything(seed)
            result = _fit_experiment(f"{variant}_s{seed}", model, loaders, loss_fn,
                                     paired_config, output_root, device, logger)
            result.update(variant=variant, seed=seed)
            runs.append(result)
            atomic_write_json(output_root / "reports/ablation_runs.json", runs)
            del model
            torch.cuda.empty_cache()
    summary = {}
    for variant in variants:
        subset = [r for r in runs if r["variant"] == variant]
        summary[variant] = {}
        for split in ("validation", "test"):
            groups = {"overall": [r[split]["overall"] for r in subset]}
            for category in config["data"]["categories"]:
                groups[category] = [r[split]["by_category"][category] for r in subset]
            summary[variant][split] = {
                group: {metric: {"mean": float(np.mean([r[metric] for r in items])),
                                  "std": float(np.std([r[metric] for r in items], ddof=1))}
                        for metric in ("miou", "pv_iou", "f1", "precision", "recall", "fp", "overlap_miss_rate", "image_false_alarm_rate")}
                for group, items in groups.items()}
    # Selection uses validation only; historical test is explicitly exploratory.
    selected = max(variants, key=lambda v: summary[v]["validation"]["overall"]["miou"]["mean"])
    report = {"protocol": {"seeds": config["ablation"]["seeds"], "epochs": config["training"]["epochs"],
                           "split_counts": {split: sum(s.split == split for s in samples) for split in ("train", "val", "test")},
                           "paired_decoder_init_and_epoch_orders": True,
                           "teacher_calibration_split": "train", "selection_split": "val",
                           "test_status": "Previously inspected pilot test, exploratory, not a new independent confirmation",
                           "no_pseudo_mask_supervision": True},
              "selected_by_validation": selected, "summary": summary, "filtering": filtering_reports}
    atomic_write_json(output_root / "reports/ablation_summary.json", report)
    logger.info("Ablations completed. Validation-selected variant=%s", selected)
    return report
