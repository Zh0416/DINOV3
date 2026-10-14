from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pvseg.attention import save_attention_cache
from pvseg.config import seed_everything
from pvseg.data import CachedFeatureDataset, load_split_manifest
from pvseg.losses import DiceCrossEntropyLoss
from pvseg.metrics import SegmentationMetrics
from pvseg.models import CachedFeatureSegmenter
from pvseg.utils import atomic_write_json, read_json


def train_experiments(config: dict[str, Any], output_root: Path, logger) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Training is configured for a Kaggle GPU")
    samples = load_split_manifest(output_root / "split_manifest.json")
    teacher_payload = read_json(output_root / "semantic_prompt.json")
    teacher_records = teacher_payload["records"]
    expected_ids = {sample.sample_id for sample in samples}
    available_ids = {record["sample_id"] for record in teacher_records}
    missing = expected_ids - available_ids
    if missing:
        raise RuntimeError(f"Teacher JSON is missing {len(missing)} selected samples")
    attention_dir = output_root / "attention_cache"
    attention_report = save_attention_cache(
        teacher_records,
        attention_dir,
        image_size=int(config["data"]["image_size"]),
        feature_size=int(config["attention"]["feature_size"]),
        sigma=float(config["attention"]["gaussian_sigma"]),
    )
    atomic_write_json(output_root / "reports" / "attention_report.json", attention_report)
    loaders = _build_loaders(config, output_root, attention_dir)
    training = config["training"]
    loss_fn = DiceCrossEntropyLoss(
        float(training["dice_weight"]), float(training["cross_entropy_weight"])
    )

    seed_everything(int(config["seed"]))
    baseline = CachedFeatureSegmenter(
        feature_channels=int(config["dino"]["expected_hidden_dim"]),
        decoder_channels=int(training["decoder_channels"]),
        adapter_bottleneck=int(training["adapter_bottleneck"]),
        use_adapter=False,
    )
    common_decoder_initialization = copy.deepcopy(baseline.decoder.state_dict())
    baseline_result = _fit_experiment(
        "baseline", baseline, loaders, loss_fn, config, output_root, device, logger
    )

    seed_everything(int(config["seed"]))
    adapter = CachedFeatureSegmenter(
        feature_channels=int(config["dino"]["expected_hidden_dim"]),
        decoder_channels=int(training["decoder_channels"]),
        adapter_bottleneck=int(training["adapter_bottleneck"]),
        use_adapter=True,
    )
    adapter.decoder.load_state_dict(common_decoder_initialization)
    adapter_result = _fit_experiment(
        "internvl_adapter", adapter, loaders, loss_fn, config, output_root, device, logger
    )
    result = {
        "attention": attention_report,
        "baseline": baseline_result,
        "internvl_adapter": adapter_result,
    }
    atomic_write_json(output_root / "reports" / "experiments.json", result)
    return result


def _build_loaders(config, output_root, attention_dir):
    image_size = int(config["data"]["image_size"])
    training = config["training"]
    feature_dir = output_root / "feature_cache"
    manifest = output_root / "split_manifest.json"
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = CachedFeatureDataset(
            load_split_manifest(manifest, split), feature_dir, attention_dir, image_size
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=split == "train",
            num_workers=int(training["num_workers"]),
            pin_memory=True,
            persistent_workers=int(training["num_workers"]) > 0,
        )
    return loaders


def _fit_experiment(name, model, loaders, loss_fn, config, output_root, device, logger):
    model = model.to(device)
    training = config["training"]
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_best.pt"
    history = []
    best_miou = -1.0
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in loaders["train"]:
            features = batch["feature"].to(device, non_blocking=True)
            attention = batch["attention"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(features, attention, target.shape[-2:])
                loss, _ = loss_fn(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            batches += 1
        validation = _evaluate(model, loaders["val"], device, float(training["threshold"]))
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / max(batches, 1),
            "validation": validation,
        }
        history.append(epoch_result)
        validation_miou = float(validation["overall"]["miou"])
        logger.info(
            "%s epoch=%d train_loss=%.6f val_mIoU=%.6f val_F1=%.6f",
            name,
            epoch,
            epoch_result["train_loss"],
            validation_miou,
            validation["overall"]["f1"],
        )
        if validation_miou > best_miou:
            best_miou = validation_miou
            torch.save(
                {
                    "experiment": name,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "validation": validation,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = _evaluate(model, loaders["test"], device, float(training["threshold"]))
    visualization_dir = output_root / "visualizations" / name
    _save_prediction_visualizations(
        model,
        loaders["test"],
        device,
        visualization_dir,
        int(training["visualization_samples"]),
    )
    result = {
        "experiment": name,
        "best_epoch": int(checkpoint["epoch"]),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output_root / "reports" / f"{name}_results.json", result)
    logger.info("%s test metrics: %s", name, test_metrics)
    return result


@torch.inference_mode()
def _evaluate(model, loader, device, threshold):
    model.eval()
    metrics = SegmentationMetrics(threshold)
    for batch in loader:
        features = batch["feature"].to(device, non_blocking=True)
        attention = batch["attention"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(features, attention, target.shape[-2:])
        metrics.update(logits, target, list(batch["category"]))
    return metrics.compute()


@torch.inference_mode()
def _save_prediction_visualizations(model, loader, device, destination, maximum):
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        features = batch["feature"].to(device, non_blocking=True)
        attention = batch["attention"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(features, attention, target.shape[-2:])
        predictions = (logits.softmax(dim=1)[:, 1] >= 0.5).cpu().numpy()
        for index in range(predictions.shape[0]):
            image = np.asarray(Image.open(batch["image_path"][index]).convert("RGB"))
            ground_truth = np.asarray(Image.open(batch["mask_path"][index]).convert("L")) > 0
            prediction = np.asarray(
                Image.fromarray(predictions[index].astype(np.uint8) * 255).resize(
                    (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
                )
            ) > 0
            overlay = image.copy()
            overlay[ground_truth] = (
                0.45 * overlay[ground_truth] + 0.55 * np.array([255, 0, 0])
            ).astype(np.uint8)
            overlay[prediction] = (
                0.45 * overlay[prediction] + 0.55 * np.array([0, 255, 0])
            ).astype(np.uint8)
            figure, axes = plt.subplots(1, 4, figsize=(16, 4))
            for axis, value, title in zip(
                axes,
                (image, ground_truth, prediction, overlay),
                ("RGB", "Ground truth", "Prediction", "GT red / prediction green"),
            ):
                axis.imshow(value, cmap="gray" if value.ndim == 2 else None)
                axis.set_title(title)
                axis.axis("off")
            category = {
                "正样本": "Positive",
                "困难正样本": "Hard positive",
                "负样本": "Negative",
            }.get(str(batch["category"][index]), str(batch["category"][index]))
            figure.suptitle(category)
            figure.tight_layout()
            figure.savefig(destination / f"{batch['sample_id'][index]}.png", dpi=140, bbox_inches="tight")
            plt.close(figure)
            saved += 1
            if saved >= maximum:
                return

