"""Paired, fixed-epoch weak-supervision experiments; no annotation evaluation."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pvseg.config import seed_everything
from pvseg.label_free_data import PseudoFeatureDataset, load_rgb_manifest
from pvseg.losses import IgnoreAwareDiceFocalLoss, PartialPseudoLoss
from pvseg.models import CachedFeatureSegmenter
from pvseg.utils import atomic_write_json


# All choices are fixed before any new evaluation annotation is read.
VARIANTS = {
    "heuristic_decoder": (False, "heuristic", "zero"),
    "heuristic_capacity": (True, "heuristic", "ones"),
    "raw_adapter": (True, "raw", "raw"),
    "filtered_decoder": (False, "filtered", "zero"),
    "filtered_capacity": (True, "filtered", "ones"),
    "filtered_adapter": (True, "filtered", "filtered"),
    "filtered_shuffled": (True, "filtered", "filtered"),
}

SOFT_VARIANTS = {
    # A/B share exactly the same machine targets; B only adds soft semantics.
    "baseline_decoder": (False, "soft", "zero"),
    "semantic_adapter": (True, "soft", "soft"),
    "semantic_capacity": (True, "soft", "base"),
    "shuffled_semantic": (True, "soft", "shuffled"),
}


def variants_for_config(config):
    if config.get("protocol", {}).get("mode") != "soft_semantic":
        return VARIANTS
    requested = config.get("ablation", {}).get("variants")
    if not requested:
        return SOFT_VARIANTS
    unknown = set(requested) - set(SOFT_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown soft-semantic variants: {sorted(unknown)}")
    return {name: SOFT_VARIANTS[name] for name in requested}


def shuffle_donors(samples, seed):
    result = {}
    rng = np.random.default_rng(seed)
    for split in sorted({s.split for s in samples}):
        ids = sorted(s.sample_id for s in samples if s.split == split)
        if len(ids) < 2:
            raise ValueError("Attention shuffle requires at least two images in each split")
        rng.shuffle(ids)
        result.update(dict(zip(ids, ids[1:]+ids[:1])))
    return result


def _optimizer_step(loss, model, optimizer, scaler):
    """Return false on AMP overflow without updating parameters or step counts."""
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        if not scaler.is_enabled():
            raise RuntimeError("Non-finite training gradients")
        # Explicit scale update clears unscale state so the same batch can be
        # recomputed. No optimizer.step is performed on invalid gradients.
        scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        return False
    scaler.step(optimizer)
    scaler.update()
    return True


def train_label_free(config, output_root, logger, device=None):
    if config["protocol"]["label_usage"] != "final_evaluation_only":
        raise ValueError("This entry point only implements final-evaluation-only labels")
    if (output_root / "prediction_seal.json").exists():
        raise RuntimeError("Predictions have been sealed; use a new output directory for another protocol")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    samples = load_rgb_manifest(output_root / "rgb_manifest.json")
    training = config["training"]
    soft_mode = config.get("protocol", {}).get("mode") == "soft_semantic"
    if soft_mode:
        from pvseg.soft_semantic import validate_soft_pseudo_cache
        validate_soft_pseudo_cache(samples, output_root, config["reliability"])
    loss_fn = (IgnoreAwareDiceFocalLoss(training["dice_weight"], training["focal_weight"],
                                        training.get("focal_gamma", 2.0), training.get("focal_alpha", .25))
               if soft_mode else PartialPseudoLoss(training["dice_weight"], training["cross_entropy_weight"]))
    seeds = config["ablation"]["seeds"]
    variants = variants_for_config(config)
    result = {"manual_labels_used": False, "checkpoint_rule": "last epoch, fixed in protocol",
              "threshold": training["threshold"], "variants": list(variants), "runs": []}
    for seed in seeds:
        seed_everything(seed)
        template = CachedFeatureSegmenter(config["dino"]["expected_hidden_dim"], training["decoder_channels"],
                                          training["adapter_bottleneck"], use_adapter=False,
                                          adapter_mode=training.get("adapter_mode", "gated"))
        decoder_initialization = copy.deepcopy(template.decoder.state_dict())
        needs_shuffle = any(name == "filtered_shuffled" or spec[2] == "shuffled" for name, spec in variants.items())
        donors = shuffle_donors(samples, seed) if needs_shuffle else {}
        if needs_shuffle:
            atomic_write_json(output_root / "reports" / f"shuffle_seed{seed}.json", donors)
        for name, (use_adapter, pseudo_kind, attention_kind) in variants.items():
            seed_everything(seed)
            model = CachedFeatureSegmenter(config["dino"]["expected_hidden_dim"], training["decoder_channels"],
                                            training["adapter_bottleneck"], use_adapter=use_adapter,
                                            adapter_mode=training.get("adapter_mode", "gated")).to(device)
            model.decoder.load_state_dict(decoder_initialization)
            mapping = donors if attention_kind == "shuffled" or name == "filtered_shuffled" else None
            datasets = {split: PseudoFeatureDataset([s for s in samples if s.split == split], output_root,
                                                     pseudo_kind, attention_kind, mapping,
                                                     base_attention=config["reliability"].get("base_attention", .2))
                        for split in ("train", "test")}
            train_loader = DataLoader(datasets["train"], batch_size=training["batch_size"], shuffle=True,
                                      generator=torch.Generator(), num_workers=training["num_workers"],
                                      pin_memory=device.type == "cuda")
            optimizer = AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"])
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            history, started = [], time.perf_counter()
            for epoch in range(1, training["epochs"]+1):
                model.train()
                train_loader.generator.manual_seed(seed+epoch)
                total, batches, valid_pixels, overflow_retries = 0., 0, 0, 0
                for batch in train_loader:
                    feature, attention = batch["feature"].to(device), batch["attention"].to(device)
                    pixel_weight = batch.get("pseudo_weight")
                    if pixel_weight is not None:
                        pixel_weight = pixel_weight.to(device)
                    # Always train/infer the decoder at 4x patch resolution (256),
                    # then upsample logits. This keeps its refine stage identical.
                    output_size = ((int(config["data"]["image_size"]), int(config["data"]["image_size"]))
                                   if training.get("decode_at_image_size", False)
                                   else tuple(d*4 for d in feature.shape[-2:]))
                    target = F.interpolate(batch["pseudo_target"][:, None].float(), size=output_size,
                                           mode="nearest")[:, 0].long().to(device)
                    if pixel_weight is not None:
                        pixel_weight = F.interpolate(pixel_weight[:, None].float(), size=output_size,
                                                     mode="bilinear", align_corners=False)[:, 0]
                    if not (target != 255).any():
                        continue
                    for attempt in range(8):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                            logits = model(feature, attention, output_size)
                            loss, components = (loss_fn(logits, target, pixel_weight)
                                                if soft_mode else loss_fn(logits, target))
                            # Equivariance of the trainable head, not a claim that
                            # flipping cached features equals re-extracting DINO.
                            flipped = model(feature.flip(-1), attention.flip(-1), output_size).flip(-1)
                            probability, other = logits.float().softmax(1), flipped.float().softmax(1)
                            consistency = (probability-other).square().mean()
                            loss = loss + float(training["consistency_weight"])*consistency
                        if _optimizer_step(loss, model, optimizer, scaler):
                            break
                        overflow_retries += 1
                    else:
                        raise RuntimeError(f"{name}: persistent AMP gradient overflow at epoch {epoch}")
                    total += float(loss.detach())
                    valid_pixels += components["valid_pixels"]
                    batches += 1
                scheduler.step()
                if not batches:
                    raise RuntimeError(f"{name}: every training pseudo-label was ignored")
                row = {"epoch": epoch, "pseudo_train_loss": total/batches, "valid_pixel_uses": valid_pixels,
                       "optimizer_steps": batches, "amp_overflow_retries": overflow_retries}
                history.append(row)
                logger.info("%s seed=%d epoch=%d/%d pseudo_loss=%.5f steps=%d", name, seed, epoch,
                            training["epochs"], row["pseudo_train_loss"], batches)
            run_name = f"{name}_seed{seed}"
            checkpoint = output_root / "checkpoints" / f"{run_name}_last.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"variant": name, "seed": seed, "epoch": training["epochs"],
                        "manual_labels_used": False, "model_state": model.cpu().state_dict()}, checkpoint)
            model.to(device)
            save_predictions(model, datasets["test"], device, output_root / "predictions" / run_name,
                             config["data"]["image_size"], float(training["threshold"]),
                             decode_at_image_size=training.get("decode_at_image_size", False))
            run = {"variant": name, "seed": seed, "history": history, "checkpoint": str(checkpoint),
                   "elapsed_seconds": time.perf_counter()-started}
            result["runs"].append(run)
            atomic_write_json(output_root / "reports" / "label_free_training.json", result)
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def save_predictions(model, dataset, device, destination, image_size, threshold, decode_at_image_size=False):
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    for item in dataset:
        feature, attention = item["feature"][None].to(device), item["attention"][None].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output_size = (image_size, image_size) if decode_at_image_size else tuple(d*4 for d in feature.shape[-2:])
            logits = model(feature, attention, output_size)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite prediction logits: {item['sample_id']}")
        logits = F.interpolate(logits.float(), (image_size, image_size), mode="bilinear", align_corners=False)
        prediction = (logits.softmax(1)[0, 1] >= threshold).cpu().numpy()
        Image.fromarray(prediction.astype(np.uint8)*255).save(destination / f"{item['sample_id']}.png")
