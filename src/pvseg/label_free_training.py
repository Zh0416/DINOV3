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
from pvseg.losses import PartialPseudoLoss
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


def train_label_free(config, output_root, logger, device=None):
    if config["protocol"]["label_usage"] != "final_evaluation_only":
        raise ValueError("This entry point only implements final-evaluation-only labels")
    if (output_root / "prediction_seal.json").exists():
        raise RuntimeError("Predictions have been sealed; use a new output directory for another protocol")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    samples = load_rgb_manifest(output_root / "rgb_manifest.json")
    training = config["training"]
    loss_fn = PartialPseudoLoss(training["dice_weight"], training["cross_entropy_weight"])
    seeds = config["ablation"]["seeds"]
    result = {"manual_labels_used": False, "checkpoint_rule": "last epoch, fixed in protocol",
              "threshold": training["threshold"], "runs": []}
    for seed in seeds:
        seed_everything(seed)
        template = CachedFeatureSegmenter(config["dino"]["expected_hidden_dim"], training["decoder_channels"],
                                          training["adapter_bottleneck"], use_adapter=False)
        decoder_initialization = copy.deepcopy(template.decoder.state_dict())
        donors = shuffle_donors(samples, seed)
        atomic_write_json(output_root / "reports" / f"shuffle_seed{seed}.json", donors)
        for name, (use_adapter, pseudo_kind, attention_kind) in VARIANTS.items():
            seed_everything(seed)
            model = CachedFeatureSegmenter(config["dino"]["expected_hidden_dim"], training["decoder_channels"],
                                            training["adapter_bottleneck"], use_adapter=use_adapter).to(device)
            model.decoder.load_state_dict(decoder_initialization)
            mapping = donors if name == "filtered_shuffled" else None
            datasets = {split: PseudoFeatureDataset([s for s in samples if s.split == split], output_root,
                                                     pseudo_kind, attention_kind, mapping)
                        for split in ("train", "test")}
            train_loader = DataLoader(datasets["train"], batch_size=training["batch_size"], shuffle=True,
                                      generator=torch.Generator(), num_workers=training["num_workers"],
                                      pin_memory=device.type == "cuda")
            optimizer = AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            history, started = [], time.perf_counter()
            for epoch in range(1, training["epochs"]+1):
                model.train()
                train_loader.generator.manual_seed(seed+epoch)
                total, batches, valid_pixels = 0., 0, 0
                for batch in train_loader:
                    feature, attention = batch["feature"].to(device), batch["attention"].to(device)
                    # Always train/infer the decoder at 4x patch resolution (256),
                    # then upsample logits. This keeps its refine stage identical.
                    output_size = tuple(d*4 for d in feature.shape[-2:])
                    target = F.interpolate(batch["pseudo_target"][:, None].float(), size=output_size,
                                           mode="nearest")[:, 0].long().to(device)
                    if not (target != 255).any():
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(feature, attention, output_size)
                        loss, components = loss_fn(logits, target)
                        # Equivariance of the trainable head, not a claim that
                        # flipping cached features equals re-extracting DINO.
                        flipped = model(feature.flip(-1), attention.flip(-1), output_size).flip(-1)
                        probability, other = logits.float().softmax(1), flipped.float().softmax(1)
                        consistency = (probability-other).square().mean()
                        loss = loss + float(training["consistency_weight"])*consistency
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    total += float(loss.detach())
                    valid_pixels += components["valid_pixels"]
                    batches += 1
                if not batches:
                    raise RuntimeError(f"{name}: every training pseudo-label was ignored")
                row = {"epoch": epoch, "pseudo_train_loss": total/batches, "valid_pixel_uses": valid_pixels,
                       "optimizer_steps": batches}
                history.append(row)
                logger.info("%s seed=%d epoch=%d/%d pseudo_loss=%.5f steps=%d labels=NONE", name, seed, epoch,
                            training["epochs"], row["pseudo_train_loss"], batches)
            run_name = f"{name}_seed{seed}"
            checkpoint = output_root / "checkpoints" / f"{run_name}_last.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"variant": name, "seed": seed, "epoch": training["epochs"],
                        "manual_labels_used": False, "model_state": model.cpu().state_dict()}, checkpoint)
            model.to(device)
            save_predictions(model, datasets["test"], device, output_root / "predictions" / run_name,
                             config["data"]["image_size"], float(training["threshold"]))
            run = {"variant": name, "seed": seed, "history": history, "checkpoint": str(checkpoint),
                   "elapsed_seconds": time.perf_counter()-started}
            result["runs"].append(run)
            atomic_write_json(output_root / "reports" / "label_free_training.json", result)
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def save_predictions(model, dataset, device, destination, image_size, threshold):
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    for item in dataset:
        feature, attention = item["feature"][None].to(device), item["attention"][None].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(feature, attention, tuple(d*4 for d in feature.shape[-2:]))
        logits = F.interpolate(logits.float(), (image_size, image_size), mode="bilinear", align_corners=False)
        prediction = (logits.softmax(1)[0, 1] >= threshold).cpu().numpy()
        Image.fromarray(prediction.astype(np.uint8)*255).save(destination / f"{item['sample_id']}.png")
