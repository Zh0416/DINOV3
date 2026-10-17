"""Frozen CLIP crop semantics; region diagnostics, never pixel training labels."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from pvseg.utils import model_reference

UNKNOWN, NEGATIVE, POSITIVE = -1, 0, 1


def validate_settings(settings):
    for key in ("window_size", "stride", "batch_size", "max_windows_per_view"):
        value = settings[key]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["stride"] > settings["window_size"]:
        raise ValueError("stride must not exceed window_size")
    if not -1 <= settings["min_similarity"] <= 1 or not 0 < settings["min_margin"] <= 2:
        raise ValueError("Invalid cosine similarity/margin")
    if not 0 <= settings["max_pair_conflict"] <= 1:
        raise ValueError("Invalid pair conflict bound")
    groups = {"photovoltaic": settings["foreground"], **settings["backgrounds"]}
    if len(groups) < 2 or "photovoltaic" in settings["backgrounds"]:
        raise ValueError("Need foreground and distinct explicit background groups")
    for prompts in groups.values():
        if not isinstance(prompts, list) or not prompts or any(
                not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("Prompt groups must contain nonempty strings")
    return groups


def symmetric_starts(length, window, stride):
    """Integer positions closed under reflection, including both edges."""
    if not 0 < window <= length or not 0 < stride <= window:
        raise ValueError("Invalid window geometry")
    span = length - window
    count = max(1, math.ceil(span / stride))
    starts = {round(i * span / count) for i in range(count + 1)}
    return sorted(starts | {span - start for start in starts})


def window_boxes(width, height, settings):
    if min(width, height) < 1 or width * height > 4_194_304:
        raise ValueError("Probe RGB must have 1..4194304 pixels")
    size = min(int(settings["window_size"]), width, height)
    stride = min(int(settings["stride"]), size)
    boxes = [(x, y, x + size, y + size)
             for y in symmetric_starts(height, size, stride)
             for x in symmetric_starts(width, size, stride)]
    if len(boxes) > settings["max_windows_per_view"]:
        raise ValueError("Window budget exceeded; do not silently change the frozen probe")
    return boxes


def classify_windows(scores, settings):
    """Require absolute similarity AND a winning margin over competing concepts."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError("Invalid CLIP cosine scores")
    if np.any(np.abs(scores) > 1.0001):
        raise ValueError("CLIP scores must be normalized cosines")
    foreground, background = scores[:, 0], scores[:, 1:].max(axis=1)
    margin = foreground - background
    state = np.full(len(scores), UNKNOWN, np.int8)
    state[(foreground >= settings["min_similarity"]) & (margin >= settings["min_margin"])] = POSITIVE
    state[(background >= settings["min_similarity"]) & (-margin >= settings["min_margin"])] = NEGATIVE
    return state, margin


def rasterize_regions(width, height, boxes, states):
    """Broadcast REGION votes solely for diagnostic coverage, not occupancy labels."""
    positive = np.zeros((height, width), bool)
    negative = np.zeros_like(positive)
    if len(boxes) != len(states):
        raise ValueError("Window/state length mismatch")
    for (x0, y0, x1, y1), state in zip(boxes, states):
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Out-of-bounds crop")
        if state not in (UNKNOWN, NEGATIVE, POSITIVE):
            raise ValueError("Invalid region state")
        if state == POSITIVE:
            positive[y0:y1, x0:x1] = True
        elif state == NEGATIVE:
            negative[y0:y1, x0:x1] = True
    result = np.full((height, width), UNKNOWN, np.int8)
    result[positive & ~negative] = POSITIVE
    result[negative & ~positive] = NEGATIVE
    return result, positive & negative


def state_summary(state):
    return {
        "positive_region_coverage": float(np.mean(state == POSITIVE)),
        "negative_region_coverage": float(np.mean(state == NEGATIVE)),
        "valid_region_evidence_coverage": float(np.mean(state != UNKNOWN)),
        "no_positive_regions": bool(not np.any(state == POSITIVE)),
        "all_positive_regions": bool(np.all(state == POSITIVE)),
        "all_negative_regions": bool(np.all(state == NEGATIVE)),
        "all_unknown": bool(np.all(state == UNKNOWN)),
    }


def diagnose_pair(original, flipped):
    """Input second view in flipped coordinates; require both views for evidence."""
    if original.shape != flipped.shape or original.ndim != 2:
        raise ValueError("View map shape mismatch")
    aligned = flipped[:, ::-1]
    joint = (original != UNKNOWN) & (aligned != UNKNOWN)
    conflict = joint & (original != aligned)
    consensus = np.full(original.shape, UNKNOWN, np.int8)
    agree = joint & (original == aligned)
    consensus[agree] = original[agree]
    union = (original == POSITIVE) | (aligned == POSITIVE)
    intersection = (original == POSITIVE) & (aligned == POSITIVE)
    report = {
        "transform_three_state_agreement": float(np.mean(original == aligned)),
        "joint_valid_coverage": float(np.mean(joint)),
        "joint_valid_agreement": float(np.mean(original[joint] == aligned[joint])) if joint.any() else None,
        "joint_valid_conflict_fraction": float(np.mean(conflict[joint])) if joint.any() else None,
        "conflict_fraction": float(np.mean(conflict)),
        "positive_region_iou": float(intersection.sum() / union.sum()) if union.any() else None,
        "consensus": state_summary(consensus),
    }
    return report, consensus


def diagnose_window_pair(original, flipped):
    """Compare identical crop locations after reflection, before region broadcasting.

    Different overlapping windows can create pixel-region consensus even when
    no same-location window repeats a known decision. Report both levels.
    """
    size = original.get("rgb_size")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(v) is not int or v < 1 for v in size)
            or flipped.get("rgb_size") != size):
        raise ValueError("Window pair RGB size mismatch or invalid size")
    width, height = size

    def index_windows(view, reflect):
        windows = view.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError("Window pair needs nonempty window lists")
        indexed = {}
        for window in windows:
            box = window.get("box_xyxy")
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(type(v) is not int for v in box)):
                raise ValueError("Invalid window coordinates")
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("Out-of-bounds window coordinates")
            label = window.get("state")
            margin = window.get("foreground_minus_best_background")
            if type(label) is not int or label not in (UNKNOWN, NEGATIVE, POSITIVE):
                raise ValueError("Invalid window state")
            if (isinstance(margin, bool) or not isinstance(margin, (int, float))
                    or not math.isfinite(margin) or abs(margin) > 2.0002):
                raise ValueError("Invalid window cosine margin")
            key = (width - x1, y0, width - x0, y1) if reflect else tuple(box)
            if key in indexed:
                raise ValueError("Duplicate window coordinates")
            indexed[key] = (label, margin)
        return indexed

    first, second = index_windows(original, False), index_windows(flipped, True)
    if first.keys() != second.keys():
        raise ValueError("Window coordinate sets differ after horizontal alignment")
    keys = sorted(first)
    states = np.array([[first[k][0], second[k][0]] for k in keys], dtype=np.int8)
    margins = np.array([[first[k][1], second[k][1]] for k in keys], dtype=np.float64)
    joint = np.all(states != UNKNOWN, axis=1)
    agree = states[:, 0] == states[:, 1]
    change = np.abs(margins[:, 0] - margins[:, 1])
    return {
        "matched_windows": len(keys),
        "transform_three_state_agreement": float(np.mean(agree)),
        "joint_valid_windows": int(joint.sum()),
        "joint_valid_fraction": float(np.mean(joint)),
        "joint_valid_agreement": float(np.mean(agree[joint])) if joint.any() else None,
        "joint_valid_conflict_fraction": float(np.mean(~agree[joint])) if joint.any() else None,
        "repeat_positive_windows": int(np.all(states == POSITIVE, axis=1).sum()),
        "repeat_negative_windows": int(np.all(states == NEGATIVE, axis=1).sum()),
        "known_unknown_changed_windows": int(np.logical_xor(states[:, 0] == UNKNOWN,
                                                            states[:, 1] == UNKNOWN).sum()),
        "mean_absolute_margin_change": float(change.mean()),
        "max_absolute_margin_change": float(change.max()),
    }


def bounded_decision(rows, max_conflict):
    """Only a gate for further inspection. No automatic pseudo-label or training run."""
    reasons = []
    if not rows:
        reasons.append("no_records")
    if any(row["pair"]["joint_valid_conflict_fraction"] is not None
           and row["pair"]["joint_valid_conflict_fraction"] > max_conflict for row in rows):
        reasons.append("transform_conflict")
    if any(row["pair"]["consensus"]["valid_region_evidence_coverage"] == 0 for row in rows):
        reasons.append("no_repeatable_region_evidence_for_a_sample")
    if any(view["regions"]["all_unknown"] for row in rows for view in row["views"].values()):
        reasons.append("all_unknown_view")
    if any(view["regions"]["all_positive_regions"] or view["regions"]["all_negative_regions"]
           for row in rows for view in row["views"].values()):
        reasons.append("uniform_region_output_requires_review")
    if not any(row["pair"]["consensus"]["positive_region_coverage"] > 0 for row in rows):
        reasons.append("no_repeatable_positive_region_evidence")
    if not any(row["pair"]["consensus"]["negative_region_coverage"] > 0 for row in rows):
        reasons.append("no_repeatable_explicit_background_evidence")
    return {
        "eligible_for_bounded_followup": not reasons,
        "reasons": reasons,
        "decision": "inspect_regions_before_any_pseudo_label_design" if not reasons else "stop_and_review_diagnostic",
        "accuracy_established": False,
    }


def save_region_preview(image, original, flipped, consensus, destination):
    """RGB/region overlays for inspection; deliberately no label-mask export."""
    image = image.convert("RGB")
    base = np.asarray(image)
    panels = [("RGB", image)]
    for title, states in (("Original regions", original),
                          ("Flipped regions (aligned)", flipped[:, ::-1]),
                          ("Two-view consensus", consensus)):
        color = np.full_like(base, 128)
        color[states == POSITIVE] = [255, 80, 30]
        color[states == NEGATIVE] = [20, 140, 255]
        overlay = Image.fromarray((base.astype(np.float32) * .55 + color * .45).astype(np.uint8))
        panels.append((title, overlay))
    target_width = min(384, image.width)
    target_height = max(1, round(image.height * target_width / image.width))
    canvas = Image.new("RGB", (target_width * len(panels), target_height + 54), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 3), "REGION EVIDENCE ONLY | orange=PV, blue=background, gray=unknown | NOT pixel labels", fill="black")
    for index, (title, panel) in enumerate(panels):
        x = index * target_width
        draw.text((x + 5, 27), title, fill="black")
        canvas.paste(panel.resize((target_width, target_height)), (x, 54))
    canvas.save(destination)


class FrozenCLIPWindowTeacher:
    def __init__(self, model, processor, settings, runtime_spec):
        self.model, self.processor, self.settings = model, processor, settings
        self.groups = validate_settings(settings)
        self.class_names = list(self.groups)
        self.runtime_spec = runtime_spec
        self.model.eval().requires_grad_(False)
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        with torch.inference_mode():
            tokens = processor(text=[p for prompts in self.groups.values() for p in prompts],
                               return_tensors="pt", padding=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            # Explicit projection also works across differing get_*_features APIs.
            embeddings = model.text_projection(model.text_model(**tokens).pooler_output).float()
            embeddings = self._normalize(embeddings)
            classes, offset = [], 0
            for prompts in self.groups.values():
                classes.append(embeddings[offset:offset + len(prompts)].mean(0))
                offset += len(prompts)
            self.text_features = self._normalize(torch.stack(classes))

    @staticmethod
    def _normalize(features):
        norm = features.norm(dim=-1, keepdim=True)
        if not torch.isfinite(features).all() or torch.any(norm <= 1e-12):
            raise ValueError("Nonfinite/zero CLIP embeddings")
        return features / norm

    @classmethod
    def load(cls, settings, cache_dir):
        from transformers import CLIPModel, CLIPProcessor
        import transformers
        reference, local = model_reference(settings)
        device = settings.get("device", "cuda")
        if device not in ("cuda", "cpu") or (device == "cuda" and not torch.cuda.is_available()):
            raise RuntimeError("Requested CLIP device unavailable")
        dtype_name = settings.get("dtype", "float32")
        if dtype_name not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported CLIP dtype")
        if device == "cpu" and dtype_name != "float32":
            raise ValueError("Use float32 for the CPU diagnostic")
        kwargs = {"local_files_only": local, "cache_dir": str(cache_dir)}
        if not local:
            kwargs["revision"] = settings["revision"]
        processor = CLIPProcessor.from_pretrained(reference, **kwargs)
        model = CLIPModel.from_pretrained(reference, torch_dtype=getattr(torch, dtype_name), **kwargs)
        model.to(device)
        # Local directories are recorded but not falsely certified as the pinned remote bytes.
        runtime = {
            "model_reference": reference, "local_files_only": local,
            "requested_remote_revision": None if local else settings["revision"],
            "resolved_commit": getattr(model.config, "_commit_hash", None),
            "local_checkpoint_bytes_verified": False,
            "dtype": str(next(model.parameters()).dtype), "device": device,
            "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
            "processor": processor.image_processor.to_dict(),
            "tokenizer_class": type(processor.tokenizer).__name__,
            "tokenizer_max_length": processor.tokenizer.model_max_length,
            "model_config": model.config.to_dict(),
            "score_semantics": "normalized CLIP cosine; no softmax, no calibrated probability",
        }
        return cls(model, processor, settings, runtime)

    def infer(self, image):
        started = time.perf_counter()
        image = image.convert("RGB")
        width, height = image.size
        boxes = window_boxes(width, height, self.settings)
        scores = []
        batch_size = int(self.settings["batch_size"])
        with torch.inference_mode():
            for start in range(0, len(boxes), batch_size):
                crops = [image.crop(box) for box in boxes[start:start + batch_size]]
                batch = self.processor(images=crops, return_tensors="pt")["pixel_values"]
                embeddings = self.model.visual_projection(self.model.vision_model(
                    pixel_values=batch.to(device=self.device, dtype=self.dtype)).pooler_output).float()
                embeddings = self._normalize(embeddings)
                scores.append((embeddings @ self.text_features.T).cpu().numpy())
        scores = np.concatenate(scores)
        states, margins = classify_windows(scores, self.settings)
        state, conflict = rasterize_regions(width, height, boxes, states)
        windows = [{"box_xyxy": list(box), "state": int(label),
                    "foreground_minus_best_background": float(margin),
                    "cosine_scores": dict(zip(self.class_names, map(float, score)))}
                   for box, label, margin, score in zip(boxes, states, margins, scores)]
        return {
            "status": "ok", "rgb_size": [width, height], "windows": windows,
            "regions": state_summary(state),
            "overlap_conflict_fraction": float(np.mean(conflict)),
            "elapsed_seconds": time.perf_counter() - started,
        }, state
