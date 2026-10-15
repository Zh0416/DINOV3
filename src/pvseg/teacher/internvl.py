from __future__ import annotations

import gc
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
from PIL import Image

from pvseg.attention import sanitize_boxes
from pvseg.utils import model_reference


def _probability(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    if result > 1.0:
        result /= 100.0
    return max(0.0, min(1.0, result))


def parse_teacher_response(raw: str, image_size: int = 1024) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    warning = None
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        warning = "fallback_parser"
        exists_match = re.search(r"(?:pv_exist|存在|光伏).{0,20}(true|false|是|否)", text, re.I)
        confidence_match = re.search(r"(?:confidence|置信度).{0,20}(\d+(?:\.\d+)?)", text, re.I)
        bbox_match = re.search(r"bbox.{0,10}(\[.*?\])", text, re.I | re.DOTALL)
        bbox: Any = []
        if bbox_match:
            try:
                bbox = json.loads(bbox_match.group(1))
            except json.JSONDecodeError:
                bbox = []
        value = {
            "pv_exist": bool(exists_match and exists_match.group(1).lower() in {"true", "是"}),
            "confidence": confidence_match.group(1) if confidence_match else 0.0,
            "bbox": bbox,
            "reason": text[:500],
        }
    if not isinstance(value, dict):
        value = {}
        warning = "not_json_object"
    raw_exists = value.get("pv_exist", value.get("pv_exists", False))
    if isinstance(raw_exists, str):
        exists = raw_exists.strip().lower() in {"true", "1", "yes", "是"}
    else:
        exists = bool(raw_exists)
    confidence = _probability(value.get("confidence", value.get("image_confidence", value.get("pv_probability", 0.0))))
    boxes = sanitize_boxes(value.get("bbox", value.get("bboxes", [])), image_size) if exists else []
    grid = []
    raw_grid = value.get("grid", [])
    if isinstance(raw_grid, list):
        # Accept either a flat list of {row,col,...} objects or a nested grid.
        if raw_grid and all(isinstance(row, list) for row in raw_grid):
            for row_index, row in enumerate(raw_grid):
                for col_index, item in enumerate(row):
                    if isinstance(item, dict):
                        item = {**item, "row": row_index, "col": col_index}
                    if isinstance(item, dict):
                        grid.append({
                            "row": row_index, "col": col_index,
                            "pv_probability": _probability(item.get("pv_probability", item.get("confidence", 0.0))),
                            "status": str(item.get("status", "unknown")).lower()
                            if str(item.get("status", "unknown")).lower() in {"positive", "negative", "unknown"}
                            else "unknown",
                            "reason": str(item.get("reason", ""))[:300],
                        })
        else:
            for item in raw_grid:
                if not isinstance(item, dict):
                    continue
                try:
                    row, col = int(item["row"]), int(item["col"])
                except (KeyError, TypeError, ValueError):
                    continue
                grid.append({
                    "row": row, "col": col,
                    "pv_probability": _probability(item.get("pv_probability", item.get("confidence", 0.0))),
                    "status": str(item.get("status", "unknown")).lower()
                    if str(item.get("status", "unknown")).lower() in {"positive", "negative", "unknown"}
                    else "unknown",
                    "reason": str(item.get("reason", ""))[:300],
                })
    result = {
        "pv_exist": exists,
        "confidence": confidence,
        "bbox": boxes,
        "grid": grid,
        "confusion_negatives": value.get("confusion_negatives", []),
        "reason": str(value.get("reason", ""))[:1000],
        "raw_response": raw,
    }
    if warning:
        result["parse_warning"] = warning
    if exists and not boxes:
        result["parse_warning"] = "pv_exists_without_valid_bbox"
    return result


def _dynamic_preprocess(image: Image.Image, image_size: int, max_patches: int) -> list[Image.Image]:
    width, height = image.size
    aspect_ratio = width / height
    layouts = {(1, 1)}
    for rows in range(1, max_patches + 1):
        for columns in range(1, max_patches + 1):
            if 1 <= rows * columns <= max_patches:
                layouts.add((rows, columns))
    # Official area tie-break: choose extra tiles only when the source has enough
    # pixels. Ordering a set previously made equal-aspect-ratio choices arbitrary.
    rows, columns = 1, 1
    best_difference = float("inf")
    for candidate in sorted(layouts, key=lambda layout: (layout[0]*layout[1], layout)):
        difference = abs(aspect_ratio - candidate[1] / candidate[0])
        if difference < best_difference or (
            difference == best_difference
            and width*height > .5 * image_size**2 * candidate[0]*candidate[1]
        ):
            rows, columns = candidate
            best_difference = difference
    resized = image.resize((image_size * columns, image_size * rows), Image.Resampling.BICUBIC)
    return [
        resized.crop(
            (
                column * image_size,
                row * image_size,
                (column + 1) * image_size,
                (row + 1) * image_size,
            )
        )
        for row in range(rows)
        for column in range(columns)
    ]


def _pixel_values(image_path: Union[str, Path], model: Any, max_patches: int,
                  view: str = "original", crop_box: Optional[list[int]] = None) -> tuple[torch.Tensor, int]:
    from torchvision import transforms

    image_size = int(getattr(model.config, "force_image_size", 448) or 448)
    image = Image.open(image_path).convert("RGB")
    if crop_box is not None:
        left, top, right, bottom = crop_box
        scale_x, scale_y = image.width / 1024.0, image.height / 1024.0
        image = image.crop((
            int(max(0, left) * scale_x), int(max(0, top) * scale_y),
            int(min(1024, right) * scale_x), int(min(1024, bottom) * scale_y),
        ))
    if view == "hflip":
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif view != "original":
        raise ValueError(f"Unknown teacher view: {view}")
    tiles = _dynamic_preprocess(image, image_size, max_patches)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    values = torch.stack([transform(tile) for tile in tiles])
    device = next(model.parameters()).device
    dtype = next(
        (parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()),
        torch.float16,
    )
    return values.to(device=device, dtype=dtype), len(tiles)


@dataclass
class InternVLTeacher:
    model: Any
    tokenizer: Any
    config: dict[str, Any]

    @classmethod
    def load(cls, config: dict[str, Any], logger: logging.Logger) -> "InternVLTeacher":
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        from transformers.modeling_utils import PreTrainedModel

        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            def get_tied_weights(model):
                return (
                    getattr(model, "_all_tied_weights_keys", None)
                    or getattr(model, "_tied_weights_keys", None)
                    or {}
                )

            def set_tied_weights(model, value):
                model._all_tied_weights_keys = value

            PreTrainedModel.all_tied_weights_keys = property(get_tied_weights, set_tied_weights)

        reference, local_only = model_reference(config)
        token = None if local_only else __import__("os").getenv("HF_TOKEN")
        common = {
            "trust_remote_code": True,
            "local_files_only": local_only,
            "token": token,
        }
        tokenizer = AutoTokenizer.from_pretrained(
            reference, use_fast=False, fix_mistral_regex=True, **common
        )
        dtype = torch.float16
        if str(config.get("dtype", "float16")).lower() == "bfloat16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        requested = str(config.get("quantization", "auto")).lower()
        attempts = ["8bit", "4bit", "none"] if requested == "auto" else [requested]
        last_error: Optional[Exception] = None
        for quantization in attempts:
            kwargs: dict[str, Any] = {
                **common,
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "device_map": "auto",
            }
            if quantization == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            try:
                logger.info("Loading InternVL from %s with quantization=%s", reference, quantization)
                model = AutoModel.from_pretrained(reference, **kwargs).eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                return cls(model=model, tokenizer=tokenizer, config=config)
            except (RuntimeError, ValueError, ImportError, OSError, AttributeError) as error:
                last_error = error
                logger.exception("InternVL load failed with quantization=%s", quantization)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        raise RuntimeError(f"Could not load InternVL teacher: {last_error}")

    @torch.inference_mode()
    def infer(self, image_path: Union[str, Path], view: str = "original",
              crop_box: Optional[list[int]] = None, prompt: Optional[str] = None) -> dict[str, Any]:
        values, patch_count = _pixel_values(
            image_path, self.model, int(self.config.get("max_dynamic_patches", 6)), view, crop_box
        )
        generation = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 256)),
            "do_sample": False,
            "temperature": 0.0,
        }
        raw = self.model.chat(
            self.tokenizer,
            values,
            prompt or self.config["prompt"],
            generation,
            num_patches_list=[patch_count],
            verbose=False,
        )
        result = parse_teacher_response(raw, image_size=1024)
        result["patch_count"] = patch_count
        if crop_box is not None:
            result["crop_box"] = list(crop_box)
        result["flipped"] = view == "hflip"
        del values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

